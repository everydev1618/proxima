// GGUF metadata + tensor-table reader (stdlib only). Port of hermes-agent's
// local_runtime/gguf.py.
//
// Reads the header only (metadata + tensor infos); never touches tensor data,
// so it is fast enough to run at picker time on multi-GB files.

package localrt

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// Split GGUF naming: "<stem>-00001-of-00003.gguf"; the part suffix is not
// part of the model id.
var (
	SplitPartRE   = regexp.MustCompile(`-(\d{5})-of-(\d{5})\.gguf$`)
	partSuffixRE  = regexp.MustCompile(`-\d{5}-of-\d{5}$`)
	ggufMagic     = [4]byte{'G', 'G', 'U', 'F'}
	errNotGGUF    = fmt.Errorf("not a GGUF file")
	maxHeaderKV   = uint64(1 << 20) // sanity bound on counts in a hostile header
	maxHeaderTens = uint64(1 << 24)
)

// ModelIDFromStem returns the model id from a GGUF file stem (split-part
// suffix stripped).
func ModelIDFromStem(stem string) string {
	return partSuffixRE.ReplaceAllString(stem, "")
}

// ggml tensor type sizes: type_id -> (block_bytes, block_elems). IQ-family
// verified against ggml-common.h.
var ggmlTypeSizes = map[uint32][2]int64{
	0: {4, 1}, 1: {2, 1}, 2: {18, 32}, 3: {20, 32}, 6: {22, 32}, 7: {24, 32},
	8: {34, 32}, 9: {36, 32}, 10: {84, 256}, 11: {110, 256}, 12: {144, 256},
	13: {176, 256}, 14: {210, 256}, 15: {292, 256}, 16: {66, 256},
	17: {74, 256}, 18: {98, 256}, 19: {50, 256}, 20: {18, 32},
	21: {110, 256}, 22: {82, 256}, 23: {136, 256}, 24: {1, 1}, 25: {2, 1},
	26: {4, 1}, 27: {8, 1}, 28: {8, 1}, 29: {56, 256}, 30: {2, 1},
}

// GGUF metadata value types.
const (
	vUint8 = iota
	vInt8
	vUint16
	vInt16
	vUint32
	vInt32
	vFloat32
	vBool
	vString
	vArray
	vUint64
	vInt64
	vFloat64
)

// general.sampling.* metadata key -> preset INI key.
var samplingINIKey = map[string]string{
	"temp": "temp", "temperature": "temp", "top_p": "top-p",
	"top_k": "top-k", "min_p": "min-p",
	"repeat_penalty": "repeat-penalty", "presence_penalty": "presence-penalty",
}

// GGUFHeader is the parsed header of one GGUF file.
type GGUFHeader struct {
	Path           string
	Version        uint32
	Metadata       map[string]any
	NTensors       uint64
	TensorBytes    int64 // exact sum over the tensor table
	EmbdTableBytes int64 // token_embd.weight (duplicated host-side when fully offloaded)
}

// ── typed accessors ──────────────────────────────────────────

func (h *GGUFHeader) Architecture() string {
	s, _ := h.Metadata["general.architecture"].(string)
	return s
}

func (h *GGUFHeader) archKey(suffix string) any {
	return h.Metadata[h.Architecture()+"."+suffix]
}

func (h *GGUFHeader) archInt(suffix string) int {
	return toInt(h.archKey(suffix))
}

func (h *GGUFHeader) NLayer() int         { return h.archInt("block_count") }
func (h *GGUFHeader) NCtxTrain() int      { return h.archInt("context_length") }
func (h *GGUFHeader) NEmbd() int          { return h.archInt("embedding_length") }
func (h *GGUFHeader) SlidingWindow() int  { return h.archInt("attention.sliding_window") }
func (h *GGUFHeader) ExpertCount() int    { return h.archInt("expert_count") }

// FullAttentionInterval is the GDN-hybrid discriminator (qwen35 family):
// every Nth layer is full attention, the rest are linear/recurrent. 0 = not
// present.
func (h *GGUFHeader) FullAttentionInterval() int { return h.archInt("full_attention_interval") }

// NVocab is the vocabulary size (prices the GPU logits buffers): vocab_size
// metadata when present, else the tokenizer list length.
func (h *GGUFHeader) NVocab() int {
	if v := h.archInt("vocab_size"); v > 0 {
		return v
	}
	if toks, ok := h.Metadata["tokenizer.ggml.tokens"].([]any); ok {
		return len(toks)
	}
	return 0
}

func (h *GGUFHeader) NHead() int {
	v := h.archKey("attention.head_count")
	if list, ok := v.([]any); ok {
		best := 0
		for _, x := range list {
			if n := toInt(x); n > best {
				best = n
			}
		}
		return best
	}
	return toInt(v)
}

// HeadCountsKV returns per-layer KV head counts; 0 marks a recurrent/linear
// layer (n_head_kv == 0).
//
// Three GGUF shapes: a per-layer array (nemotron_h_moe) is used as-is; a
// scalar plus full_attention_interval (qwen35) applies to every N-th layer
// (1-indexed) and is zero elsewhere — pricing all layers as attention was a
// 4x overestimate; a plain scalar (dense) broadcasts to every layer.
func (h *GGUFHeader) HeadCountsKV() []int {
	v := h.archKey("attention.head_count_kv")
	if list, ok := v.([]any); ok {
		out := make([]int, len(list))
		for i, x := range list {
			out[i] = toInt(x)
		}
		return out
	}
	scalar := toInt(v)
	n := h.NLayer()
	out := make([]int, n)
	if interval := h.FullAttentionInterval(); interval > 1 {
		for i := range out {
			if (i+1)%interval == 0 {
				out[i] = scalar
			}
		}
		return out
	}
	for i := range out {
		out[i] = scalar
	}
	return out
}

func (h *GGUFHeader) HeadDimK() int {
	if v := h.archInt("attention.key_length"); v > 0 {
		return v
	}
	if n := h.NHead(); n > 0 {
		return h.NEmbd() / n
	}
	return 0
}

func (h *GGUFHeader) HeadDimV() int {
	if v := h.archInt("attention.value_length"); v > 0 {
		return v
	}
	return h.HeadDimK()
}

// SamplingDefaults returns upstream's recommended sampling as preset INI
// keys, when the file carries it. Publishers bake general.sampling.* keys
// into the GGUF (llama-server reads them as that model's defaults), so the
// file is the source of truth. Empty if absent.
func (h *GGUFHeader) SamplingDefaults() map[string]string {
	out := map[string]string{}
	for key, value := range h.Metadata {
		if !strings.HasPrefix(key, "general.sampling.") {
			continue
		}
		short := key[strings.LastIndex(key, ".")+1:]
		name, ok := samplingINIKey[short]
		if !ok {
			continue
		}
		var num float64
		switch v := value.(type) {
		case float64:
			num = v
		case int64:
			num = float64(v)
		case uint64:
			num = float64(v)
		default:
			continue
		}
		num = math.Round(num*10000) / 10000
		if num == math.Trunc(num) {
			out[name] = strconv.Itoa(int(num))
		} else {
			out[name] = strconv.FormatFloat(num, 'f', -1, 64)
		}
	}
	return out
}

// toInt converts any GGUF scalar to int (0 for non-numeric).
func toInt(v any) int {
	switch x := v.(type) {
	case uint64:
		return int(x)
	case int64:
		return int(x)
	case float64:
		return int(x)
	case bool:
		if x {
			return 1
		}
		return 0
	}
	return 0
}

// ── reader ───────────────────────────────────────────────────

type ggufReader struct {
	r   io.Reader
	err error
}

func (g *ggufReader) read(v any) {
	if g.err == nil {
		g.err = binary.Read(g.r, binary.LittleEndian, v)
	}
}

func (g *ggufReader) u32() uint32 { var v uint32; g.read(&v); return v }
func (g *ggufReader) u64() uint64 { var v uint64; g.read(&v); return v }

func (g *ggufReader) str() string {
	n := g.u64()
	if g.err != nil {
		return ""
	}
	if n > 1<<24 {
		g.err = fmt.Errorf("gguf string length %d exceeds sanity bound", n)
		return ""
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(g.r, buf); err != nil {
		g.err = err
		return ""
	}
	return string(buf)
}

func (g *ggufReader) value(vtype uint32) any {
	if g.err != nil {
		return nil
	}
	switch vtype {
	case vString:
		return g.str()
	case vArray:
		etype, n := g.u32(), g.u64()
		if g.err != nil {
			return nil
		}
		if n > maxHeaderKV {
			g.err = fmt.Errorf("gguf array length %d exceeds sanity bound", n)
			return nil
		}
		out := make([]any, 0, n)
		for i := uint64(0); i < n && g.err == nil; i++ {
			out = append(out, g.value(etype))
		}
		return out
	case vUint8:
		var v uint8
		g.read(&v)
		return uint64(v)
	case vInt8:
		var v int8
		g.read(&v)
		return int64(v)
	case vUint16:
		var v uint16
		g.read(&v)
		return uint64(v)
	case vInt16:
		var v int16
		g.read(&v)
		return int64(v)
	case vUint32:
		var v uint32
		g.read(&v)
		return uint64(v)
	case vInt32:
		var v int32
		g.read(&v)
		return int64(v)
	case vFloat32:
		var v float32
		g.read(&v)
		return float64(v)
	case vBool:
		var v uint8
		g.read(&v)
		return v != 0
	case vUint64:
		return g.u64()
	case vInt64:
		var v int64
		g.read(&v)
		return v
	case vFloat64:
		var v float64
		g.read(&v)
		return v
	}
	g.err = fmt.Errorf("unknown gguf value type %d", vtype)
	return nil
}

// ReadGGUFHeader parses the header (metadata + tensor table) of a GGUF file.
func ReadGGUFHeader(path string) (*GGUFHeader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return nil, err
	}
	if magic != ggufMagic {
		return nil, fmt.Errorf("%w: %s", errNotGGUF, path)
	}

	g := &ggufReader{r: f}
	version := g.u32()
	nTensors := g.u64()
	nKV := g.u64()
	if g.err == nil && (nKV > maxHeaderKV || nTensors > maxHeaderTens) {
		return nil, fmt.Errorf("gguf header counts out of range in %s", path)
	}

	metadata := make(map[string]any, nKV)
	for i := uint64(0); i < nKV && g.err == nil; i++ {
		key := g.str()
		vtype := g.u32()
		metadata[key] = g.value(vtype)
	}

	var tensorBytes, embdBytes int64
	for i := uint64(0); i < nTensors && g.err == nil; i++ {
		name := g.str()
		nDims := g.u32()
		if g.err != nil {
			break
		}
		if nDims > 8 {
			return nil, fmt.Errorf("gguf tensor with %d dims in %s", nDims, path)
		}
		elems := int64(1)
		for d := uint32(0); d < nDims; d++ {
			elems *= int64(g.u64())
		}
		ttype := g.u32()
		g.u64() // offset
		if g.err != nil {
			break
		}
		size, ok := ggmlTypeSizes[ttype]
		if !ok {
			return nil, fmt.Errorf("unknown ggml tensor type %d in %s", ttype, path)
		}
		nbytes := (elems / size[1]) * size[0]
		tensorBytes += nbytes
		if name == "token_embd.weight" {
			embdBytes = nbytes
		}
	}
	if g.err != nil {
		return nil, fmt.Errorf("reading gguf header of %s: %w", path, g.err)
	}

	return &GGUFHeader{Path: path, Version: version, Metadata: metadata,
		NTensors: nTensors, TensorBytes: tensorBytes, EmbdTableBytes: embdBytes}, nil
}
