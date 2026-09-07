package localrt

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// ggufBuilder assembles a minimal valid GGUF header for tests.
type ggufBuilder struct {
	buf     bytes.Buffer
	kv      bytes.Buffer
	tensors bytes.Buffer
	nKV     uint64
	nTens   uint64
}

func (b *ggufBuilder) writeStr(w *bytes.Buffer, s string) {
	binary.Write(w, binary.LittleEndian, uint64(len(s)))
	w.WriteString(s)
}

func (b *ggufBuilder) str(key, val string) *ggufBuilder {
	b.writeStr(&b.kv, key)
	binary.Write(&b.kv, binary.LittleEndian, uint32(vString))
	b.writeStr(&b.kv, val)
	b.nKV++
	return b
}

func (b *ggufBuilder) u32(key string, val uint32) *ggufBuilder {
	b.writeStr(&b.kv, key)
	binary.Write(&b.kv, binary.LittleEndian, uint32(vUint32))
	binary.Write(&b.kv, binary.LittleEndian, val)
	b.nKV++
	return b
}

func (b *ggufBuilder) f32(key string, val float32) *ggufBuilder {
	b.writeStr(&b.kv, key)
	binary.Write(&b.kv, binary.LittleEndian, uint32(vFloat32))
	binary.Write(&b.kv, binary.LittleEndian, val)
	b.nKV++
	return b
}

func (b *ggufBuilder) u32Array(key string, vals []uint32) *ggufBuilder {
	b.writeStr(&b.kv, key)
	binary.Write(&b.kv, binary.LittleEndian, uint32(vArray))
	binary.Write(&b.kv, binary.LittleEndian, uint32(vUint32))
	binary.Write(&b.kv, binary.LittleEndian, uint64(len(vals)))
	for _, v := range vals {
		binary.Write(&b.kv, binary.LittleEndian, v)
	}
	b.nKV++
	return b
}

func (b *ggufBuilder) tensor(name string, dims []uint64, ttype uint32) *ggufBuilder {
	b.writeStr(&b.tensors, name)
	binary.Write(&b.tensors, binary.LittleEndian, uint32(len(dims)))
	for _, d := range dims {
		binary.Write(&b.tensors, binary.LittleEndian, d)
	}
	binary.Write(&b.tensors, binary.LittleEndian, ttype)
	binary.Write(&b.tensors, binary.LittleEndian, uint64(0)) // offset
	b.nTens++
	return b
}

func (b *ggufBuilder) write(t *testing.T) string {
	t.Helper()
	b.buf.Write(ggufMagic[:])
	binary.Write(&b.buf, binary.LittleEndian, uint32(3)) // version
	binary.Write(&b.buf, binary.LittleEndian, b.nTens)
	binary.Write(&b.buf, binary.LittleEndian, b.nKV)
	b.buf.Write(b.kv.Bytes())
	b.buf.Write(b.tensors.Bytes())
	path := filepath.Join(t.TempDir(), "test.gguf")
	if err := os.WriteFile(path, b.buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadGGUFHeader(t *testing.T) {
	b := &ggufBuilder{}
	b.str("general.architecture", "qwen3").
		u32("qwen3.block_count", 4).
		u32("qwen3.context_length", 32768).
		u32("qwen3.embedding_length", 1024).
		u32("qwen3.attention.head_count", 8).
		u32("qwen3.attention.head_count_kv", 2).
		f32("general.sampling.temp", 1.0).
		f32("general.sampling.top_p", 0.95).
		// F16 (type 1): 1024*100 elems * 2 bytes
		tensor("token_embd.weight", []uint64{1024, 100}, 1).
		// q8_0 (type 8): 2048 elems / 32 * 34 = 2176 bytes
		tensor("blk.0.attn_q.weight", []uint64{64, 32}, 8)
	path := b.write(t)

	h, err := ReadGGUFHeader(path)
	if err != nil {
		t.Fatal(err)
	}
	if h.Architecture() != "qwen3" {
		t.Errorf("architecture = %q", h.Architecture())
	}
	if h.NLayer() != 4 || h.NCtxTrain() != 32768 || h.NEmbd() != 1024 {
		t.Errorf("dims: layer=%d ctx=%d embd=%d", h.NLayer(), h.NCtxTrain(), h.NEmbd())
	}
	if h.NHead() != 8 || h.HeadDimK() != 128 || h.HeadDimV() != 128 {
		t.Errorf("heads: n=%d dk=%d dv=%d", h.NHead(), h.HeadDimK(), h.HeadDimV())
	}
	wantEmbd := int64(1024 * 100 * 2)
	if h.EmbdTableBytes != wantEmbd {
		t.Errorf("embd bytes = %d, want %d", h.EmbdTableBytes, wantEmbd)
	}
	if want := wantEmbd + 2176; h.TensorBytes != want {
		t.Errorf("tensor bytes = %d, want %d", h.TensorBytes, want)
	}
	kv := h.HeadCountsKV()
	if len(kv) != 4 || kv[0] != 2 || kv[3] != 2 {
		t.Errorf("head counts kv = %v", kv)
	}
	samp := h.SamplingDefaults()
	if samp["temp"] != "1" || samp["top-p"] != "0.95" {
		t.Errorf("sampling defaults = %v", samp)
	}
}

func TestHeadCountsKVShapes(t *testing.T) {
	// GDN-hybrid: scalar + full_attention_interval -> every Nth layer.
	b := &ggufBuilder{}
	b.str("general.architecture", "qwen35").
		u32("qwen35.block_count", 6).
		u32("qwen35.full_attention_interval", 3).
		u32("qwen35.attention.head_count_kv", 4)
	h, err := ReadGGUFHeader(b.write(t))
	if err != nil {
		t.Fatal(err)
	}
	want := []int{0, 0, 4, 0, 0, 4}
	got := h.HeadCountsKV()
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("interval kv heads = %v, want %v", got, want)
		}
	}

	// Per-layer array is used as-is.
	b2 := &ggufBuilder{}
	b2.str("general.architecture", "hyb").
		u32("hyb.block_count", 3).
		u32Array("hyb.attention.head_count_kv", []uint32{8, 0, 8})
	h2, err := ReadGGUFHeader(b2.write(t))
	if err != nil {
		t.Fatal(err)
	}
	got2 := h2.HeadCountsKV()
	if len(got2) != 3 || got2[0] != 8 || got2[1] != 0 || got2[2] != 8 {
		t.Errorf("array kv heads = %v", got2)
	}
}

func TestReadGGUFHeaderRejectsNonGGUF(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.gguf")
	os.WriteFile(path, []byte("NOPE and some trailing bytes"), 0o644)
	if _, err := ReadGGUFHeader(path); err == nil {
		t.Fatal("expected error for non-GGUF file")
	}
}

func TestModelIDFromStem(t *testing.T) {
	if got := ModelIDFromStem("Qwen3.8-Flash-UD-Q4_K_XL-00001-of-00004"); got != "Qwen3.8-Flash-UD-Q4_K_XL" {
		t.Errorf("split stem: %q", got)
	}
	if got := ModelIDFromStem("Qwen3-32B-Q4_K_M"); got != "Qwen3-32B-Q4_K_M" {
		t.Errorf("plain stem: %q", got)
	}
}
