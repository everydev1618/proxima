// Ollama-specific honesty probes. Port of the local-endpoint pieces of
// hermes-agent's agent/model_metadata.py (MIT, Nous Research — see NOTICE).

package localrt

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
)

var modelfileNumCtxRE = regexp.MustCompile(`(?im)^PARAMETER\s+num_ctx\s+(\d+)`)

// QueryOllamaNumCtx returns the context window Ollama will actually serve
// for a model: the Modelfile's num_ctx parameter when set, else the GGUF's
// trained context length, else 0. Ollama's silent 2048-token default
// truncates agent contexts, so callers warn (or upstream, inject
// options.num_ctx) when this comes back small.
func QueryOllamaNumCtx(ctx context.Context, baseURL, model string) int {
	body, _ := json.Marshal(map[string]string{"model": model})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		strings.TrimRight(baseURL, "/")+"/api/show", bytes.NewReader(body))
	if err != nil {
		return 0
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := probeClient.Do(req)
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil || resp.StatusCode != http.StatusOK {
		return 0
	}
	var show struct {
		Parameters string         `json:"parameters"`
		Modelfile  string         `json:"modelfile"`
		ModelInfo  map[string]any `json:"model_info"`
	}
	if json.Unmarshal(raw, &show) != nil {
		return 0
	}
	// Modelfile parameter first: it is what the server enforces.
	for _, text := range []string{show.Parameters, show.Modelfile} {
		if m := modelfileNumCtxRE.FindStringSubmatch(text); m != nil {
			if n, err := strconv.Atoi(m[1]); err == nil {
				return n
			}
		}
	}
	// Fall back to the GGUF's trained window (model_info.<arch>.context_length).
	for key, value := range show.ModelInfo {
		if strings.HasSuffix(key, ".context_length") {
			if f, ok := value.(float64); ok {
				return int(f)
			}
		}
	}
	return 0
}
