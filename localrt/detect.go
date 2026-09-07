// Package localrt detects and describes local model servers (Ollama,
// LM Studio, llama.cpp, vLLM) so proxima can wire govega's OpenAI-compat
// backend to whatever is already running on the user's machine.
//
// The probe strategy is a port of hermes-agent's detect_local_server_type
// (agent/model_metadata.py): each server type has a distinguishing endpoint,
// and probe ORDER matters — LM Studio answers 200 on Ollama's /api/tags, so
// its own /api/v1/models must be checked first.
package localrt

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"
)

// ServerType identifies a local model server implementation.
type ServerType string

const (
	ServerUnknown  ServerType = ""
	ServerOllama   ServerType = "ollama"
	ServerLMStudio ServerType = "lmstudio"
	ServerLlamaCpp ServerType = "llamacpp"
	ServerVLLM     ServerType = "vllm"
)

// Server is a detected local model server.
type Server struct {
	Type    ServerType
	BaseURL string // origin, e.g. http://127.0.0.1:11434 — no /v1 suffix
}

// OpenAIBaseURL returns the OpenAI-compatible API root for the server; all
// four supported servers serve chat completions under /v1.
func (s Server) OpenAIBaseURL() string {
	return strings.TrimRight(s.BaseURL, "/") + "/v1"
}

// DefaultEndpoints are the well-known local listen addresses, probed in
// order: Ollama, LM Studio, llama.cpp (also LiteLLM's default :4000).
var DefaultEndpoints = []string{
	"http://127.0.0.1:11434",
	"http://127.0.0.1:1234",
	"http://127.0.0.1:8080",
	"http://127.0.0.1:4000",
}

var probeClient = &http.Client{Timeout: 2 * time.Second}

// DetectServerType identifies what kind of model server listens at baseURL,
// or ServerUnknown if nothing recognizable answers.
func DetectServerType(ctx context.Context, baseURL string) ServerType {
	base := strings.TrimRight(baseURL, "/")

	// LM Studio first: it 200s on /api/tags too.
	if _, ok := get(ctx, base+"/api/v1/models"); ok {
		return ServerLMStudio
	}
	if body, ok := get(ctx, base+"/api/tags"); ok {
		var tags map[string]json.RawMessage
		if json.Unmarshal(body, &tags) == nil {
			if _, has := tags["models"]; has {
				return ServerOllama
			}
		}
	}
	if body, ok := get(ctx, base+"/props"); ok {
		var props map[string]json.RawMessage
		if json.Unmarshal(body, &props) == nil {
			if _, has := props["default_generation_settings"]; has {
				return ServerLlamaCpp
			}
		}
	}
	if _, ok := get(ctx, base+"/version"); ok {
		return ServerVLLM
	}
	return ServerUnknown
}

// Detect probes the given endpoints (DefaultEndpoints when none are passed)
// and returns the first recognizable server.
func Detect(ctx context.Context, endpoints ...string) (Server, bool) {
	if len(endpoints) == 0 {
		endpoints = DefaultEndpoints
	}
	for _, ep := range endpoints {
		if t := DetectServerType(ctx, ep); t != ServerUnknown {
			return Server{Type: t, BaseURL: strings.TrimRight(ep, "/")}, true
		}
	}
	return Server{}, false
}

// ListModels returns the model ids the server advertises on the
// OpenAI-compatible /models endpoint. openaiBaseURL includes the /v1.
func ListModels(ctx context.Context, openaiBaseURL string) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		strings.TrimRight(openaiBaseURL, "/")+"/models", nil)
	if err != nil {
		return nil, err
	}
	resp, err := probeClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	models := make([]string, 0, len(out.Data))
	for _, m := range out.Data {
		models = append(models, m.ID)
	}
	return models, nil
}

// get performs a GET and returns (body, true) only on HTTP 200.
func get(ctx context.Context, url string) ([]byte, bool) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, false
	}
	resp, err := probeClient.Do(req)
	if err != nil {
		return nil, false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, false
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, false
	}
	return body, true
}
