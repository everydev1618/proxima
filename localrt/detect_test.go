package localrt

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

// fakeServer builds an httptest server that responds 200 with body on the
// given paths and 404 everywhere else.
func fakeServer(t *testing.T, routes map[string]string) *httptest.Server {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if body, ok := routes[r.URL.Path]; ok {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(body))
			return
		}
		http.NotFound(w, r)
	}))
	t.Cleanup(ts.Close)
	return ts
}

func TestDetectServerType(t *testing.T) {
	ctx := context.Background()

	// LM Studio: native API under /api/v1/models. It also 200s on /api/tags,
	// which is why the LM Studio probe must run before the Ollama probe.
	lmstudio := fakeServer(t, map[string]string{
		"/api/v1/models": `{"data":[{"id":"qwen3-30b"}]}`,
		"/api/tags":      `{}`,
	})
	if got := DetectServerType(ctx, lmstudio.URL); got != ServerLMStudio {
		t.Errorf("lmstudio: got %q", got)
	}

	// Ollama: /api/tags with a "models" key.
	ollama := fakeServer(t, map[string]string{
		"/api/tags": `{"models":[{"name":"qwen3:30b"}]}`,
	})
	if got := DetectServerType(ctx, ollama.URL); got != ServerOllama {
		t.Errorf("ollama: got %q", got)
	}

	// llama.cpp: /props with default_generation_settings.
	llamacpp := fakeServer(t, map[string]string{
		"/props": `{"default_generation_settings":{"n_ctx":65536},"model_path":"/m.gguf"}`,
	})
	if got := DetectServerType(ctx, llamacpp.URL); got != ServerLlamaCpp {
		t.Errorf("llamacpp: got %q", got)
	}

	// vLLM: /version.
	vllm := fakeServer(t, map[string]string{
		"/version": `{"version":"0.8.0"}`,
	})
	if got := DetectServerType(ctx, vllm.URL); got != ServerVLLM {
		t.Errorf("vllm: got %q", got)
	}

	// Nothing recognizable.
	blank := fakeServer(t, map[string]string{})
	if got := DetectServerType(ctx, blank.URL); got != ServerUnknown {
		t.Errorf("unknown: got %q", got)
	}
}

func TestDetect(t *testing.T) {
	ctx := context.Background()
	ollama := fakeServer(t, map[string]string{
		"/api/tags": `{"models":[{"name":"qwen3:30b"}]}`,
	})

	// First endpoint is dead, second is the fake Ollama.
	srv, ok := Detect(ctx, "http://127.0.0.1:1", ollama.URL)
	if !ok {
		t.Fatal("Detect found nothing")
	}
	if srv.Type != ServerOllama || srv.BaseURL != ollama.URL {
		t.Errorf("got %+v", srv)
	}
	if want := ollama.URL + "/v1"; srv.OpenAIBaseURL() != want {
		t.Errorf("OpenAIBaseURL = %q, want %q", srv.OpenAIBaseURL(), want)
	}

	if _, ok := Detect(ctx, "http://127.0.0.1:1"); ok {
		t.Error("Detect reported a server on a dead port")
	}
}

func TestListModels(t *testing.T) {
	ts := fakeServer(t, map[string]string{
		"/v1/models": `{"object":"list","data":[{"id":"qwen3-30b"},{"id":"nomic-embed-text"}]}`,
	})
	models, err := ListModels(context.Background(), ts.URL+"/v1")
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0] != "qwen3-30b" {
		t.Errorf("got %v", models)
	}
}
