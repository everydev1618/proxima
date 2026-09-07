package main

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func testHub() *setupHub {
	return newSetupHub(setupInfo{
		Hostname:    "studio.local",
		UsableGB:    24,
		UMA:         true,
		Recommended: "qwen3.5-4b-q4",
		Models: []setupModel{
			{ID: "qwen3.5-4b-q4", DisplayName: "Qwen3.5 4B", DownloadGB: 3.6, Fits: true, Starter: true},
			{ID: "big-70b-q4", DisplayName: "Big 70B", DownloadGB: 40, Fits: false},
		},
	})
}

func TestSetupStateEndpoint(t *testing.T) {
	hub := testHub()
	srv := httptest.NewServer(newSetupMux(hub))
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/api/v1/local/state")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var got struct {
		Phase       string       `json:"phase"`
		Hostname    string       `json:"hostname"`
		Recommended string       `json:"recommended"`
		Models      []setupModel `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.Phase != "setup" {
		t.Fatalf("phase = %q, want setup", got.Phase)
	}
	if got.Recommended != "qwen3.5-4b-q4" || len(got.Models) != 2 {
		t.Fatalf("unexpected state: %+v", got)
	}
}

func TestBootstrapConsent(t *testing.T) {
	hub := testHub()
	srv := httptest.NewServer(newSetupMux(hub))
	defer srv.Close()

	// Unknown model is a 400.
	resp, _ := http.Post(srv.URL+"/api/v1/local/bootstrap", "application/json",
		strings.NewReader(`{"model_id":"nope"}`))
	if resp.StatusCode != 400 {
		t.Fatalf("unknown model: status %d, want 400", resp.StatusCode)
	}
	// A model that doesn't fit is a 400 too.
	resp, _ = http.Post(srv.URL+"/api/v1/local/bootstrap", "application/json",
		strings.NewReader(`{"model_id":"big-70b-q4"}`))
	if resp.StatusCode != 400 {
		t.Fatalf("unfitting model: status %d, want 400", resp.StatusCode)
	}

	// Empty body means "the recommended one".
	resp, _ = http.Post(srv.URL+"/api/v1/local/bootstrap", "application/json", strings.NewReader(`{}`))
	if resp.StatusCode != 202 {
		t.Fatalf("consent: status %d, want 202", resp.StatusCode)
	}
	select {
	case id := <-hub.consent:
		if id != "qwen3.5-4b-q4" {
			t.Fatalf("consented model = %q", id)
		}
	case <-time.After(time.Second):
		t.Fatal("no consent delivered")
	}

	// A second consent is a conflict.
	resp, _ = http.Post(srv.URL+"/api/v1/local/bootstrap", "application/json", strings.NewReader(`{}`))
	if resp.StatusCode != 409 {
		t.Fatalf("second consent: status %d, want 409", resp.StatusCode)
	}
}

func TestConsentOnceRace(t *testing.T) {
	hub := testHub()
	ok1 := hub.offerConsent("a")
	ok2 := hub.offerConsent("b")
	if !ok1 || ok2 {
		t.Fatalf("offerConsent race: first=%v second=%v, want true/false", ok1, ok2)
	}
	if id := <-hub.consent; id != "a" {
		t.Fatalf("winner = %q, want a", id)
	}
}

func TestProgressSSE(t *testing.T) {
	hub := testHub()
	srv := httptest.NewServer(newSetupMux(hub))
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/api/v1/local/progress")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type = %q", ct)
	}

	hub.setPhase(phaseDownloading)
	hub.progressFn()("model", 1e9, 2e9, "qwen.gguf")

	r := bufio.NewReader(resp.Body)
	deadline := time.After(2 * time.Second)
	var events []string
	for len(events) < 2 {
		lineCh := make(chan string, 1)
		go func() {
			for {
				line, err := r.ReadString('\n')
				if err != nil {
					return
				}
				if strings.HasPrefix(line, "data: ") {
					lineCh <- strings.TrimSpace(strings.TrimPrefix(line, "data: "))
					return
				}
			}
		}()
		select {
		case l := <-lineCh:
			events = append(events, l)
		case <-deadline:
			t.Fatalf("timed out; got events: %v", events)
		}
	}
	if !strings.Contains(events[0], `"phase":"downloading"`) {
		t.Fatalf("first event = %s, want phase change", events[0])
	}
	if !strings.Contains(events[1], `"label":"qwen.gguf"`) {
		t.Fatalf("second event = %s, want progress", events[1])
	}
}
