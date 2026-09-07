package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func approvalTestMux(hub *approvalHub) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/v1/local/approvals", hub.handleList)
	mux.HandleFunc("POST /api/v1/local/approvals/{id}", hub.handleResolve)
	return mux
}

// waitPending polls until the hub shows n pending approvals.
func waitPending(t *testing.T, hub *approvalHub, n int) []approvalView {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if got := hub.snapshot(); len(got) == n {
			return got
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("never reached %d pending approvals", n)
	return nil
}

func TestApprovalResolvedFromPhone(t *testing.T) {
	hub := newApprovalHub()
	out := &bytes.Buffer{}
	// TTY never answers; the phone does.
	a := &approver{in: &blockingReader{}, out: out, always: map[string]bool{},
		timeout: 5 * time.Second, hub: hub}
	srv := httptest.NewServer(approvalTestMux(hub))
	defer srv.Close()

	got := make(chan bool, 1)
	go func() { got <- a.allow("exec", map[string]any{"command": "ls"}) }()

	pending := waitPending(t, hub, 1)
	if pending[0].Tool != "exec" || pending[0].Summary != "ls" {
		t.Fatalf("pending = %+v", pending[0])
	}

	resp, err := http.Post(srv.URL+"/api/v1/local/approvals/"+pending[0].ID,
		"application/json", strings.NewReader(`{"allow":true}`))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("resolve status = %d", resp.StatusCode)
	}
	select {
	case allowed := <-got:
		if !allowed {
			t.Fatal("phone allow=true should approve")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("allow() did not return after phone resolution")
	}
	waitPending(t, hub, 0)
	if !strings.Contains(out.String(), "phone") {
		t.Errorf("terminal not told about phone answer: %q", out.String())
	}
}

func TestApprovalPhoneDeny(t *testing.T) {
	hub := newApprovalHub()
	a := &approver{in: &blockingReader{}, out: &bytes.Buffer{}, always: map[string]bool{},
		timeout: 5 * time.Second, hub: hub}
	got := make(chan bool, 1)
	go func() { got <- a.allow("exec", map[string]any{"command": "rm -rf /"}) }()
	p := waitPending(t, hub, 1)
	if !hub.resolve(p[0].ID, false) {
		t.Fatal("resolve failed")
	}
	if allowed := <-got; allowed {
		t.Fatal("phone deny should deny")
	}
}

func TestApprovalResolveOnce(t *testing.T) {
	hub := newApprovalHub()
	a := &approver{in: &blockingReader{}, out: &bytes.Buffer{}, always: map[string]bool{},
		timeout: 5 * time.Second, hub: hub}
	go a.allow("exec", map[string]any{"command": "ls"})
	p := waitPending(t, hub, 1)
	if !hub.resolve(p[0].ID, true) {
		t.Fatal("first resolve should succeed")
	}
	if hub.resolve(p[0].ID, true) {
		t.Fatal("second resolve should fail")
	}
}

func TestApprovalResolveUnknownIs404(t *testing.T) {
	hub := newApprovalHub()
	srv := httptest.NewServer(approvalTestMux(hub))
	defer srv.Close()
	resp, _ := http.Post(srv.URL+"/api/v1/local/approvals/nope",
		"application/json", strings.NewReader(`{"allow":true}`))
	if resp.StatusCode != 404 {
		t.Fatalf("status = %d, want 404", resp.StatusCode)
	}
}

func TestApprovalTTYAnswerClearsHub(t *testing.T) {
	hub := newApprovalHub()
	a := &approver{in: strings.NewReader("y\n"), out: &bytes.Buffer{},
		always: map[string]bool{}, timeout: 2 * time.Second, hub: hub}
	if !a.allow("exec", map[string]any{"command": "ls"}) {
		t.Fatal("y should approve")
	}
	if got := hub.snapshot(); len(got) != 0 {
		t.Fatalf("pending after TTY answer: %+v", got)
	}
}

func TestApprovalListEndpoint(t *testing.T) {
	hub := newApprovalHub()
	a := &approver{in: &blockingReader{}, out: &bytes.Buffer{}, always: map[string]bool{},
		timeout: 5 * time.Second, hub: hub}
	go a.allow("exec", map[string]any{"command": "ls"})
	waitPending(t, hub, 1)

	srv := httptest.NewServer(approvalTestMux(hub))
	defer srv.Close()
	resp, err := http.Get(srv.URL + "/api/v1/local/approvals")
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		Pending []approvalView `json:"pending"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if len(body.Pending) != 1 || body.Pending[0].Tool != "exec" {
		t.Fatalf("list = %+v", body)
	}
}

// Hub-only approver (headless Mac, phone paired): no terminal at all.
func TestApprovalHubOnlyApprover(t *testing.T) {
	hub := newApprovalHub()
	a := &approver{out: &bytes.Buffer{}, always: map[string]bool{},
		timeout: 5 * time.Second, hub: hub}
	got := make(chan bool, 1)
	go func() { got <- a.allow("exec", map[string]any{"command": "ls"}) }()
	p := waitPending(t, hub, 1)
	hub.resolve(p[0].ID, true)
	if allowed := <-got; !allowed {
		t.Fatal("hub-only approval failed")
	}
}
