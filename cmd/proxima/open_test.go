package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

func TestOpenWhenReady(t *testing.T) {
	var ready atomic.Bool
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !ready.Load() {
			http.Error(w, "starting", http.StatusServiceUnavailable)
			return
		}
	}))
	defer ts.Close()

	opened := make(chan string, 1)
	oldOpen, oldPoll := openBrowser, openPollInterval
	openBrowser = func(u string) { opened <- u }
	openPollInterval = 5 * time.Millisecond
	defer func() { openBrowser, openPollInterval = oldOpen, oldPoll }()

	openWhenReady(context.Background(), ts.URL)
	select {
	case <-opened:
		t.Fatal("browser opened before the server was ready")
	case <-time.After(50 * time.Millisecond):
	}
	ready.Store(true)
	select {
	case u := <-opened:
		if u != ts.URL {
			t.Fatalf("opened %q, want %q", u, ts.URL)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("browser never opened after the server came up")
	}
}
