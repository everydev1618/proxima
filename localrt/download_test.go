package localrt

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// flakyServer serves payload but cuts every response off after chunk bytes,
// honoring Range so a resuming client makes progress on each attempt. A
// Content-Length longer than the body makes the client see the truncation
// as an unexpected EOF, like a dropped connection.
func flakyServer(t *testing.T, payload []byte, chunk int) (*httptest.Server, *int) {
	t.Helper()
	requests := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		from := 0
		if rg := r.Header.Get("Range"); strings.HasPrefix(rg, "bytes=") {
			from, _ = strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(rg, "bytes="), "-"))
			w.Header().Set("Content-Range",
				fmt.Sprintf("bytes %d-%d/%d", from, len(payload)-1, len(payload)))
			w.Header().Set("Content-Length", strconv.Itoa(len(payload)-from))
			w.WriteHeader(http.StatusPartialContent)
		} else {
			w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
		}
		end := from + chunk
		if end > len(payload) {
			end = len(payload)
		}
		w.Write(payload[from:end])
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	t.Cleanup(ts.Close)
	return ts, &requests
}

func TestDownloadFileRetriesAndResumes(t *testing.T) {
	old := downloadRetryDelay
	downloadRetryDelay = time.Millisecond
	defer func() { downloadRetryDelay = old }()

	payload := bytes.Repeat([]byte("proxima!"), 8192) // 64 KiB
	ts, requests := flakyServer(t, payload, 20000)    // ~4 truncated responses needed

	dest := filepath.Join(t.TempDir(), "model.gguf")
	if err := downloadFile(ts.URL+"/w.gguf", dest, nil); err != nil {
		t.Fatalf("downloadFile: %v (after %d requests)", err, *requests)
	}
	got, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("payload corrupted across resumes: got %d bytes, want %d", len(got), len(payload))
	}
	if *requests < 3 {
		t.Errorf("expected multiple resumed attempts, saw %d requests", *requests)
	}
	if _, err := os.Stat(dest + ".part"); !os.IsNotExist(err) {
		t.Error(".part not cleaned up after completion")
	}
}

func TestDownloadFileNoRetryOnClientError(t *testing.T) {
	old := downloadRetryDelay
	downloadRetryDelay = time.Millisecond
	defer func() { downloadRetryDelay = old }()

	requests := 0
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		http.NotFound(w, r)
	}))
	defer ts.Close()

	dest := filepath.Join(t.TempDir(), "gone.gguf")
	if err := downloadFile(ts.URL+"/gone.gguf", dest, nil); err == nil {
		t.Fatal("expected error for 404")
	}
	if requests != 1 {
		t.Errorf("404 must not be retried: saw %d requests", requests)
	}
}
