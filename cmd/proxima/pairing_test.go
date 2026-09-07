package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadOrCreatePairingToken(t *testing.T) {
	dir := t.TempDir()

	tok, err := loadOrCreatePairingToken(dir)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if len(tok) != 64 { // 32 random bytes, hex
		t.Fatalf("token length = %d, want 64", len(tok))
	}

	// Second call returns the SAME token (pairing survives restarts).
	tok2, err := loadOrCreatePairingToken(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if tok2 != tok {
		t.Fatalf("token changed across loads: %q vs %q", tok, tok2)
	}

	// Token file is private to the user.
	info, err := os.Stat(filepath.Join(dir, "mobile-token"))
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Fatalf("token file mode = %o, want 600", perm)
	}
}

func TestTokenMiddleware(t *testing.T) {
	const tok = "sekret"
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	h := tokenMiddleware(tok)(inner)

	cases := []struct {
		name   string
		remote string
		header string
		query  string
		want   int
	}{
		{"loopback exempt", "127.0.0.1:5555", "", "", 200},
		{"loopback v6 exempt", "[::1]:5555", "", "", 200},
		{"lan no token", "192.168.1.20:5555", "", "", 401},
		{"lan bearer ok", "192.168.1.20:5555", "Bearer sekret", "", 200},
		{"lan bearer wrong", "192.168.1.20:5555", "Bearer nope", "", 401},
		{"lan query ok", "192.168.1.20:5555", "", "sekret", 200},
		{"lan query wrong", "192.168.1.20:5555", "", "nope", 401},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			url := "/api/v1/agents"
			if tc.query != "" {
				url += "?token=" + tc.query
			}
			r := httptest.NewRequest("GET", url, nil)
			r.RemoteAddr = tc.remote
			if tc.header != "" {
				r.Header.Set("Authorization", tc.header)
			}
			w := httptest.NewRecorder()
			h.ServeHTTP(w, r)
			if w.Code != tc.want {
				t.Fatalf("status = %d, want %d", w.Code, tc.want)
			}
		})
	}
}

func TestPairURL(t *testing.T) {
	u := pairURL("192.168.1.10", "7769", "abc123", "studio.local")
	for _, want := range []string{"proxima://pair?", "h=192.168.1.10", "p=7769", "t=abc123", "n=studio.local"} {
		if !strings.Contains(u, want) {
			t.Fatalf("pairURL = %q, missing %q", u, want)
		}
	}
}

func TestLanIPs(t *testing.T) {
	// Can't assert specific addresses on an arbitrary machine, but whatever
	// comes back must be private/CGNAT IPv4 and never loopback.
	for _, ip := range lanIPs() {
		if strings.Contains(ip, ":") {
			t.Fatalf("lanIPs returned non-IPv4 %q", ip)
		}
		if strings.HasPrefix(ip, "127.") {
			t.Fatalf("lanIPs returned loopback %q", ip)
		}
	}
}
