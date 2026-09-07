package localrt

import "testing"

func TestIsLocalEndpoint(t *testing.T) {
	cases := []struct {
		raw  string
		want bool
	}{
		{"http://localhost:11434", true},
		{"http://127.0.0.1:1234/v1", true},
		{"http://[::1]:8080", true},
		{"localhost:1234", true}, // scheme-less
		{"http://foo.localhost:9999", true},
		{"http://192.168.1.50:8080", true},
		{"http://10.0.0.5", true},
		{"http://172.20.3.4:8000", true},
		{"http://169.254.1.1", true},
		{"http://100.101.102.103:11434", true}, // Tailscale CGNAT 100.64/10
		{"http://100.30.1.1", false},           // just below CGNAT range — public
		{"http://host.docker.internal:11434", true},
		{"http://host.lima.internal:11434", true},
		{"http://host.containers.internal:11434", true},
		{"https://api.openai.com/v1", false},
		{"http://example.com:8080", false},
		{"http://8.8.8.8", false},
		{"", false},
	}
	for _, c := range cases {
		if got := IsLocalEndpoint(c.raw); got != c.want {
			t.Errorf("IsLocalEndpoint(%q) = %v, want %v", c.raw, got, c.want)
		}
	}
}
