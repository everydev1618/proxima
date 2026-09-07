package localrt

import (
	"net"
	"net/url"
	"strings"
)

// containerHostNames are the well-known DNS names container runtimes give
// the host machine — a server there is the user's own box.
var containerHostNames = map[string]bool{
	"host.docker.internal":     true,
	"host.lima.internal":       true,
	"host.containers.internal": true,
}

// cgnat is 100.64.0.0/10 — the CGNAT range Tailscale assigns. A model server
// reached over a tailnet is remote-but-trusted and gets local treatment
// (generous timeouts, no API key), same call hermes-agent makes.
var cgnat = func() *net.IPNet {
	_, n, _ := net.ParseCIDR("100.64.0.0/10")
	return n
}()

// IsLocalEndpoint reports whether raw points at the user's own machine or
// private network: loopback, RFC-1918/ULA, link-local, Tailscale CGNAT,
// localhost names, and container-host DNS names.
func IsLocalEndpoint(raw string) bool {
	if raw == "" {
		return false
	}
	if !strings.Contains(raw, "//") {
		raw = "http://" + raw
	}
	u, err := url.Parse(raw)
	if err != nil {
		return false
	}
	host := strings.ToLower(u.Hostname())
	if host == "" {
		return false
	}
	if host == "localhost" || strings.HasSuffix(host, ".localhost") || containerHostNames[host] {
		return true
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return false
	}
	return ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || cgnat.Contains(ip)
}
