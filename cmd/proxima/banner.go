package main

import (
	"net"
	"net/url"
	"strings"
)

// bootBanner renders the startup lines. Server internals (type, port) stay
// out of it — the endpoint's location only appears when the model lives on
// another box, where "on this machine" would be a lie worth debugging.
func bootBanner(model, origin string) []string {
	where := "on this machine"
	if !onThisMachine(origin) {
		where = "at " + origin
	}
	return []string{
		"model ready — " + model + ", " + where,
		"agents up — tools, memory",
	}
}

// onThisMachine reports whether origin points at this machine itself:
// loopback, localhost names, and container-host aliases. Narrower than
// localrt.IsLocalEndpoint, which also admits tailnet and LAN boxes.
func onThisMachine(origin string) bool {
	u, err := url.Parse(origin)
	if err != nil {
		return false
	}
	host := strings.ToLower(u.Hostname())
	if host == "localhost" || strings.HasSuffix(host, ".localhost") ||
		strings.HasSuffix(host, ".internal") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
