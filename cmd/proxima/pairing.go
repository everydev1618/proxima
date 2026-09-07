// Mobile pairing: a per-machine secret token gates every non-loopback request,
// and a QR code on the terminal carries host+port+token to the phone. The
// desktop dashboard keeps working untouched because loopback is exempt.
package main

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/mdp/qrterminal/v3"
)

// loadOrCreatePairingToken returns the machine's mobile pairing token,
// creating it on first use. One token per machine, persisted so pairing
// survives restarts; delete the file to force every phone to re-pair.
func loadOrCreatePairingToken(dir string) (string, error) {
	path := filepath.Join(dir, "mobile-token")
	if b, err := os.ReadFile(path); err == nil {
		if tok := strings.TrimSpace(string(b)); tok != "" {
			return tok, nil
		}
	}
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", fmt.Errorf("generating pairing token: %w", err)
	}
	tok := hex.EncodeToString(raw)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(path, []byte(tok+"\n"), 0o600); err != nil {
		return "", fmt.Errorf("persisting pairing token: %w", err)
	}
	return tok, nil
}

// tokenMiddleware rejects non-loopback requests that don't present the
// pairing token. Accepted as `Authorization: Bearer <tok>` or `?token=<tok>`
// (React Native SSE reconnects can't always resend headers).
func tokenMiddleware(token string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if isLoopbackRequest(r) || presentsToken(r, token) {
				next.ServeHTTP(w, r)
				return
			}
			http.Error(w, `{"error":"pairing token required"}`, http.StatusUnauthorized)
		})
	}
}

func isLoopbackRequest(r *http.Request) bool {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return false
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func presentsToken(r *http.Request, token string) bool {
	got := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	if got == r.Header.Get("Authorization") { // no Bearer prefix
		got = ""
	}
	if got == "" {
		got = r.URL.Query().Get("token")
	}
	return got != "" && subtle.ConstantTimeCompare([]byte(got), []byte(token)) == 1
}

// pairURL builds the payload the QR code carries. The app parses the custom
// scheme; h/p/t/n = host, port, token, machine name.
func pairURL(host, port, token, name string) string {
	q := url.Values{"h": {host}, "p": {port}, "t": {token}, "n": {name}}
	return "proxima://pair?" + q.Encode()
}

// lanIPs lists this machine's IPv4 addresses a phone on the same network can
// reach: RFC-1918 private ranges plus Tailscale CGNAT (100.64/10) — the same
// notion of "local" as localrt.IsLocalEndpoint. Loopback and link-local are
// excluded; IPv6 is skipped (QR payloads stay short, and every network a
// phone shares with a Mac has IPv4).
func lanIPs() []string {
	var out []string
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return nil
	}
	cgnat := net.IPNet{IP: net.IPv4(100, 64, 0, 0), Mask: net.CIDRMask(10, 32)}
	for _, a := range addrs {
		ipnet, ok := a.(*net.IPNet)
		if !ok {
			continue
		}
		ip4 := ipnet.IP.To4()
		if ip4 == nil || ip4.IsLoopback() || ip4.IsLinkLocalUnicast() {
			continue
		}
		if ip4.IsPrivate() || cgnat.Contains(ip4) {
			out = append(out, ip4.String())
		}
	}
	return out
}

// printPairing renders the pairing block: a scannable QR plus a manual
// fallback for phones without cameras pointed at terminals.
func printPairing(out io.Writer, port, token string) {
	ips := lanIPs()
	if len(ips) == 0 {
		fmt.Fprintln(out, "mobile: no LAN address found — phone pairing unavailable (WiFi off?)")
		return
	}
	host, _ := os.Hostname()
	fmt.Fprintf(out, "\nPair your phone (Proxima app → scan):\n\n")
	qrterminal.GenerateWithConfig(pairURL(ips[0], port, token, host), qrterminal.Config{
		Level: qrterminal.L, Writer: out,
		BlackChar: qrterminal.BLACK, WhiteChar: qrterminal.WHITE,
		HalfBlocks: true, QuietZone: 2,
	})
	fmt.Fprintf(out, "\n  or enter manually — host %s  port %s\n  token %s\n\n", ips[0], port, token)
}
