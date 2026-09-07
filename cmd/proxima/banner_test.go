package main

import (
	"strings"
	"testing"
)

func TestBootBannerOnThisMachine(t *testing.T) {
	for _, origin := range []string{
		"http://127.0.0.1:11434",
		"http://localhost:1234",
		"http://[::1]:8080",
		"http://host.docker.internal:11434",
	} {
		lines := bootBanner("qwen3:30b", origin)
		if len(lines) != 2 {
			t.Fatalf("bootBanner(%q): got %d lines, want 2", origin, len(lines))
		}
		if want := "model ready — qwen3:30b, on this machine"; lines[0] != want {
			t.Errorf("bootBanner(%q)[0] = %q, want %q", origin, lines[0], want)
		}
		if strings.Contains(lines[0], origin) {
			t.Errorf("bootBanner(%q)[0] leaks the origin: %q", origin, lines[0])
		}
	}
}

func TestBootBannerNamesRemoteBoxes(t *testing.T) {
	// Tailnet and LAN endpoints count as local for detection, but they are
	// not this machine — the banner must say where the model actually is.
	for _, origin := range []string{
		"http://100.101.1.2:11434",
		"http://192.168.1.50:8080",
	} {
		lines := bootBanner("qwen3:30b", origin)
		if want := "model ready — qwen3:30b, at " + origin; lines[0] != want {
			t.Errorf("bootBanner(%q)[0] = %q, want %q", origin, lines[0], want)
		}
	}
}

func TestBootBannerAgentsLine(t *testing.T) {
	lines := bootBanner("qwen3:30b", "http://127.0.0.1:11434")
	if want := "agents up — tools, memory"; lines[1] != want {
		t.Errorf("bootBanner[1] = %q, want %q", lines[1], want)
	}
}

func TestWelcomeLines(t *testing.T) {
	lines := welcomeLines("http://localhost:7769", true, true, "/x/proxima.log")
	joined := strings.Join(lines, "\n")
	for _, want := range []string{
		"http://localhost:7769", "opening in your browser", "proxima pair", "/x/proxima.log",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("welcome card missing %q:\n%s", want, joined)
		}
	}
	// The chat line leads — it is the answer to "what do I do now".
	if !strings.HasPrefix(lines[0], "chat — ") {
		t.Errorf("first line is not the chat address: %q", lines[0])
	}

	// Headless, no mobile, no log file: no browser claim, no phone line.
	joined = strings.Join(welcomeLines("http://localhost:7769", false, false, ""), "\n")
	for _, banned := range []string{"opening", "proxima pair", "logs"} {
		if strings.Contains(joined, banned) {
			t.Errorf("minimal card must not mention %q:\n%s", banned, joined)
		}
	}
}
