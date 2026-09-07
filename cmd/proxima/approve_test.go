package main

import (
	"bytes"
	"context"
	"strings"
	"testing"
	"time"

	"github.com/everydev1618/govega/tools"
)

// gateHarness builds a tools registry with the approval middleware installed
// and scripted terminal input.
func gateHarness(t *testing.T, input string) (*tools.Tools, *bytes.Buffer, *approver) {
	t.Helper()
	out := &bytes.Buffer{}
	a := &approver{in: strings.NewReader(input), out: out,
		always: map[string]bool{}, timeout: 2 * time.Second}
	tl := tools.NewTools()
	for _, name := range []string{"exec", "recall"} {
		n := name
		if err := tl.Register(n, func(ctx context.Context) string { return n + " ran" }); err != nil {
			t.Fatal(err)
		}
	}
	tl.Use(a.middleware(approvalSets["exec"]))
	return tl, out, a
}

func TestApprovalGate(t *testing.T) {
	ctx := context.Background()

	// Ungated tools pass without a prompt.
	tl, out, _ := gateHarness(t, "")
	got, err := tl.Execute(ctx, "recall", nil)
	if err != nil || got != "recall ran" || out.Len() != 0 {
		t.Fatalf("ungated: %q err=%v prompt=%q", got, err, out.String())
	}

	// "y" approves once; the next call prompts again and "n" denies.
	tl2, out2, _ := gateHarness(t, "y\nn\n")
	if got, err := tl2.Execute(ctx, "exec", map[string]any{"command": "ls"}); err != nil || got != "exec ran" {
		t.Fatalf("approved: %q err=%v", got, err)
	}
	if _, err := tl2.Execute(ctx, "exec", map[string]any{"command": "rm -rf /"}); err == nil ||
		!strings.Contains(err.Error(), "declined") {
		t.Fatalf("denied exec returned err=%v", err)
	}
	if !strings.Contains(out2.String(), "rm -rf /") {
		t.Error("prompt did not show the command")
	}

	// "a" persists for the session.
	tl3, _, a3 := gateHarness(t, "a\n")
	if _, err := tl3.Execute(ctx, "exec", map[string]any{"command": "ls"}); err != nil {
		t.Fatal(err)
	}
	if _, err := tl3.Execute(ctx, "exec", map[string]any{"command": "ls again"}); err != nil {
		t.Fatalf("always did not persist: %v", err)
	}
	if !a3.always["exec"] {
		t.Error("always flag not recorded")
	}

	// Exhausted input (closed terminal) denies.
	tl4, _, _ := gateHarness(t, "")
	if _, err := tl4.Execute(ctx, "exec", map[string]any{"command": "ls"}); err == nil {
		t.Fatal("closed input should deny")
	}
}

func TestApprovalTimeoutDenies(t *testing.T) {
	out := &bytes.Buffer{}
	// A reader that never delivers a line.
	blocked := &blockingReader{}
	a := &approver{in: blocked, out: out, always: map[string]bool{}, timeout: 150 * time.Millisecond}
	if a.allow("exec", map[string]any{"command": "ls"}) {
		t.Fatal("timeout should deny")
	}
	if !strings.Contains(out.String(), "denied") {
		t.Errorf("output: %q", out.String())
	}
}

type blockingReader struct{}

func (b *blockingReader) Read(p []byte) (int, error) {
	time.Sleep(time.Hour)
	return 0, nil
}

func TestSummarize(t *testing.T) {
	if s := summarize("exec", map[string]any{"command": "ls -la"}); s != "ls -la" {
		t.Errorf("command: %q", s)
	}
	if s := summarize("write_file", map[string]any{"path": "/tmp/x", "content": "..."}); s != "/tmp/x" {
		t.Errorf("path: %q", s)
	}
	long := strings.Repeat("x", 500)
	if s := summarize("exec", map[string]any{"command": long}); len(s) > 410 {
		t.Errorf("not truncated: %d", len(s))
	}
}
