// Tool-call approval gate: a small local model's judgment does not get to
// run shell commands on this machine unattended. Installed as govega tool
// middleware; gated calls block on a y/n/a prompt on the controlling
// terminal, and a denial returns a normal tool error the model can adapt to.
package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/everydev1618/govega/tools"
)

// approvalSets maps the -approve mode to the gated tool names. "exec" gates
// what runs code; "all" adds file writes. Everything else always passes.
var approvalSets = map[string]map[string]bool{
	"exec": {"exec": true, "start_service": true, "stop_service": true},
	"all": {"exec": true, "start_service": true, "stop_service": true,
		"write_file": true, "append_file": true},
}

// promptTimeout: an unanswered prompt is a denial — an agent must never hang
// forever on a wandered-off human, and auto-approve on timeout would defeat
// the gate.
const promptTimeout = 120 * time.Second

type approver struct {
	in  io.Reader
	out io.Writer

	mu      sync.Mutex // serializes prompts (tool calls run in parallel)
	always  map[string]bool
	timeout time.Duration
	reader  *bufio.Reader
}

// newApprover wires the gate to the controlling terminal. ok=false when no
// terminal is available (headless run) — the caller decides what that means.
func newApprover() (*approver, bool) {
	tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		return nil, false
	}
	return &approver{in: tty, out: tty, always: map[string]bool{}, timeout: promptTimeout}, true
}

// summarize renders the one param a human needs to judge the call.
func summarize(name string, params map[string]any) string {
	for _, key := range []string{"command", "path", "name"} {
		if v, ok := params[key].(string); ok && v != "" {
			if len(v) > 400 {
				v = v[:400] + "…"
			}
			return v
		}
	}
	return fmt.Sprintf("%v", params)
}

// allow prompts for one gated call. Answers: y (once), n (deny), a (always
// for this tool, this session).
func (a *approver) allow(name string, params map[string]any) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.always[name] {
		return true
	}
	if a.reader == nil {
		a.reader = bufio.NewReader(a.in)
	}
	fmt.Fprintf(a.out, "\n┌─ approval: agent wants to run %s\n│  %s\n└─ allow? [y]es / [n]o / [a]lways this session: ",
		name, summarize(name, params))

	type answer struct {
		line string
		err  error
	}
	ch := make(chan answer, 1)
	go func() {
		line, err := a.reader.ReadString('\n')
		ch <- answer{line, err}
	}()
	select {
	case ans := <-ch:
		if ans.err != nil {
			fmt.Fprintln(a.out, "→ denied (input closed)")
			return false
		}
		switch strings.ToLower(strings.TrimSpace(ans.line)) {
		case "y", "yes":
			return true
		case "a", "always":
			a.always[name] = true
			return true
		default:
			return false
		}
	case <-time.After(a.timeout):
		fmt.Fprintln(a.out, "→ denied (no answer)")
		return false
	}
}

// middleware builds the govega tool middleware for one gated set.
func (a *approver) middleware(gated map[string]bool) tools.ToolMiddleware {
	return func(next tools.ToolFunc) tools.ToolFunc {
		return func(ctx context.Context, params map[string]any) (string, error) {
			name := tools.ToolNameFromContext(ctx)
			if !gated[name] {
				return next(ctx, params)
			}
			if a.allow(name, params) {
				return next(ctx, params)
			}
			return "", fmt.Errorf("the user declined to approve this %s call; ask them or take another approach", name)
		}
	}
}
