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
	in  io.Reader // nil when there is no controlling terminal (phone-only gate)
	out io.Writer

	mu      sync.Mutex // serializes prompts (tool calls run in parallel)
	always  map[string]bool
	timeout time.Duration
	reader  *bufio.Reader
	hub     *approvalHub // optional: mirrors prompts to paired phones
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

// newHubApprover builds a phone-only gate for headless runs: no terminal,
// prompts answered exclusively through the approvals API.
func newHubApprover(hub *approvalHub) *approver {
	return &approver{out: os.Stderr, always: map[string]bool{}, timeout: promptTimeout, hub: hub}
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

// allow prompts for one gated call and blocks until the terminal, a paired
// phone, or the timeout answers — whichever comes first. Terminal answers:
// y (once), n (deny), a (always for this tool, this session).
func (a *approver) allow(name string, params map[string]any) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.always[name] {
		return true
	}
	summary := summarize(name, params)

	var pending *pendingApproval
	var phoneCh chan bool // nil (blocks forever in select) without a hub
	if a.hub != nil {
		pending = a.hub.add(name, summary)
		phoneCh = pending.result
	}
	// decide settles the outcome everywhere: clears the phone queue when the
	// terminal or the timeout answered first (no-op if the phone already won).
	decide := func(allowed bool) bool {
		if pending != nil {
			a.hub.resolve(pending.ID, allowed)
		}
		return allowed
	}

	ttyCh := a.promptTTY(name, summary)
	select {
	case ans := <-ttyCh:
		if ans.err != nil {
			fmt.Fprintln(a.out, "→ denied (input closed)")
			return decide(false)
		}
		switch strings.ToLower(strings.TrimSpace(ans.line)) {
		case "y", "yes":
			return decide(true)
		case "a", "always":
			a.always[name] = true
			return decide(true)
		default:
			return decide(false)
		}
	case allowed := <-phoneCh:
		verdict := "denied"
		if allowed {
			verdict = "allowed"
		}
		fmt.Fprintf(a.out, "→ %s from phone\n", verdict)
		return allowed
	case <-time.After(a.timeout):
		fmt.Fprintln(a.out, "→ denied (no answer)")
		return decide(false)
	}
}

type ttyAnswer struct {
	line string
	err  error
}

// promptTTY prints the prompt and reads one line off the terminal in the
// background. Returns nil (a channel that never delivers) when the approver
// has no terminal.
func (a *approver) promptTTY(name, summary string) chan ttyAnswer {
	if a.in == nil {
		fmt.Fprintf(a.out, "\n┌─ approval: agent wants to run %s\n│  %s\n└─ waiting for the phone…\n", name, summary)
		return nil
	}
	if a.reader == nil {
		a.reader = bufio.NewReader(a.in)
	}
	fmt.Fprintf(a.out, "\n┌─ approval: agent wants to run %s\n│  %s\n└─ allow? [y]es / [n]o / [a]lways this session: ",
		name, summary)
	ch := make(chan ttyAnswer, 1)
	go func() {
		line, err := a.reader.ReadString('\n')
		ch <- ttyAnswer{line, err}
	}()
	return ch
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
