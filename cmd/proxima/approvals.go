// Approvals over HTTP: the same gate approve.go enforces, answerable from a
// paired phone. Pending requests fan out over SSE; the first answer — terminal
// keystroke or phone tap — wins.
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// approvalView is the wire shape of one pending request.
type approvalView struct {
	ID        string    `json:"id"`
	Tool      string    `json:"tool"`
	Summary   string    `json:"summary"`
	CreatedAt time.Time `json:"created_at"`
}

type pendingApproval struct {
	approvalView
	// result delivers the phone's answer to the blocked allow() call.
	// Buffered so resolve never blocks on a caller that already returned.
	result chan bool
}

type approvalHub struct {
	mu      sync.Mutex
	seq     int
	pending map[string]*pendingApproval
	order   []string
	subs    map[chan []byte]struct{}
}

func newApprovalHub() *approvalHub {
	return &approvalHub{
		pending: map[string]*pendingApproval{},
		subs:    map[chan []byte]struct{}{},
	}
}

// add registers a pending approval and announces it to subscribers.
func (h *approvalHub) add(tool, summary string) *pendingApproval {
	h.mu.Lock()
	h.seq++
	p := &pendingApproval{
		approvalView: approvalView{
			ID: fmt.Sprintf("a%d", h.seq), Tool: tool, Summary: summary,
			CreatedAt: time.Now(),
		},
		result: make(chan bool, 1),
	}
	h.pending[p.ID] = p
	h.order = append(h.order, p.ID)
	h.mu.Unlock()
	h.broadcast(map[string]any{"type": "approval", "approval": p.approvalView})
	return p
}

// resolve answers one pending approval. false when the id is unknown or
// already answered — resolution happens exactly once.
func (h *approvalHub) resolve(id string, allow bool) bool {
	h.mu.Lock()
	p, ok := h.pending[id]
	if ok {
		delete(h.pending, id)
		for i, oid := range h.order {
			if oid == id {
				h.order = append(h.order[:i], h.order[i+1:]...)
				break
			}
		}
	}
	h.mu.Unlock()
	if !ok {
		return false
	}
	p.result <- allow
	h.broadcast(map[string]any{"type": "approval_resolved", "id": id, "allow": allow})
	return true
}

func (h *approvalHub) snapshot() []approvalView {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]approvalView, 0, len(h.order))
	for _, id := range h.order {
		out = append(out, h.pending[id].approvalView)
	}
	return out
}

func (h *approvalHub) broadcast(v any) {
	b, err := json.Marshal(v)
	if err != nil {
		return
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	for ch := range h.subs {
		select {
		case ch <- b:
		default:
		}
	}
}

func (h *approvalHub) subscribe() chan []byte {
	ch := make(chan []byte, 32)
	h.mu.Lock()
	h.subs[ch] = struct{}{}
	h.mu.Unlock()
	return ch
}

func (h *approvalHub) unsubscribe(ch chan []byte) {
	h.mu.Lock()
	delete(h.subs, ch)
	h.mu.Unlock()
}

// ── HTTP handlers (registered on serve via RegisterRoute) ────

func (h *approvalHub) handleList(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"pending": h.snapshot()})
}

func (h *approvalHub) handleResolve(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Allow bool `json:"allow"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid json"}`, http.StatusBadRequest)
		return
	}
	if !h.resolve(r.PathValue("id"), req.Allow) {
		http.Error(w, `{"error":"unknown or already resolved"}`, http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"ok": true})
}

func (h *approvalHub) handleStream(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	fmt.Fprintf(w, ": connected\n\n")
	// Replay the current queue so a late subscriber sees what's waiting.
	for _, v := range h.snapshot() {
		if b, err := json.Marshal(map[string]any{"type": "approval", "approval": v}); err == nil {
			fmt.Fprintf(w, "data: %s\n\n", b)
		}
	}
	flusher.Flush()

	ch := h.subscribe()
	defer h.unsubscribe(ch)
	ticker := time.NewTicker(25 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
			fmt.Fprintf(w, ": heartbeat\n\n")
			flusher.Flush()
		case b := <-ch:
			fmt.Fprintf(w, "data: %s\n\n", b)
			flusher.Flush()
		}
	}
}
