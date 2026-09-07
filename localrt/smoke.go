// Tool-call smoke eval. The readiness touch (TouchGenerate) proves a model
// can talk; this proves it can DRIVE: emit a well-formed call to an obvious
// tool. Open HF pulls especially need it — quant damage and broken chat
// templates ruin tool calling in ways no leaderboard shows. The verdict is
// stamped per model id and is advice, never a gate.

package localrt

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	smokeToolName = "lookup_capital"
	smokePrompt   = "Use the lookup_capital tool to find the capital of France."
)

// TouchToolCall asks for one forced tool call and verifies shape and intent:
// the right tool, parseable JSON arguments, and the argument the prompt
// obviously implies.
func (s *Supervisor) TouchToolCall(modelID string, timeout time.Duration) bool {
	var resp struct {
		Choices []struct {
			Message struct {
				ToolCalls []struct {
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	err := s.request("/v1/chat/completions", map[string]any{
		"model":    modelID,
		"messages": []map[string]string{{"role": "user", "content": smokePrompt}},
		"tools": []map[string]any{{
			"type": "function",
			"function": map[string]any{
				"name":        smokeToolName,
				"description": "Return the capital city of a country.",
				"parameters": map[string]any{
					"type":       "object",
					"properties": map[string]any{"country": map[string]any{"type": "string"}},
					"required":   []string{"country"},
				},
			},
		}},
		"tool_choice": "required",
		"max_tokens":  512, "temperature": 0,
	}, timeout, &resp)
	if err != nil || len(resp.Choices) == 0 || len(resp.Choices[0].Message.ToolCalls) == 0 {
		return false
	}
	call := resp.Choices[0].Message.ToolCalls[0].Function
	if call.Name != smokeToolName {
		return false
	}
	var args map[string]any
	if json.Unmarshal([]byte(call.Arguments), &args) != nil {
		return false
	}
	country, _ := args["country"].(string)
	return strings.Contains(strings.ToLower(country), "france")
}

// ── stamps: persisted verdicts, one per model id ─────────────

func smokeStampsPath() string {
	return filepath.Join(RuntimesRoot(), "smoke.json")
}

func loadSmokeStamps() map[string]bool {
	out := map[string]bool{}
	raw, err := os.ReadFile(smokeStampsPath())
	if err != nil {
		return out
	}
	_ = json.Unmarshal(raw, &out)
	return out
}

func writeSmokeStamps(stamps map[string]bool) error {
	path := smokeStampsPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(stamps, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, raw, 0o644)
}

// SmokeStamp returns a model's recorded verdict; known=false when the model
// was never smoked.
func SmokeStamp(modelID string) (passed, known bool) {
	passed, known = loadSmokeStamps()[modelID]
	return
}

// SaveSmokeStamp records a model's smoke verdict.
func SaveSmokeStamp(modelID string, passed bool) error {
	stamps := loadSmokeStamps()
	stamps[modelID] = passed
	return writeSmokeStamps(stamps)
}

// ClearSmokeStamp drops a model's verdict (delete/re-download paths).
func ClearSmokeStamp(modelID string) error {
	stamps := loadSmokeStamps()
	if _, ok := stamps[modelID]; !ok {
		return nil
	}
	delete(stamps, modelID)
	return writeSmokeStamps(stamps)
}
