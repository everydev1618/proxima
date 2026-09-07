package localrt

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"
	"time"
)

// smokeSupervisor wires a Supervisor at a fake /v1/chat/completions endpoint.
func smokeSupervisor(t *testing.T, handler http.HandlerFunc) *Supervisor {
	t.Helper()
	t.Setenv("VEGA_HOME", t.TempDir())
	ts := httptest.NewServer(handler)
	t.Cleanup(ts.Close)
	u, err := url.Parse(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	sup := NewSupervisor(t.TempDir(), t.TempDir())
	sup.Port, _ = strconv.Atoi(u.Port())
	return sup
}

func toolCallResponse(name, arguments string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{
				"tool_calls": []map[string]any{{
					"type": "function",
					"function": map[string]string{
						"name": name, "arguments": arguments,
					},
				}},
			}}},
		})
	}
}

func TestTouchToolCallPass(t *testing.T) {
	var gotReq map[string]any
	sup := smokeSupervisor(t, func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		json.Unmarshal(raw, &gotReq)
		toolCallResponse("lookup_capital", `{"country": "France"}`)(w, r)
	})
	if !sup.TouchToolCall("m", 5*time.Second) {
		t.Error("well-formed tool call should pass")
	}
	// The request must actually offer the tool and force tool choice.
	if gotReq["tool_choice"] != "required" || gotReq["tools"] == nil {
		t.Errorf("request did not force a tool call: %v", gotReq)
	}
}

func TestTouchToolCallFailures(t *testing.T) {
	cases := map[string]http.HandlerFunc{
		"wrong tool":        toolCallResponse("get_weather", `{"country": "France"}`),
		"garbage arguments": toolCallResponse("lookup_capital", `{not json`),
		"wrong argument":    toolCallResponse("lookup_capital", `{"country": "Spain"}`),
		"no tool call": func(w http.ResponseWriter, r *http.Request) {
			json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{{"message": map[string]string{"content": "Paris"}}},
			})
		},
	}
	for name, handler := range cases {
		sup := smokeSupervisor(t, handler)
		if sup.TouchToolCall("m", 5*time.Second) {
			t.Errorf("%s should fail the smoke eval", name)
		}
	}
}

func TestSmokeStamps(t *testing.T) {
	t.Setenv("VEGA_HOME", t.TempDir())
	if _, known := SmokeStamp("m1"); known {
		t.Error("unstamped model reported known")
	}
	if err := SaveSmokeStamp("m1", true); err != nil {
		t.Fatal(err)
	}
	if err := SaveSmokeStamp("m2", false); err != nil {
		t.Fatal(err)
	}
	if passed, known := SmokeStamp("m1"); !known || !passed {
		t.Errorf("m1 stamp = %v %v", passed, known)
	}
	if passed, known := SmokeStamp("m2"); !known || passed {
		t.Errorf("m2 stamp = %v %v", passed, known)
	}
	if err := ClearSmokeStamp("m1"); err != nil {
		t.Fatal(err)
	}
	if _, known := SmokeStamp("m1"); known {
		t.Error("cleared stamp survived")
	}
}
