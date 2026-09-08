package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/everydev1618/proxima/localrt"
)

func starterFixture() (*localrt.CatalogEntry, localrt.QuantVariant) {
	e := &localrt.CatalogEntry{
		ID: "tiny-chat", DisplayName: "Tiny Chat 4B",
		Variants: []localrt.QuantVariant{{Quant: "Q4_K_M",
			Files: []localrt.AssetFile{{Path: "Tiny-Chat-4B-Q4_K_M.gguf", SizeBytes: 2_900_000_000}}}},
	}
	return e, e.Variants[0]
}

func TestConfirmFirstRun(t *testing.T) {
	e, v := starterFixture()

	cases := []struct {
		input string
		want  bool
	}{
		{"y\n", true},
		{"yes\n", true},
		{"\n", true}, // enter = yes: the whole point is zero-friction
		{"n\n", false},
		{"no\n", false},
		{"", false}, // closed input (piped run) must not auto-download
	}
	for _, c := range cases {
		out := &bytes.Buffer{}
		got := confirmFirstRun(strings.NewReader(c.input), out, e, v)
		if got != c.want {
			t.Errorf("input %q: got %v, want %v", c.input, got, c.want)
		}
		if !strings.Contains(out.String(), "Tiny Chat 4B") {
			t.Errorf("prompt does not name the model: %q", out.String())
		}
		if !strings.Contains(out.String(), "2.9 GB") {
			t.Errorf("prompt does not state the download size: %q", out.String())
		}
	}
}
