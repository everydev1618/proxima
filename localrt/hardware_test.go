package localrt

import "testing"

func TestParseVMStat(t *testing.T) {
	out := `Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              100000.
Pages active:                            500000.
Pages inactive:                          200000.
Pages speculative:                        50000.
Pages purgeable:                          25000.
`
	// (100000 + 200000 + 25000 + 50000) * 16384
	if got, want := parseVMStat(out), int64(375000)*16384; got != want {
		t.Errorf("parseVMStat = %d, want %d", got, want)
	}
	if got := parseVMStat("garbage"); got != 0 {
		t.Errorf("garbage = %d", got)
	}
}

func TestParseNvidiaSmi(t *testing.T) {
	total, free, ok := parseNvidiaSmi("24576, 20480\n")
	if !ok || total != 24576<<20 || free != 20480<<20 {
		t.Errorf("parse: %d %d %v", total, free, ok)
	}
	if _, _, ok := parseNvidiaSmi(""); ok {
		t.Error("empty should fail")
	}
	if _, _, ok := parseNvidiaSmi("NVIDIA-SMI has failed"); ok {
		t.Error("error output should fail")
	}
}

func TestUMABudget(t *testing.T) {
	b := umaBudget(100*gib, 128*gib)
	if !b.UMA || b.UsableVRAMBytes != 80*gib || b.RAMAvailableBytes != 0 {
		t.Errorf("uma budget: %+v", b)
	}
}

func TestProbeBudgetRuns(t *testing.T) {
	// Smoke: on any machine the probe returns a non-degenerate budget.
	b := ProbeBudget(true)
	if b.UsableVRAMBytes <= 0 {
		t.Errorf("planning budget unusable: %+v", b)
	}
}
