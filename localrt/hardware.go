// Live hardware budget probe. Port of hermes-agent's
// local_runtime/hardware.py (MIT, Nous Research — see NOTICE).
//
// Budget-source rule: discrete cards may trust the device query (measured
// honest within rounding); unified-memory devices must budget from OS free
// physical memory minus headroom — their device queries have been observed
// off by 3x in both directions. Every probe here must work under a stripped
// PATH.
//
// Divergence from the Python original: the CUDA-driver-API unified-pool
// probe (nvcuda/libcuda via ctypes, for NVIDIA carve-out devices like
// DGX/Jetson) is not ported — it needs dlopen. On those machines the
// nvidia-smi carve-out is what gets budgeted: smaller than the real pool,
// which errs safe. Revisit with purego if the hardware shows up.

package localrt

import (
	"os/exec"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"
)

const (
	// Reserve carved off the card before any grant: the desktop's own
	// co-residents (compositor, browser, Electron) measure ~2-2.5 GiB, and a
	// window granted into that space demotes silently. 7-9% covers big
	// cards; the 2 GiB floor is what a 512 MiB floor failed to cover.
	marginFloor    = int64(2) << 30
	marginFraction = 0.09
	// UMA headroom: on unified-memory machines the model shares physical
	// memory with the OS and every app, so budget from RAM minus this
	// fraction.
	umaHeadroomFraction = 0.20
)

func cmdOutput(timeout time.Duration, argv ...string) string {
	cmd := exec.Command(argv[0], argv[1:]...)
	done := make(chan struct{})
	var out []byte
	go func() { out, _ = cmd.Output(); close(done) }()
	select {
	case <-done:
	case <-time.After(timeout):
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
		<-done
	}
	return string(out)
}

var vmStatPageRE = regexp.MustCompile(`page size of (\d+)`)

// parseVMStat derives reclaimable-on-demand bytes from macOS vm_stat output:
// free + inactive + purgeable + speculative (the speculative pool is dropped
// by the OS under pressure too). 0 when unparseable.
func parseVMStat(out string) int64 {
	page := int64(16384)
	if m := vmStatPageRE.FindStringSubmatch(out); m != nil {
		page, _ = strconv.ParseInt(m[1], 10, 64)
	}
	var pages int64
	found := false
	for _, key := range []string{"Pages free", "Pages inactive", "Pages purgeable", "Pages speculative"} {
		re := regexp.MustCompile(key + `:\s+(\d+)\.`)
		if m := re.FindStringSubmatch(out); m != nil {
			n, _ := strconv.ParseInt(m[1], 10, 64)
			pages += n
			found = true
		}
	}
	if !found {
		return 0
	}
	return pages * page
}

// ramBytes returns (total, available) physical memory.
func ramBytes() (int64, int64) {
	if runtime.GOOS == "darwin" {
		// macOS getconf has no _PHYS_PAGES — sysctl is the platform truth.
		total, _ := strconv.ParseInt(strings.TrimSpace(
			cmdOutput(5*time.Second, "/usr/sbin/sysctl", "-n", "hw.memsize")), 10, 64)
		if total <= 0 {
			return 0, 0
		}
		avail := total / 2 // conservative fallback
		if a := parseVMStat(cmdOutput(5*time.Second, "/usr/bin/vm_stat")); a > 0 {
			avail = a
		}
		return total, avail
	}
	// POSIX
	page, _ := strconv.ParseInt(strings.TrimSpace(cmdOutput(5*time.Second, "getconf", "PAGE_SIZE")), 10, 64)
	if page <= 0 {
		page = 4096
	}
	phys, _ := strconv.ParseInt(strings.TrimSpace(cmdOutput(5*time.Second, "getconf", "_PHYS_PAGES")), 10, 64)
	total := phys * page
	avail := total / 2
	if ap, _ := strconv.ParseInt(strings.TrimSpace(cmdOutput(5*time.Second, "getconf", "_AVPHYS_PAGES")), 10, 64); ap > 0 {
		avail = ap * page
	}
	return total, avail
}

// nvidiaSmiPath finds nvidia-smi: PATH first (respects user overrides), then
// the WSL install location the Windows driver exposes without guaranteeing
// PATH presence.
func nvidiaSmiPath() string {
	if p, err := exec.LookPath("nvidia-smi"); err == nil {
		return p
	}
	if runtime.GOOS == "linux" {
		candidate := "/usr/lib/wsl/lib/nvidia-smi"
		if _, err := exec.LookPath(candidate); err == nil {
			return candidate
		}
	}
	return ""
}

// parseNvidiaSmi parses "total, free" MiB CSV into bytes; (0,0,false) when
// unparseable.
func parseNvidiaSmi(out string) (total, free int64, ok bool) {
	line := strings.TrimSpace(out)
	if i := strings.IndexByte(line, '\n'); i >= 0 {
		line = line[:i]
	}
	parts := strings.Split(line, ",")
	if len(parts) != 2 {
		return 0, 0, false
	}
	t, err1 := strconv.ParseInt(strings.TrimSpace(parts[0]), 10, 64)
	f, err2 := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
	if err1 != nil || err2 != nil {
		return 0, 0, false
	}
	return t << 20, f << 20, true
}

// nvidiaVRAM returns (total, free) bytes from nvidia-smi, or ok=false.
func nvidiaVRAM() (total, free int64, ok bool) {
	exe := nvidiaSmiPath()
	if exe == "" {
		return 0, 0, false
	}
	return parseNvidiaSmi(cmdOutput(10*time.Second, exe,
		"--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"))
}

// DetectGPUVendor returns a best-effort GPU vendor string for backend
// selection ("" when unknown).
func DetectGPUVendor() string {
	exe := nvidiaSmiPath()
	if exe == "" {
		return ""
	}
	out := strings.TrimSpace(cmdOutput(10*time.Second, exe, "--query-gpu=name", "--format=csv,noheader"))
	if out == "" {
		return ""
	}
	if i := strings.IndexByte(out, '\n'); i >= 0 {
		out = out[:i]
	}
	return "nvidia " + out
}

func umaBudget(base, total int64) HardwareBudget {
	usable := int64(float64(base) * (1 - umaHeadroomFraction))
	if usable < 0 {
		usable = 0
	}
	return HardwareBudget{UsableVRAMBytes: usable, TotalDeviceBytes: total, UMA: true}
}

// ProbeBudget constructs the budget per the source rules above.
//
// planning=false: LIVE budget (free VRAM now) for launch-time fit and growth
// re-grants. planning=true: CAPACITY budget (total minus margin) for catalog
// pricing and quant selection — pricing against live-free while a model was
// loaded made every row read as too large. The managed server
// unloads/relaunches itself, so capacity is real.
func ProbeBudget(planning bool) HardwareBudget {
	ramTotal, ramAvail := ramBytes()
	total, free, hasNvidia := nvidiaVRAM()

	if !hasNvidia {
		// No NVIDIA device visible: Metal/Vulkan/CPU paths budget from RAM
		// as UMA (Apple Silicon) — conservative for discrete AMD until a
		// vendor probe lands.
		base := ramAvail
		if planning {
			base = ramTotal
		}
		return umaBudget(base, ramTotal)
	}

	margin := int64(float64(total) * marginFraction)
	if margin < marginFloor {
		margin = marginFloor
	}
	vram := free
	ram := ramAvail
	if planning {
		vram = total
		ram = ramTotal
	}
	usable := vram - margin
	if usable < 0 {
		usable = 0
	}
	return HardwareBudget{UsableVRAMBytes: usable, TotalDeviceBytes: total,
		RAMAvailableBytes: ram, UMA: false}
}
