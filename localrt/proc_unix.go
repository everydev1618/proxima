//go:build !windows

package localrt

import (
	"os/exec"
	"syscall"
	"time"
)

// setProcessGroup puts the child in its own process group so a group kill
// reaches the router's model children too — even after a router crash
// orphans them (they stay in the group).
func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// killProcessGroup terminates the process group: SIGTERM, a grace wait, then
// SIGKILL. grace 0 skips straight to SIGKILL (crash-cleanup path).
func killProcessGroup(cmd *exec.Cmd, grace time.Duration) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	pgid := -cmd.Process.Pid
	if grace <= 0 {
		syscall.Kill(pgid, syscall.SIGKILL)
		return
	}
	syscall.Kill(pgid, syscall.SIGTERM)
	deadline := time.Now().Add(grace)
	for time.Now().Before(deadline) {
		// Signal 0 probes group liveness.
		if err := syscall.Kill(pgid, 0); err != nil {
			return
		}
		time.Sleep(200 * time.Millisecond)
	}
	syscall.Kill(pgid, syscall.SIGKILL)
}
