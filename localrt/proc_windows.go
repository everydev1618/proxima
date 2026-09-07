//go:build windows

package localrt

import (
	"os/exec"
	"time"
)

// Windows lacks POSIX process groups; the managed runtime is not yet
// supported there (detection of external servers still works). Kill the
// router only.
func setProcessGroup(cmd *exec.Cmd) {}

func killProcessGroup(cmd *exec.Cmd, grace time.Duration) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	cmd.Process.Kill()
}
