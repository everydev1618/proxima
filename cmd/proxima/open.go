// Browser auto-open: the boot card names the chat address immediately; the
// tab appears only once the server actually answers, so the user never lands
// on a connection-refused page.
package main

import (
	"context"
	"net/http"
	"os/exec"
	"runtime"
	"time"
)

// openBrowser launches the platform browser. A var so tests stub it.
var openBrowser = func(url string) {
	switch runtime.GOOS {
	case "darwin":
		exec.Command("open", url).Start()
	case "windows":
		exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	default:
		exec.Command("xdg-open", url).Start()
	}
}

var openPollInterval = 300 * time.Millisecond

// openWhenReady polls url in the background and opens the browser on the
// first non-5xx answer. Gives up quietly after 90s — a failed boot already
// reports itself on the terminal.
func openWhenReady(ctx context.Context, url string) {
	go func() {
		client := &http.Client{Timeout: time.Second}
		deadline := time.Now().Add(90 * time.Second)
		for time.Now().Before(deadline) && ctx.Err() == nil {
			if resp, err := client.Get(url); err == nil {
				resp.Body.Close()
				if resp.StatusCode < 500 {
					openBrowser(url)
					return
				}
			}
			time.Sleep(openPollInterval)
		}
	}()
}
