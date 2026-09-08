// First-run bootstrap: a machine with no model server and no staged models
// gets offered the catalog's starter — a small, chatty model — so the first
// `proxima` ends in a chat, not an error telling the user to run three more
// commands. Consent lives on the controlling terminal; a headless run keeps
// the old error path (nothing multi-gigabyte downloads without a human).
package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/everydev1618/proxima/localrt"
)

// bootstrapStarter offers to stage the starter model and returns its model
// id. "" means the offer was declined or unavailable (no starter fits, no
// terminal) — the caller falls back to the no-server error and its help.
func bootstrapStarter(ctx context.Context) (string, error) {
	entry, choice := localrt.StarterEntry(localrt.Catalog(), localrt.ProbeBudget(true))
	if entry == nil {
		return "", nil
	}
	tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		return "", nil // headless: never auto-download gigabytes
	}
	defer tty.Close()
	if !confirmFirstRun(tty, tty, entry, choice.Variant) {
		return "", nil
	}
	if err := installRuntimeAndModel(ctx, entry, choice.Variant, localrt.DefaultRuntimeTag); err != nil {
		return "", err
	}
	return choice.Variant.ModelID(), nil
}

// confirmFirstRun asks for download consent. Enter defaults to yes — the
// point of the starter is zero friction — but closed input is a no.
func confirmFirstRun(in io.Reader, out io.Writer, entry *localrt.CatalogEntry, v localrt.QuantVariant) bool {
	fmt.Fprintf(out,
		"\nNo local model server found. proxima can set itself up:\n"+
			"  download %s (%.1f GB) + the llama.cpp runtime, then start chatting.\n"+
			"Continue? [Y/n] ",
		entry.DisplayName, float64(entry.DownloadBytes(v))/1e9)
	line, err := bufio.NewReader(in).ReadString('\n')
	if err != nil {
		fmt.Fprintln(out, "→ skipped (input closed)")
		return false
	}
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "", "y", "yes":
		return true
	default:
		return false
	}
}
