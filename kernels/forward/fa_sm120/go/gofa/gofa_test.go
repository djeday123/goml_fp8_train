package gofa

import (
	"testing"
)

func TestVersionNotEmpty(t *testing.T) {
	v := Version()
	if v == "" {
		t.Fatalf("Version() returned empty")
	}
	t.Logf("libfa_sm120 version: %s", v)
}

func TestDispatchTable(t *testing.T) {
	tests := []struct {
		bh, sl, hd, ca, wnd int
		expectName          string
	}{
		{64, 8192, 128, 0, 0, "v121r"}, // primary peak
		{128, 8192, 128, 0, 0, "v121r"},
		{4, 1024, 128, 0, 0, "v122"},
		{4, 4096, 128, 0, 0, "v118"},
		{4, 8192, 128, 1, 1024, "v117b"},
		{64, 8192, 128, 1, 1024, "v121"},
		{16, 4096, 128, 0, 0, "v96b"},
		{64, 8192, 64, 0, 0, "v89"},
		{4, 1024, 64, 0, 0, "v80b"},
	}
	for _, tc := range tests {
		_, name := DispatchSelect(tc.bh, tc.sl, tc.hd, tc.ca, tc.wnd)
		if name != tc.expectName {
			t.Errorf("DispatchSelect(bh=%d sl=%d hd=%d ca=%d wnd=%d) = %s, want %s",
				tc.bh, tc.sl, tc.hd, tc.ca, tc.wnd, name, tc.expectName)
		}
	}
}

func TestCreateOnNonBlackwellSkips(t *testing.T) {
	ctx, err := Create()
	if err != nil {
		// On non-sm_120a, this is expected.
		if e, ok := err.(*Error); ok && e.Status == ErrUnsupportedArch {
			t.Skipf("non-sm_120a card: %v", err)
		} else {
			t.Fatalf("Create failed unexpectedly: %v", err)
		}
		return
	}
	defer ctx.Destroy()
	t.Logf("created context on sm_120a")
}
