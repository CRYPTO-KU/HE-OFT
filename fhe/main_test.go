package main

import (
	"fmt"
	"testing"
)

// TestProtocolCorrectness validates the end-to-end multiparty-CKKS aggregation
// (DKG → encrypt → server linear combine → threshold decrypt) against the
// plaintext float64 reference for the headline head dimensions and client
// counts, plus a larger d that spans multiple ciphertexts per client.
func TestProtocolCorrectness(t *testing.T) {
	cases := []config{
		{d: 5130, n: 5, logN: 14},   // 512→10 head, N=5
		{d: 5130, n: 10, logN: 14},  // 512→10 head, N=10
		{d: 7700, n: 5, logN: 14},   // 768→10 head, N=5
		{d: 7700, n: 10, logN: 14},  // 768→10 head, N=10
		{d: 20000, n: 10, logN: 14}, // spans 3 ciphertexts/client (cost sanity)
	}
	for _, c := range cases {
		c := c
		t.Run(fmt.Sprintf("d%d_N%d", c.d, c.n), func(t *testing.T) {
			r := run(c)
			if !r.Passed {
				t.Fatalf("relative L2 %.3e exceeds 1e-3", r.RelL2Error)
			}
			if r.MultDepth != 1 {
				t.Fatalf("multiplicative depth = %d, want 1 (PT×CT once + Rescale)", r.MultDepth)
			}
			if c.d > r.Slots && r.CtPerClient < 2 {
				t.Fatalf("d=%d > slots=%d but ctPerClient=%d", c.d, r.Slots, r.CtPerClient)
			}
			t.Logf("rel L2 %.3e, depth %d, %d ct/client, %d B/ct",
				r.RelL2Error, r.MultDepth, r.CtPerClient, r.BytesPerCt)
		})
	}
}
