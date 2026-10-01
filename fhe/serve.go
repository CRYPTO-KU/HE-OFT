// serve.go: the per-query cost units of serving under encryption.
//
// The aggregate head stays in ciphertext and answers queries under encryption,
// so a client learns only the labels it asks for. Per query the serving path is
//
//	logits_enc = W_enc · φ(x)          a depth-1 linear head
//	label_enc  = argmax(logits_enc)    a minimax sign circuit whose bootstraps,
//	                                    in the multiparty setting, can be served
//	                                    by collective refreshes
//	label      = threshold-decrypt(label_enc), or a key switch of label_enc
//	             to the querier's key (serve_real.go)
//
// This file measures two reusable units: one collective refresh (a multiparty
// bootstrap) over N parties, and the threshold decryption of the C-dimensional
// result. serve_argmax.go composes the refresh into the full argmax.
package main

import (
	"encoding/json"
	"fmt"
	"math"
	mrand "math/rand"
	"os"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/multiparty/mpckks"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// serveConfig is one primitive-cost scenario.
type serveConfig struct {
	n    int // number of parties in the threshold quorum
	logN int // log2 ring degree
	C    int // number of classes / logits in a query output
}

// serveResult captures the timed cost of the collective-refresh unit + decrypt.
type serveResult struct {
	Scenario     string  `json:"scenario"`
	N            int     `json:"n_parties"`
	LogN         int     `json:"log_ring_degree"`
	C            int     `json:"num_classes"`
	Levels       int     `json:"mult_levels"`
	MinLevel     int     `json:"refresh_min_level"`
	MaxLevel     int     `json:"max_level"`
	LogBound     int     `json:"refresh_log_bound"`
	RefreshMs    float64 `json:"collective_refresh_ms"` // one multiparty bootstrap (all N)
	DecryptMs    float64 `json:"threshold_decrypt_ms"`
	RefreshRelL2 float64 `json:"refresh_relative_l2"`
	Passed       bool    `json:"passed_1e-2"`
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000.0 }

// runServeSuite sweeps the quorum size N (the refresh scales with the number of
// partial decryptions) at a fixed deep modulus chain representative of the sign
// circuit's per-stage depth, and prints a paste-ready CSV block.
func runServeSuite(jsonOut string) {
	scenarios := []serveConfig{
		{n: 5, logN: 15, C: 100},
		{n: 10, logN: 15, C: 100},
		{n: 20, logN: 15, C: 100},
	}
	results := make([]serveResult, 0, len(scenarios))
	allPass := true
	for _, c := range scenarios {
		r := runServe(c)
		results = append(results, r)
		allPass = allPass && r.Passed
		printServe(r)
	}

	fmt.Println("\n--- CSV (record: results/fhe/serve_primitives.csv) ---")
	fmt.Println("n_parties,logN,C,mult_levels,min_level,max_level,log_bound,collective_refresh_ms,threshold_decrypt_ms,refresh_rel_l2,passed")
	for _, r := range results {
		fmt.Printf("%d,%d,%d,%d,%d,%d,%d,%.2f,%.2f,%.3e,%v\n",
			r.N, r.LogN, r.C, r.Levels, r.MinLevel, r.MaxLevel, r.LogBound,
			r.RefreshMs, r.DecryptMs, r.RefreshRelL2, r.Passed)
	}

	if jsonOut != "" {
		b, err := json.MarshalIndent(results, "", "  ")
		check(err)
		check(os.WriteFile(jsonOut, b, 0o644))
		fmt.Printf("\nwrote %s\n", jsonOut)
	}
	if !allPass {
		fmt.Println("\nWARN: a refresh exceeded the 1e-2 relative-L2 sanity bound")
	}
}

// runServe times one collective refresh (multiparty bootstrap) + a threshold
// decrypt for a single query's encrypted logit vector.
func runServe(c serveConfig) serveResult {
	// A deep modulus chain: one 55-bit base + twelve 45-bit levels + two 61-bit
	// key-switch primes. About 13 multiplicative levels, representative of the
	// depth a minimax sign stage consumes before it must bootstrap. Secure at logN=15
	// (QP ≈ 762 bits < the 128-bit-security budget for 2^15).
	logQ := []int{55}
	for i := 0; i < 12; i++ {
		logQ = append(logQ, 45)
	}
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            c.logN,
		LogQ:            logQ,
		LogP:            []int{61, 61},
		LogDefaultScale: 45,
	})
	check(err)

	prng, err := sampling.NewKeyedPRNG([]byte("he-ifd-serve-crs"))
	check(err)
	crs := prng

	// ---- DKG: N parties → collective public key (ideal secret = Σ sk_i) -------
	kgen := rlwe.NewKeyGenerator(params)
	sks := make([]*rlwe.SecretKey, c.n)
	for i := range sks {
		sks[i] = kgen.GenSecretKeyNew()
	}
	ckg := multiparty.NewPublicKeyGenProtocol(params)
	ckgCRP := ckg.SampleCRP(crs)
	ckgCombined := ckg.AllocateShare()
	for i := 0; i < c.n; i++ {
		share := ckg.AllocateShare()
		ckg.GenShare(sks[i], ckgCRP, &share)
		if i == 0 {
			ckgCombined = share
		} else {
			ckg.AggregateShares(share, ckgCombined, &ckgCombined)
		}
	}
	pk := rlwe.NewPublicKey(params)
	ckg.GenPublicKey(ckgCombined, ckgCRP, pk)

	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	slots := params.MaxSlots()

	// ---- encrypt one query's C logits ------------------------------------------
	rng := mrand.New(mrand.NewSource(20260713))
	logits := make([]float64, c.C)
	for j := range logits {
		logits[j] = rng.NormFloat64()
	}
	vec := make([]float64, slots)
	copy(vec, logits)
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	check(encoder.Encode(vec, pt))
	ct, err := encryptor.EncryptNew(pt)
	check(err)

	// Drop to the refresh minimum level: simulate a ciphertext that has consumed
	// its budget on a sign stage and now needs a bootstrap before the next one.
	minLevel, logBound, ok := mpckks.GetMinimumLevelForRefresh(128, params.DefaultScale(), c.n, params.Q())
	if !ok || minLevel+1 > params.MaxLevel() {
		panic(fmt.Sprintf("refresh not possible: minLevel=%d maxLevel=%d", minLevel, params.MaxLevel()))
	}
	eval := ckks.NewEvaluator(params, nil)
	ctMin := ct.CopyNew()
	eval.DropLevel(ctMin, params.MaxLevel()-minLevel)

	// ---- collective refresh (multiparty bootstrap): the per-query cost unit ---
	rfp, err := mpckks.NewRefreshProtocol(params, uint(params.LogDefaultScale()), params.Xe())
	check(err)
	crp := rfp.SampleCRP(params.MaxLevel(), crs)

	tRef := time.Now()
	var combinedR multiparty.RefreshShare
	for i := 0; i < c.n; i++ {
		share := rfp.AllocateShare(ctMin.Level(), params.MaxLevel())
		check(rfp.GenShare(sks[i], logBound, ctMin, crp, &share))
		if i == 0 {
			combinedR = share
		} else {
			check(rfp.AggregateShares(&share, &combinedR, &combinedR))
		}
	}
	refreshed := ckks.NewCiphertext(params, 1, params.MaxLevel())
	check(rfp.Finalize(ctMin, crp, combinedR, refreshed))
	refreshMs := ms(time.Since(tRef))

	// ---- threshold decrypt the refreshed logits (collective key-switch to 0) ---
	cks, err := multiparty.NewKeySwitchProtocol(params, keySwitchNoise())
	check(err)
	zeroSk := rlwe.NewSecretKey(params)
	tDec := time.Now()
	var combinedK multiparty.KeySwitchShare
	for i := 0; i < c.n; i++ {
		share := cks.AllocateShare(refreshed.Level())
		cks.GenShare(sks[i], zeroSk, refreshed, &share)
		if i == 0 {
			combinedK = share
		} else {
			check(cks.AggregateShares(share, combinedK, &combinedK))
		}
	}
	switched := ckks.NewCiphertext(params, 1, refreshed.Level())
	cks.KeySwitch(refreshed, combinedK, switched)
	dec := rlwe.NewDecryptor(params, zeroSk)
	out := make([]float64, slots)
	check(encoder.Decode(dec.DecryptNew(switched), out))
	decryptMs := ms(time.Since(tDec))

	// ---- sanity: refreshed logits must still match the plaintext logits --------
	var num, den float64
	for j := 0; j < c.C; j++ {
		e := out[j] - logits[j]
		num += e * e
		den += logits[j] * logits[j]
	}
	relL2 := math.Sqrt(num) / math.Sqrt(den)

	return serveResult{
		Scenario:     fmt.Sprintf("N=%d logN=%d C=%d", c.n, c.logN, c.C),
		N:            c.n,
		LogN:         c.logN,
		C:            c.C,
		Levels:       params.MaxLevel(),
		MinLevel:     minLevel,
		MaxLevel:     params.MaxLevel(),
		LogBound:     int(logBound),
		RefreshMs:    refreshMs,
		DecryptMs:    decryptMs,
		RefreshRelL2: relL2,
		Passed:       relL2 <= 1e-2,
	}
}

func printServe(r serveResult) {
	fmt.Printf("\n── serve %s ──────────────────────────────\n", r.Scenario)
	fmt.Printf("  ring degree           : 2^%d, %d levels\n", r.LogN, r.MaxLevel)
	fmt.Printf("  refresh min level     : %d  (logBound=%d)\n", r.MinLevel, r.LogBound)
	fmt.Printf("  collective refresh    : %.2f ms  (multiparty bootstrap, N=%d)\n", r.RefreshMs, r.N)
	fmt.Printf("  threshold decrypt     : %.2f ms\n", r.DecryptMs)
	fmt.Printf("  refresh relative L2   : %.3e\n", r.RefreshRelL2)
	fmt.Printf("  PASS (≤1e-2)          : %v\n", r.Passed)
}
