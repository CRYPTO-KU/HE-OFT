// serve_tournament.go: the encrypted argmax as a log-depth SIMD tournament.
//
// The sequential fold (serve_argmax.go) does C-1 comparisons in sequence. This
// packs the C logits into one ciphertext and reduces them with a
// ceil(log2 C)-round rotate-and-Max tournament, so the number of sequential
// comparisons drops from C-1 to ceil(log2 C). Each round's pairwise comparisons
// run in parallel across slots, and the rotations use one-time Galois keys with
// no bootstrap. This is the multiparty-CKKS form of the QuickMax reduction of
// NEXUS. It measures the per-query latency and the collective-refresh count.
package main

import (
	"encoding/json"
	"fmt"
	"math"
	mrand "math/rand"
	"os"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/comparison"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/minimax"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/multiparty/mpckks"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

type tournamentResult struct {
	Scenario     string  `json:"scenario"`
	N            int     `json:"n_parties"`
	LogN         int     `json:"log_ring_degree"`
	C            int     `json:"num_classes"`
	Cpad         int     `json:"c_padded"`
	Rounds       int     `json:"tournament_rounds"`
	RefreshCount int     `json:"collective_refreshes"`
	ArgmaxMs     float64 `json:"argmax_total_ms"`
	RefreshMs    float64 `json:"in_refresh_ms"`
	LocalMs      float64 `json:"local_eval_ms"`
	CorrectMax   bool    `json:"correct_max"`
	MaxAbsErr    float64 `json:"max_abs_error"`
}

// runTournamentSuite sweeps C at fixed N=10, logN=15, recovering per-C.
func runTournamentSuite(jsonOut string) {
	configs := []argmaxConfig{
		{10, 15, 4}, {10, 15, 6}, {10, 15, 14}, {10, 15, 77}, {10, 15, 100},
	}
	results := make([]tournamentResult, 0, len(configs))
	for _, c := range configs {
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("\n[tournament] C=%d FAILED: %v\n", c.C, r)
				}
			}()
			res := runTournament(c)
			results = append(results, res)
			printTournament(res)
		}()
	}
	fmt.Println("\n--- CSV (record: results/fhe/argmax_tournament.csv) ---")
	fmt.Println("n_parties,logN,C,c_padded,tournament_rounds,collective_refreshes,argmax_total_ms,in_refresh_ms,local_eval_ms,correct_max,max_abs_error")
	for _, r := range results {
		fmt.Printf("%d,%d,%d,%d,%d,%d,%.1f,%.1f,%.1f,%v,%.4f\n",
			r.N, r.LogN, r.C, r.Cpad, r.Rounds, r.RefreshCount,
			r.ArgmaxMs, r.RefreshMs, r.LocalMs, r.CorrectMax, r.MaxAbsErr)
	}
	if jsonOut != "" {
		b, err := json.MarshalIndent(results, "", "  ")
		check(err)
		check(os.WriteFile(jsonOut, b, 0o644))
		fmt.Printf("\nwrote %s\n", jsonOut)
	}
}

func runTournament(c argmaxConfig) tournamentResult {
	logQ := []int{55}
	for i := 0; i < 14; i++ {
		logQ = append(logQ, 45)
	}
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: c.logN, LogQ: logQ, LogP: []int{61, 61}, LogDefaultScale: 45,
	})
	check(err)

	prng, err := sampling.NewKeyedPRNG([]byte("he-ifd-serve-tournament-crs"))
	check(err)
	crs := prng

	// ---- DKG ------------------------------------------------------------------
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

	// ---- ideal secret + eval keys: relin, conjugation, power-of-2 rotations ----
	idealSk := rlwe.NewSecretKey(params)
	rQP := params.RingQP()
	for i := 0; i < c.n; i++ {
		rQP.Add(idealSk.Value, sks[i].Value, idealSk.Value)
	}
	Cpad := 1
	for Cpad < c.C {
		Cpad *= 2
	}
	galEls := []uint64{params.GaloisElementForComplexConjugation()}
	for step := 1; step < Cpad; step *= 2 {
		galEls = append(galEls, params.GaloisElement(step))
	}
	relinKey := kgen.GenRelinearizationKeyNew(idealSk)
	galKeys := kgen.GenGaloisKeysNew(galEls, idealSk)
	evk := rlwe.NewMemEvaluationKeySet(relinKey, galKeys...)

	// ---- collective-refresh bootstrapper + comparison evaluator ---------------
	minLevel, logBound, ok := mpckks.GetMinimumLevelForRefresh(128, params.DefaultScale(), c.n, params.Q())
	if !ok || minLevel+1 > params.MaxLevel() {
		panic(fmt.Sprintf("refresh not possible: minLevel=%d maxLevel=%d", minLevel, params.MaxLevel()))
	}
	rfp, err := mpckks.NewRefreshProtocol(params, uint(params.LogDefaultScale()), params.Xe())
	check(err)
	btp := &collectiveBootstrapper{params: params, sks: sks, rfp: rfp, crs: crs, minLevel: minLevel, logBound: logBound}
	eval := ckks.NewEvaluator(params, evk)
	minimaxEvl := minimax.NewEvaluator(params, eval, btp)
	cmp := comparison.NewEvaluator(params, minimaxEvl, minimax.NewPolynomial(comparison.DefaultCompositePolynomialForSign))

	// ---- pack C logits into slots 0..C-1; pad the rest just below the logits ----
	// The comparison circuit is only accurate for inputs in [-0.5, 0.5] (|a-b| <= 1),
	// so the padding must lie INSIDE that range yet strictly below every logit. A
	// far-out sentinel (e.g. -10) leaves the sign polynomial's valid domain and makes
	// it diverge. Logits in [-0.4, 0.5), pad -0.5.
	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	slots := params.MaxSlots()
	rng := mrand.New(mrand.NewSource(int64(20260714 + c.C)))
	logits := make([]float64, c.C)
	vec := make([]float64, slots)
	for i := range vec {
		vec[i] = -0.5
	}
	for j := 0; j < c.C; j++ {
		logits[j] = rng.Float64()*0.9 - 0.4
		vec[j] = logits[j]
	}
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	check(encoder.Encode(vec, pt))
	m, err := encryptor.EncryptNew(pt)
	check(err)

	// ---- ceil(log2 Cpad)-round rotate-and-Max tournament; slot 0 ends = max ----
	tArg := time.Now()
	for step := 1; step < Cpad; step *= 2 {
		// keep the accumulator at full level before each comparison (same reason as
		// the sequential fold: a depleted ciphertext drives the sign refresh below the
		// secure minimum). Rotate the refreshed ciphertext so both operands align.
		if m.Level() < params.MaxLevel() {
			m, err = btp.Bootstrap(m)
			check(err)
		}
		rot, err := eval.RotateNew(m, step)
		check(err)
		m, err = cmp.Max(m, rot)
		check(err)
	}
	argmaxMs := ms(time.Since(tArg))

	// ---- verify slot 0 holds the plaintext max --------------------------------
	dec := rlwe.NewDecryptor(params, idealSk)
	out := make([]float64, slots)
	check(encoder.Decode(dec.DecryptNew(m), out))
	trueMax := logits[0]
	for _, v := range logits {
		if v > trueMax {
			trueMax = v
		}
	}
	absErr := math.Abs(out[0] - trueMax)
	rounds := 0
	for s := 1; s < Cpad; s *= 2 {
		rounds++
	}

	return tournamentResult{
		Scenario:     fmt.Sprintf("N=%d logN=%d C=%d", c.n, c.logN, c.C),
		N:            c.n,
		LogN:         c.logN,
		C:            c.C,
		Cpad:         Cpad,
		Rounds:       rounds,
		RefreshCount: btp.count,
		ArgmaxMs:     argmaxMs,
		RefreshMs:    btp.totalMs,
		LocalMs:      argmaxMs - btp.totalMs,
		CorrectMax:   absErr < 0.05,
		MaxAbsErr:    absErr,
	}
}

func printTournament(r tournamentResult) {
	fmt.Printf("\n── tournament %s ─────────────────────────────\n", r.Scenario)
	fmt.Printf("  C padded              : %d  (%d rounds)\n", r.Cpad, r.Rounds)
	fmt.Printf("  collective refreshes  : %d\n", r.RefreshCount)
	fmt.Printf("  argmax total          : %.1f ms  (%.2f s)\n", r.ArgmaxMs, r.ArgmaxMs/1000)
	fmt.Printf("    in refreshes        : %.1f ms\n", r.RefreshMs)
	fmt.Printf("    local eval          : %.1f ms\n", r.LocalMs)
	fmt.Printf("  correct max           : %v  (abs err %.4f)\n", r.CorrectMax, r.MaxAbsErr)
}
