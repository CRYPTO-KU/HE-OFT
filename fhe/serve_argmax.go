// serve_argmax.go: the encrypted argmax as a sequential fold.
//
// The querier must learn the label and nothing else, so the argmax is computed
// under encryption and only the label leaves the ciphertext domain. The argmax
// is a minimax sign circuit (comparison.Max folded over the C logits), and in
// the multiparty setting its bootstraps can be served by collective refreshes.
// This file wires mpckks.RefreshProtocol into the bootstrapping.Bootstrapper that
// the minimax and comparison evaluators require, then measures the per-query
// argmax latency and the number of refreshes it costs, over C in
// {4, 6, 14, 77, 100}.
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

// collectiveBootstrapper realises bootstrapping.Bootstrapper via the multiparty
// collective refresh: every bootstrap the sign circuit needs becomes one refresh
// round over the N parties (the per-query multiparty cost). It counts and times
// them so we can separate refresh cost from local evaluator work.
type collectiveBootstrapper struct {
	params   ckks.Parameters
	sks      []*rlwe.SecretKey
	rfp      mpckks.RefreshProtocol
	crs      sampling.PRNG
	minLevel int
	logBound uint
	count    int
	totalMs  float64
}

func (b *collectiveBootstrapper) Bootstrap(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	crp := b.rfp.SampleCRP(b.params.MaxLevel(), b.crs)
	t := time.Now()
	var combined multiparty.RefreshShare
	for i := range b.sks {
		share := b.rfp.AllocateShare(ct.Level(), b.params.MaxLevel())
		if err := b.rfp.GenShare(b.sks[i], b.logBound, ct, crp, &share); err != nil {
			return nil, err
		}
		if i == 0 {
			combined = share
		} else if err := b.rfp.AggregateShares(&share, &combined, &combined); err != nil {
			return nil, err
		}
	}
	out := ckks.NewCiphertext(b.params, 1, b.params.MaxLevel())
	if err := b.rfp.Finalize(ct, crp, combined, out); err != nil {
		return nil, err
	}
	b.totalMs += ms(time.Since(t))
	b.count++
	return out, nil
}

func (b *collectiveBootstrapper) Count() int      { return b.count }
func (b *collectiveBootstrapper) Millis() float64 { return b.totalMs }

func (b *collectiveBootstrapper) BootstrapMany(cts []rlwe.Ciphertext) ([]rlwe.Ciphertext, error) {
	for i := range cts {
		r, err := b.Bootstrap(&cts[i])
		if err != nil {
			return nil, err
		}
		cts[i] = *r
	}
	return cts, nil
}

func (b *collectiveBootstrapper) Depth() int             { return 0 }
func (b *collectiveBootstrapper) MinimumInputLevel() int { return b.minLevel }
func (b *collectiveBootstrapper) OutputLevel() int       { return b.params.MaxLevel() }

type argmaxConfig struct{ n, logN, C int }

type argmaxResult struct {
	Scenario     string  `json:"scenario"`
	N            int     `json:"n_parties"`
	LogN         int     `json:"log_ring_degree"`
	C            int     `json:"num_classes"`
	MaxLevel     int     `json:"max_level"`
	Comparisons  int     `json:"pairwise_max_ops"`
	RefreshCount int     `json:"collective_refreshes"`
	ArgmaxMs     float64 `json:"argmax_total_ms"` // wall time of the C-1 Max fold
	RefreshMs    float64 `json:"in_refresh_ms"`   // time inside collective refreshes
	LocalMs      float64 `json:"local_eval_ms"`   // server-local comparison work
	CorrectMax   bool    `json:"correct_max"`
	MaxAbsErr    float64 `json:"max_abs_error"`
}

// runArgmaxSuite sweeps the label-space size C (fixed N=10, logN=15), recovering
// gracefully per-C so a slow/failed large-C case still leaves the smaller ones.
func runArgmaxSuite(jsonOut string) {
	configs := []argmaxConfig{
		{10, 15, 4}, {10, 15, 6}, {10, 15, 14}, {10, 15, 77}, {10, 15, 100},
	}
	results := make([]argmaxResult, 0, len(configs))
	for _, c := range configs {
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("\n[argmax] C=%d FAILED: %v\n", c.C, r)
				}
			}()
			res := runArgmax(c)
			results = append(results, res)
			printArgmax(res)
		}()
	}

	fmt.Println("\n--- CSV (record: results/fhe/argmax_cost.csv) ---")
	fmt.Println("n_parties,logN,C,max_level,pairwise_max_ops,collective_refreshes,argmax_total_ms,in_refresh_ms,local_eval_ms,correct_max,max_abs_error")
	for _, r := range results {
		fmt.Printf("%d,%d,%d,%d,%d,%d,%.1f,%.1f,%.1f,%v,%.4f\n",
			r.N, r.LogN, r.C, r.MaxLevel, r.Comparisons, r.RefreshCount,
			r.ArgmaxMs, r.RefreshMs, r.LocalMs, r.CorrectMax, r.MaxAbsErr)
	}
	if jsonOut != "" {
		b, err := json.MarshalIndent(results, "", "  ")
		check(err)
		check(os.WriteFile(jsonOut, b, 0o644))
		fmt.Printf("\nwrote %s\n", jsonOut)
	}
}

func runArgmax(c argmaxConfig) argmaxResult {
	// Deep chain: 55 + 14×45 + 2×61(P) ≈ 807 bits, secure at logN=15. More usable
	// levels between refreshes → fewer refreshes per sign evaluation.
	logQ := []int{55}
	for i := 0; i < 14; i++ {
		logQ = append(logQ, 45)
	}
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: c.logN, LogQ: logQ, LogP: []int{61, 61}, LogDefaultScale: 45,
	})
	check(err)

	prng, err := sampling.NewKeyedPRNG([]byte("he-ifd-serve-argmax-crs"))
	check(err)
	crs := prng

	// ---- DKG: N parties → collective public key -------------------------------
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

	// ---- eval keys from the ideal secret Σ sk_i (one-time server setup; a
	//      deployment generates them collectively, which gives the same key and
	//      the same per-query evaluation speed) -------------------------------
	idealSk := rlwe.NewSecretKey(params)
	rQP := params.RingQP()
	for i := 0; i < c.n; i++ {
		rQP.Add(idealSk.Value, sks[i].Value, idealSk.Value)
	}
	relinKey := kgen.GenRelinearizationKeyNew(idealSk)
	galKey := kgen.GenGaloisKeyNew(params.GaloisElementForComplexConjugation(), idealSk)
	evk := rlwe.NewMemEvaluationKeySet(relinKey, galKey)

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

	// ---- encrypt C logits in [-0.5,0.5], one ciphertext each ------------------
	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	slots := params.MaxSlots()
	rng := mrand.New(mrand.NewSource(int64(20260714 + c.C)))
	logits := make([]float64, c.C)
	cts := make([]*rlwe.Ciphertext, c.C)
	for j := 0; j < c.C; j++ {
		logits[j] = rng.Float64() - 0.5
		vec := make([]float64, slots)
		for k := range vec {
			vec[k] = logits[j]
		}
		pt := ckks.NewPlaintext(params, params.MaxLevel())
		check(encoder.Encode(vec, pt))
		ct, err := encryptor.EncryptNew(pt)
		check(err)
		cts[j] = ct
	}

	// ---- argmax: fold Max over the C logits (C-1 encrypted comparisons; the
	//      minimax evaluator calls a collective refresh when levels run out).
	tArg := time.Now()
	m := cts[0]
	for i := 1; i < c.C; i++ {
		// Keep the running max at full level before each comparison. The Max circuit
		// leaves its output a level or two down, and that drift accumulates over the
		// fold. Left unchecked it drives the sign circuit's internal refresh below the
		// secure minimum level after a few comparisons, and GenShare fails.
		// Refreshing the accumulator resets it, at one extra collective refresh per
		// comparison, which makes this an upper bound on the argmax cost.
		if m.Level() < params.MaxLevel() {
			m, err = btp.Bootstrap(m)
			check(err)
		}
		m, err = cmp.Max(m, cts[i])
		check(err)
	}
	argmaxMs := ms(time.Since(tArg))

	// ---- verify: decrypted max value matches the plaintext max -----------------
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

	return argmaxResult{
		Scenario:     fmt.Sprintf("N=%d logN=%d C=%d", c.n, c.logN, c.C),
		N:            c.n,
		LogN:         c.logN,
		C:            c.C,
		MaxLevel:     params.MaxLevel(),
		Comparisons:  c.C - 1,
		RefreshCount: btp.count,
		ArgmaxMs:     argmaxMs,
		RefreshMs:    btp.totalMs,
		LocalMs:      argmaxMs - btp.totalMs,
		CorrectMax:   absErr < 0.05,
		MaxAbsErr:    absErr,
	}
}

func printArgmax(r argmaxResult) {
	fmt.Printf("\n── argmax %s ─────────────────────────────\n", r.Scenario)
	fmt.Printf("  pairwise Max ops      : %d\n", r.Comparisons)
	fmt.Printf("  collective refreshes  : %d\n", r.RefreshCount)
	fmt.Printf("  argmax total          : %.1f ms  (%.2f s)\n", r.ArgmaxMs, r.ArgmaxMs/1000)
	fmt.Printf("    in refreshes        : %.1f ms  (%.0f%%)\n", r.RefreshMs, 100*r.RefreshMs/math.Max(r.ArgmaxMs, 1))
	fmt.Printf("    local eval          : %.1f ms\n", r.LocalMs)
	fmt.Printf("  correct max           : %v  (abs err %.4f)\n", r.CorrectMax, r.MaxAbsErr)
}
