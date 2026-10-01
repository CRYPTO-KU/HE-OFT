// serve_btp.go: the tournament argmax with level restoration done by the serving
// party alone, under collectively generated bootstrapping keys.
//
// serve_tournament.go restores levels with a collective refresh, which needs a
// share from every client for every restoration. The protocol specifies the other
// arrangement: the parties generate bootstrapping keys once, collectively, and the
// serving party refreshes on its own afterwards. Evaluation under a collectively
// generated key is identical to evaluation under a single key, so the two are the
// same circuit with the same output; they differ in who does the work and in what
// crosses the network per query. This file measures the second one, so the latency
// the paper prices its per-query traffic against is the latency of the design it
// describes.
//
// The bootstrapping ring is 2^16 even when the evaluated ring is 2^15: Lattigo's
// bootstrapping circuit switches the ciphertext up for the modulus raising and back
// down afterwards. The ring the tournament arithmetic runs in is reported alongside
// the ring the bootstrapping runs in, and so is the fresh ciphertext size, which is
// what sets the per-query traffic.
package main

import (
	"encoding/json"
	"fmt"
	"math"
	mrand "math/rand"
	"os"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/comparison"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/minimax"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// countingBootstrapper times and counts the server-local bootstraps, so the
// refresh time can be separated from the local evaluator work exactly as the
// collective version separates them.
type countingBootstrapper struct {
	inner   bootstrapping.Bootstrapper
	count   int
	totalMs float64
}

func (b *countingBootstrapper) Bootstrap(ct *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	t := time.Now()
	out, err := b.inner.Bootstrap(ct)
	b.totalMs += ms(time.Since(t))
	b.count++
	return out, err
}

func (b *countingBootstrapper) Count() int      { return b.count }
func (b *countingBootstrapper) Millis() float64 { return b.totalMs }

func (b *countingBootstrapper) BootstrapMany(cts []rlwe.Ciphertext) ([]rlwe.Ciphertext, error) {
	for i := range cts {
		r, err := b.Bootstrap(&cts[i])
		if err != nil {
			return nil, err
		}
		cts[i] = *r
	}
	return cts, nil
}

func (b *countingBootstrapper) Depth() int             { return b.inner.Depth() }
func (b *countingBootstrapper) MinimumInputLevel() int { return b.inner.MinimumInputLevel() }
func (b *countingBootstrapper) OutputLevel() int       { return b.inner.OutputLevel() }

type btpServeRow struct {
	N              int     `json:"n_parties"`
	EvalLogN       int     `json:"eval_log_ring_degree"`
	BtpLogN        int     `json:"bootstrapping_log_ring_degree"`
	ResidualLevels int     `json:"residual_max_level"`
	C              int     `json:"num_classes"`
	Cpad           int     `json:"c_padded"`
	Rounds         int     `json:"tournament_rounds"`
	Bootstraps     int     `json:"server_bootstraps"`
	TotalMs        float64 `json:"argmax_total_ms"`
	BtpMs          float64 `json:"in_bootstrap_ms"`
	LocalMs        float64 `json:"local_eval_ms"`
	CorrectMax     bool    `json:"correct_max"`
	MaxAbsErr      float64 `json:"max_abs_error"`
	CtBytes        int     `json:"fresh_ciphertext_bytes"`
	BtpKeyBytes    int     `json:"bootstrapping_key_bytes"`
	BtpKeyGenMs    float64 `json:"bootstrapping_key_gen_ms"`
}

// runServeBtpSuite measures the tournament under server-side bootstrapping over
// the same C sweep the collective-refresh tournament used.
func runServeBtpSuite(jsonOut string) {
	fmt.Println("=== tournament argmax with server-side bootstrapping ===")
	fmt.Println()

	// The residual chain is the one the collective-refresh tournament used, 55 +
	// 14x45 at ring 2^15. Its moduli must be NTT-friendly for the bootstrapping
	// ring as well, which is what LogNthRoot = LogN + 2 buys. The deepest chain
	// whose bootstrapping parameters stay inside the 128-bit ceiling for a ring of
	// 2^16 is the one measured; every candidate's modulus is printed either way.
	const btpLogQPCeiling = 1553.0
	btpLogN := 16
	var (
		residual  ckks.Parameters
		btpParams bootstrapping.Parameters
		nPrimes   int
		built     bool
		overCeil  bool
	)
	type cand struct {
		np       int
		r        ckks.Parameters
		bp       bootstrapping.Parameters
		btpLogQP float64
	}
	var cands []cand
	for _, np := range []int{14, 12, 10, 8} {
		logQ := []int{55}
		for i := 0; i < np; i++ {
			logQ = append(logQ, 45)
		}
		r, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN: 15, LogNthRoot: 17, LogQ: logQ, LogP: []int{61, 61}, LogDefaultScale: 45,
		})
		if err != nil {
			fmt.Printf("  residual 55+%dx45 at 2^15 rejected: %v\n", np, err)
			continue
		}
		bp, err := bootstrapping.NewParametersFromLiteral(r, bootstrapping.ParametersLiteral{LogN: &btpLogN})
		if err != nil {
			fmt.Printf("  bootstrapping over 55+%dx45 rejected: %v\n", np, err)
			continue
		}
		q := bp.BootstrappingParameters.LogQP()
		fmt.Printf("  candidate 55+%dx45 : residual logQP %.1f, bootstrapping logQP %.1f\n",
			np, r.LogQP(), q)
		cands = append(cands, cand{np, r, bp, q})
	}
	for _, c := range cands {
		if c.btpLogQP <= btpLogQPCeiling {
			residual, btpParams, nPrimes, built = c.r, c.bp, c.np, true
			break
		}
	}
	if !built && len(cands) > 0 {
		best := cands[len(cands)-1]
		for _, c := range cands {
			if c.btpLogQP < best.btpLogQP {
				best = c
			}
		}
		residual, btpParams, nPrimes, built, overCeil = best.r, best.bp, best.np, true, true
		fmt.Printf("  NOTE: no candidate stayed under logQP %.0f at ring 2^%d; measuring the smallest (%.1f).\n",
			btpLogQPCeiling, btpLogN, best.btpLogQP)
		fmt.Println("  The latency stands; the parameter set is NOT at the 128-bit ceiling.")
	}
	if !built {
		fmt.Println("FAIL: no residual chain admitted a bootstrapping circuit")
		return
	}
	_ = overCeil

	fmt.Printf("  residual chain         : 55 + %dx45 at ring 2^%d, max level %d\n",
		nPrimes, residual.LogN(), residual.MaxLevel())
	fmt.Printf("  bootstrapping ring     : 2^%d\n", btpParams.BootstrappingParameters.LogN())
	fmt.Printf("  bootstrapping logQP    : %.1f\n", btpParams.BootstrappingParameters.LogQP())
	fmt.Printf("  residual logQP         : %.1f\n", residual.LogQP())
	fmt.Println()

	const n = 10
	prng, err := sampling.NewKeyedPRNG([]byte("he-ifd-serve-btp-crs"))
	check(err)
	crs := prng

	// One DKG for the whole sweep: the bootstrapping keys are tied to the ideal
	// secret, so the secret keys cannot be resampled per case.
	kgen := rlwe.NewKeyGenerator(residual)
	sks := make([]*rlwe.SecretKey, n)
	for i := range sks {
		sks[i] = kgen.GenSecretKeyNew()
	}
	ckg := multiparty.NewPublicKeyGenProtocol(residual)
	ckgCRP := ckg.SampleCRP(crs)
	ckgCombined := ckg.AllocateShare()
	for i := 0; i < n; i++ {
		share := ckg.AllocateShare()
		ckg.GenShare(sks[i], ckgCRP, &share)
		if i == 0 {
			ckgCombined = share
		} else {
			ckg.AggregateShares(share, ckgCombined, &ckgCombined)
		}
	}
	pk := rlwe.NewPublicKey(residual)
	ckg.GenPublicKey(ckgCombined, ckgCRP, pk)

	idealSk := rlwe.NewSecretKey(residual)
	rQP := residual.RingQP()
	for i := 0; i < n; i++ {
		rQP.Add(idealSk.Value, sks[i].Value, idealSk.Value)
	}

	// The bootstrapping key material. In deployment the parties generate this
	// jointly; generating it here from the ideal secret gives the same key and so
	// the same per-query evaluation cost.
	fmt.Println("  generating bootstrapping keys ...")
	t0 := time.Now()
	btpKeys, skN2, err := btpParams.GenEvaluationKeys(idealSk)
	check(err)
	keyGenMs := ms(time.Since(t0))
	keyBytes := btpKeys.BinarySize()
	fmt.Printf("  bootstrapping keys     : %s\n", human(keyBytes))
	fmt.Printf("  generation             : %.1f s\n", keyGenMs/1000)
	fmt.Printf("  (for reference, the extended secret key alone is %s)\n", human(skN2.BinarySize()))
	fmt.Println()

	btpEval, err := bootstrapping.NewEvaluator(btpParams, btpKeys)
	check(err)

	results := make([]btpServeRow, 0, 5)
	for _, C := range []int{4, 6, 14, 77, 100} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("\n[serve-btp] C=%d FAILED: %v\n", C, r)
				}
			}()
			r := runServeBtp(residual, btpParams, btpEval, kgen, idealSk, pk, n, C)
			r.BtpKeyBytes = keyBytes
			r.BtpKeyGenMs = keyGenMs
			results = append(results, r)
			printBtpServe(r)
		}()
	}

	fmt.Println("\n--- CSV (record: results/fhe/argmax_tournament_btp.csv) ---")
	fmt.Println("n_parties,eval_logN,btp_logN,residual_max_level,C,c_padded,tournament_rounds,server_bootstraps,argmax_total_ms,in_bootstrap_ms,local_eval_ms,correct_max,max_abs_error,fresh_ciphertext_bytes,bootstrapping_key_bytes,bootstrapping_key_gen_ms")
	for _, r := range results {
		fmt.Printf("%d,%d,%d,%d,%d,%d,%d,%d,%.1f,%.1f,%.1f,%v,%.3e,%d,%d,%.1f\n",
			r.N, r.EvalLogN, r.BtpLogN, r.ResidualLevels, r.C, r.Cpad, r.Rounds,
			r.Bootstraps, r.TotalMs, r.BtpMs, r.LocalMs, r.CorrectMax, r.MaxAbsErr,
			r.CtBytes, r.BtpKeyBytes, r.BtpKeyGenMs)
	}
	if jsonOut != "" {
		b, err := json.MarshalIndent(results, "", "  ")
		check(err)
		check(os.WriteFile(jsonOut, b, 0o644))
		fmt.Printf("\nwrote %s\n", jsonOut)
	}
}

func runServeBtp(params ckks.Parameters, btpParams bootstrapping.Parameters,
	inner bootstrapping.Bootstrapper, kgen *rlwe.KeyGenerator, idealSk *rlwe.SecretKey,
	pk *rlwe.PublicKey, n, C int) btpServeRow {

	Cpad := 1
	for Cpad < C {
		Cpad *= 2
	}
	galEls := []uint64{params.GaloisElementForComplexConjugation()}
	for step := 1; step < Cpad; step *= 2 {
		galEls = append(galEls, params.GaloisElement(step))
	}
	relinKey := kgen.GenRelinearizationKeyNew(idealSk)
	galKeys := kgen.GenGaloisKeysNew(galEls, idealSk)
	evk := rlwe.NewMemEvaluationKeySet(relinKey, galKeys...)

	btp := &countingBootstrapper{inner: inner}
	eval := ckks.NewEvaluator(params, evk)
	minimaxEvl := minimax.NewEvaluator(params, eval, btp)
	cmp := comparison.NewEvaluator(params, minimaxEvl, minimax.NewPolynomial(comparison.DefaultCompositePolynomialForSign))

	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	slots := params.MaxSlots()
	rng := mrand.New(mrand.NewSource(int64(20260714 + C)))
	logits := make([]float64, C)
	vec := make([]float64, slots)
	for i := range vec {
		vec[i] = -0.5
	}
	for j := 0; j < C; j++ {
		logits[j] = rng.Float64()*0.9 - 0.4
		vec[j] = logits[j]
	}
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	check(encoder.Encode(vec, pt))
	m, err := encryptor.EncryptNew(pt)
	check(err)
	ctBytes := m.BinarySize()

	tArg := time.Now()
	for step := 1; step < Cpad; step *= 2 {
		if m.Level() < params.MaxLevel() {
			m, err = btp.Bootstrap(m)
			check(err)
		}
		rot, err := eval.RotateNew(m, step)
		check(err)
		m, err = cmp.Max(m, rot)
		check(err)
	}
	totalMs := ms(time.Since(tArg))

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

	return btpServeRow{
		N: n, EvalLogN: params.LogN(), BtpLogN: btpParams.BootstrappingParameters.LogN(),
		ResidualLevels: params.MaxLevel(), C: C, Cpad: Cpad, Rounds: rounds,
		Bootstraps: btp.count, TotalMs: totalMs, BtpMs: btp.totalMs,
		LocalMs: totalMs - btp.totalMs, CorrectMax: absErr < 0.05, MaxAbsErr: absErr,
		CtBytes: ctBytes,
	}
}

func printBtpServe(r btpServeRow) {
	fmt.Printf("\n-- serve-btp N=%d eval 2^%d btp 2^%d C=%d (pad %d, %d rounds)\n",
		r.N, r.EvalLogN, r.BtpLogN, r.C, r.Cpad, r.Rounds)
	fmt.Printf("   server bootstraps    : %d\n", r.Bootstraps)
	fmt.Printf("   argmax total         : %.1f ms  (%.2f s)\n", r.TotalMs, r.TotalMs/1000)
	fmt.Printf("     in bootstraps      : %.1f ms\n", r.BtpMs)
	fmt.Printf("     local eval         : %.1f ms\n", r.LocalMs)
	fmt.Printf("   correct max          : %v  (abs err %.3e)\n", r.CorrectMax, r.MaxAbsErr)
	fmt.Printf("   fresh ciphertext     : %s\n", human(r.CtBytes))
}
