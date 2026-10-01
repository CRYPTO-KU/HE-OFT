// serve_index_btp.go: the argmax INDEX under server-side bootstrapping.
//
// serve_index.go measures the index with its levels restored by a collective
// refresh, which needs a share from every client for every restoration. The
// protocol specifies the other arrangement: the parties generate bootstrapping
// keys once, collectively, and the serving party restores levels on its own.
// serve_btp.go measures the value-only tournament that way. This file measures the
// index the same way, so the price of returning the label rather than the largest
// logit is priced under the mechanism the paper describes.
//
// The circuit is unchanged. Only the bootstrapper differs, which is why the whole
// measurement is one call to runIndexOn over a context built here, and why the
// tournament_max rows reproduce argmax_tournament_btp.csv as a control.
package main

import (
	"encoding/json"
	"fmt"
	"math"
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

// countingBtp is a bootstrapper that reports how much work it has done. Both
// mechanisms that restore a level satisfy it, the collective refresh and the
// serving party's own bootstrap, so one measurement runs under either.
type countingBtp interface {
	bootstrapping.Bootstrapper
	Count() int
	Millis() float64
}

// indexBtpRow is one measured (case, method) pair under server-side bootstrapping.
type indexBtpRow struct {
	indexRow
	BtpLogN        int     `json:"bootstrapping_log_ring_degree"`
	ResidualLevels int     `json:"residual_max_level"`
	CtBytes        int     `json:"fresh_ciphertext_bytes"`
	BtpKeyBytes    int     `json:"bootstrapping_key_bytes"`
	BtpKeyGenMs    float64 `json:"bootstrapping_key_gen_ms"`
}

// btpIndexParams builds the residual chain and the bootstrapping circuit over it.
// The chain is the one serve_btp.go's search selected for the value-only
// tournament, 55 + 8x45 at ring 2^15 with a bootstrapping ring of 2^16, recorded
// in argmax_tournament_btp.csv as residual_max_level 8. The index has to run in
// the same parameters or its cost is not comparable to that control, so the chain
// is stated here rather than searched for a second time. The bootstrapping modulus
// is printed, and the ceiling that search enforced was 1553 bits.
//
// LogNthRoot = LogN + 2 is what makes the residual moduli NTT-friendly for the
// bootstrapping ring as well.
func btpIndexParams() (ckks.Parameters, bootstrapping.Parameters) {
	return btpResidualParams(15)
}

// btpResidualParams is the same chain at evaluation ring 2^logN, with the
// bootstrapping ring one larger. The paper's parameters are logN = 15; the
// -smudge-noise and -cost-vs-d modes accept a smaller ring for smoke tests only,
// where the chain is far outside the 128-bit budget.
func btpResidualParams(logN int) (ckks.Parameters, bootstrapping.Parameters) {
	logQ := []int{55}
	for i := 0; i < 8; i++ {
		logQ = append(logQ, 45)
	}
	residual, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: logN, LogNthRoot: logN + 2, LogQ: logQ, LogP: []int{61, 61}, LogDefaultScale: 45,
	})
	check(err)
	btpLogN := logN + 1
	btpParams, err := bootstrapping.NewParametersFromLiteral(residual,
		bootstrapping.ParametersLiteral{LogN: &btpLogN})
	check(err)
	return residual, btpParams
}

// runIndexBtpSuite sweeps C at fixed N=10 on the residual chain serve_btp.go picks,
// measuring the value-only tournament and then the index by both constructions.
func runIndexBtpSuite(jsonOut string) {
	fmt.Println("=== argmax index with server-side bootstrapping ===")
	fmt.Println()

	residual, btpParams := btpIndexParams()
	fmt.Printf("  residual chain         : 55 + 8x45 at ring 2^%d, max level %d\n",
		residual.LogN(), residual.MaxLevel())
	fmt.Printf("  bootstrapping ring     : 2^%d\n", btpParams.BootstrappingParameters.LogN())
	fmt.Printf("  bootstrapping logQP    : %.1f\n", btpParams.BootstrappingParameters.LogQP())
	fmt.Printf("  residual logQP         : %.1f\n", residual.LogQP())
	fmt.Println()

	const n = 10
	crs, err := sampling.NewKeyedPRNG([]byte("he-ifd-serve-btp-crs"))
	check(err)

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
	btpKeys, _, err := btpParams.GenEvaluationKeys(idealSk)
	check(err)
	keyGenMs := ms(time.Since(t0))
	keyBytes := btpKeys.BinarySize()
	fmt.Printf("  bootstrapping keys     : %s\n", human(keyBytes))
	fmt.Printf("  generation             : %.1f s\n", keyGenMs/1000)
	fmt.Println()

	btpEval, err := bootstrapping.NewEvaluator(btpParams, btpKeys)
	check(err)

	taus := []float64{1e-4, 1e-3}
	rows := make([]indexBtpRow, 0, 20)
	for _, C := range []int{4, 6, 14, 77, 100} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("\n[index-btp] C=%d FAILED: %v\n", C, r)
				}
			}()
			c := argmaxConfig{n, residual.LogN(), C}
			ctx, logits, ct0 := newIdxCtxBtp(residual, btpEval, kgen, idealSk, pk, c)
			ctBytes := ct0.BinarySize()
			for _, r := range runIndexOn(ctx, logits, ct0, c, taus) {
				rows = append(rows, indexBtpRow{
					indexRow:       r,
					BtpLogN:        btpParams.BootstrappingParameters.LogN(),
					ResidualLevels: residual.MaxLevel(),
					CtBytes:        ctBytes,
					BtpKeyBytes:    keyBytes,
					BtpKeyGenMs:    keyGenMs,
				})
			}
		}()
	}

	fmt.Println("\n--- CSV (record: results/fhe/argmax_index_btp.csv) ---")
	fmt.Println("method,n_parties,eval_logN,btp_logN,residual_max_level,C,c_padded,tournament_rounds,tau,server_bootstraps,total_ms,in_bootstrap_ms,local_eval_ms,extra_bootstraps,extra_ms,correct_max,max_abs_error,true_index,decoded_index,index_exact,index_abs_error,onehot_sum,top1_top2_gap,fresh_ciphertext_bytes,bootstrapping_key_bytes,bootstrapping_key_gen_ms")
	for _, r := range rows {
		fmt.Printf("%s,%d,%d,%d,%d,%d,%d,%d,%.5f,%d,%.1f,%.1f,%.1f,%d,%.1f,%v,%.3e,%d,%.6f,%v,%.3e,%.6f,%.6f,%d,%d,%.1f\n",
			r.Method, r.N, r.LogN, r.BtpLogN, r.ResidualLevels, r.C, r.Cpad, r.Rounds,
			r.Tau, r.Refreshes, r.TotalMs, r.RefreshMs, r.LocalMs, r.ExtraRefresh,
			r.ExtraMs, r.CorrectMax, r.MaxAbsErr, r.TrueIndex, r.DecodedIndex,
			r.IndexExact, r.IndexAbsErr, r.OneHotSum, r.Gap,
			r.CtBytes, r.BtpKeyBytes, r.BtpKeyGenMs)
	}

	if jsonOut != "" {
		// A row that has no index, or no one-hot, carries NaN so the CSV shows the
		// field is absent. JSON has no NaN, so those become zero here.
		clean := make([]indexBtpRow, len(rows))
		copy(clean, rows)
		for i := range clean {
			if math.IsNaN(clean[i].DecodedIndex) {
				clean[i].DecodedIndex = 0
			}
			if math.IsNaN(clean[i].OneHotSum) {
				clean[i].OneHotSum = 0
			}
		}
		b, err := json.MarshalIndent(clean, "", "  ")
		check(err)
		check(os.WriteFile(jsonOut, b, 0o644))
		fmt.Printf("\nwrote %s\n", jsonOut)
	}
}

// newIdxCtxBtp builds one case's evaluation keys and packed ciphertext over the
// residual chain, with level restoration done by the serving party alone. The
// secret keys, the collective public key and the bootstrapping keys are the
// suite's, because the bootstrapping keys are tied to the ideal secret and cannot
// be resampled per case. extraGal adds rotation keys for circuits that run beside
// the argmax, such as the head application of -cost-vs-d; none are added when it
// is empty, so the argmax suites build exactly the keys they always did.
func newIdxCtxBtp(params ckks.Parameters, inner bootstrapping.Bootstrapper,
	kgen *rlwe.KeyGenerator, idealSk *rlwe.SecretKey, pk *rlwe.PublicKey,
	c argmaxConfig, extraGal ...uint64) (*idxCtx, []float64, *rlwe.Ciphertext) {

	Cpad := 1
	for Cpad < c.C {
		Cpad *= 2
	}
	// Positive steps drive the tournament and the inner product; negative steps
	// broadcast the maximum back across the label slots. The value-only tournament
	// needs the positive ones alone, so the index doubles the rotation-key setup.
	galEls := []uint64{params.GaloisElementForComplexConjugation()}
	for step := 1; step < Cpad; step *= 2 {
		galEls = append(galEls, params.GaloisElement(step))
		galEls = append(galEls, params.GaloisElement(-step))
	}
	have := make(map[uint64]bool, len(galEls))
	for _, g := range galEls {
		have[g] = true
	}
	for _, g := range extraGal {
		if !have[g] {
			galEls = append(galEls, g)
			have[g] = true
		}
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
	logits, ct0 := packLogits(params, encoder, encryptor, c.C)

	return &idxCtx{
		params: params, encoder: encoder, eval: eval, cmp: cmp, btp: btp,
		encryptor: encryptor, dec: rlwe.NewDecryptor(params, idealSk),
		Cpad: Cpad, C: c.C, slots: params.MaxSlots(),
	}, logits, ct0
}
