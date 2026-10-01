// serve_index.go: the argmax INDEX under encryption, not only the maximum value.
//
// serve_argmax.go and serve_tournament.go both reduce the encrypted logit vector
// to the largest logit VALUE. The protocol specifies something else: the server
// reduces the logits to Enc(y) where y = argmax_c l_c, an INDEX. This file measures
// the index, by two constructions, against the value-only tournament as its control.
//
//	one-hot   the tournament produces the max in slot 0; broadcast it across the
//	          label slots, form l_c - M + tau, and evaluate ONE step circuit. The
//	          result is an encrypted one-hot vector, and an inner product with the
//	          plaintext index vector collapses it to a single encrypted index. Costs
//	          one extra sign evaluation. Correct whenever the top-1/top-2 gap
//	          exceeds tau and tau exceeds the smooth-max error, both of which are
//	          measured and reported per case.
//
//	tracked   carry an index ciphertext through the tournament. At each round the
//	          comparison bit b = step(v_a - v_b) is already computed for the value
//	          update; reusing it for i_new = b*(i_a - i_b) + i_b costs one extra
//	          ciphertext product per round and no extra sign evaluation. Correct
//	          under the same condition the max itself needs, with no tau.
//
// Both keep the label encrypted throughout. Nothing is decrypted here except for
// verification, which is what a benchmark is for.
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
	"github.com/tuneinsight/lattigo/v6/multiparty/mpckks"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// indexRow is one measured (case, method) pair.
type indexRow struct {
	Method       string  `json:"method"`
	N            int     `json:"n_parties"`
	LogN         int     `json:"log_ring_degree"`
	C            int     `json:"num_classes"`
	Cpad         int     `json:"c_padded"`
	Rounds       int     `json:"tournament_rounds"`
	Tau          float64 `json:"tau"`
	Refreshes    int     `json:"collective_refreshes"`
	TotalMs      float64 `json:"total_ms"`
	RefreshMs    float64 `json:"in_refresh_ms"`
	LocalMs      float64 `json:"local_eval_ms"`
	ExtraRefresh int     `json:"extra_refreshes_vs_max_only"`
	ExtraMs      float64 `json:"extra_ms_vs_max_only"`
	CorrectMax   bool    `json:"correct_max"`
	MaxAbsErr    float64 `json:"max_abs_error"`
	TrueIndex    int     `json:"true_index"`
	DecodedIndex float64 `json:"decoded_index"`
	IndexExact   bool    `json:"index_exact"`
	IndexAbsErr  float64 `json:"index_abs_error"`
	OneHotSum    float64 `json:"onehot_sum"`
	Gap          float64 `json:"top1_top2_gap"`
}

// idxCtx holds one case's cryptographic material, shared by every method so the
// setup is paid once and the measured times are the serving work alone.
type idxCtx struct {
	params    ckks.Parameters
	encoder   *ckks.Encoder
	eval      *ckks.Evaluator
	cmp       *comparison.Evaluator
	btp       countingBtp
	encryptor *rlwe.Encryptor
	dec       *rlwe.Decryptor
	Cpad      int
	C         int
	slots     int
}

// runIndexSuite sweeps C at fixed N=10, logN=15, recovering per case.
func runIndexSuite(jsonOut string) {
	configs := []argmaxConfig{
		{10, 15, 4}, {10, 15, 6}, {10, 15, 14}, {10, 15, 77}, {10, 15, 100},
	}
	taus := []float64{1e-4, 1e-3}
	rows := make([]indexRow, 0, len(configs)*4)
	for _, c := range configs {
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("\n[index] C=%d FAILED: %v\n", c.C, r)
				}
			}()
			rows = append(rows, runIndex(c, taus)...)
		}()
	}
	fmt.Println("\n--- CSV (record: results/fhe/argmax_index.csv) ---")
	fmt.Println("method,n_parties,logN,C,c_padded,tournament_rounds,tau,collective_refreshes,total_ms,in_refresh_ms,local_eval_ms,extra_refreshes,extra_ms,correct_max,max_abs_error,true_index,decoded_index,index_exact,index_abs_error,onehot_sum,top1_top2_gap")
	for _, r := range rows {
		fmt.Printf("%s,%d,%d,%d,%d,%d,%.5f,%d,%.1f,%.1f,%.1f,%d,%.1f,%v,%.3e,%d,%.6f,%v,%.3e,%.6f,%.6f\n",
			r.Method, r.N, r.LogN, r.C, r.Cpad, r.Rounds, r.Tau, r.Refreshes,
			r.TotalMs, r.RefreshMs, r.LocalMs, r.ExtraRefresh, r.ExtraMs,
			r.CorrectMax, r.MaxAbsErr, r.TrueIndex, r.DecodedIndex, r.IndexExact,
			r.IndexAbsErr, r.OneHotSum, r.Gap)
	}
	if jsonOut != "" {
		// A row that has no index, or no one-hot, carries NaN so the CSV shows the
		// field is absent. JSON has no NaN, so those become zero here.
		clean := make([]indexRow, len(rows))
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

// runIndex measures one case: the value-only tournament, then the index by both
// constructions.
func runIndex(c argmaxConfig, taus []float64) []indexRow {
	ctx, logits, ct0 := newIdxCtx(c)
	return runIndexOn(ctx, logits, ct0, c, taus)
}

// runIndexOn is the measurement itself, over a context somebody else built. The
// collective-refresh suite and the server-bootstrapping suite differ only in that
// context, so the circuit they measure is the same one.
func runIndexOn(ctx *idxCtx, logits []float64, ct0 *rlwe.Ciphertext, c argmaxConfig, taus []float64) []indexRow {
	params := ctx.params
	rounds := 0
	for s := 1; s < ctx.Cpad; s *= 2 {
		rounds++
	}

	// plaintext reference: the max, its index, and the top-1/top-2 gap.
	trueIdx, trueMax := 0, logits[0]
	for j, v := range logits {
		if v > trueMax {
			trueMax, trueIdx = v, j
		}
	}
	second := math.Inf(-1)
	for j, v := range logits {
		if j != trueIdx && v > second {
			second = v
		}
	}
	gap := trueMax - second

	base := indexRow{
		N: c.n, LogN: c.logN, C: c.C, Cpad: ctx.Cpad, Rounds: rounds,
		TrueIndex: trueIdx, Gap: gap,
	}
	rows := make([]indexRow, 0, 2+len(taus))

	// ---- control: the value-only tournament, exactly as serve_tournament.go --
	r0, ms0 := ctx.btp.Count(), ctx.btp.Millis()
	tA := time.Now()
	m, err := ctx.tournamentMax(ct0)
	check(err)
	maxMs := ms(time.Since(tA))
	maxRefresh := ctx.btp.Count() - r0
	maxRefreshMs := ctx.btp.Millis() - ms0

	out := make([]float64, ctx.slots)
	check(ctx.encoder.Decode(ctx.dec.DecryptNew(m), out))
	maxErr := math.Abs(out[0] - trueMax)

	ctrl := base
	ctrl.Method = "tournament_max"
	ctrl.Refreshes = maxRefresh
	ctrl.TotalMs = maxMs
	ctrl.RefreshMs = maxRefreshMs
	ctrl.LocalMs = maxMs - maxRefreshMs
	ctrl.CorrectMax = maxErr < 0.05
	ctrl.MaxAbsErr = maxErr
	ctrl.DecodedIndex = math.NaN()
	rows = append(rows, ctrl)
	printIndexRow(ctrl)

	// ---- one-hot index, per tau -------------------------------------------
	for _, tau := range taus {
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("\n[index] C=%d onehot tau=%g FAILED: %v\n", c.C, tau, r)
				}
			}()
			r1, ms1 := ctx.btp.Count(), ctx.btp.Millis()
			tB := time.Now()
			idxVal, hotSum, err := ctx.oneHotIndex(m, ct0, tau)
			check(err)
			extraMs := ms(time.Since(tB))
			extraRef := ctx.btp.Count() - r1
			extraRefMs := ctx.btp.Millis() - ms1

			row := base
			row.Method = "onehot_index"
			row.Tau = tau
			row.Refreshes = maxRefresh + extraRef
			row.TotalMs = maxMs + extraMs
			row.RefreshMs = maxRefreshMs + extraRefMs
			row.LocalMs = row.TotalMs - row.RefreshMs
			row.ExtraRefresh = extraRef
			row.ExtraMs = extraMs
			row.CorrectMax = maxErr < 0.05
			row.MaxAbsErr = maxErr
			row.DecodedIndex = idxVal
			row.IndexAbsErr = math.Abs(idxVal - float64(trueIdx))
			row.IndexExact = int(math.Round(idxVal)) == trueIdx
			row.OneHotSum = hotSum
			rows = append(rows, row)
			printIndexRow(row)
		}()
	}

	// ---- tracked index: one index ciphertext carried through the tournament --
	func() {
		defer func() {
			if r := recover(); r != nil {
				fmt.Printf("\n[index] C=%d tracked FAILED: %v\n", c.C, r)
			}
		}()
		r2, ms2 := ctx.btp.Count(), ctx.btp.Millis()
		tC := time.Now()
		mv, mi, err := ctx.trackedTournament(ct0)
		check(err)
		totMs := ms(time.Since(tC))
		ref := ctx.btp.Count() - r2
		refMs := ctx.btp.Millis() - ms2

		vOut := make([]float64, ctx.slots)
		check(ctx.encoder.Decode(ctx.dec.DecryptNew(mv), vOut))
		iOut := make([]float64, ctx.slots)
		check(ctx.encoder.Decode(ctx.dec.DecryptNew(mi), iOut))
		idxVal := iOut[0] * float64(ctx.Cpad)

		row := base
		row.Method = "tracked_index"
		row.Refreshes = ref
		row.TotalMs = totMs
		row.RefreshMs = refMs
		row.LocalMs = totMs - refMs
		row.ExtraRefresh = ref - maxRefresh
		row.ExtraMs = totMs - maxMs
		row.CorrectMax = math.Abs(vOut[0]-trueMax) < 0.05
		row.MaxAbsErr = math.Abs(vOut[0] - trueMax)
		row.DecodedIndex = idxVal
		row.IndexAbsErr = math.Abs(idxVal - float64(trueIdx))
		row.IndexExact = int(math.Round(idxVal)) == trueIdx
		row.OneHotSum = math.NaN()
		rows = append(rows, row)
		printIndexRow(row)
	}()

	_ = params
	return rows
}

// newIdxCtx builds the parameters, the DKG, the evaluation keys and the packed
// ciphertext. The logit vector and the padding match serve_tournament.go exactly,
// so the value-only row reproduces the recorded control.
func newIdxCtx(c argmaxConfig) (*idxCtx, []float64, *rlwe.Ciphertext) {
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

	idealSk := rlwe.NewSecretKey(params)
	rQP := params.RingQP()
	for i := 0; i < c.n; i++ {
		rQP.Add(idealSk.Value, sks[i].Value, idealSk.Value)
	}

	Cpad := 1
	for Cpad < c.C {
		Cpad *= 2
	}
	// Positive steps drive the tournament and the inner product; negative steps
	// broadcast the maximum back across the label slots.
	galEls := []uint64{params.GaloisElementForComplexConjugation()}
	for step := 1; step < Cpad; step *= 2 {
		galEls = append(galEls, params.GaloisElement(step))
		galEls = append(galEls, params.GaloisElement(-step))
	}
	relinKey := kgen.GenRelinearizationKeyNew(idealSk)
	galKeys := kgen.GenGaloisKeysNew(galEls, idealSk)
	evk := rlwe.NewMemEvaluationKeySet(relinKey, galKeys...)

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

	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	slots := params.MaxSlots()
	logits, ct0 := packLogits(params, encoder, encryptor, c.C)

	return &idxCtx{
		params: params, encoder: encoder, eval: eval, cmp: cmp, btp: btp,
		encryptor: encryptor, dec: rlwe.NewDecryptor(params, idealSk),
		Cpad: Cpad, C: c.C, slots: slots,
	}, logits, ct0
}

// packLogits builds one case's logit vector and encrypts it. The seed, the range
// and the padding value are the ones serve_tournament.go used, so every suite that
// calls this compares against the same numbers.
//
// The comparison circuit is accurate on [-0.5, 0.5], so the padding lies inside
// that range and strictly below every logit.
func packLogits(params ckks.Parameters, encoder *ckks.Encoder, encryptor *rlwe.Encryptor, C int) ([]float64, *rlwe.Ciphertext) {
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
	ct, err := encryptor.EncryptNew(pt)
	check(err)
	return logits, ct
}

// tournamentMax is the recorded control: ceil(log2 Cpad) rounds of rotate-and-Max,
// slot 0 ending at the largest logit.
func (ctx *idxCtx) tournamentMax(ct0 *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	eval, params := ctx.eval, ctx.params
	m := ct0.CopyNew()
	var err error
	for step := 1; step < ctx.Cpad; step *= 2 {
		if m.Level() < params.MaxLevel() {
			if m, err = ctx.btp.Bootstrap(m); err != nil {
				return nil, err
			}
		}
		rot, err := eval.RotateNew(m, step)
		if err != nil {
			return nil, err
		}
		if m, err = ctx.cmp.Max(m, rot); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// oneHotIndex turns the encrypted maximum into an encrypted index. It broadcasts
// the maximum across the label slots, evaluates one step circuit on l_c - M + tau
// to obtain an encrypted one-hot vector, and collapses that against the plaintext
// index vector. Returns the decoded index and the decoded one-hot mass, both for
// verification only.
func (ctx *idxCtx) oneHotIndex(m, ct0 *rlwe.Ciphertext, tau float64) (float64, float64, error) {
	eval, params, btp := ctx.eval, ctx.params, ctx.btp

	// A refresh puts the tournament output back at the full level and at the
	// default scale, which is what the masking multiplication below assumes.
	m2, err := btp.Bootstrap(m)
	if err != nil {
		return 0, 0, err
	}

	mask := make([]float64, ctx.slots)
	mask[0] = 1
	M, err := eval.MulNew(m2, mask)
	if err != nil {
		return 0, 0, err
	}
	if err = eval.Rescale(M, M); err != nil {
		return 0, 0, err
	}
	// Doubling broadcast: slot 0 spreads over slots 0..Cpad-1. Slots beyond that
	// stay zero, so their difference below stays inside the sign circuit's domain.
	for step := 1; step < ctx.Cpad; step *= 2 {
		r, err := eval.RotateNew(M, -step)
		if err != nil {
			return 0, 0, err
		}
		if err = eval.Add(M, r, M); err != nil {
			return 0, 0, err
		}
	}

	// d = (l - M + tau)/2. The halving keeps every slot inside [-0.5, 0.25], well
	// within the interval on which the composite sign polynomial is a minimax
	// approximation.
	d, err := eval.SubNew(ct0, M)
	if err != nil {
		return 0, 0, err
	}
	if err = eval.Add(d, tau, d); err != nil {
		return 0, 0, err
	}
	if err = eval.Mul(d, 0.5, d); err != nil {
		return 0, 0, err
	}
	if err = eval.Rescale(d, d); err != nil {
		return 0, 0, err
	}

	onehot, err := ctx.cmp.Step(d)
	if err != nil {
		return 0, 0, err
	}
	if onehot.Level() < params.LevelsConsumedPerRescaling() {
		if onehot, err = btp.Bootstrap(onehot); err != nil {
			return 0, 0, err
		}
	}

	// Inner product with (0, 1, ..., C-1)/Cpad, then a rotate-and-sum into slot 0.
	idxPT := make([]float64, ctx.slots)
	for j := 0; j < ctx.C; j++ {
		idxPT[j] = float64(j) / float64(ctx.Cpad)
	}
	p, err := eval.MulNew(onehot, idxPT)
	if err != nil {
		return 0, 0, err
	}
	if err = eval.Rescale(p, p); err != nil {
		return 0, 0, err
	}
	for step := 1; step < ctx.Cpad; step *= 2 {
		r, err := eval.RotateNew(p, step)
		if err != nil {
			return 0, 0, err
		}
		if err = eval.Add(p, r, p); err != nil {
			return 0, 0, err
		}
	}

	hot := make([]float64, ctx.slots)
	if err = ctx.encoder.Decode(ctx.dec.DecryptNew(onehot), hot); err != nil {
		return 0, 0, err
	}
	sum := 0.0
	for j := 0; j < ctx.C; j++ {
		sum += hot[j]
	}
	res := make([]float64, ctx.slots)
	if err = ctx.encoder.Decode(ctx.dec.DecryptNew(p), res); err != nil {
		return 0, 0, err
	}
	return res[0] * float64(ctx.Cpad), sum, nil
}

// trackedTournament runs the same rotate-and-Max reduction while carrying an index
// ciphertext, updated at each round with the comparison bit the value update
// already computed. Returns the value and the index ciphertexts.
func (ctx *idxCtx) trackedTournament(ct0 *rlwe.Ciphertext) (*rlwe.Ciphertext, *rlwe.Ciphertext, error) {
	eval, params, btp := ctx.eval, ctx.params, ctx.btp

	// The index vector is scaled by 1/Cpad so every encoded value stays in [0,1),
	// the same magnitude as a logit, which keeps the refresh bound valid.
	idxVec := make([]float64, ctx.slots)
	for j := 0; j < ctx.slots; j++ {
		idxVec[j] = float64(j%ctx.Cpad) / float64(ctx.Cpad)
	}
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	if err := ctx.encoder.Encode(idxVec, pt); err != nil {
		return nil, nil, err
	}
	idx, err := ctx.encryptor.EncryptNew(pt)
	if err != nil {
		return nil, nil, err
	}

	m := ct0.CopyNew()
	for step := 1; step < ctx.Cpad; step *= 2 {
		if m.Level() < params.MaxLevel() {
			if m, err = btp.Bootstrap(m); err != nil {
				return nil, nil, err
			}
		}
		rotM, err := eval.RotateNew(m, step)
		if err != nil {
			return nil, nil, err
		}
		newM, b, err := ctx.maxAndStep(m, rotM)
		if err != nil {
			return nil, nil, err
		}

		if idx.Level() < params.MaxLevel() {
			if idx, err = btp.Bootstrap(idx); err != nil {
				return nil, nil, err
			}
		}
		rotI, err := eval.RotateNew(idx, step)
		if err != nil {
			return nil, nil, err
		}
		dI, err := eval.SubNew(idx, rotI)
		if err != nil {
			return nil, nil, err
		}
		if idx, err = ctx.combine(b, dI, rotI); err != nil {
			return nil, nil, err
		}
		m = newM
	}
	return m, idx, nil
}

// maxAndStep returns max(op0, op1) and the comparison bit step(op0-op1) that
// produced it, so a second quantity can be gated on the same bit.
func (ctx *idxCtx) maxAndStep(op0, op1 *rlwe.Ciphertext) (*rlwe.Ciphertext, *rlwe.Ciphertext, error) {
	eval, params, btp := ctx.eval, ctx.params, ctx.btp
	diff, err := eval.SubNew(op0, op1)
	if err != nil {
		return nil, nil, err
	}
	if diff.Level() < params.LevelsConsumedPerRescaling()*2 {
		if diff, err = btp.Bootstrap(diff); err != nil {
			return nil, nil, err
		}
	}
	b, err := ctx.cmp.Step(diff)
	if err != nil {
		return nil, nil, err
	}
	if b.Level() < params.LevelsConsumedPerRescaling() {
		if b, err = btp.Bootstrap(b); err != nil {
			return nil, nil, err
		}
	}
	mx, err := ctx.combine(b, diff.CopyNew(), op1)
	if err != nil {
		return nil, nil, err
	}
	return mx, b, nil
}

// combine returns b*d + base at the default scale. It reproduces the scale
// discipline of comparison.Evaluator.Max, which relabels d's scale so that the
// product lands back where an addition against base is exact.
func (ctx *idxCtx) combine(b, d, base *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	eval, params, btp := ctx.eval, ctx.params, ctx.btp
	var err error
	if d.Level() < params.LevelsConsumedPerRescaling()*2 {
		if d, err = btp.Bootstrap(d); err != nil {
			return nil, err
		}
	}
	s := b
	if s.Level() > d.Level() {
		s = s.CopyNew()
		eval.DropLevel(s, s.Level()-d.Level())
	}
	if d.Level() > s.Level() {
		eval.DropLevel(d, d.Level()-s.Level())
	}
	level := d.Level()
	ratio := rlwe.NewScale(1)
	for i := 0; i < params.LevelsConsumedPerRescaling(); i++ {
		ratio = ratio.Mul(rlwe.NewScale(params.Q()[level-i]))
	}
	ratio = ratio.Div(d.Scale)
	if err = eval.Mul(d, &ratio.Value, d); err != nil {
		return nil, err
	}
	if err = eval.Rescale(d, d); err != nil {
		return nil, err
	}
	d.Scale = d.Scale.Mul(ratio)
	if err = eval.MulRelin(d, s, d); err != nil {
		return nil, err
	}
	if err = eval.Rescale(d, d); err != nil {
		return nil, err
	}
	if err = eval.Add(d, base, d); err != nil {
		return nil, err
	}
	return d, nil
}

func printIndexRow(r indexRow) {
	fmt.Printf("\n-- %s  N=%d logN=%d C=%d (pad %d, %d rounds)\n", r.Method, r.N, r.LogN, r.C, r.Cpad, r.Rounds)
	if r.Tau != 0 {
		fmt.Printf("   tau                  : %g\n", r.Tau)
	}
	fmt.Printf("   level restorations   : %d  (extra %d)\n", r.Refreshes, r.ExtraRefresh)
	fmt.Printf("   total                : %.1f ms  (extra %.1f ms)\n", r.TotalMs, r.ExtraMs)
	fmt.Printf("     in refreshes       : %.1f ms\n", r.RefreshMs)
	fmt.Printf("     local eval         : %.1f ms\n", r.LocalMs)
	fmt.Printf("   correct max          : %v  (abs err %.3e)\n", r.CorrectMax, r.MaxAbsErr)
	if !math.IsNaN(r.DecodedIndex) {
		fmt.Printf("   index                : decoded %.6f, true %d, exact %v (abs err %.3e)\n",
			r.DecodedIndex, r.TrueIndex, r.IndexExact, r.IndexAbsErr)
	}
	if !math.IsNaN(r.OneHotSum) && r.OneHotSum != 0 {
		fmt.Printf("   one-hot mass         : %.6f  (1 is correct)\n", r.OneHotSum)
	}
	fmt.Printf("   top1-top2 gap        : %.6f\n", r.Gap)
}

// keep the bootstrapping import honest: the collective bootstrapper satisfies the
// interface the minimax evaluator requires.
var _ bootstrapping.Bootstrapper = (*collectiveBootstrapper)(nil)
