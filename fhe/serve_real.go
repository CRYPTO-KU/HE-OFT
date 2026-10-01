// serve_real.go: one real query, end to end, against a real trained head.
//
// Every other benchmark in this directory runs on synthetic vectors: a random
// head, uniform logits, and whatever top-1/top-2 gap the seed produced. This
// file answers the question those cannot. It loads a head that was actually
// trained (experiments/export_head.py writes it out of a trained artifact),
// loads real query features under the same frozen backbone, and runs the
// serving path the method specifies:
//
//	the client encrypts phi(x) under the collective public key;
//	the serving party applies the ENCRYPTED head ciphertext-by-ciphertext,
//	  gathering the C logits into the first C slots of one ciphertext;
//	the argmax INDEX is computed under encryption by the tracked tournament
//	  of serve_index.go, its bootstraps served by collective refreshes;
//	a quorum key-switches that index to the querier's public key, and the
//	  querier alone decrypts it.
//
// Nothing else is decrypted. No logit, no score and no maximum value leaves the
// ciphertext domain, which is the property the security section claims. The
// plaintext label the encrypted answer is compared against comes from the
// export, computed in float64 by numpy, never from a decryption here.
//
// Two public constants make the circuit run and both are reported rather than
// hidden.
//
//	The bias is carried in homogeneous coordinates: the head row is
//	[W_c | b_c] and the encrypted query is [phi(x) | 1], so the bias needs no
//	separate ciphertext and no separate scale.
//
//	The sign circuit is a minimax approximation on [-1,1], so the logits have
//	to be mapped into that interval by a public scale gamma. The argmax is
//	invariant under multiplication by a positive constant, so gamma changes
//	nothing about which label is correct; what it does change is the margin the
//	circuit must resolve, since gamma multiplies the top-1/top-2 gap as well.
//	gamma is fixed once, before the run, as 0.4 divided by the largest absolute
//	logit in the export, and it is printed with every row.
//
// Run: go run . -serve-real -logn 15 -real-export ../outputs/export_head/ag_news_s42_A.json
package main

import (
	"encoding/json"
	"fmt"
	"math"
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

// realQuery is one exported test example.
type realQuery struct {
	TestIndex  int       `json:"test_index"`
	TrueLabel  int       `json:"true_label"`
	PlainLabel int       `json:"plain_label"`
	Margin     float64   `json:"margin"`
	Features   []float64 `json:"features"`
	Logits     []float64 `json:"logits"`
}

// realExport is what experiments/export_head.py writes.
type realExport struct {
	Task        string      `json:"task"`
	Seed        int         `json:"seed"`
	Arrangement string      `json:"arrangement"`
	Backbone    string      `json:"backbone"`
	Artifact    string      `json:"artifact"`
	Client      int         `json:"client"`
	N           int         `json:"N"`
	Alpha       float64     `json:"alpha"`
	K           int         `json:"K"`
	R           int         `json:"r"`
	C           int         `json:"C"`
	D           int         `json:"d"`
	W           [][]float64 `json:"W"`
	B           []float64   `json:"b"`
	LogitAbsMax float64     `json:"logit_abs_max"`
	MinMargin   float64     `json:"min_margin"`
	PlainAcc    float64     `json:"plain_accuracy"`
	Queries     []realQuery `json:"queries"`
}

// realRow is one answered query.
type realRow struct {
	Task         string  `json:"task"`
	Arrangement  string  `json:"arrangement"`
	N            int     `json:"n_parties"`
	LogN         int     `json:"log_ring_degree"`
	LogScale     int     `json:"log_scale"`
	C            int     `json:"num_classes"`
	D            int     `json:"feature_dim"`
	TestIndex    int     `json:"test_index"`
	TrueLabel    int     `json:"true_label"`
	PlainLabel   int     `json:"plaintext_label"`
	EncLabel     int     `json:"encrypted_label"`
	Agree        bool    `json:"agree"`
	Margin       float64 `json:"plaintext_margin"`
	ScaledMargin float64 `json:"scaled_margin"`
	Gamma        float64 `json:"gamma"`
	DecodedIndex float64 `json:"decoded_index"`
	IndexAbsErr  float64 `json:"index_abs_error"`
	Refreshes    int     `json:"collective_refreshes"`
	HeadMs       float64 `json:"head_apply_ms"`
	ArgmaxMs     float64 `json:"argmax_ms"`
	SwitchMs     float64 `json:"key_switch_to_querier_ms"`
	TotalMs      float64 `json:"total_ms"`
	OtherMaxAbs  float64 `json:"other_slots_max_abs"`
}

// realCtx carries the serving party's material for one export.
type realCtx struct {
	idx       *idxCtx
	ctHead    []*rlwe.Ciphertext // one ciphertext per head row, [W_c | b_c]
	dPad      int                // power of two >= d+1, the rotate-and-sum window
	gamma     float64
	pcks      multiparty.PublicKeySwitchProtocol
	sks       []*rlwe.SecretKey
	querierSk *rlwe.SecretKey
	querierPk *rlwe.PublicKey
}

// runServeReal answers every exported query end to end and reports agreement.
func runServeReal(path string, nParties, logN, maxQ int, jsonOut string) {
	raw, err := os.ReadFile(path)
	check(err)
	var ex realExport
	check(json.Unmarshal(raw, &ex))
	if len(ex.Queries) == 0 {
		panic("export carries no queries")
	}
	if maxQ > 0 && maxQ < len(ex.Queries) {
		ex.Queries = ex.Queries[:maxQ]
	}

	fmt.Printf("=== one real query, end to end ===\n")
	fmt.Printf("  export       : %s\n", path)
	fmt.Printf("  task         : %s seed %d arrangement %s (%s, artifact %s)\n",
		ex.Task, ex.Seed, ex.Arrangement, ex.Backbone, ex.Artifact)
	fmt.Printf("  head         : C=%d, d=%d, trained with N=%d alpha=%g K=%d r=%d\n",
		ex.C, ex.D, ex.N, ex.Alpha, ex.K, ex.R)
	fmt.Printf("  queries      : %d, plaintext accuracy on them %.3f\n",
		len(ex.Queries), ex.PlainAcc)
	fmt.Printf("  largest |logit| in the export : %.6f\n", ex.LogitAbsMax)
	fmt.Printf("  smallest plaintext margin     : %.6f\n\n", ex.MinMargin)

	// The fifteen-modulus chain below is ~807 bits of QP, which reaches 128-bit
	// security only at ring degree 2^15. Refuse to run it at a smaller ring
	// rather than report a number measured under a broken parameter set.
	if logN < 15 {
		panic(fmt.Sprintf("-serve-real needs -logn 15 or larger; got %d", logN))
	}

	ctx := newRealCtx(&ex, nParties, logN)
	fmt.Printf("  gamma (public logit scale)    : %.8f\n", ctx.gamma)
	fmt.Printf("  ring degree 2^%d, %d slots, log scale %d, %d levels\n\n",
		logN, ctx.idx.slots, ctx.idx.params.LogDefaultScale(), ctx.idx.params.MaxLevel())

	rows := make([]realRow, 0, len(ex.Queries))
	agree := 0
	for i := range ex.Queries {
		r := ctx.answer(&ex, i, logN)
		rows = append(rows, r)
		if r.Agree {
			agree++
		}
		fmt.Printf("  q%02d test#%-5d plaintext %d  encrypted %d  %s"+
			"  margin %.6f (scaled %.6f)  index %.6f  %d refreshes  %.1f s\n",
			i, r.TestIndex, r.PlainLabel, r.EncLabel,
			map[bool]string{true: "AGREE   ", false: "DISAGREE"}[r.Agree],
			r.Margin, r.ScaledMargin, r.DecodedIndex, r.Refreshes, r.TotalMs/1000)
	}

	fmt.Printf("\n  %d of %d queries agree\n", agree, len(rows))
	if agree < len(rows) {
		fmt.Println("  DISAGREEMENTS, with the plaintext margin at which each occurred:")
		for _, r := range rows {
			if !r.Agree {
				fmt.Printf("    test#%d: plaintext %d, encrypted %d, margin %.6f "+
					"(scaled %.6f), decoded index %.6f\n",
					r.TestIndex, r.PlainLabel, r.EncLabel, r.Margin,
					r.ScaledMargin, r.DecodedIndex)
			}
		}
	}

	fmt.Println("\n--- CSV (record: results/fhe/real_query.csv) ---")
	fmt.Println("task,arrangement,n_parties,logN,log_scale,C,d,test_index,true_label," +
		"plaintext_label,encrypted_label,agree,plaintext_margin,scaled_margin,gamma," +
		"decoded_index,index_abs_error,collective_refreshes,head_apply_ms,argmax_ms," +
		"key_switch_ms,total_ms,other_slots_max_abs")
	for _, r := range rows {
		fmt.Printf("%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%v,%.6f,%.6f,%.8f,%.6f,%.3e,%d,%.1f,%.1f,%.1f,%.1f,%.3e\n",
			r.Task, r.Arrangement, r.N, r.LogN, r.LogScale, r.C, r.D, r.TestIndex,
			r.TrueLabel, r.PlainLabel, r.EncLabel, r.Agree, r.Margin, r.ScaledMargin,
			r.Gamma, r.DecodedIndex, r.IndexAbsErr, r.Refreshes, r.HeadMs,
			r.ArgmaxMs, r.SwitchMs, r.TotalMs, r.OtherMaxAbs)
	}

	if jsonOut != "" {
		b, err := json.MarshalIndent(rows, "", "  ")
		check(err)
		check(os.WriteFile(jsonOut, b, 0o644))
		fmt.Printf("\nwrote %s\n", jsonOut)
	}
	if agree < len(rows) {
		os.Exit(1)
	}
}

// newRealCtx runs the distributed key generation, encrypts the head, and fixes
// the public logit scale. Everything here is paid once, not per query.
func newRealCtx(ex *realExport, n, logN int) *realCtx {
	// The same fifteen-modulus chain serve_index.go uses, so the tournament this
	// file calls is the one already recorded in argmax_index.csv.
	logQ := []int{55}
	for i := 0; i < 14; i++ {
		logQ = append(logQ, 45)
	}
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: logN, LogQ: logQ, LogP: []int{61, 61}, LogDefaultScale: 45,
	})
	check(err)
	slots := params.MaxSlots()

	dPad := 1
	for dPad < ex.D+1 { // +1 for the homogeneous coordinate carrying the bias
		dPad *= 2
	}
	Cpad := 1
	for Cpad < ex.C {
		Cpad *= 2
	}
	if dPad > slots || Cpad > slots {
		panic(fmt.Sprintf("head does not fit: dPad=%d Cpad=%d slots=%d", dPad, Cpad, slots))
	}

	crs, err := sampling.NewKeyedPRNG([]byte("he-oft-serve-real-crs"))
	check(err)

	// ---- distributed key generation: the collective public key ---------------
	kgen := rlwe.NewKeyGenerator(params)
	sks := make([]*rlwe.SecretKey, n)
	for i := range sks {
		sks[i] = kgen.GenSecretKeyNew()
	}
	ckg := multiparty.NewPublicKeyGenProtocol(params)
	ckgCRP := ckg.SampleCRP(crs)
	var ckgCombined multiparty.PublicKeyGenShare
	for i := 0; i < n; i++ {
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

	// The ideal secret is formed here only to generate the evaluation keys, which
	// a deployment generates collectively in a one-time round that changes neither
	// the keys' contents nor the per-query evaluation speed. It is never used to
	// decrypt anything in this file.
	idealSk := rlwe.NewSecretKey(params)
	rQP := params.RingQP()
	for i := 0; i < n; i++ {
		rQP.Add(idealSk.Value, sks[i].Value, idealSk.Value)
	}
	galEls := []uint64{params.GaloisElementForComplexConjugation()}
	for step := 1; step < dPad; step *= 2 { // rotate-and-sum over the feature window
		galEls = append(galEls, params.GaloisElement(step))
	}
	for c := 1; c < ex.C; c++ { // gather logit c from slot 0 into slot c
		galEls = append(galEls, params.GaloisElement(-c))
	}
	relinKey := kgen.GenRelinearizationKeyNew(idealSk)
	galKeys := kgen.GenGaloisKeysNew(galEls, idealSk)
	evk := rlwe.NewMemEvaluationKeySet(relinKey, galKeys...)

	// ---- the collective refresh that serves the sign circuit's bootstraps -----
	minLevel, logBound, ok := mpckks.GetMinimumLevelForRefresh(128, params.DefaultScale(), n, params.Q())
	if !ok || minLevel+1 > params.MaxLevel() {
		panic(fmt.Sprintf("refresh not possible: minLevel=%d maxLevel=%d", minLevel, params.MaxLevel()))
	}
	rfp, err := mpckks.NewRefreshProtocol(params, uint(params.LogDefaultScale()), params.Xe())
	check(err)
	btp := &collectiveBootstrapper{params: params, sks: sks, rfp: rfp, crs: crs,
		minLevel: minLevel, logBound: logBound}

	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	eval := ckks.NewEvaluator(params, evk)
	minimaxEvl := minimax.NewEvaluator(params, eval, btp)
	cmp := comparison.NewEvaluator(params, minimaxEvl, minimax.NewPolynomial(comparison.DefaultCompositePolynomialForSign))

	// ---- the head, encrypted once, one ciphertext per class row --------------
	// Row c holds [W_c | b_c] in slots 0..d, zero elsewhere, so the query's
	// homogeneous coordinate supplies the bias with no extra ciphertext.
	ctHead := make([]*rlwe.Ciphertext, ex.C)
	for c := 0; c < ex.C; c++ {
		row := make([]float64, slots)
		copy(row, ex.W[c])
		row[ex.D] = ex.B[c]
		pt := ckks.NewPlaintext(params, params.MaxLevel())
		check(encoder.Encode(row, pt))
		ct, err := encryptor.EncryptNew(pt)
		check(err)
		ctHead[c] = ct
	}

	// ---- the querier's own key pair, and the key switch that reaches it ------
	querierSk := kgen.GenSecretKeyNew()
	querierPk := kgen.GenPublicKeyNew(querierSk)
	pcks, err := multiparty.NewPublicKeySwitchProtocol(params, keySwitchNoise())
	check(err)

	return &realCtx{
		idx: &idxCtx{
			params: params, encoder: encoder, eval: eval, cmp: cmp, btp: btp,
			encryptor: encryptor, dec: nil, Cpad: Cpad, C: ex.C, slots: slots,
		},
		ctHead: ctHead,
		dPad:   dPad,
		// The argmax is invariant under a positive scalar, so gamma is free to
		// choose; it is fixed once from the export and never tuned per query.
		gamma:     0.4 / ex.LogitAbsMax,
		pcks:      pcks,
		sks:       sks,
		querierSk: querierSk,
		querierPk: querierPk,
	}
}

// applyHead is the serving party's application of the encrypted head to an
// encrypted query, with the head held one ciphertext per class row. Per class:
// one ciphertext-by-ciphertext product with relinearization, a rotate-and-sum
// over the feature window of dPad slots, then a masked rotation that places the
// logit in slot c. The mask carries gamma, so the public scale costs no extra
// level. Every slot outside the label range is then pushed below the smallest
// attainable scaled logit, so the tournament can never return a padding slot.
// -cost-vs-d times this same function.
func applyHead(params ckks.Parameters, eval *ckks.Evaluator, encoder *ckks.Encoder,
	ctHead []*rlwe.Ciphertext, ctPhi *rlwe.Ciphertext, dPad int, gamma float64, C, slots int) *rlwe.Ciphertext {

	mask := make([]float64, slots)
	mask[0] = gamma
	var acc *rlwe.Ciphertext
	for c := 0; c < C; c++ {
		p, err := eval.MulRelinNew(ctHead[c], ctPhi)
		check(err)
		check(eval.Rescale(p, p))
		for step := 1; step < dPad; step *= 2 {
			r, err := eval.RotateNew(p, step)
			check(err)
			check(eval.Add(p, r, p))
		}
		check(eval.Mul(p, mask, p))
		check(eval.Rescale(p, p))
		if c > 0 {
			p, err = eval.RotateNew(p, -c)
			check(err)
		}
		if c == 0 {
			acc = p
		} else {
			check(eval.Add(acc, p, acc))
		}
	}
	pad := make([]float64, slots)
	for j := C; j < slots; j++ {
		pad[j] = -0.5
	}
	ptPad := ckks.NewPlaintext(params, acc.Level())
	ptPad.Scale = acc.Scale
	check(encoder.Encode(pad, ptPad))
	check(eval.Add(acc, ptPad, acc))
	return acc
}

// answer runs the full serving path for one exported query.
func (rc *realCtx) answer(ex *realExport, i, logN int) realRow {
	q := &ex.Queries[i]
	ctx := rc.idx
	params, eval := ctx.params, ctx.eval
	refresh0 := ctx.btp.Count()
	tAll := time.Now()

	// ---- the client encrypts [phi(x) | 1] ------------------------------------
	vec := make([]float64, ctx.slots)
	copy(vec, q.Features)
	vec[ex.D] = 1 // homogeneous coordinate, so the head row carries its own bias
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	check(ctx.encoder.Encode(vec, pt))
	ctPhi, err := ctx.encryptor.EncryptNew(pt)
	check(err)

	// ---- the serving party applies the ENCRYPTED head -------------------------
	tHead := time.Now()
	acc := applyHead(params, eval, ctx.encoder, rc.ctHead, ctPhi, rc.dPad, rc.gamma, ex.C, ctx.slots)
	headMs := ms(time.Since(tHead))

	// ---- the argmax INDEX, under encryption ----------------------------------
	tArg := time.Now()
	_, ctIdx, err := ctx.trackedTournament(acc)
	check(err)
	// Keep the label slot alone. The tournament leaves in every slot the argmax
	// of a window of logits, and switching the whole ciphertext would give the
	// querier that ranking along with its label. The protocol returns the label
	// only, so every other slot is zeroed before the key switch.
	if ctIdx.Level() < 1 {
		ctIdx, err = ctx.btp.Bootstrap(ctIdx)
		check(err)
	}
	keep := make([]float64, ctx.slots)
	keep[0] = 1
	check(eval.Mul(ctIdx, keep, ctIdx))
	check(eval.Rescale(ctIdx, ctIdx))
	argmaxMs := ms(time.Since(tArg))

	// ---- a quorum switches the label to the querier, who alone decrypts it ----
	tSw := time.Now()
	var pcksAgg multiparty.PublicKeySwitchShare
	for j := range rc.sks {
		share := rc.pcks.AllocateShare(ctIdx.Level())
		rc.pcks.GenShare(rc.sks[j], rc.querierPk, ctIdx, &share)
		if j == 0 {
			pcksAgg = share
		} else {
			check(rc.pcks.AggregateShares(share, pcksAgg, &pcksAgg))
		}
	}
	switched := ckks.NewCiphertext(params, 1, ctIdx.Level())
	rc.pcks.KeySwitch(ctIdx, pcksAgg, switched)
	out := make([]float64, ctx.slots)
	check(ctx.encoder.Decode(rlwe.NewDecryptor(params, rc.querierSk).DecryptNew(switched), out))
	switchMs := ms(time.Since(tSw))

	decoded := out[0] * float64(ctx.Cpad)
	encLabel := int(math.Round(decoded))
	// What the querier reads anywhere but slot 0, in index units, so it compares
	// directly with the rounding margin of 0.5. The mask should leave noise only.
	otherMax := 0.0
	for j := 1; j < ctx.slots; j++ {
		if a := math.Abs(out[j]) * float64(ctx.Cpad); a > otherMax {
			otherMax = a
		}
	}
	if otherMax > 0.25 {
		fmt.Fprintf(os.Stderr, "WARNING: query %d, a slot other than the label decodes to %.3f index units\n", q.TestIndex, otherMax)
	}

	return realRow{
		Task: ex.Task, Arrangement: ex.Arrangement, N: len(rc.sks), LogN: logN,
		LogScale: params.LogDefaultScale(), C: ex.C, D: ex.D,
		TestIndex: q.TestIndex, TrueLabel: q.TrueLabel, PlainLabel: q.PlainLabel,
		EncLabel: encLabel, Agree: encLabel == q.PlainLabel,
		Margin: q.Margin, ScaledMargin: q.Margin * rc.gamma, Gamma: rc.gamma,
		DecodedIndex: decoded, IndexAbsErr: math.Abs(decoded - float64(q.PlainLabel)),
		Refreshes: ctx.btp.Count() - refresh0,
		HeadMs:    headMs, ArgmaxMs: argmaxMs, SwitchMs: switchMs,
		TotalMs: ms(time.Since(tAll)), OtherMaxAbs: otherMax,
	}
}
