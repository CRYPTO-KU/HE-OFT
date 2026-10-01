// cost_vs_d.go: the cost of every encrypted operation that depends on the
// feature dimension d or the label-space size C, over a grid of both.
//
// Each cell runs the operations the paper describes, on the code paths the other
// modes measure, at the serving parameters of
// serve_index_btp.go: residual chain 55 + 8x45 at ring 2^15, bootstrapping ring
// 2^16, N parties.
//
//	training upload   one client's head displacement, C rows of d weights and a
//	                  bias, weighted by its per-class counts before encryption and
//	                  packed flat across ceil(C(d+1)/slots) ciphertexts by
//	                  main.go's encryptVector. The per-client upload is reported as
//	                  the paper's communication table counts it: two arrangements
//	                  and one ciphertext of per-class counts.
//	server merge      the N uploads added ciphertext by ciphertext, multiplied by
//	                  the encrypted reciprocal of the class totals, and the public
//	                  initializer added as a plaintext.
//	query             the client encrypts [phi(x) | 1]; the server applies the
//	                  encrypted head with serve_real.go's applyHead, takes the
//	                  argmax index with the tracked tournament of serve_index.go
//	                  under server-side bootstrapping, and a quorum switches the
//	                  index to the querier's key with serve_real.go's public key
//	                  switch.
//
// Weights, counts and features are synthetic, seeded per cell. The cost of every
// operation here depends on the number and level of the ciphertexts and not on
// the values they hold, so synthetic values measure the same cost.
//
// Two steps of the merge have no implementation anywhere in this directory, and
// this file does not supply them. The reciprocal of the class totals has to be
// laid out like the flat-packed numerator before the slot-wise product; here it
// is encrypted in that layout directly, at the numerator's level, and the circuit
// that evaluates it (measureReciprocal) is not rerun, since its cost does not
// depend on d. The merged head, flat-packed, has to be repacked one row per
// ciphertext before applyHead can use it; here the served head is encrypted in
// that layout directly from the plaintext merge, as serve_real.go does from its
// export. Neither repacking is timed.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type costDRow struct {
	D, C, DPad, Cpad, Slots int
	N, EvalLogN, BtpLogN    int
	ResidualMaxLevel        int

	HeadValues, HeadCts          int
	CtBytes, UploadArr, UploadCl int
	AggCtBytes, UploadClAgg      int
	ClientEncMs                  float64

	ServerAdds          int
	ServerAddMs, MulMs  float64
	MergeRelErr         float64
	GaloisKeys          int
	GaloisBytes         int
	KeyGenMs            float64
	QueryEncMs          float64
	QueryBytes          int
	HeadProducts        int
	HeadRotations       int
	HeadMs              float64
	ArgmaxMeasured      bool
	ArgmaxMs            float64
	Bootstraps          int
	InBtpMs             float64
	KSMs                float64
	KSShareBytes        int
	LabelBytes          int
	DecMs, QueryTotalMs float64
	PlainLabel, Served  int
	Agree               bool
	ScaledMargin        float64
	SmudgeLog2          float64
}

const costDHeader = "d,C,d_padded,c_padded,slots,n_parties,eval_logN,btp_logN,residual_max_level," +
	"head_values,head_ciphertexts_per_arrangement,ciphertext_bytes,upload_bytes_per_arrangement," +
	"upload_bytes_per_client,agg_chain_ciphertext_bytes,upload_bytes_per_client_agg_chain," +
	"client_encrypt_ms,server_additions,server_add_ms,server_recip_mul_ms,merge_rel_l2," +
	"galois_keys,galois_key_bytes_total,eval_keygen_ms,query_encrypt_ms,query_upload_bytes," +
	"head_ct_ct_products,head_rotations,head_apply_ms,argmax_measured,argmax_ms,server_bootstraps," +
	"in_bootstrap_ms,key_switch_ms,key_switch_share_bytes,label_download_bytes,querier_decrypt_ms," +
	"query_total_ms,plaintext_label,served_label,label_agree,scaled_margin,smudge_log2_sigma"

func (r costDRow) csv() string {
	head := fmt.Sprintf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.1f,%d,%.1f,%.1f,%.3e,%d,%d,%.1f,%.1f,%d,%d,%d,%.1f,%v",
		r.D, r.C, r.DPad, r.Cpad, r.Slots, r.N, r.EvalLogN, r.BtpLogN, r.ResidualMaxLevel,
		r.HeadValues, r.HeadCts, r.CtBytes, r.UploadArr, r.UploadCl, r.AggCtBytes, r.UploadClAgg,
		r.ClientEncMs, r.ServerAdds, r.ServerAddMs, r.MulMs, r.MergeRelErr,
		r.GaloisKeys, r.GaloisBytes, r.KeyGenMs, r.QueryEncMs, r.QueryBytes,
		r.HeadProducts, r.HeadRotations, r.HeadMs, r.ArgmaxMeasured)
	if !r.ArgmaxMeasured {
		return head + fmt.Sprintf(",,,,,,,,,%d,,,%.6f,%.3f", r.PlainLabel, r.ScaledMargin, r.SmudgeLog2)
	}
	return head + fmt.Sprintf(",%.1f,%d,%.1f,%.1f,%d,%d,%.1f,%.1f,%d,%d,%v,%.6f,%.3f",
		r.ArgmaxMs, r.Bootstraps, r.InBtpMs, r.KSMs, r.KSShareBytes, r.LabelBytes, r.DecMs,
		r.QueryTotalMs, r.PlainLabel, r.Served, r.Agree, r.ScaledMargin, r.SmudgeLog2)
}

func nextPow2(x int) int {
	p := 1
	for p < x {
		p *= 2
	}
	return p
}

// runCostVsD sweeps d and C. The bootstrapping keys are generated once, since they
// are tied to the ideal secret; the rotation keys are generated per cell, since
// the head application needs log2(d_padded) of them and C-1 more to gather the
// logits.
func runCostVsD(evalLogN, n int, dims, classes []int, argmaxEvery bool, csvPath string) {
	fmt.Println("=== encrypted-path cost against the feature dimension and the label space ===")
	fmt.Println()

	residual, btpParams := btpResidualParams(evalLogN)
	slots := residual.MaxSlots()
	fmt.Printf("  residual chain         : 55 + 8x45 at ring 2^%d, max level %d, %d slots\n",
		residual.LogN(), residual.MaxLevel(), slots)
	fmt.Printf("  bootstrapping ring     : 2^%d, logQP %.1f\n",
		btpParams.BootstrappingParameters.LogN(), btpParams.BootstrappingParameters.LogQP())

	// The chain the paper's communication table prices the training upload on,
	// for comparison. Nothing is evaluated on it.
	aggQ := []int{55}
	for i := 0; i < 7; i++ {
		aggQ = append(aggQ, 45)
	}
	aggParams, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: evalLogN, LogQ: aggQ, LogP: []int{61}, LogDefaultScale: 45,
	})
	check(err)
	aggCtBytes := ckks.NewCiphertext(aggParams, 1, aggParams.MaxLevel()).BinarySize()
	fmt.Printf("  aggregation chain      : 55 + 7x45, a fresh ciphertext is %s (sizes only)\n", human(aggCtBytes))
	fmt.Printf("  parties                : %d\n", n)
	fmt.Printf("  key-switch smudging    : sigma %.6g (2^%.3f)\n", ksSmudgeSigma, math.Log2(ksSmudgeSigma))
	fmt.Println()

	kgen, sks, pk, idealSk := collectiveKeys(residual, n, "he-oft-cost-vs-d")

	fmt.Println("  generating bootstrapping keys ...")
	t0 := time.Now()
	btpKeys, _, err := btpParams.GenEvaluationKeys(idealSk)
	check(err)
	fmt.Printf("  bootstrapping keys     : %s in %.1f s\n", human(btpKeys.BinarySize()), ms(time.Since(t0))/1000)
	btpEval, err := bootstrapping.NewEvaluator(btpParams, btpKeys)
	check(err)

	querierSk := kgen.GenSecretKeyNew()
	querierPk := kgen.GenPublicKeyNew(querierSk)
	pcks, err := multiparty.NewPublicKeySwitchProtocol(residual, keySwitchNoise())
	check(err)

	// Every rotation key at this chain has the same size.
	galoisKeyBytes := kgen.GenGaloisKeyNew(residual.GaloisElement(1), idealSk).BinarySize()
	fmt.Printf("  one rotation key       : %s\n\n", human(galoisKeyBytes))

	var rows []string
	for _, C := range classes {
		for i, d := range dims {
			func() {
				defer func() {
					if r := recover(); r != nil {
						fmt.Printf("\n[cost-vs-d] d=%d C=%d FAILED: %v\n", d, C, r)
					}
				}()
				row := measureCostVsD(residual, btpParams, btpEval, kgen, sks, pk, idealSk,
					querierSk, querierPk, pcks, d, C, argmaxEvery || i == 0)
				row.AggCtBytes = aggCtBytes
				row.UploadClAgg = (2*row.HeadCts + 1) * aggCtBytes
				row.GaloisBytes = row.GaloisKeys * galoisKeyBytes
				fmt.Printf("   row: %s\n", row.csv())
				rows = append(rows, row.csv())
				writeCSV(csvPath, costDHeader, rows)
			}()
		}
	}

	fmt.Println("\n--- CSV (record: results/fhe/cost_vs_d.csv) ---")
	fmt.Println(costDHeader)
	for _, r := range rows {
		fmt.Println(r)
	}
	if csvPath != "" {
		fmt.Printf("\nwrote %s\n", csvPath)
	}
}

// measureCostVsD measures one (d, C) cell.
func measureCostVsD(params ckks.Parameters, btpParams bootstrapping.Parameters,
	btpEval bootstrapping.Bootstrapper, kgen *rlwe.KeyGenerator, sks []*rlwe.SecretKey,
	pk *rlwe.PublicKey, idealSk *rlwe.SecretKey, querierSk *rlwe.SecretKey, querierPk *rlwe.PublicKey,
	pcks multiparty.PublicKeySwitchProtocol, d, C int, withArgmax bool) costDRow {

	n := len(sks)
	slots := params.MaxSlots()
	width := d + 1 // d weights and the bias
	dPad := nextPow2(width)
	Cpad := nextPow2(C)
	if dPad > slots || Cpad > slots {
		panic(fmt.Sprintf("does not fit: d_padded=%d C_padded=%d slots=%d", dPad, Cpad, slots))
	}
	fmt.Printf("-- d=%d C=%d (d padded %d, C padded %d)\n", d, C, dPad, Cpad)

	// ---- evaluation keys: relinearization, the tournament's rotations, and the
	// head application's rotate-and-sum and gather rotations ------------------
	var extra []uint64
	for step := 1; step < dPad; step *= 2 {
		extra = append(extra, params.GaloisElement(step))
	}
	for c := 1; c < C; c++ {
		extra = append(extra, params.GaloisElement(-c))
	}
	galSet := map[uint64]bool{params.GaloisElementForComplexConjugation(): true}
	for step := 1; step < Cpad; step *= 2 {
		galSet[params.GaloisElement(step)] = true
		galSet[params.GaloisElement(-step)] = true
	}
	for _, g := range extra {
		galSet[g] = true
	}
	tKey := time.Now()
	ctx, _, _ := newIdxCtxBtp(params, btpEval, kgen, idealSk, pk, argmaxConfig{n, params.LogN(), C}, extra...)
	keyGenMs := ms(time.Since(tKey))
	eval, ecd, enc := ctx.eval, ctx.encoder, ctx.encryptor

	rng := rand.New(rand.NewSource(int64(20260926 + 1000*C + d)))
	scale := 1 / math.Sqrt(float64(d))

	// ---- training upload ------------------------------------------------------
	D := C * width
	K := (D + slots - 1) / slots
	theta0 := make([]float64, D)
	for i := range theta0 {
		theta0[i] = rng.NormFloat64() * scale
	}
	numPlain := make([]float64, D)
	totals := make([]float64, C)
	uploads := make([][]*rlwe.Ciphertext, n)
	var encSum float64
	for j := 0; j < n; j++ {
		v := make([]float64, D)
		for c := 0; c < C; c++ {
			g := float64(1 + rng.Intn(50))
			totals[c] += g
			for f := 0; f < width; f++ {
				x := g * 0.1 * rng.NormFloat64() * scale
				v[c*width+f] = x
				numPlain[c*width+f] += x
			}
		}
		tEnc := time.Now()
		uploads[j] = encryptVector(params, ecd, enc, v, slots, K)
		encSum += ms(time.Since(tEnc))
	}
	ctBytes := uploads[0][0].BinarySize()

	// The reciprocal of the class totals in the numerator's flat layout. See the
	// file comment: this layout is assumed, not produced.
	recip := make([]float64, D)
	for c := 0; c < C; c++ {
		for f := 0; f < width; f++ {
			recip[c*width+f] = 1 / totals[c]
		}
	}
	recipCts := encryptVector(params, ecd, enc, recip, slots, K)

	// ---- server merge -------------------------------------------------------
	tAdd := time.Now()
	merged := make([]*rlwe.Ciphertext, K)
	for k := 0; k < K; k++ {
		acc := uploads[0][k].CopyNew()
		for j := 1; j < n; j++ {
			check(eval.Add(acc, uploads[j][k], acc))
		}
		merged[k] = acc
	}
	addMs := ms(time.Since(tAdd))

	tMul := time.Now()
	for k := range merged {
		p, err := eval.MulRelinNew(merged[k], recipCts[k])
		check(err)
		check(eval.Rescale(p, p))
		merged[k] = p
	}
	mulMs := ms(time.Since(tMul))

	tInit := time.Now()
	for k := range merged {
		pt := ckks.NewPlaintext(params, merged[k].Level())
		pt.Scale = merged[k].Scale
		check(ecd.Encode(chunkSlice(theta0, k, slots), pt))
		check(eval.Add(merged[k], pt, merged[k]))
	}
	addMs += ms(time.Since(tInit))

	// verification only: the merged head against the plaintext merge
	thetaStar := make([]float64, D)
	for c := 0; c < C; c++ {
		for f := 0; f < width; f++ {
			i := c*width + f
			thetaStar[i] = theta0[i] + numPlain[i]/totals[c]
		}
	}
	var num, den float64
	got := make([]float64, slots)
	for k := range merged {
		check(ecd.Decode(ctx.dec.DecryptNew(merged[k]), got))
		for s := 0; s < slots && k*slots+s < D; s++ {
			e := got[s] - thetaStar[k*slots+s]
			num += e * e
			den += thetaStar[k*slots+s] * thetaStar[k*slots+s]
		}
	}
	mergeRelErr := math.Sqrt(num) / math.Sqrt(den)

	// ---- the served head, one ciphertext per class row [W_c | b_c] ------------
	ctHead := make([]*rlwe.Ciphertext, C)
	for c := 0; c < C; c++ {
		row := make([]float64, slots)
		copy(row, thetaStar[c*width:(c+1)*width])
		pt := ckks.NewPlaintext(params, params.MaxLevel())
		check(ecd.Encode(row, pt))
		ct, err := enc.EncryptNew(pt)
		check(err)
		ctHead[c] = ct
	}

	// the query, and its plaintext answer
	phi := make([]float64, d)
	for f := range phi {
		phi[f] = rng.NormFloat64()
	}
	logits := make([]float64, C)
	maxAbs := 0.0
	for c := 0; c < C; c++ {
		s := thetaStar[c*width+d] // the bias, carried by the homogeneous coordinate
		for f := 0; f < d; f++ {
			s += thetaStar[c*width+f] * phi[f]
		}
		logits[c] = s
		maxAbs = math.Max(maxAbs, math.Abs(s))
	}
	gamma := 0.4 / maxAbs
	plainLabel := 0
	for c, v := range logits {
		if v > logits[plainLabel] {
			plainLabel = c
		}
	}
	second := math.Inf(-1)
	for c, v := range logits {
		if c != plainLabel && v > second {
			second = v
		}
	}

	// ---- per query: encrypt, apply the head --------------------------------
	tQ := time.Now()
	vec := make([]float64, slots)
	copy(vec, phi)
	vec[d] = 1
	ptQ := ckks.NewPlaintext(params, params.MaxLevel())
	check(ecd.Encode(vec, ptQ))
	ctPhi, err := enc.EncryptNew(ptQ)
	check(err)
	qEncMs := ms(time.Since(tQ))

	tHead := time.Now()
	acc := applyHead(params, eval, ecd, ctHead, ctPhi, dPad, gamma, C, slots)
	headMs := ms(time.Since(tHead))

	logDPad := 0
	for s := 1; s < dPad; s *= 2 {
		logDPad++
	}
	row := costDRow{
		D: d, C: C, DPad: dPad, Cpad: Cpad, Slots: slots,
		N: n, EvalLogN: params.LogN(), BtpLogN: btpParams.BootstrappingParameters.LogN(),
		ResidualMaxLevel: params.MaxLevel(),
		HeadValues:       D, HeadCts: K, CtBytes: ctBytes,
		UploadArr: K * ctBytes, UploadCl: (2*K + 1) * ctBytes,
		ClientEncMs: encSum / float64(n),
		ServerAdds:  (n-1)*K + K, ServerAddMs: addMs, MulMs: mulMs, MergeRelErr: mergeRelErr,
		GaloisKeys: len(galSet), KeyGenMs: keyGenMs,
		QueryEncMs: qEncMs, QueryBytes: ctPhi.BinarySize(),
		HeadProducts: C, HeadRotations: C*logDPad + (C - 1), HeadMs: headMs,
		ArgmaxMeasured: withArgmax, PlainLabel: plainLabel,
		ScaledMargin: (logits[plainLabel] - second) * gamma,
		SmudgeLog2:   math.Log2(ksSmudgeSigma),
	}
	fmt.Printf("   upload %d ciphertexts per arrangement, client encrypt %.1f ms, merge add %.1f ms, "+
		"reciprocal product %.1f ms, merge rel err %.2e\n", K, row.ClientEncMs, addMs, mulMs, mergeRelErr)
	fmt.Printf("   %d rotation keys in %.1f s, query encrypt %.1f ms, head %.1f ms\n",
		row.GaloisKeys, keyGenMs/1000, qEncMs, headMs)
	if !withArgmax {
		return row
	}

	// ---- argmax index, then the switch to the querier ---------------------
	b0, m0 := ctx.btp.Count(), ctx.btp.Millis()
	tArg := time.Now()
	_, ctIdx, err := ctx.trackedTournament(acc)
	check(err)
	row.ArgmaxMs = ms(time.Since(tArg))
	row.Bootstraps, row.InBtpMs = ctx.btp.Count()-b0, ctx.btp.Millis()-m0

	tKS := time.Now()
	sw, shareBytes := switchToQuerier(params, pcks, sks, querierPk, ctIdx)
	row.KSMs = ms(time.Since(tKS))
	row.KSShareBytes = shareBytes
	row.LabelBytes = sw.BinarySize()

	tDec := time.Now()
	out := make([]float64, slots)
	check(ecd.Decode(rlwe.NewDecryptor(params, querierSk).DecryptNew(sw), out))
	row.DecMs = ms(time.Since(tDec))
	row.Served = int(math.Round(out[0] * float64(Cpad)))
	row.Agree = row.Served == plainLabel
	row.QueryTotalMs = qEncMs + headMs + row.ArgmaxMs + row.KSMs + row.DecMs
	fmt.Printf("   argmax %.1f s (%d bootstraps), key switch %.1f ms, label %d against %d\n",
		row.ArgmaxMs/1000, row.Bootstraps, row.KSMs, row.Served, plainLabel)
	return row
}
