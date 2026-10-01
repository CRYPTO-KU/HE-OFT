// protocol_cost.go: the operations the encrypted-serving protocol needs beyond
// the depth-one aggregation. Four of them are measured here:
//
//  1. HEAD APPLICATION under a private query. The client encrypts its features,
//     so the head is applied ciphertext-by-ciphertext, not plaintext-by-ciphertext.
//     Costs one relinearization and one rescale on top of the multiply. The
//     serving party never sees phi(x), so it cannot invert the features back to
//     the client's input.
//
//  2. RECIPROCAL under encryption. The count-weighted head merge divides by the
//     per-class totals. Decrypting that denominator would let a coalition of N-1
//     clients subtract its own counts and recover the honest client's class
//     histogram exactly, so the division has to happen under encryption. Lattigo's
//     inversion circuit over the positive domain, with its bootstraps served by
//     collective refreshes. Paid once at aggregation, never per query.
//
//  3. KEY SWITCH TO THE QUERIER. The label must reach the client that asked and
//     nobody else. Switching to the querier's public key lets the other clients
//     contribute decryption shares without learning the answer.
//
//  4. SELECTION SCORING. One plaintext-mask multiply to pick out the logit of the
//     true class, then one comparison against the encrypted maximum. The client
//     knows its own label, so the mask is public and the multiply stays at depth
//     one. Summing the indicators and decrypting a single scalar is what lets the
//     federation choose between candidates without seeing either of them.
//
// Rates are measured per operation and every reported configuration is computed
// from them.
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/comparison"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/inverse"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/minimax"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/multiparty/mpckks"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

type protoResult struct {
	N           int     `json:"n_parties"`
	LogN        int     `json:"log_n"`
	Slots       int     `json:"slots"`
	CtxCtMs     float64 `json:"ct_x_ct_mul_relin_rescale_ms"`
	PtxCtMs     float64 `json:"pt_x_ct_mul_rescale_ms"`
	RecipMs     float64 `json:"encrypted_reciprocal_ms"`
	RecipIters  int     `json:"reciprocal_refreshes"`
	RecipRelErr float64 `json:"reciprocal_rel_err"`
	PCKSMs      float64 `json:"key_switch_to_querier_ms"`
	PCKSBytes   int     `json:"key_switch_share_bytes_total"`
	MaskMs      float64 `json:"selection_mask_ms"`
	AddMs       float64 `json:"ciphertext_add_ms"`
	RotMs       float64 `json:"rotation_ms"`
}

// ---------------------------------------------------------------------------

func runProtocolCost(jsonPath string) {
	fmt.Println("=== encrypted-serving protocol: per-operation cost ===")
	fmt.Println()
	var out []protoResult
	for _, n := range []int{5, 10, 20} {
		r := measureProtocol(n, 14)
		fmt.Printf("  measuring the encrypted reciprocal at N=%d ...\n", n)
		r.RecipMs, r.RecipIters, r.RecipRelErr = measureReciprocal(n, 15)
		out = append(out, r)
		printProto(r)
	}

	fmt.Println()
	fmt.Println("n_parties,log_n,slots,ct_x_ct_ms,pt_x_ct_ms,reciprocal_ms,reciprocal_refreshes,reciprocal_rel_err,key_switch_querier_ms,key_switch_bytes,selection_mask_ms,add_ms,rotation_ms")
	for _, r := range out {
		fmt.Printf("%d,%d,%d,%.3f,%.3f,%.2f,%d,%.3e,%.2f,%d,%.3f,%.4f,%.3f\n",
			r.N, r.LogN, r.Slots, r.CtxCtMs, r.PtxCtMs, r.RecipMs, r.RecipIters,
			r.RecipRelErr, r.PCKSMs, r.PCKSBytes, r.MaskMs, r.AddMs, r.RotMs)
	}
	if jsonPath != "" {
		b, _ := json.MarshalIndent(out, "", "  ")
		check(os.WriteFile(jsonPath, b, 0o644))
		fmt.Printf("\nwrote %s\n", jsonPath)
	}
}

func printProto(r protoResult) {
	fmt.Printf("N=%d  (ring 2^%d, %d slots)\n", r.N, r.LogN, r.Slots)
	fmt.Printf("  head applied to an encrypted query : %.3f ms   (ct x ct, relin + rescale)\n", r.CtxCtMs)
	fmt.Printf("  head applied to a plaintext query  : %.3f ms   (pt x ct, for contrast)\n", r.PtxCtMs)
	fmt.Printf("  reciprocal under encryption        : %.2f ms   (%d collective refreshes, rel err %.2e)\n",
		r.RecipMs, r.RecipIters, r.RecipRelErr)
	fmt.Printf("  key switch to the querier          : %.2f ms   (%s of shares)\n", r.PCKSMs, human(r.PCKSBytes))
	fmt.Printf("  selection mask + accumulate        : %.3f ms\n", r.MaskMs)
	fmt.Printf("  ciphertext addition                : %.4f ms\n", r.AddMs)
	fmt.Println()
}

// ---------------------------------------------------------------------------

func measureProtocol(n, logN int) protoResult {
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            []int{55, 45, 45, 45, 45, 45, 45, 45},
		LogP:            []int{61},
		LogDefaultScale: 45,
	})
	check(err)
	slots := params.MaxSlots()

	// --- distributed key generation: collective public key + evaluation keys ---
	kgen := rlwe.NewKeyGenerator(params)
	sks := make([]*rlwe.SecretKey, n)
	for i := range sks {
		sks[i] = kgen.GenSecretKeyNew()
	}
	crs, err := sampling.NewKeyedPRNG([]byte("he-oft-protocol-cost"))
	check(err)

	ckg := multiparty.NewPublicKeyGenProtocol(params)
	ckgCRP := ckg.SampleCRP(crs)
	var combined multiparty.PublicKeyGenShare
	for i := 0; i < n; i++ {
		share := ckg.AllocateShare()
		ckg.GenShare(sks[i], ckgCRP, &share)
		if i == 0 {
			combined = share
		} else {
			ckg.AggregateShares(share, combined, &combined)
		}
	}
	pk := rlwe.NewPublicKey(params)
	ckg.GenPublicKey(combined, ckgCRP, pk)

	// The ideal secret exists only for the correctness checks below; the protocol
	// never forms it. Relinearization keys would be generated collectively in a
	// deployment, which costs a one-time round and does not change per-op timings.
	idealSk := rlwe.NewSecretKey(params)
	ringQP := params.RingQP()
	for i := 0; i < n; i++ {
		ringQP.Add(idealSk.Value, sks[i].Value, idealSk.Value)
	}
	ikgen := rlwe.NewKeyGenerator(params)
	rlk := ikgen.GenRelinearizationKeyNew(idealSk)
	gks := ikgen.GenGaloisKeysNew([]uint64{params.GaloisElement(1)}, idealSk)
	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	enc := rlwe.NewEncryptor(params, pk)
	ecd := ckks.NewEncoder(params)
	eval := ckks.NewEvaluator(params, evk)

	rng := rand.New(rand.NewSource(1))
	vecA := make([]float64, slots)
	vecB := make([]float64, slots)
	for i := range vecA {
		vecA[i] = rng.Float64()*2 - 1
		vecB[i] = rng.Float64()*2 - 1
	}
	ptA := ckks.NewPlaintext(params, params.MaxLevel())
	ptB := ckks.NewPlaintext(params, params.MaxLevel())
	check(ecd.Encode(vecA, ptA))
	check(ecd.Encode(vecB, ptB))
	ctA, err := enc.EncryptNew(ptA)
	check(err)
	ctB, err := enc.EncryptNew(ptB)
	check(err)

	const reps = 20

	// --- 1. head applied to an ENCRYPTED query: ct x ct + relin + rescale -----
	t0 := time.Now()
	for i := 0; i < reps; i++ {
		p, err := eval.MulRelinNew(ctA, ctB)
		check(err)
		check(eval.Rescale(p, p))
	}
	ctxct := ms(time.Since(t0)) / reps

	// --- head applied to a PLAINTEXT query, for contrast ----------------------
	t0 = time.Now()
	for i := 0; i < reps; i++ {
		p, err := eval.MulNew(ctA, ptB)
		check(err)
		check(eval.Rescale(p, p))
	}
	ptxct := ms(time.Since(t0)) / reps

	// --- ciphertext addition and rotation ------------------------------------
	t0 = time.Now()
	for i := 0; i < reps; i++ {
		_, err := eval.AddNew(ctA, ctB)
		check(err)
	}
	addMs := ms(time.Since(t0)) / reps

	t0 = time.Now()
	for i := 0; i < reps; i++ {
		_, err := eval.RotateNew(ctA, 1)
		check(err)
	}
	rotMs := ms(time.Since(t0)) / reps

	// --- 3. key switch to the querier's public key ---------------------------
	// Every client contributes a share, but the result is re-encrypted under the
	// asking client's key, so only that client can decrypt the label.
	querierSk := kgen.GenSecretKeyNew()
	querierPk := kgen.GenPublicKeyNew(querierSk)
	pcks, err := multiparty.NewPublicKeySwitchProtocol(params, keySwitchNoise())
	check(err)
	pcksBytes := 0
	t0 = time.Now()
	var pcksAgg multiparty.PublicKeySwitchShare
	for i := 0; i < n; i++ {
		share := pcks.AllocateShare(ctA.Level())
		pcks.GenShare(sks[i], querierPk, ctA, &share)
		if i == 0 {
			pcksAgg = share
			pcksBytes = share.BinarySize() * n
		} else {
			check(pcks.AggregateShares(share, pcksAgg, &pcksAgg))
		}
	}
	switched := ckks.NewCiphertext(params, 1, ctA.Level())
	pcks.KeySwitch(ctA, pcksAgg, switched)
	pcksMs := ms(time.Since(t0))

	// --- 4. selection scoring: plaintext mask, then accumulate ---------------
	// The client knows its own label, so the one-hot mask is public and the
	// multiply stays at depth one.
	mask := make([]float64, slots)
	for i := range mask {
		if i%16 == 0 {
			mask[i] = 1
		}
	}
	ptM := ckks.NewPlaintext(params, ctA.Level())
	check(ecd.Encode(mask, ptM))
	acc, err := eval.MulNew(ctA, ptM)
	check(err)
	t0 = time.Now()
	for i := 0; i < reps; i++ {
		p, err := eval.MulNew(ctA, ptM)
		check(err)
		check(eval.Add(acc, p, acc))
	}
	maskMs := ms(time.Since(t0)) / reps

	return protoResult{
		N: n, LogN: logN, Slots: slots,
		CtxCtMs: ctxct, PtxCtMs: ptxct,
		PCKSMs: pcksMs, PCKSBytes: pcksBytes,
		MaskMs: maskMs, AddMs: addMs, RotMs: rotMs,
	}
}

// ---------------------------------------------------------------------------
// The encrypted reciprocal, measured on the deep chain the sign circuit needs.
//
// The count-weighted head merge divides the accumulated numerator by the
// per-class totals. Those totals are the sum of every client's per-class count,
// so decrypting them would let a coalition of N-1 clients subtract its own
// counts and read the remaining client's class histogram. The division is
// therefore evaluated under encryption, with Lattigo's inversion circuit over
// the positive domain, and its bootstraps are collective refreshes.
//
// This is a one-time cost at aggregation. It is not paid per query.
func measureReciprocal(n, logN int) (float64, int, float64) {
	logQ := []int{55}
	for i := 0; i < 14; i++ {
		logQ = append(logQ, 45)
	}
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: logN, LogQ: logQ, LogP: []int{61, 61}, LogDefaultScale: 45,
	})
	check(err)

	kgen := rlwe.NewKeyGenerator(params)
	sks := make([]*rlwe.SecretKey, n)
	for i := range sks {
		sks[i] = kgen.GenSecretKeyNew()
	}
	crs, err := sampling.NewKeyedPRNG([]byte("he-oft-reciprocal"))
	check(err)

	ckg := multiparty.NewPublicKeyGenProtocol(params)
	ckgCRP := ckg.SampleCRP(crs)
	var combined multiparty.PublicKeyGenShare
	for i := 0; i < n; i++ {
		share := ckg.AllocateShare()
		ckg.GenShare(sks[i], ckgCRP, &share)
		if i == 0 {
			combined = share
		} else {
			ckg.AggregateShares(share, combined, &combined)
		}
	}
	pk := rlwe.NewPublicKey(params)
	ckg.GenPublicKey(combined, ckgCRP, pk)

	idealSk := rlwe.NewSecretKey(params)
	ringQP := params.RingQP()
	for i := 0; i < n; i++ {
		ringQP.Add(idealSk.Value, sks[i].Value, idealSk.Value)
	}
	ikgen := rlwe.NewKeyGenerator(params)
	rlk := ikgen.GenRelinearizationKeyNew(idealSk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)

	minLevel, logBound, ok := mpckks.GetMinimumLevelForRefresh(128, params.DefaultScale(), n, params.Q())
	if !ok {
		panic("no valid refresh level for the reciprocal chain")
	}
	rfp, err := mpckks.NewRefreshProtocol(params, uint(params.LogDefaultScale()), params.Xe())
	check(err)
	btp := &collectiveBootstrapper{
		params: params, sks: sks, rfp: rfp, crs: crs,
		minLevel: minLevel, logBound: logBound,
	}

	enc := rlwe.NewEncryptor(params, pk)
	dec := rlwe.NewDecryptor(params, idealSk)
	ecd := ckks.NewEncoder(params)
	eval := ckks.NewEvaluator(params, evk)
	invEval := inverse.NewEvaluator(params, minimax.NewEvaluator(params, eval, btp))

	// Per-class totals, rescaled by the public federation size into (2^-6, 1].
	// The total number of examples is public, since the per-client sample counts
	// are public, so this normalisation reveals no per-class value.
	slots := params.MaxSlots()
	rng := rand.New(rand.NewSource(7))
	den := make([]float64, slots)
	for i := range den {
		den[i] = 0.02 + 0.97*rng.Float64()
	}
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	check(ecd.Encode(den, pt))
	ct, err := enc.EncryptNew(pt)
	check(err)

	t0 := time.Now()
	inv, err := invEval.EvaluatePositiveDomainNew(ct, -6.0, 0.0)
	check(err)
	elapsed := ms(time.Since(t0))

	got := make([]float64, slots)
	check(ecd.Decode(dec.DecryptNew(inv), got))
	var num, dnm float64
	for i := 0; i < 2048; i++ {
		want := 1.0 / den[i]
		num += (got[i] - want) * (got[i] - want)
		dnm += want * want
	}
	return elapsed, btp.count, math.Sqrt(num) / math.Sqrt(dnm)
}

// ---------------------------------------------------------------------------
// Communication accounting.
//
// "One-shot" here means that no intermediate training artifact is ever exposed:
// there is a single training exchange and no per-round snapshot for anyone to
// observe. It does not mean the parties never speak again. Key generation
// precedes the protocol, selection costs a bounded exchange, and serving costs
// traffic per query. This measures that traffic.
//
// The per-query key-switching share is the quantity to watch, because it is the
// only cost that recurs. Its size follows the level of the ciphertext being
// switched, and the label leaving the argmax sits near the bottom of the chain,
// so we report the share at both ends of the range.
type commResult struct {
	N            int    `json:"n_parties"`
	LogN         int    `json:"log_n"`
	Chain        string `json:"chain"`
	NumModuli    int    `json:"num_moduli"`
	NumAux       int    `json:"num_aux_moduli"`
	PubKeyShare  int    `json:"public_key_share_bytes"`
	RelinShare   int    `json:"relin_key_share_bytes_two_rounds"`
	GaloisShare  int    `json:"galois_key_share_bytes_per_rotation"`
	CtFullLevel  int    `json:"ciphertext_bytes_full_level"`
	CtLowLevel   int    `json:"ciphertext_bytes_level_1"`
	KSShareFull  int    `json:"key_switch_share_bytes_full_level"`
	KSShareLow   int    `json:"key_switch_share_bytes_level_1"`
	RefreshShare int    `json:"collective_refresh_share_bytes"`
}

func runCommCost(jsonPath string) {
	fmt.Println("=== encrypted-serving protocol: communication ===")
	fmt.Println()
	// Two chains are measured, because the protocol uses two. The aggregation
	// chain carries the head merge, which is depth one. The serving chain is the
	// one runTournament builds, and it is what a query ciphertext actually lives
	// in. Reporting only the first understates the per-query traffic.
	aggQ := []int{55, 45, 45, 45, 45, 45, 45, 45}
	serveQ := []int{55}
	for i := 0; i < 14; i++ {
		serveQ = append(serveQ, 45)
	}
	chains := []struct {
		label string
		logQ  []int
		logP  []int
		logNs []int
	}{
		{"aggregation", aggQ, []int{61}, []int{14, 15, 16}},
		{"serving", serveQ, []int{61, 61}, []int{14, 15, 16}},
	}
	var out []commResult
	for _, ch := range chains {
		for _, logN := range ch.logNs {
			for _, n := range []int{5, 10, 20} {
				r := measureComm(n, logN, ch.logQ, ch.logP, ch.label)
				out = append(out, r)
				fmt.Printf("N=%d  (ring 2^%d, %s chain, %d moduli)\n",
					r.N, r.LogN, r.Chain, r.NumModuli)
				fmt.Printf("  setup, per client: public key share %s, relinearization %s, one rotation key %s\n",
					human(r.PubKeyShare), human(r.RelinShare), human(r.GaloisShare))
				fmt.Printf("  one ciphertext   : %s at full level, %s near the bottom of the chain\n",
					human(r.CtFullLevel), human(r.CtLowLevel))
				fmt.Printf("  key-switch share : %s at full level, %s near the bottom\n",
					human(r.KSShareFull), human(r.KSShareLow))
				fmt.Printf("  refresh share    : %s\n\n", human(r.RefreshShare))
			}
		}
	}
	fmt.Println("n_parties,log_n,chain,num_moduli,pubkey_share_B,relin_share_B,galois_share_B,ct_full_B,ct_low_B,ks_share_full_B,ks_share_low_B,refresh_share_B")
	for _, r := range out {
		fmt.Printf("%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d\n", r.N, r.LogN, r.Chain,
			r.NumModuli, r.PubKeyShare,
			r.RelinShare, r.GaloisShare, r.CtFullLevel, r.CtLowLevel,
			r.KSShareFull, r.KSShareLow, r.RefreshShare)
	}
	if jsonPath != "" {
		b, _ := json.MarshalIndent(out, "", "  ")
		check(os.WriteFile(jsonPath, b, 0o644))
		fmt.Printf("\nwrote %s\n", jsonPath)
	}
}

func measureComm(n, logN int, logQ, logP []int, chain string) commResult {
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            logN,
		LogQ:            logQ,
		LogP:            logP,
		LogDefaultScale: 45,
	})
	check(err)
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	ckg := multiparty.NewPublicKeyGenProtocol(params)
	pkShare := ckg.AllocateShare()

	ekg := multiparty.NewRelinearizationKeyGenProtocol(params)
	_, r1, r2 := ekg.AllocateShare()

	gkg := multiparty.NewGaloisKeyGenProtocol(params)
	gkShare := gkg.AllocateShare()

	ctFull := ckks.NewCiphertext(params, 1, params.MaxLevel())
	ctLow := ckks.NewCiphertext(params, 1, 1)

	cks, err := multiparty.NewKeySwitchProtocol(params, keySwitchNoise())
	check(err)
	ksFull := cks.AllocateShare(params.MaxLevel())
	ksLow := cks.AllocateShare(1)

	minLevel, _, ok := mpckks.GetMinimumLevelForRefresh(128, params.DefaultScale(), n, params.Q())
	refreshBytes := 0
	if ok {
		rfp, err := mpckks.NewRefreshProtocol(params, uint(params.LogDefaultScale()), params.Xe())
		check(err)
		rs := rfp.AllocateShare(minLevel, params.MaxLevel())
		refreshBytes = rs.BinarySize()
	}
	_ = sk

	return commResult{
		N: n, LogN: logN, Chain: chain,
		NumModuli:    len(logQ),
		NumAux:       len(logP),
		PubKeyShare:  pkShare.BinarySize(),
		RelinShare:   r1.BinarySize() + r2.BinarySize(),
		GaloisShare:  gkShare.BinarySize(),
		CtFullLevel:  ctFull.BinarySize(),
		CtLowLevel:   ctLow.BinarySize(),
		KSShareFull:  ksFull.BinarySize(),
		KSShareLow:   ksLow.BinarySize(),
		RefreshShare: refreshBytes,
	}
}

// ---------------------------------------------------------------------------
// Bootstrapping keys.
//
// The argmax spends most of its levels, so it needs its budget restored several
// times per query. There are two ways to do that, and the choice decides whether
// serving is practical.
//
// A COLLECTIVE REFRESH needs a share from every client for every refresh. At a
// hundred classes the tournament refreshes tens of times per query, so the
// traffic would run to hundreds of megabytes for a single label. That is not a
// service anyone would operate.
//
// The alternative is to generate bootstrapping keys ONCE, collectively, and let
// the serving party bootstrap on its own afterwards. Homomorphic evaluation under
// a collectively generated key is identical to evaluation under a single key, so
// this is sound, and it moves the entire cost into setup: the refreshes then cost
// no communication at all and the recurring per-query traffic is just the query,
// the label, and one key-switching share per client.
//
// This function times the generation of that one-time key material at a
// residual ring of 2^16. The value it reports as bootstrapping_key_bytes is the
// size of the second return value of GenEvaluationKeys, which is the extended
// secret key and not the evaluation key set. -serve-btp reports the size of the
// key set itself.
func runBootstrapKeys(jsonPath string) {
	fmt.Println("=== bootstrapping keys: the one-time price of local refreshes ===")
	fmt.Println()

	residual, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            16,
		LogQ:            append([]int{55}, repeat(45, 10)...),
		LogP:            []int{61, 61, 61},
		LogDefaultScale: 45,
	})
	check(err)

	btpParams, err := bootstrapping.NewParametersFromLiteral(residual,
		bootstrapping.ParametersLiteral{})
	check(err)

	kgen := rlwe.NewKeyGenerator(btpParams.BootstrappingParameters)
	sk := kgen.GenSecretKeyNew()

	t0 := time.Now()
	_, evk, err := btpParams.GenEvaluationKeys(sk)
	check(err)
	genMs := ms(time.Since(t0))

	size := evk.BinarySize()
	fmt.Printf("  residual ring          : 2^%d\n", residual.LogN())
	fmt.Printf("  bootstrapping ring     : 2^%d\n", btpParams.BootstrappingParameters.LogN())
	fmt.Printf("  key material           : %s\n", human(size))
	fmt.Printf("  generation             : %.1f s\n", genMs/1000)
	fmt.Println()
	fmt.Println("  Generated once, collectively. Refreshes afterwards are local to the")
	fmt.Println("  serving party and cost no communication per query.")
	fmt.Println()
	fmt.Println("metric,value")
	fmt.Printf("bootstrapping_key_bytes,%d\n", size)
	fmt.Printf("bootstrapping_key_gen_ms,%.1f\n", genMs)
	fmt.Printf("residual_log_n,%d\n", residual.LogN())
	fmt.Printf("bootstrapping_log_n,%d\n", btpParams.BootstrappingParameters.LogN())

	if jsonPath != "" {
		b, _ := json.MarshalIndent(map[string]any{
			"bootstrapping_key_bytes":  size,
			"bootstrapping_key_gen_ms": genMs,
			"residual_log_n":           residual.LogN(),
			"bootstrapping_log_n":      btpParams.BootstrappingParameters.LogN(),
		}, "", "  ")
		check(os.WriteFile(jsonPath, b, 0o644))
		fmt.Printf("\nwrote %s\n", jsonPath)
	}
}

func repeat(v, n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// ---------------------------------------------------------------------------
// runCostGrid measures every protocol operation over the full cross product of
// ring degree and federation size, so one figure can carry both axes. GPU CKKS
// figures are usually reported at ring degree 2^16, so the grid also lets a CPU
// figure be compared at a matched ring.
func runCostGrid(jsonPath string) {
	fmt.Println("=== per-operation cost against ring degree and federation size (CPU) ===")
	fmt.Println()
	var out []protoResult
	for _, logN := range []int{14, 15, 16} {
		for _, n := range []int{5, 10, 20} {
			fmt.Printf("  measuring at ring 2^%d, N=%d ...\n", logN, n)
			r := measureProtocol(n, logN)
			out = append(out, r)
			printProto(r)
			if jsonPath != "" { // write after every cell so a wall-clock kill keeps the work
				b, _ := json.MarshalIndent(out, "", "  ")
				check(os.WriteFile(jsonPath, b, 0o644))
			}
		}
	}
	fmt.Println()
	fmt.Println("log_n,n_parties,slots,ct_x_ct_ms,pt_x_ct_ms,add_ms,rotation_ms," +
		"key_switch_ms,key_switch_bytes,reciprocal_ms,mask_ms")
	for _, r := range out {
		fmt.Printf("%d,%d,%d,%.3f,%.3f,%.4f,%.3f,%.3f,%d,%.1f,%.4f\n",
			r.LogN, r.N, r.Slots, r.CtxCtMs, r.PtxCtMs, r.AddMs, r.RotMs,
			r.PCKSMs, r.PCKSBytes, r.RecipMs, r.MaskMs)
	}
	if jsonPath != "" {
		fmt.Printf("\nwrote %s\n", jsonPath)
	}
}

// ---------------------------------------------------------------------------
// Selection.
//
// The federation holds two arrangements, the shared head alone and the shared
// head under each client's own adapter, and it must choose between them without
// decrypting either. Every client scores both arrangements
// on its held-out data under encryption, reports encrypted per-class counts, the
// server combines those into one encrypted score per arrangement by the
// prior-weighted estimator, compares the two scores, and decrypts exactly one
// value, the index of the winner.
//
// A simple count bounds the step by 2NC encrypted comparisons, for N clients and
// C classes. This measures its time and traffic.
//
// One held-out example is scored by the serving circuit itself: reduce its logits
// to the maximum, then test whether the logit of its true label is that maximum.
// The client knows its own label, so the mask that selects that logit is public
// and the multiply stays at depth one. Held-out examples pack: a client's whole
// held-out set, one example per class, occupies C blocks of Cpad slots, which at
// ring degree 2^15 is one ciphertext for every class count the paper reports. So a
// client scores its entire held-out set with ONE tournament and ONE step circuit,
// and the federation's selection costs 2N of them rather than 2NC.
//
// The score circuit is measured once per arrangement for one client, and the 2N
// figure is computed from that rate, which is how the other protocol costs in this
// file are reported. The server combine, the final comparison and the single
// decryption happen once for the federation and are measured as they stand.

type selectionRow struct {
	N           int `json:"n_parties"`
	LogN        int `json:"log_n"`
	C           int `json:"num_classes"`
	Cpad        int `json:"c_padded"`
	Slots       int `json:"slots"`
	HeldOut     int `json:"held_out_per_client"`
	Blocks      int `json:"held_out_blocks_per_ciphertext"`
	CtPerClient int `json:"score_ciphertexts_per_client"`
	Rounds      int `json:"tournament_rounds"`

	// measured: one client scoring one arrangement over its whole held-out set
	ScoreMs        float64 `json:"one_score_ms"`
	ScoreRefreshes int     `json:"one_score_refreshes"`
	ScoreRefreshMs float64 `json:"one_score_refresh_ms"`

	// measured: the steps the federation runs once
	CombineMs        float64 `json:"server_combine_ms"`
	CompareMs        float64 `json:"final_compare_ms"`
	CompareRefreshes int     `json:"final_compare_refreshes"`
	DecryptMs        float64 `json:"threshold_decrypt_ms"`

	// computed from the rate above
	TotalMs        float64 `json:"selection_total_ms"`
	TotalRefreshes int     `json:"selection_total_refreshes"`

	// how many comparisons that is, two ways of counting
	SignCircuits      int `json:"sequential_sign_circuits"`
	ScalarComparisons int `json:"scalar_comparisons"`
	PaperComparisons  int `json:"paper_claim_2nc"`

	// communication
	ScoreCtBytes      int `json:"score_ciphertext_bytes"`
	UploadBytes       int `json:"score_upload_bytes"`
	RefreshShareBytes int `json:"refresh_share_bytes"`
	RefreshBytes      int `json:"refresh_traffic_bytes"`
	KSShareBytes      int `json:"key_switch_share_bytes"`
	DecryptBytes      int `json:"decrypt_share_bytes"`
	CommBytes         int `json:"selection_total_bytes"`

	// correctness
	WinnerCorrect bool    `json:"winner_correct"`
	DecodedWinner float64 `json:"decoded_winner_indicator"`
	CountRelErr   float64 `json:"per_class_count_rel_err"`
	ScoreGap      float64 `json:"plaintext_score_gap"`
}

func runSelectionCost(jsonPath string) {
	fmt.Println("=== selection: choosing an arrangement without decrypting either ===")
	fmt.Println()
	var out []selectionRow
	for _, n := range []int{5, 10, 20} {
		for _, C := range []int{4, 14, 77, 100} {
			fmt.Printf("  measuring selection at N=%d, C=%d ...\n", n, C)
			func() {
				defer func() {
					if r := recover(); r != nil {
						fmt.Printf("\n[selection] N=%d C=%d FAILED: %v\n", n, C, r)
					}
				}()
				r := measureSelection(n, 15, C)
				out = append(out, r)
				printSelection(r)
				fmt.Print("  row: ")
				printSelectionCSV(r)
				if jsonPath != "" { // write after every cell so a kill keeps the work
					b, _ := json.MarshalIndent(out, "", "  ")
					check(os.WriteFile(jsonPath, b, 0o644))
				}
			}()
		}
	}

	fmt.Println("\n--- CSV (record: results/fhe/selection_cost.csv) ---")
	fmt.Println(selectionHeader)
	for _, r := range out {
		printSelectionCSV(r)
	}
	if jsonPath != "" {
		fmt.Printf("\nwrote %s\n", jsonPath)
	}
}

const selectionHeader = "n_parties,logN,C,c_padded,slots,held_out_per_client,blocks_per_ciphertext," +
	"score_ciphertexts_per_client,tournament_rounds,one_score_ms,one_score_refreshes," +
	"one_score_refresh_ms,server_combine_ms,final_compare_ms,final_compare_refreshes," +
	"threshold_decrypt_ms,selection_total_ms,selection_total_refreshes,sequential_sign_circuits," +
	"scalar_comparisons,paper_claim_2nc,score_ciphertext_bytes,score_upload_bytes," +
	"refresh_share_bytes,refresh_traffic_bytes,key_switch_share_bytes,decrypt_share_bytes," +
	"selection_total_bytes,winner_correct,decoded_winner,per_class_count_rel_err,plaintext_score_gap"

func printSelectionCSV(r selectionRow) {
	fmt.Printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%.1f,%d,%.1f,%.2f,%.1f,%d,%.2f,%.1f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%v,%.6f,%.3e,%.6f\n",
		r.N, r.LogN, r.C, r.Cpad, r.Slots, r.HeldOut, r.Blocks, r.CtPerClient, r.Rounds,
		r.ScoreMs, r.ScoreRefreshes, r.ScoreRefreshMs, r.CombineMs, r.CompareMs,
		r.CompareRefreshes, r.DecryptMs, r.TotalMs, r.TotalRefreshes, r.SignCircuits,
		r.ScalarComparisons, r.PaperComparisons, r.ScoreCtBytes, r.UploadBytes,
		r.RefreshShareBytes, r.RefreshBytes, r.KSShareBytes, r.DecryptBytes, r.CommBytes,
		r.WinnerCorrect, r.DecodedWinner, r.CountRelErr, r.ScoreGap)
}

func printSelection(r selectionRow) {
	fmt.Printf("\n-- selection N=%d C=%d (pad %d, %d held-out examples in %d ciphertext)\n",
		r.N, r.C, r.Cpad, r.HeldOut, r.CtPerClient)
	fmt.Printf("   one client, one arrangement : %.1f ms  (%d refreshes, %.1f ms in them)\n",
		r.ScoreMs, r.ScoreRefreshes, r.ScoreRefreshMs)
	fmt.Printf("   server combine              : %.2f ms\n", r.CombineMs)
	fmt.Printf("   final comparison            : %.1f ms  (%d refreshes)\n", r.CompareMs, r.CompareRefreshes)
	fmt.Printf("   threshold decrypt, one value: %.2f ms\n", r.DecryptMs)
	fmt.Printf("   selection, all 2N scores    : %.1f s  (%d refreshes)\n", r.TotalMs/1000, r.TotalRefreshes)
	fmt.Printf("   sequential sign circuits    : %d\n", r.SignCircuits)
	fmt.Printf("   scalar comparisons          : %d   (the paper's 2NC is %d)\n",
		r.ScalarComparisons, r.PaperComparisons)
	fmt.Printf("   communication               : %s  (upload %s, refresh %s, decrypt %s)\n",
		human(r.CommBytes), human(r.UploadBytes), human(r.RefreshBytes), human(r.DecryptBytes))
	fmt.Printf("   winner correct              : %v  (indicator %.6f, plaintext gap %.6f)\n",
		r.WinnerCorrect, r.DecodedWinner, r.ScoreGap)
	fmt.Printf("   per-class counts, rel err   : %.3e\n", r.CountRelErr)
	fmt.Println()
}

// selCtx is one selection cell's cryptographic material.
type selCtx struct {
	params   ckks.Parameters
	encoder  *ckks.Encoder
	eval     *ckks.Evaluator
	cmp      *comparison.Evaluator
	btp      *collectiveBootstrapper
	dec      *rlwe.Decryptor
	Cpad     int
	C        int
	examples int
	blocks   int
	slots    int
	tau      float64
}

// measureSelection measures one (N, C) cell of the selection step.
func measureSelection(n, logN, C int) selectionRow {
	logQ := []int{55}
	for i := 0; i < 14; i++ {
		logQ = append(logQ, 45)
	}
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN: logN, LogQ: logQ, LogP: []int{61, 61}, LogDefaultScale: 45,
	})
	check(err)
	slots := params.MaxSlots()

	// The held-out set. The paper requires one example of every class the client
	// holds, so a client that holds every class reserves C of them, which is the
	// largest held-out set the claim covers.
	Cpad := 1
	for Cpad < C {
		Cpad *= 2
	}
	heldOut := C
	blockCap := slots / Cpad
	blocks := 1
	for blocks < heldOut {
		blocks *= 2
	}
	ctPerClient := 1
	if blocks > blockCap {
		ctPerClient = blocks / blockCap
		blocks = blockCap
	}

	crs, err := sampling.NewKeyedPRNG([]byte("he-oft-selection"))
	check(err)
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

	idealSk := rlwe.NewSecretKey(params)
	ringQP := params.RingQP()
	for i := 0; i < n; i++ {
		ringQP.Add(idealSk.Value, sks[i].Value, idealSk.Value)
	}

	// Rotation keys. Steps inside a block reduce it to its maximum and broadcast
	// that maximum back; steps of a whole block fold the per-example indicators
	// into one per-class count vector.
	galSet := map[uint64]bool{params.GaloisElementForComplexConjugation(): true}
	for step := 1; step < Cpad; step *= 2 {
		galSet[params.GaloisElement(step)] = true
		galSet[params.GaloisElement(-step)] = true
	}
	for step := Cpad; step < Cpad*blocks; step *= 2 {
		galSet[params.GaloisElement(step)] = true
	}
	galEls := make([]uint64, 0, len(galSet))
	for el := range galSet {
		galEls = append(galEls, el)
	}
	relinKey := kgen.GenRelinearizationKeyNew(idealSk)
	galKeys := kgen.GenGaloisKeysNew(galEls, idealSk)
	evk := rlwe.NewMemEvaluationKeySet(relinKey, galKeys...)

	minLevel, logBound, ok := mpckks.GetMinimumLevelForRefresh(128, params.DefaultScale(), n, params.Q())
	if !ok || minLevel+1 > params.MaxLevel() {
		panic(fmt.Sprintf("refresh not possible: minLevel=%d maxLevel=%d", minLevel, params.MaxLevel()))
	}
	rfp, err := mpckks.NewRefreshProtocol(params, uint(params.LogDefaultScale()), params.Xe())
	check(err)
	btp := &collectiveBootstrapper{params: params, sks: sks, rfp: rfp, crs: crs, minLevel: minLevel, logBound: logBound}
	eval := ckks.NewEvaluator(params, evk)
	cmp := comparison.NewEvaluator(params, minimax.NewEvaluator(params, eval, btp),
		minimax.NewPolynomial(comparison.DefaultCompositePolynomialForSign))

	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	ctx := &selCtx{params: params, encoder: encoder, eval: eval, cmp: cmp, btp: btp,
		dec: rlwe.NewDecryptor(params, idealSk), Cpad: Cpad, C: C, examples: heldOut,
		blocks: blocks, slots: slots, tau: 1e-3}

	// Two arrangements over the same held-out set. The logits differ, so the two
	// arrangements are right about different examples and the scores differ.
	// Logit range and padding value are the serving benchmark's, which is the
	// interval the sign circuit approximates on.
	rng := rand.New(rand.NewSource(int64(20260901 + C)))
	labels := make([]int, heldOut)
	for b := range labels {
		labels[b] = b % C
	}
	ctA, refA := ctx.encodeArrangement(rng, encryptor, labels, 0.80)
	ctB, refB := ctx.encodeArrangement(rng, encryptor, labels, 0.45)
	labelMask := make([]float64, slots)
	for b := 0; b < heldOut; b++ {
		labelMask[b*Cpad+labels[b]] = 1
	}

	// --- measured: one client scores one arrangement over its held-out set -----
	r0, ms0 := btp.count, btp.totalMs
	t0 := time.Now()
	countsA, err := ctx.score(ctA, labelMask)
	check(err)
	scoreAMs := ms(time.Since(t0))
	t0 = time.Now()
	countsB, err := ctx.score(ctB, labelMask)
	check(err)
	scoreBMs := ms(time.Since(t0))
	scoreMs := (scoreAMs + scoreBMs) / 2
	scoreRefreshes := (btp.count - r0) / 2
	scoreRefreshMs := (btp.totalMs - ms0) / 2

	// verification only: the encrypted per-class counts against the plaintext ones
	countRelErr := ctx.checkCounts(countsA, refA)

	// --- measured: the server combines N of those into one score per arrangement
	// Every client's count vector has the same shape and the same level, so
	// summing one client's vector N times costs what summing N of them costs.
	// The estimator's scalars are public, so the combine stays at depth one.
	prior := make([]float64, slots)
	for c := 0; c < C; c++ {
		prior[c] = 1.0 / float64(C*n)
	}
	t0 = time.Now()
	eA, err := ctx.combineScores(countsA, prior, n)
	check(err)
	eB, err := ctx.combineScores(countsB, prior, n)
	check(err)
	combineMs := ms(time.Since(t0))

	// --- measured: compare the two encrypted scores ---------------------------
	r1, _ := btp.count, btp.totalMs
	t0 = time.Now()
	winner, err := ctx.compareScores(eA, eB)
	check(err)
	compareMs := ms(time.Since(t0))
	compareRefreshes := btp.count - r1

	// --- measured: decrypt exactly one value, the index of the winner ---------
	cks, err := multiparty.NewKeySwitchProtocol(params, keySwitchNoise())
	check(err)
	zeroSk := rlwe.NewSecretKey(params)
	ksBytes := 0
	t0 = time.Now()
	var ksAgg multiparty.KeySwitchShare
	for i := 0; i < n; i++ {
		share := cks.AllocateShare(winner.Level())
		cks.GenShare(sks[i], zeroSk, winner, &share)
		if i == 0 {
			ksAgg = share
			ksBytes = share.BinarySize()
		} else {
			check(cks.AggregateShares(share, ksAgg, &ksAgg))
		}
	}
	switched := ckks.NewCiphertext(params, 1, winner.Level())
	cks.KeySwitch(winner, ksAgg, switched)
	dec0 := rlwe.NewDecryptor(params, zeroSk)
	outv := make([]float64, slots)
	check(encoder.Decode(dec0.DecryptNew(switched), outv))
	decryptMs := ms(time.Since(t0))

	// The indicator is 1 when the first arrangement scores higher, so the index of
	// the winner is 1 minus it.
	plainA, plainB := ctx.plainScore(refA, labels, n, C), ctx.plainScore(refB, labels, n, C)
	decoded := outv[0]
	winnerCorrect := (decoded > 0.5) == (plainA > plainB)

	rounds := 0
	for s := 1; s < Cpad; s *= 2 {
		rounds++
	}
	scoreCtBytes := countsA.BinarySize()
	refreshShareBytes := rfp.AllocateShare(minLevel, params.MaxLevel()).BinarySize()

	totalScores := 2 * n * ctPerClient
	totalRefreshes := totalScores*scoreRefreshes + compareRefreshes
	uploadBytes := totalScores * scoreCtBytes
	refreshBytes := totalRefreshes * n * refreshShareBytes
	decryptBytes := n * ksBytes

	return selectionRow{
		N: n, LogN: logN, C: C, Cpad: Cpad, Slots: slots,
		HeldOut: heldOut, Blocks: blocks, CtPerClient: ctPerClient, Rounds: rounds,
		ScoreMs: scoreMs, ScoreRefreshes: scoreRefreshes, ScoreRefreshMs: scoreRefreshMs,
		CombineMs: combineMs / 2, CompareMs: compareMs, CompareRefreshes: compareRefreshes,
		DecryptMs: decryptMs,
		TotalMs:   float64(totalScores)*scoreMs + combineMs + compareMs + decryptMs,

		TotalRefreshes: totalRefreshes,
		// one tournament and one step circuit per score, plus the final comparison
		SignCircuits: totalScores*(rounds+1) + 1,
		// what "2NC encrypted comparisons" counts: every pairwise comparison the
		// argmaxes and the equality tests perform, over every held-out example
		ScalarComparisons: totalScores*heldOut*Cpad + 1,
		PaperComparisons:  2 * n * C,

		ScoreCtBytes: scoreCtBytes, UploadBytes: uploadBytes,
		RefreshShareBytes: refreshShareBytes, RefreshBytes: refreshBytes,
		KSShareBytes: ksBytes, DecryptBytes: decryptBytes,
		CommBytes: uploadBytes + refreshBytes + decryptBytes,

		WinnerCorrect: winnerCorrect, DecodedWinner: decoded,
		CountRelErr: countRelErr, ScoreGap: math.Abs(plainA - plainB),
	}
}

// encodeArrangement packs one arrangement's held-out logits, one example per block
// of Cpad slots, and encrypts them. The arrangement predicts the true label on a
// hit fraction of the examples, which is what makes the two scores differ.
// It returns the plaintext logits as well, for the verification.
func (s *selCtx) encodeArrangement(rng *rand.Rand, enc *rlwe.Encryptor,
	labels []int, hit float64) (*rlwe.Ciphertext, [][]float64) {

	vec := make([]float64, s.slots)
	for i := range vec {
		vec[i] = -0.5
	}
	ref := make([][]float64, s.examples)
	for b := 0; b < s.examples; b++ {
		ref[b] = make([]float64, s.C)
		for c := 0; c < s.C; c++ {
			ref[b][c] = rng.Float64()*0.8 - 0.4
		}
		// Put the maximum on the true label, or on a different class, according to
		// the arrangement's hit rate.
		top, at := ref[b][0], 0
		for c, v := range ref[b] {
			if v > top {
				top, at = v, c
			}
		}
		want := labels[b]
		if rng.Float64() >= hit {
			want = (labels[b] + 1) % s.C
		}
		ref[b][at], ref[b][want] = ref[b][want], ref[b][at]
		for c := 0; c < s.C; c++ {
			vec[b*s.Cpad+c] = ref[b][c]
		}
	}
	pt := ckks.NewPlaintext(s.params, s.params.MaxLevel())
	check(s.encoder.Encode(vec, pt))
	ct, err := enc.EncryptNew(pt)
	check(err)
	return ct, ref
}

// score is one client's whole held-out set under one arrangement: reduce each
// block to its maximum, test whether the true label's logit is that maximum, and
// fold the per-example indicators into one per-class count vector.
func (s *selCtx) score(ct *rlwe.Ciphertext, labelMask []float64) (*rlwe.Ciphertext, error) {
	eval, params, btp := s.eval, s.params, s.btp

	// 1. the tournament, exactly as the serving circuit runs it, over every block
	//    at once. Slot b*Cpad ends at the maximum of block b.
	m := ct.CopyNew()
	var err error
	for step := 1; step < s.Cpad; step *= 2 {
		if m.Level() < params.MaxLevel() {
			if m, err = btp.Bootstrap(m); err != nil {
				return nil, err
			}
		}
		rot, err := eval.RotateNew(m, step)
		if err != nil {
			return nil, err
		}
		if m, err = s.cmp.Max(m, rot); err != nil {
			return nil, err
		}
	}

	// 2. isolate each block's maximum and broadcast it back across the block.
	if m, err = btp.Bootstrap(m); err != nil {
		return nil, err
	}
	head := make([]float64, s.slots)
	for b := 0; b < s.blocks; b++ {
		head[b*s.Cpad] = 1
	}
	M, err := eval.MulNew(m, head)
	if err != nil {
		return nil, err
	}
	if err = eval.Rescale(M, M); err != nil {
		return nil, err
	}
	for step := 1; step < s.Cpad; step *= 2 {
		r, err := eval.RotateNew(M, -step)
		if err != nil {
			return nil, err
		}
		if err = eval.Add(M, r, M); err != nil {
			return nil, err
		}
	}

	// 3. the indicator 1[l_c = max]. Halving keeps every slot inside the interval
	//    the composite sign polynomial approximates on.
	d, err := eval.SubNew(ct, M)
	if err != nil {
		return nil, err
	}
	if err = eval.Add(d, s.tau, d); err != nil {
		return nil, err
	}
	if err = eval.Mul(d, 0.5, d); err != nil {
		return nil, err
	}
	if err = eval.Rescale(d, d); err != nil {
		return nil, err
	}
	hot, err := s.cmp.Step(d)
	if err != nil {
		return nil, err
	}
	if hot.Level() < params.LevelsConsumedPerRescaling()*2 {
		if hot, err = btp.Bootstrap(hot); err != nil {
			return nil, err
		}
	}

	// 4. keep the true label's slot only. The client knows its own label, so the
	//    mask is public and the multiply is depth one.
	acc, err := eval.MulNew(hot, labelMask)
	if err != nil {
		return nil, err
	}
	if err = eval.Rescale(acc, acc); err != nil {
		return nil, err
	}

	// 5. fold the blocks together, leaving the per-class counts in block 0.
	for step := s.Cpad; step < s.Cpad*s.blocks; step *= 2 {
		r, err := eval.RotateNew(acc, step)
		if err != nil {
			return nil, err
		}
		if err = eval.Add(acc, r, acc); err != nil {
			return nil, err
		}
	}
	return acc, nil
}

// combineScores sums n clients' count vectors and applies the prior-weighted
// estimator, leaving the arrangement's score in slot 0. Depth one.
func (s *selCtx) combineScores(counts *rlwe.Ciphertext, prior []float64, n int) (*rlwe.Ciphertext, error) {
	eval := s.eval
	acc := counts.CopyNew()
	for i := 1; i < n; i++ {
		if err := eval.Add(acc, counts, acc); err != nil {
			return nil, err
		}
	}
	e, err := eval.MulNew(acc, prior)
	if err != nil {
		return nil, err
	}
	if err = eval.Rescale(e, e); err != nil {
		return nil, err
	}
	for step := 1; step < s.Cpad; step *= 2 {
		r, err := eval.RotateNew(e, step)
		if err != nil {
			return nil, err
		}
		if err = eval.Add(e, r, e); err != nil {
			return nil, err
		}
	}
	return e, nil
}

// compareScores returns the encrypted indicator of the first arrangement scoring
// higher. It is the one value the federation decrypts.
func (s *selCtx) compareScores(a, b *rlwe.Ciphertext) (*rlwe.Ciphertext, error) {
	eval, params, btp := s.eval, s.params, s.btp
	d, err := eval.SubNew(a, b)
	if err != nil {
		return nil, err
	}
	if d.Level() < params.LevelsConsumedPerRescaling()*2 {
		if d, err = btp.Bootstrap(d); err != nil {
			return nil, err
		}
	}
	if err = eval.Mul(d, 0.5, d); err != nil {
		return nil, err
	}
	if err = eval.Rescale(d, d); err != nil {
		return nil, err
	}
	return s.cmp.Step(d)
}

// checkCounts decrypts the per-class counts and compares them to the plaintext
// ones. Verification only: the protocol never decrypts this vector.
func (s *selCtx) checkCounts(counts *rlwe.Ciphertext, ref [][]float64) float64 {
	got := make([]float64, s.slots)
	check(s.encoder.Decode(s.dec.DecryptNew(counts), got))
	want := make([]float64, s.C)
	for b := 0; b < s.examples; b++ {
		top, at := ref[b][0], 0
		for c, v := range ref[b] {
			if v > top {
				top, at = v, c
			}
		}
		if at == b%s.C {
			want[at]++
		}
	}
	var num, den float64
	for c := 0; c < s.C; c++ {
		e := got[c] - want[c]
		num += e * e
		den += want[c] * want[c]
	}
	if den == 0 {
		return math.Sqrt(num)
	}
	return math.Sqrt(num) / math.Sqrt(den)
}

// plainScore is the estimator evaluated in the clear, for the verification of the
// decrypted winner.
func (s *selCtx) plainScore(ref [][]float64, labels []int, n, C int) float64 {
	correct := 0.0
	for b := 0; b < s.examples; b++ {
		top, at := ref[b][0], 0
		for c, v := range ref[b] {
			if v > top {
				top, at = v, c
			}
		}
		if at == labels[b] {
			correct++
		}
	}
	return correct / float64(C)
}
