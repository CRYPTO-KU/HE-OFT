// Command heoft-fhe measures the encrypted operations of HE-OFT under
// multiparty CKKS with Lattigo. Each mode flag selects one measurement and
// writes one record. The modes, and the record each one writes, are listed in
// README.md.
//
// With no mode flag it runs the depth-one aggregation check
//
//	θ = θ₀ + Σ_i w_i · Δ_i ,   w_i = n_i / Σ_j n_j
//
// end to end and validates it against the plaintext float64 computation:
//
//  1. Key generation. N parties each sample a secret key sk_i, whose sum is
//     the ideal secret, and jointly generate a collective public key over a
//     common reference polynomial. No single party holds the decryption key.
//  2. Encryption. Each client encodes its length-d vector Δ_i, chunked across
//     ciphertexts of ring/2 slots, under the collective public key.
//  3. Aggregation. The server computes θ₀ + Σ_i w_i·Δ_i with plaintext-scalar
//     by ciphertext products and ciphertext additions only. Multiplicative
//     depth is one, with no relinearization and no bootstrapping.
//  4. Threshold decryption. The parties run a collective key switch from the
//     joint key to a zero target key. All N shares are required.
//
// It reports the relative L2 error against the plaintext reference and the
// ciphertext cost (ring degree, scale, depth, ciphertext count, bytes per
// ciphertext, upload and download bytes, timings).
//
// Run:  go run .  [-d 5130] [-n 5] [-logn 14]
package main

import (
	"crypto/rand"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	mrand "math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// config is one PoC scenario.
type config struct {
	d    int // head parameter dimension (length of each Δ_i)
	n    int // number of clients
	logN int // log2 ring degree
}

// result captures the validated correctness + cost numbers for one scenario.
type result struct {
	Scenario       string  `json:"scenario"`
	LattigoVersion string  `json:"lattigo_version"`
	D              int     `json:"d_head_params"`
	N              int     `json:"n_clients"`
	LogN           int     `json:"log_ring_degree"`
	RingDegree     int     `json:"ring_degree"`
	Slots          int     `json:"slots_per_ciphertext"`
	CtPerClient    int     `json:"ciphertexts_per_client"`
	LogScale       int     `json:"log_scale"`
	MultDepth      int     `json:"multiplicative_depth_used"`
	BytesPerCt     int     `json:"bytes_per_ciphertext_fresh"`
	UploadBytes    int     `json:"total_upload_bytes"`        // N clients × Δ ciphertexts
	DownloadBytes  int     `json:"total_download_bytes"`      // final result ciphertexts → all clients
	DecShareBytes  int     `json:"decrypt_share_bytes_total"` // CKS shares (threshold-decrypt traffic)
	RelL2Error     float64 `json:"relative_l2_error"`
	MaxAbsError    float64 `json:"max_abs_error"`
	Passed         bool    `json:"passed_1e-3"`
	EncryptMs      float64 `json:"client_encrypt_ms_total"`
	AggregateMs    float64 `json:"server_aggregate_ms"`
	DecryptMs      float64 `json:"threshold_decrypt_ms"`
	DkgMs          float64 `json:"dkg_keygen_ms"`
}

const lattigoVersion = "github.com/tuneinsight/lattigo/v6 v6.1.0"

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	var (
		dFlag               = flag.Int("d", 0, "single-scenario head dimension (0 = run default suite)")
		nFlag               = flag.Int("n", 0, "single-scenario client count")
		logNFlag            = flag.Int("logn", 14, "log2 ring degree")
		outFlag             = flag.String("out", "../outputs/fhe", "directory a mode writes its record to when -json or -csv is not given")
		jsonOut             = flag.String("json", "", "path of the JSON a mode writes (default: the mode's record name under -out; the default aggregation suite writes JSON only when this is set)")
		serveFlag           = flag.Bool("serve", false, "measure the per-query cost units of serving under encryption: one collective refresh and one threshold decryption")
		serveArgmaxFlag     = flag.Bool("serve-argmax", false, "measure the encrypted argmax over C classes as a sequential fold, its bootstraps served by collective refreshes")
		serveTournamentFlag = flag.Bool("serve-tournament", false, "measure the encrypted argmax as a log-depth rotate-and-Max tournament over one packed ciphertext, under collective refreshes")
		serveIndexFlag      = flag.Bool("serve-index", false, "measure the argmax INDEX under encryption, by a one-hot indicator and by an index carried through the tournament, against the value-only tournament as control")
		serveBtpFlag        = flag.Bool("serve-btp", false, "run the tournament argmax with level restoration done by the serving party alone, under collectively generated bootstrapping keys")
		serveIndexBtpFlag   = flag.Bool("serve-index-btp", false, "measure the argmax INDEX with level restoration done by the serving party alone, so the index is priced under the mechanism the paper describes")
		selectionCostFlag   = flag.Bool("selection-cost", false, "measure the selection step: each client scores both arrangements on its held-out data under encryption, the server combines the encrypted scores, and one value is decrypted")
		btpKeyFlag          = flag.Bool("btp-keys", false, "measure the one-time bootstrapping key material that lets the serving party refresh locally")
		commCostFlag        = flag.Bool("comm-cost", false, "measure the communication the protocol needs: key-generation shares, ciphertexts, key-switching shares, and refresh shares")
		costGridFlag        = flag.Bool("cost-grid", false, "measure every protocol operation over the cross product of ring degree and federation size")
		protocolCostFlag    = flag.Bool("protocol-cost", false, "measure the operations the encrypted-serving protocol adds: ciphertext-by-ciphertext head application, encrypted reciprocal for the head merge, key switch to the querier, and selection scoring")
		serveRealFlag       = flag.Bool("serve-real", false, "answer real queries against a real trained head, read from -real-export")
		realExportFlag      = flag.String("real-export", "../outputs/export_head/ag_news_s42_A.json", "-serve-real: the JSON written by experiments/export_head.py")
		realPartiesFlag     = flag.Int("real-parties", 10, "quorum size for -serve-real")
		realQueriesFlag     = flag.Int("real-queries", 0, "answer only the first this many exported queries (0 = all)")
		smudgeLog2Flag      = flag.Float64("smudge-log2", 0, "set the smudging sigma of every key switch in this binary to 2^k (0 keeps 8*rlwe.DefaultNoise, the value every recorded run used); under -smudge-noise it is instead the second setting measured beside 8*rlwe.DefaultNoise, and 0 there means 2^25")
		smudgeNoiseFlag     = flag.Bool("smudge-noise", false, "measure the noise of the ciphertext that enters the key switch on the bootstrapped serving path, and the served label and latency at two smudging settings (see smudge.go)")
		smudgeRepsFlag      = flag.Int("smudge-reps", 1, "-smudge-noise: independent encryptions of each case, each running the whole circuit")
		smudgeTrialsFlag    = flag.Int("smudge-trials", 5, "-smudge-noise: key switches per repetition, variant and smudging setting")
		costVsDFlag         = flag.Bool("cost-vs-d", false, "measure every encrypted operation whose cost depends on the feature dimension d or the label space C, over a grid of both (see cost_vs_d.go)")
		costVsDArgmaxFlag   = flag.Bool("cost-vs-d-argmax", true, "-cost-vs-d: run the argmax and the key switch in every cell; false runs them only at the first d of each C")
		dimsFlag            = flag.String("dims", "", "-cost-vs-d: comma-separated feature dimensions (default 768,1024,2048,4096)")
		classesFlag         = flag.String("classes", "", "comma-separated label-space sizes (default 4,6,14,77,100 for -smudge-noise and 4,14,77,100 for -cost-vs-d)")
		partiesFlag         = flag.Int("parties", 10, "number of parties for -smudge-noise and -cost-vs-d")
		btpEvalLogNFlag     = flag.Int("btp-eval-logn", 15, "evaluation ring for -smudge-noise and -cost-vs-d, with the bootstrapping ring one larger; 15 is the paper's, smaller values are for smoke tests only")
		csvFlag             = flag.String("csv", "", "CSV file written by -smudge-noise and -cost-vs-d, rewritten after every finished case (default: the mode's record name under -out)")
	)
	flag.Parse()

	// -smudge-noise sets its two settings itself; every other mode takes the flag.
	if !*smudgeNoiseFlag {
		setSmudgeLog2(*smudgeLog2Flag)
	}
	if *smudgeNoiseFlag {
		runSmudgeNoise(*btpEvalLogNFlag, *partiesFlag,
			parseInts(*classesFlag, []int{4, 6, 14, 77, 100}),
			*smudgeRepsFlag, *smudgeTrialsFlag, *smudgeLog2Flag,
			outPath(*csvFlag, *outFlag, "smudge_noise.csv"))
		return
	}
	if *costVsDFlag {
		runCostVsD(*btpEvalLogNFlag, *partiesFlag,
			parseInts(*dimsFlag, []int{768, 1024, 2048, 4096}),
			parseInts(*classesFlag, []int{4, 14, 77, 100}),
			*costVsDArgmaxFlag, outPath(*csvFlag, *outFlag, "cost_vs_d.csv"))
		return
	}

	// One real query, end to end, against a real trained head. See serve_real.go.
	if *serveRealFlag {
		tag := strings.TrimSuffix(filepath.Base(*realExportFlag), ".json")
		runServeReal(*realExportFlag, *realPartiesFlag, *logNFlag, *realQueriesFlag,
			outPath(*jsonOut, *outFlag, filepath.Join("real_query", tag+"_answers.json")))
		return
	}
	if *costGridFlag {
		runCostGrid(outPath(*jsonOut, *outFlag, "cost_grid.json"))
		return
	}
	// The per-query cost units of serving under encryption: one collective
	// refresh and one threshold decryption. See serve.go.
	if *serveFlag {
		runServeSuite(outPath(*jsonOut, *outFlag, "serve_primitives.json"))
		return
	}
	// The encrypted argmax over C classes as a sequential fold, its bootstraps
	// served by collective refreshes. See serve_argmax.go.
	if *serveArgmaxFlag {
		runArgmaxSuite(outPath(*jsonOut, *outFlag, "argmax_cost.json"))
		return
	}
	// The encrypted argmax as a log-depth rotate-and-Max tournament over one
	// packed ciphertext. See serve_tournament.go.
	if *serveTournamentFlag {
		runTournamentSuite(outPath(*jsonOut, *outFlag, "argmax_tournament.json"))
		return
	}
	// The argmax index, which is what the protocol returns. See serve_index.go.
	if *serveIndexFlag {
		runIndexSuite(outPath(*jsonOut, *outFlag, "argmax_index.json"))
		return
	}
	// Level restoration by the serving party alone. See serve_btp.go.
	if *serveBtpFlag {
		runServeBtpSuite(outPath(*jsonOut, *outFlag, "argmax_tournament_btp.json"))
		return
	}
	// The index, priced under server-side bootstrapping. See serve_index_btp.go.
	if *serveIndexBtpFlag {
		runIndexBtpSuite(outPath(*jsonOut, *outFlag, "argmax_index_btp.json"))
		return
	}
	// Choosing an arrangement without decrypting either. See protocol_cost.go.
	if *selectionCostFlag {
		runSelectionCost(outPath(*jsonOut, *outFlag, "selection_cost.json"))
		return
	}
	// The operations encrypted serving adds to the depth-one aggregation,
	// including the encrypted reciprocal. See protocol_cost.go.
	if *protocolCostFlag {
		runProtocolCost(outPath(*jsonOut, *outFlag, "protocol_cost.json"))
		return
	}
	// Communication accounting over both modulus chains. See protocol_cost.go.
	if *commCostFlag {
		runCommCost(outPath(*jsonOut, *outFlag, "comm_grid.json"))
		return
	}
	if *btpKeyFlag {
		runBootstrapKeys(outPath(*jsonOut, *outFlag, "btp_keys.json"))
		return
	}

	var scenarios []config
	if *dFlag > 0 && *nFlag > 0 {
		scenarios = []config{{d: *dFlag, n: *nFlag, logN: *logNFlag}}
	} else {
		// Default suite: head dims for a 512→10 head (d≈5130) and a 768→10
		// head (d≈7700), each at N∈{5,10}.
		scenarios = []config{
			{d: 5130, n: 5, logN: *logNFlag},
			{d: 5130, n: 10, logN: *logNFlag},
			{d: 7700, n: 5, logN: *logNFlag},
			{d: 7700, n: 10, logN: *logNFlag},
		}
	}

	results := make([]result, 0, len(scenarios))
	allPass := true
	for _, c := range scenarios {
		r := run(c)
		results = append(results, r)
		allPass = allPass && r.Passed
		printResult(r)
	}

	if *jsonOut != "" {
		b, err := json.MarshalIndent(results, "", "  ")
		check(err)
		check(os.WriteFile(*jsonOut, b, 0o644))
		fmt.Printf("\nwrote %s\n", *jsonOut)
	}

	if !allPass {
		fmt.Println("\nFAIL: at least one scenario exceeded the 1e-3 relative L2 bound")
		os.Exit(1)
	}
	fmt.Println("\nALL SCENARIOS PASSED (relative L2 ≤ 1e-3)")
}

// run executes one full DKG → encrypt → aggregate → threshold-decrypt cycle and
// validates it against the plaintext reference.
func run(c config) result {
	// ---- CKKS parameters --------------------------------------------------
	// LogQ chain: one 55-bit prime and one 45-bit prime suffice, because the
	// circuit consumes exactly one multiplicative level (the PT×CT scalar
	// multiply, followed by one Rescale). Depth = 1. P is the key-switch prime,
	// needed by the collective key-switch (decryption) step.
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            c.logN,
		LogQ:            []int{55, 45},
		LogP:            []int{61},
		LogDefaultScale: 45,
	})
	check(err)

	slots := params.MaxSlots() // N/2 for the standard (conjugate-invariant off) ring
	ctPerClient := (c.d + slots - 1) / slots

	// ---- synthetic protocol inputs ---------------------------------------
	// Realistic magnitudes: cumulative displacements Δ_i are small (~1e-2),
	// θ₀ is O(1). Sample sizes n_i drive the weights w_i.
	rng := mrand.New(mrand.NewSource(20260529))
	theta0 := make([]float64, c.d)
	for j := range theta0 {
		theta0[j] = rng.NormFloat64() * 0.3
	}
	deltas := make([][]float64, c.n)
	sampleSizes := make([]int, c.n)
	for i := 0; i < c.n; i++ {
		deltas[i] = make([]float64, c.d)
		for j := 0; j < c.d; j++ {
			deltas[i][j] = rng.NormFloat64() * 0.02
		}
		sampleSizes[i] = 100 + rng.Intn(900) // heterogeneous client data sizes
	}
	totalSamples := 0
	for _, s := range sampleSizes {
		totalSamples += s
	}
	weights := make([]float64, c.n)
	for i := range weights {
		weights[i] = float64(sampleSizes[i]) / float64(totalSamples)
	}

	// ---- plaintext reference: θ = θ₀ + Σ_i w_i·Δ_i ------------------------
	ref := make([]float64, c.d)
	copy(ref, theta0)
	for i := 0; i < c.n; i++ {
		w := weights[i]
		for j := 0; j < c.d; j++ {
			ref[j] += w * deltas[i][j]
		}
	}

	// ---- shared CRS (common reference string) -----------------------------
	prng, err := sampling.NewKeyedPRNG([]byte("he-ifd-fhe-poc-crs"))
	check(err)
	crs := prng

	encoder := ckks.NewEncoder(params)

	// =======================================================================
	// 1. DKG: N parties sample sk_i, jointly build the collective public key.
	//    Ideal secret key s = Σ_i sk_i (never reconstructed in the clear).
	// =======================================================================
	tDkg := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sks := make([]*rlwe.SecretKey, c.n)
	for i := 0; i < c.n; i++ {
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
	collectivePK := rlwe.NewPublicKey(params)
	ckg.GenPublicKey(ckgCombined, ckgCRP, collectivePK)
	dkgMs := float64(time.Since(tDkg).Microseconds()) / 1000.0

	encryptor := rlwe.NewEncryptor(params, collectivePK)

	// =======================================================================
	// 2. ENCRYPT: each client encrypts its Δ_i as ctPerClient ciphertexts.
	// =======================================================================
	encDeltas := make([][]*rlwe.Ciphertext, c.n) // encDeltas[i][chunk]
	var encryptMs float64
	t0 := time.Now()
	for i := 0; i < c.n; i++ {
		encDeltas[i] = encryptVector(params, encoder, encryptor, deltas[i], slots, ctPerClient)
	}
	encryptMs = float64(time.Since(t0).Microseconds()) / 1000.0

	bytesPerCt := encDeltas[0][0].BinarySize()

	// =======================================================================
	// 3. AGGREGATE (server, the ONLY crypto op):  θ₀ + Σ_i w_i·Δ_i
	//    PT-scalar × CT  (the w_i multiply, +1 level, then Rescale)
	//    CT + CT         (accumulation across clients, and adding θ₀ plaintext)
	//    Multiplicative depth used = 1.
	// =======================================================================
	evaluator := ckks.NewEvaluator(params, nil) // nil eval keys: no relin needed
	t0 = time.Now()
	aggCts := make([]*rlwe.Ciphertext, ctPerClient)
	for chunk := 0; chunk < ctPerClient; chunk++ {
		var acc *rlwe.Ciphertext
		for i := 0; i < c.n; i++ {
			// PT(scalar w_i) × CT, depth 1.
			scaled, err := evaluator.MulNew(encDeltas[i][chunk], weights[i])
			check(err)
			check(evaluator.Rescale(scaled, scaled)) // consume the level, restore scale
			if i == 0 {
				acc = scaled
			} else {
				check(evaluator.Add(acc, scaled, acc)) // CT + CT
			}
		}
		// + θ₀ chunk as a plaintext (PT + CT), encoded at the post-rescale scale/level.
		theta0Chunk := chunkSlice(theta0, chunk, slots)
		pt := ckks.NewPlaintext(params, acc.Level())
		pt.Scale = acc.Scale
		check(encoder.Encode(theta0Chunk, pt))
		check(evaluator.Add(acc, pt, acc))
		aggCts[chunk] = acc
	}
	aggregateMs := float64(time.Since(t0).Microseconds()) / 1000.0
	multDepth := params.MaxLevel() - aggCts[0].Level() // levels consumed = 1

	// =======================================================================
	// 4. THRESHOLD DECRYPT: collective key-switch from the joint key to a zero
	//    target key. Each party contributes a CKS share built from sk_i; the
	//    aggregate switches the ciphertext to be decryptable under sk_zero = 0
	//    (i.e. the masked plaintext can be decoded directly). No single party
	//    can do this alone. All N shares are required (N-out-of-N).
	// =======================================================================
	// Smudging noise for the collective key-switch. The canonical choice (the
	// library's own multiparty tests) is 8×the fresh-encryption noise: large
	// enough to statistically hide each party's secret-key contribution, small
	// enough that the decoded result keeps full float64-grade precision.
	// -smudge-log2 moves it; see smudge.go.
	cks, err := multiparty.NewKeySwitchProtocol(params, keySwitchNoise())
	check(err)
	zeroSk := rlwe.NewSecretKey(params) // sk_output = 0
	decShareBytes := 0
	t0 = time.Now()
	result := make([]float64, 0, c.d)
	for chunk := 0; chunk < ctPerClient; chunk++ {
		ct := aggCts[chunk]
		combined := cks.AllocateShare(ct.Level())
		for i := 0; i < c.n; i++ {
			share := cks.AllocateShare(ct.Level())
			cks.GenShare(sks[i], zeroSk, ct, &share)
			if chunk == 0 {
				decShareBytes += share.BinarySize()
			}
			if i == 0 {
				combined = share
			} else {
				check(cks.AggregateShares(share, combined, &combined))
			}
		}
		switched := ckks.NewCiphertext(params, 1, ct.Level())
		cks.KeySwitch(ct, combined, switched)
		// switched is now decryptable under sk_output = 0: decode directly.
		dec := rlwe.NewDecryptor(params, zeroSk)
		pt := dec.DecryptNew(switched)
		vals := make([]float64, slots)
		check(encoder.Decode(pt, vals))
		take := slots
		if remaining := c.d - chunk*slots; remaining < slots {
			take = remaining
		}
		result = append(result, vals[:take]...)
	}
	decryptMs := float64(time.Since(t0).Microseconds()) / 1000.0

	// ---- validation: relative L2 error vs plaintext reference -------------
	var num, den, maxAbs float64
	for j := 0; j < c.d; j++ {
		e := result[j] - ref[j]
		num += e * e
		den += ref[j] * ref[j]
		if a := math.Abs(e); a > maxAbs {
			maxAbs = a
		}
	}
	relL2 := math.Sqrt(num) / math.Sqrt(den)

	uploadBytes := c.n * ctPerClient * bytesPerCt
	downloadBytes := c.n * ctPerClient * bytesPerCt // result broadcast to all N clients

	return result_(c, params, slots, ctPerClient, bytesPerCt, uploadBytes,
		downloadBytes, decShareBytes*ctPerClient, relL2, maxAbs, multDepth,
		encryptMs, aggregateMs, decryptMs, dkgMs)
}

// result_ assembles the result struct (kept separate to keep run() readable).
func result_(c config, params ckks.Parameters, slots, ctPerClient, bytesPerCt,
	uploadBytes, downloadBytes, decShareBytes int, relL2, maxAbs float64,
	multDepth int, encMs, aggMs, decMs, dkgMs float64) result {
	return result{
		Scenario:       fmt.Sprintf("d=%d N=%d logN=%d", c.d, c.n, c.logN),
		LattigoVersion: lattigoVersion,
		D:              c.d,
		N:              c.n,
		LogN:           c.logN,
		RingDegree:     params.N(),
		Slots:          slots,
		CtPerClient:    ctPerClient,
		LogScale:       int(math.Round(math.Log2(params.DefaultScale().Float64()))),
		MultDepth:      multDepth,
		BytesPerCt:     bytesPerCt,
		UploadBytes:    uploadBytes,
		DownloadBytes:  downloadBytes,
		DecShareBytes:  decShareBytes,
		RelL2Error:     relL2,
		MaxAbsError:    maxAbs,
		Passed:         relL2 <= 1e-3,
		EncryptMs:      encMs,
		AggregateMs:    aggMs,
		DecryptMs:      decMs,
		DkgMs:          dkgMs,
	}
}

// encryptVector encodes v (length d) into ctPerClient CKKS ciphertexts of `slots`
// slots each, encrypted under the collective public key.
func encryptVector(params ckks.Parameters, encoder *ckks.Encoder, enc *rlwe.Encryptor,
	v []float64, slots, ctPerClient int) []*rlwe.Ciphertext {
	cts := make([]*rlwe.Ciphertext, ctPerClient)
	for chunk := 0; chunk < ctPerClient; chunk++ {
		pt := ckks.NewPlaintext(params, params.MaxLevel())
		check(encoder.Encode(chunkSlice(v, chunk, slots), pt))
		ct, err := enc.EncryptNew(pt)
		check(err)
		cts[chunk] = ct
	}
	return cts
}

// chunkSlice returns the `chunk`-th window of `slots` values from v, zero-padded
// if v runs out.
func chunkSlice(v []float64, chunk, slots int) []float64 {
	out := make([]float64, slots)
	start := chunk * slots
	for j := 0; j < slots && start+j < len(v); j++ {
		out[j] = v[start+j]
	}
	return out
}

func printResult(r result) {
	fmt.Printf("\n── scenario %s ──────────────────────────────\n", r.Scenario)
	fmt.Printf("  ring degree           : %d (logN=%d), %d slots/ct\n", r.RingDegree, r.LogN, r.Slots)
	fmt.Printf("  log2 scale            : %d\n", r.LogScale)
	fmt.Printf("  multiplicative depth  : %d\n", r.MultDepth)
	fmt.Printf("  ciphertexts/client    : %d  (d=%d over %d slots)\n", r.CtPerClient, r.D, r.Slots)
	fmt.Printf("  bytes/ciphertext      : %s\n", human(r.BytesPerCt))
	fmt.Printf("  total upload (N=%d)    : %s\n", r.N, human(r.UploadBytes))
	fmt.Printf("  total download (N=%d)  : %s\n", r.N, human(r.DownloadBytes))
	fmt.Printf("  decrypt-share traffic : %s\n", human(r.DecShareBytes))
	fmt.Printf("  DKG keygen time       : %.1f ms\n", r.DkgMs)
	fmt.Printf("  encrypt time (all i)  : %.1f ms\n", r.EncryptMs)
	fmt.Printf("  server aggregate time : %.1f ms\n", r.AggregateMs)
	fmt.Printf("  threshold decrypt time: %.1f ms\n", r.DecryptMs)
	fmt.Printf("  relative L2 error     : %.3e   (max abs %.3e)\n", r.RelL2Error, r.MaxAbsError)
	fmt.Printf("  PASS (≤1e-3)          : %v\n", r.Passed)
}

func human(b int) string {
	const u = 1024.0
	f := float64(b)
	switch {
	case f >= u*u:
		return fmt.Sprintf("%.2f MiB (%d B)", f/(u*u), b)
	case f >= u:
		return fmt.Sprintf("%.2f KiB (%d B)", f/u, b)
	default:
		return fmt.Sprintf("%d B", b)
	}
}

// parseInts reads a comma-separated list of integers, or returns def when s is
// empty.
func parseInts(s string, def []int) []int {
	if strings.TrimSpace(s) == "" {
		return def
	}
	var out []int
	for _, f := range strings.Split(s, ",") {
		v, err := strconv.Atoi(strings.TrimSpace(f))
		check(err)
		out = append(out, v)
	}
	return out
}

// crypto/rand is imported to keep it explicit for auditors. The keyed PRNGs in
// this package produce the public common reference polynomials only. Secret
// keys and encryption noise come from Lattigo's samplers, which crypto/rand
// seeds.
var _ = rand.Reader

// outPath returns explicit when it is set, and otherwise name inside dir. It
// creates the parent directory, so a long run does not fail at its final write.
func outPath(explicit, dir, name string) string {
	p := explicit
	if p == "" {
		p = filepath.Join(dir, name)
	}
	check(os.MkdirAll(filepath.Dir(p), 0o755))
	return p
}
