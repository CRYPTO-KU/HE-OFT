// smudge.go: the key-switch smudging, and the noise of the ciphertext it hides.
//
// Every key switch in this binary smudges with a discrete Gaussian whose standard
// deviation is ksSmudgeSigma and whose tail bound is six times that. The value
// the library uses in its own multiparty tests, 8*rlwe.DefaultNoise, is the
// default, and every mode except -smudge-noise uses it unless -smudge-log2 k
// moves it to 2^k.
//
// -smudge-noise measures the ciphertext that enters the key switch on the serving
// path the paper prices: the tracked tournament of serve_index.go, whose levels
// are restored by the serving party alone under bootstrapping keys generated from
// the ideal secret, exactly as serve_index_btp.go builds it, followed by the
// public key switch of serve_real.go that re-encrypts the label under the
// querier's key. The logits are packLogits' seeded synthetic vectors, the ones
// argmax_tournament_btp.csv and argmax_index_btp.csv were measured on. No head is
// applied, which is also true of those records.
//
// The noise is measured with the ideal secret, the sum of the parties' shares,
// which exists in this single-process simulation and in no deployment. The
// decryption under it is subtracted from the encoding, at the ciphertext's own
// level and scale, of the message the circuit computes in exact arithmetic, and
// the difference is read in the coefficient domain. Two ciphertexts are
// measured. The first is the one the implementation switches. The second is the
// same ciphertext multiplied by a plaintext mask that keeps the label slot alone,
// which is the ciphertext a switch that reveals only the label would carry. The
// second is a diagnostic and not the measured serving path.
package main

import (
	"fmt"
	"math"
	"math/big"
	"os"
	"strings"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/multiparty"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

// ksSmudgeSigma is the standard deviation of every key switch's smudging noise.
var ksSmudgeSigma = 8 * rlwe.DefaultNoise

// setSmudgeLog2 moves the smudging to 2^k. A k of zero or less keeps the default.
func setSmudgeLog2(k float64) {
	if k > 0 {
		ksSmudgeSigma = math.Exp2(k)
	}
}

// smudgeDist is the smudging distribution at standard deviation sigma, truncated
// at six standard deviations.
func smudgeDist(sigma float64) ring.DiscreteGaussian {
	return ring.DiscreteGaussian{Sigma: sigma, Bound: 6 * sigma}
}

// keySwitchNoise is the smudging every key switch in this binary uses.
func keySwitchNoise() ring.DiscreteGaussian { return smudgeDist(ksSmudgeSigma) }

// tieThreshold marks a comparison the exact circuit cannot predict the encrypted
// outcome of. Distinct logits in the seeded vectors differ by far more, so in
// practice it separates the exact ties between padding slots from the rest.
const tieThreshold = 0x1p-20

// collectiveKeys runs the distributed key generation for n parties and forms the
// ideal secret, which this simulation needs to generate evaluation keys and to
// measure noise and which no deployment forms.
func collectiveKeys(params ckks.Parameters, n int, crsLabel string) (*rlwe.KeyGenerator, []*rlwe.SecretKey, *rlwe.PublicKey, *rlwe.SecretKey) {
	crs, err := sampling.NewKeyedPRNG([]byte(crsLabel))
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
	rQP := params.RingQP()
	for i := 0; i < n; i++ {
		rQP.Add(idealSk.Value, sks[i].Value, idealSk.Value)
	}
	return kgen, sks, pk, idealSk
}

// switchToQuerier is the public key switch of serve_real.go: every party
// contributes a share, and the result decrypts under the querier's key alone.
// It returns the switched ciphertext and the size of one party's share.
func switchToQuerier(params ckks.Parameters, pcks multiparty.PublicKeySwitchProtocol,
	sks []*rlwe.SecretKey, querierPk *rlwe.PublicKey, ct *rlwe.Ciphertext) (*rlwe.Ciphertext, int) {

	var agg multiparty.PublicKeySwitchShare
	shareBytes := 0
	for j := range sks {
		share := pcks.AllocateShare(ct.Level())
		pcks.GenShare(sks[j], querierPk, ct, &share)
		if j == 0 {
			agg = share
			shareBytes = share.BinarySize()
		} else {
			check(pcks.AggregateShares(share, agg, &agg))
		}
	}
	out := ckks.NewCiphertext(params, 1, ct.Level())
	pcks.KeySwitch(ct, agg, out)
	return out, shareBytes
}

// coeffError decrypts ct, subtracts the encoding of want at ct's level and scale,
// and returns log2 of the infinity norm and of the standard deviation of the
// difference in the coefficient domain, centred modulo Q at that level. The
// statistics are rlwe.NormStats, which rlwe.Norm uses.
func coeffError(params ckks.Parameters, ecd *ckks.Encoder, dec *rlwe.Decryptor,
	ct *rlwe.Ciphertext, want []float64) (log2Inf, log2Std float64) {

	level := ct.Level()
	pt := dec.DecryptNew(ct)

	ref := ckks.NewPlaintext(params, level)
	*ref.MetaData = *ct.MetaData
	ref.IsBatched = true
	check(ecd.Encode(want, ref))

	ringQ := params.RingQ().AtLevel(level)
	ringQ.Sub(pt.Value, ref.Value, pt.Value)
	if pt.IsNTT {
		ringQ.INTT(pt.Value, pt.Value)
	}
	coeffs := make([]*big.Int, params.N())
	for i := range coeffs {
		coeffs[i] = new(big.Int)
	}
	ringQ.PolyToBigintCentered(pt.Value, 1, coeffs)
	log2Std, _, log2Inf = rlwe.NormStats(coeffs)
	return log2Inf, log2Std
}

// packedVector is the slot vector packLogits encrypts: the logits in the first C
// slots and -0.5 everywhere else.
func packedVector(logits []float64, slots int) []float64 {
	vec := make([]float64, slots)
	for i := range vec {
		vec[i] = -0.5
	}
	copy(vec, logits)
	return vec
}

// trackedTournamentPlain is idxCtx.trackedTournament in exact arithmetic over
// every slot, with Lattigo's step convention: 1 above zero, 0 below, 0.5 at zero.
// Rotation by k moves slot j+k to slot j, as ckks.Evaluator.Rotate does. It
// returns the index vector and, per slot, whether a comparison the slot depends on
// was a tie, in which case the encrypted circuit breaks it on noise and the exact
// value is not a prediction of the encrypted one.
func trackedTournamentPlain(vec []float64, Cpad int) ([]float64, []bool) {
	slots := len(vec)
	m := append([]float64(nil), vec...)
	idx := make([]float64, slots)
	for j := range idx {
		idx[j] = float64(j%Cpad) / float64(Cpad)
	}
	tie := make([]bool, slots)
	nm := make([]float64, slots)
	ni := make([]float64, slots)
	nt := make([]bool, slots)
	for step := 1; step < Cpad; step *= 2 {
		for j := 0; j < slots; j++ {
			k := (j + step) % slots
			diff := m[j] - m[k]
			b := 0.5
			if diff > 0 {
				b = 1
			} else if diff < 0 {
				b = 0
			}
			nm[j] = b*diff + m[k]
			ni[j] = b*(idx[j]-idx[k]) + idx[k]
			switch {
			case math.Abs(diff) < tieThreshold:
				nt[j] = true
			case b == 1:
				nt[j] = tie[j]
			default:
				nt[j] = tie[k]
			}
		}
		m, nm = nm, m
		idx, ni = ni, idx
		tie, nt = nt, tie
	}
	return idx, tie
}

// windowLogits is the number of logits in slot j's tournament window, the Cpad
// slots from j onward, cyclically.
func windowLogits(j, C, Cpad, slots int) int {
	n := 0
	for t := 0; t < Cpad; t++ {
		if (j+t)%slots < C {
			n++
		}
	}
	return n
}

type smudgeRow struct {
	C, Cpad, Rounds, Rep      int
	Variant, SigmaLabel       string
	Sigma, Log2Sigma          float64
	N, EvalLogN, BtpLogN      int
	ResidualMaxLevel, Level   int
	Log2Scale                 float64
	PreLog2Inf, PreLog2Std    float64
	SlotErrLabel              float64
	TieSlots                  int
	SlotErrClean, SlotErrTie  float64
	PostLog2Std               float64
	PlainLabel, Trials, Agree int
	LabelEqual                bool
	DecodedWorst, IdxErrMax   float64
	ExtraReadable, ExtraTotal int
	EncryptMs, ArgmaxMs       float64
	Bootstraps                int
	InBtpMs, MaskMs           float64
	MaskBootstraps            int
	KeySwitchMs, DecryptMs    float64
	TotalMs                   float64
	ShareBytes                int
}

const smudgeHeader = "C,c_padded,tournament_rounds,rep,variant,sigma_setting,smudge_sigma,smudge_log2_sigma," +
	"n_parties,eval_logN,btp_logN,residual_max_level,ct_level,ct_log2_scale," +
	"pre_ks_log2_inf,pre_ks_log2_std,slot_err_label,tie_slots,slot_err_max_clean,slot_err_max_tie," +
	"post_ks_log2_std_mean,plaintext_label,ks_trials,ks_label_agree,label_equal," +
	"decoded_index_worst,index_abs_error_max,extra_slots_readable,extra_slots_total," +
	"encrypt_ms,argmax_ms,server_bootstraps,in_bootstrap_ms,mask_ms,mask_bootstraps," +
	"key_switch_ms_mean,querier_decrypt_ms_mean,total_ms,key_switch_share_bytes"

func (r smudgeRow) csv() string {
	return fmt.Sprintf("%d,%d,%d,%d,%s,%s,%.6g,%.3f,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3e,%d,%.3e,%.3e,"+
		"%.3f,%d,%d,%d,%v,%.6f,%.3e,%d,%d,%.1f,%.1f,%d,%.1f,%.1f,%d,%.1f,%.1f,%.1f,%d",
		r.C, r.Cpad, r.Rounds, r.Rep, r.Variant, r.SigmaLabel, r.Sigma, r.Log2Sigma,
		r.N, r.EvalLogN, r.BtpLogN, r.ResidualMaxLevel, r.Level, r.Log2Scale,
		r.PreLog2Inf, r.PreLog2Std, r.SlotErrLabel, r.TieSlots, r.SlotErrClean, r.SlotErrTie,
		r.PostLog2Std, r.PlainLabel, r.Trials, r.Agree, r.LabelEqual,
		r.DecodedWorst, r.IdxErrMax, r.ExtraReadable, r.ExtraTotal,
		r.EncryptMs, r.ArgmaxMs, r.Bootstraps, r.InBtpMs, r.MaskMs, r.MaskBootstraps,
		r.KeySwitchMs, r.DecryptMs, r.TotalMs, r.ShareBytes)
}

// writeCSV rewrites path with the header and every row so far, so a job that is
// killed keeps the finished rows.
func writeCSV(path, header string, rows []string) {
	if path == "" {
		return
	}
	check(os.WriteFile(path, []byte(header+"\n"+strings.Join(rows, "\n")+"\n"), 0o644))
}

// runSmudgeNoise measures, per label-space size, the noise of the ciphertext that
// enters the key switch and what the switch does to the served label at two
// smudging settings. The circuit runs once per repetition, since the smudging is
// sampled inside the switch and nowhere else; the switch then runs `trials` times
// per setting on the same ciphertext.
func runSmudgeNoise(evalLogN, n int, classes []int, reps, trials int, altLog2 float64, csvPath string) {
	if altLog2 <= 0 {
		altLog2 = 25
	}
	fmt.Println("=== noise of the served ciphertext against the key-switch smudging ===")
	fmt.Println()

	residual, btpParams := btpResidualParams(evalLogN)
	slots := residual.MaxSlots()
	fmt.Printf("  residual chain         : 55 + 8x45 at ring 2^%d, max level %d, log2 scale %d\n",
		residual.LogN(), residual.MaxLevel(), residual.LogDefaultScale())
	fmt.Printf("  bootstrapping ring     : 2^%d, logQP %.1f\n",
		btpParams.BootstrappingParameters.LogN(), btpParams.BootstrappingParameters.LogQP())
	fmt.Printf("  parties                : %d\n", n)
	fmt.Printf("  secret distribution Xs : %v\n", residual.Xs())
	fmt.Printf("  error distribution Xe  : %v\n", residual.Xe())
	fmt.Println()

	kgen, sks, pk, idealSk := collectiveKeys(residual, n, "he-ifd-serve-btp-crs")

	fmt.Println("  generating bootstrapping keys ...")
	t0 := time.Now()
	btpKeys, _, err := btpParams.GenEvaluationKeys(idealSk)
	check(err)
	fmt.Printf("  bootstrapping keys     : %s in %.1f s\n", human(btpKeys.BinarySize()), ms(time.Since(t0))/1000)
	btpEval, err := bootstrapping.NewEvaluator(btpParams, btpKeys)
	check(err)

	querierSk := kgen.GenSecretKeyNew()
	querierPk := kgen.GenPublicKeyNew(querierSk)
	querierDec := rlwe.NewDecryptor(residual, querierSk)

	type setting struct {
		label string
		sigma float64
		pcks  multiparty.PublicKeySwitchProtocol
	}
	settings := []setting{
		{label: "8xDefaultNoise", sigma: 8 * rlwe.DefaultNoise},
		{label: fmt.Sprintf("2^%g", altLog2), sigma: math.Exp2(altLog2)},
	}
	for i := range settings {
		settings[i].pcks, err = multiparty.NewPublicKeySwitchProtocol(residual, smudgeDist(settings[i].sigma))
		check(err)
		fmt.Printf("  smudging setting %-14s: sigma %.6g (2^%.3f), bound %.6g\n",
			settings[i].label, settings[i].sigma, math.Log2(settings[i].sigma), 6*settings[i].sigma)
	}
	fmt.Println()

	var rows []string
	for _, C := range classes {
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("\n[smudge-noise] C=%d FAILED: %v\n", C, r)
				}
			}()
			cfg := argmaxConfig{n, residual.LogN(), C}
			ctx, logits, _ := newIdxCtxBtp(residual, btpEval, kgen, idealSk, pk, cfg)
			Cpad := ctx.Cpad
			rounds := 0
			for s := 1; s < Cpad; s *= 2 {
				rounds++
			}
			trueIdx := 0
			for j, v := range logits {
				if v > logits[trueIdx] {
					trueIdx = j
				}
			}

			// The exact circuit output over every slot, which slots it cannot
			// predict, and which slots other than the label carry the argmax of a
			// window of two or more logits.
			refIdx, tie := trackedTournamentPlain(packedVector(logits, slots), Cpad)
			tieSlots := 0
			var extraSlots []int
			for j := 0; j < slots; j++ {
				if tie[j] {
					tieSlots++
					continue
				}
				if j != 0 && windowLogits(j, C, Cpad, slots) >= 2 && math.Round(refIdx[j]*float64(Cpad)) >= 1 {
					extraSlots = append(extraSlots, j)
				}
			}
			// The functionality's output: the label in slot 0 and nothing else.
			labelOnly := make([]float64, slots)
			labelOnly[0] = float64(trueIdx) / float64(Cpad)
			if tie[0] || math.Abs(refIdx[0]-labelOnly[0]) > 1e-12 {
				fmt.Printf("   WARNING: the exact circuit does not put the label in slot 0 (tie %v, value %.9f)\n",
					tie[0], refIdx[0]*float64(Cpad))
			}
			fmt.Printf("-- C=%d (pad %d, %d rounds): label %d, %d tie slots of %d, %d further argmax slots\n",
				C, Cpad, rounds, trueIdx, tieSlots, slots, len(extraSlots))

			for rep := 0; rep < reps; rep++ {
				tEnc := time.Now()
				_, ct0 := packLogits(residual, ctx.encoder, ctx.encryptor, C)
				encMs := ms(time.Since(tEnc))

				b0, m0 := ctx.btp.Count(), ctx.btp.Millis()
				tArg := time.Now()
				_, ctIdx, err := ctx.trackedTournament(ct0)
				check(err)
				argMs := ms(time.Since(tArg))
				nBtp, inBtp := ctx.btp.Count()-b0, ctx.btp.Millis()-m0

				// The diagnostic: keep the label slot alone.
				b1 := ctx.btp.Count()
				tMask := time.Now()
				ctMask := ctIdx.CopyNew()
				if ctMask.Level() < 1 {
					ctMask, err = ctx.btp.Bootstrap(ctMask)
					check(err)
				}
				keep := make([]float64, slots)
				keep[0] = 1
				check(ctx.eval.Mul(ctMask, keep, ctMask))
				check(ctx.eval.Rescale(ctMask, ctMask))
				maskMs := ms(time.Since(tMask))
				maskBtp := ctx.btp.Count() - b1

				variants := []struct {
					name   string
					ct     *rlwe.Ciphertext
					want   []float64
					maskMs float64
					maskB  int
				}{
					{"paper_path", ctIdx, refIdx, 0, 0},
					{"label_masked", ctMask, labelOnly, maskMs, maskBtp},
				}
				for _, v := range variants {
					preInf, preStd := coeffError(residual, ctx.encoder, ctx.dec, v.ct, v.want)
					vals := make([]float64, slots)
					check(ctx.encoder.Decode(ctx.dec.DecryptNew(v.ct), vals))
					errLabel := math.Abs(vals[0] - v.want[0])
					var errClean, errTie float64
					for j := 0; j < slots; j++ {
						e := math.Abs(vals[j] - v.want[j])
						if v.name == "paper_path" && tie[j] {
							errTie = math.Max(errTie, e)
						} else {
							errClean = math.Max(errClean, e)
						}
					}
					fmt.Printf("   rep %d %-12s level %d, pre-switch noise log2 inf %.2f std %.2f, slot err label %.2e clean %.2e tie %.2e\n",
						rep, v.name, v.ct.Level(), preInf, preStd, errLabel, errClean, errTie)

					for _, s := range settings {
						row := smudgeRow{
							C: C, Cpad: Cpad, Rounds: rounds, Rep: rep, Variant: v.name,
							SigmaLabel: s.label, Sigma: s.sigma, Log2Sigma: math.Log2(s.sigma),
							N: n, EvalLogN: residual.LogN(), BtpLogN: btpParams.BootstrappingParameters.LogN(),
							ResidualMaxLevel: residual.MaxLevel(), Level: v.ct.Level(),
							Log2Scale:  math.Log2(v.ct.Scale.Float64()),
							PreLog2Inf: preInf, PreLog2Std: preStd,
							SlotErrLabel: errLabel, SlotErrClean: errClean, SlotErrTie: errTie,
							PlainLabel: trueIdx, Trials: trials,
							EncryptMs: encMs, ArgmaxMs: argMs, Bootstraps: nBtp, InBtpMs: inBtp,
							MaskMs: v.maskMs, MaskBootstraps: v.maskB, ExtraTotal: len(extraSlots),
						}
						if v.name == "paper_path" {
							row.TieSlots = tieSlots
						}
						var ksSum, decSum, postSum float64
						for trial := 0; trial < trials; trial++ {
							tKS := time.Now()
							sw, shareBytes := switchToQuerier(residual, s.pcks, sks, querierPk, v.ct)
							ksSum += ms(time.Since(tKS))
							row.ShareBytes = shareBytes

							tDec := time.Now()
							out := make([]float64, slots)
							check(ctx.encoder.Decode(querierDec.DecryptNew(sw), out))
							decSum += ms(time.Since(tDec))

							_, postStd := coeffError(residual, ctx.encoder, querierDec, sw, v.want)
							postSum += postStd

							decoded := out[0] * float64(Cpad)
							if int(math.Round(decoded)) == trueIdx {
								row.Agree++
							}
							if e := math.Abs(decoded - float64(trueIdx)); trial == 0 || e > row.IdxErrMax {
								row.IdxErrMax, row.DecodedWorst = e, decoded
							}
							if trial == 0 {
								for _, j := range extraSlots {
									if math.Round(out[j]*float64(Cpad)) == math.Round(refIdx[j]*float64(Cpad)) {
										row.ExtraReadable++
									}
								}
							}
						}
						row.LabelEqual = row.Agree == trials
						row.KeySwitchMs = ksSum / float64(trials)
						row.DecryptMs = decSum / float64(trials)
						row.PostLog2Std = postSum / float64(trials)
						row.TotalMs = encMs + argMs + v.maskMs + row.KeySwitchMs + row.DecryptMs
						fmt.Printf("     %-14s post-switch log2 std %.2f, label %d/%d agree, worst decoded %.6f, switch %.1f ms, %d/%d further slots readable\n",
							s.label, row.PostLog2Std, row.Agree, trials, row.DecodedWorst, row.KeySwitchMs,
							row.ExtraReadable, row.ExtraTotal)
						rows = append(rows, row.csv())
					}
				}
				writeCSV(csvPath, smudgeHeader, rows)
			}
		}()
	}

	fmt.Println("\n--- CSV (record: results/fhe/smudge_noise.csv) ---")
	fmt.Println(smudgeHeader)
	for _, r := range rows {
		fmt.Println(r)
	}
	if csvPath != "" {
		fmt.Printf("\nwrote %s\n", csvPath)
	}
}
