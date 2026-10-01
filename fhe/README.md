# fhe

This directory holds the Go program that measures the encrypted operations of HE-OFT under multiparty CKKS, written against Lattigo v6.1.0. N parties generate a collective public key from their secret-key shares, and no party holds the decryption key. Each client encrypts its head displacement and its per-class counts once. The server forms the coverage-weighted merge by summing the uploads, multiplying the sum by the encrypted reciprocal of the per-class totals and adding the public initializer, so the totals are never decrypted. The reciprocal is evaluated under encryption with Lattigo's inversion circuit over the positive domain. `-protocol-cost` times that circuit, and `-cost-vs-d` times the product with its result. A query is served under encryption. The querier encrypts its features, the server applies the encrypted head ciphertext by ciphertext, and the server reduces the logits to the index of the largest one with a tournament argmax that carries the index through every round. The server restores the levels this circuit consumes by bootstrapping on its own, under bootstrapping keys generated once from the parties' shares. All N parties then key-switch the label to the querier's public key, so the querier alone decrypts it. Every key switch adds smudging noise from a discrete Gaussian with standard deviation 8 times `rlwe.DefaultNoise`, truncated at six standard deviations, unless `-smudge-log2` moves it. Several modes measure the same circuits with levels restored by a collective refresh instead of server-side bootstrapping, so that the two mechanisms can be compared, and the real-query mode restores levels by collective refresh as well. All parties run in one process. The program forms the ideal secret, the sum of the shares, only to generate evaluation keys and to verify results, and no protocol step uses it.

## Build

Go 1.24 or newer is required. The first build downloads Lattigo v6.1.0 from the Go module proxy, pinned by `go.sum`.

```sh
cd fhe
go build -o heoft-fhe .
go vet ./...
go test ./...
```

The test runs the depth-one aggregation check at ring degree 2^14 and asserts a relative L2 error of at most 1e-3 and a multiplicative depth of one.

## Running a mode

Every command below runs from `fhe/`. A mode writes to `../outputs/fhe/` unless `-out` names another directory, and `-json` or `-csv` names a single output file. No mode writes to `../results/fhe/`, which holds the records the paper cites.

Six modes write their record as a file. `-cost-grid`, `-comm-cost`, `-btp-keys` and `-protocol-cost` write JSON, and `-cost-vs-d` and `-smudge-noise` write CSV, rewritten after every finished case. The other modes print their record to standard output as a CSV block that begins at the line `--- CSV (record: ...)`, and the JSON they write holds the same rows. The following commands extract that block.

```sh
./heoft-fhe -serve-tournament | tee ../outputs/fhe/argmax_tournament.log
awk '/^--- CSV/{f=1;next} f&&/^$/{exit} f' ../outputs/fhe/argmax_tournament.log \
  > ../outputs/fhe/argmax_tournament.csv
```

`-serve-real` reads the head and the query features that `experiments/export_head.py` writes to `../outputs/export_head/`. It runs once per arrangement, and the record `real_query.csv` holds both CSV blocks under one header.

## Modes and records

The hardware column gives the memory the cluster job requested, which bounds the peak from above. No mode uses a GPU. Wall times come from the record or its notes, except those marked as measured on a laptop. Where only the timed operations are recorded, the column gives their sum, which is a lower bound on the job's wall time. The records were measured on one core of a CPU cluster node.

| Mode flags | What it measures | Record it reproduces | Hardware | Wall time |
|---|---|---|---|---|
| `-cost-grid` | Per-operation cost on the eight-modulus aggregation chain: ciphertext-by-ciphertext and plaintext-by-ciphertext products, addition, rotation, key switch to the querier and selection mask, at ring degree 2^14, 2^15 and 2^16 and N in {5, 10, 20}. Its reciprocal fields are zero, since the grid does not evaluate the reciprocal | `results/fhe/cost_grid.json` | CPU, 64 GB | 16 s on a laptop |
| `-protocol-cost` | The same operations at ring degree 2^14, and the encrypted reciprocal of the per-class totals by the inversion circuit at ring degree 2^15 under collective refresh, at N in {5, 10, 20} | `results/fhe/protocol_cost.json` | CPU, 48 GB | not recorded, the reciprocal alone takes 3.2 to 6.3 s |
| `-comm-cost` | Byte sizes of the key-generation shares, ciphertexts, key-switching shares and refresh shares, on the aggregation and the serving chain, at ring degree 2^14 to 2^16 and N in {5, 10, 20} | `results/fhe/comm_grid.json` | CPU, 96 GB | under 1 s on a laptop |
| `-btp-keys` | Generation time of the bootstrapping key material at a residual ring of 2^16. The byte count it reports is that of the extended secret key and not of the key set, which `-serve-btp` reports | `results/fhe/btp_keys.json` | CPU, 96 GB | 50.8 s of key generation |
| `-serve` | One collective refresh and one threshold decryption at ring degree 2^15 and N in {5, 10, 20} | `results/fhe/serve_primitives.csv` (standard output) | CPU, 32 GB | about 4 s of timed operations |
| `-serve-argmax` | The argmax as a sequential fold of C-1 comparisons under collective refresh, N = 10, C in {4, 6, 14, 77, 100} | `results/fhe/argmax_cost.csv` (standard output) | CPU, 48 GB | 52 min of argmax |
| `-serve-tournament` | The argmax as a rotate-and-Max tournament of ceil(log2 C) rounds under collective refresh, N = 10, C in {4, 6, 14, 77, 100} | `results/fhe/argmax_tournament.csv` (standard output) | CPU, 48 GB | 6.1 min of argmax |
| `-serve-index` | The argmax index by a one-hot step circuit and by a tracked tournament, against the value-only tournament, under collective refresh | `results/fhe/argmax_index.csv` (standard output) | CPU, 64 GB | 15 min of measured circuits |
| `-serve-btp` | The tournament argmax under server-side bootstrapping, evaluation ring 2^15, bootstrapping ring 2^16, residual chain 55 + 8x45 selected by a search under a 1553-bit ceiling, and the size of the bootstrapping key set | `results/fhe/argmax_tournament_btp.csv` (standard output) | CPU, 200 GB | 80 min of argmax and 46 s of key generation |
| `-serve-index-btp` | The argmax index by both constructions under server-side bootstrapping, at the parameters of `-serve-btp` | `results/fhe/argmax_index_btp.csv` (standard output) | CPU, 200 GB | 3 h 15 min |
| `-selection-cost` | The selection step, in which each client scores both arrangements on its held-out set under encryption, the server combines the scores and compares them, and one value is decrypted. N in {5, 10, 20}, C in {4, 14, 77, 100}, collective refresh | `results/fhe/selection_cost.csv` (standard output) | CPU, 64 GB | 45 min |
| `-cost-vs-d` | Every operation whose cost depends on the feature dimension d or the number of classes C: training upload, server merge, query encryption, head application, tracked argmax under server-side bootstrapping, key switch to the querier and decryption. d in {768, 1024, 2048, 4096}, C in {4, 14, 77, 100}, N = 10 | `results/fhe/cost_vs_d.csv` | CPU, 64 GB | 5.3 h of timed operations |
| `-cost-vs-d -dims 768 -classes 4,6,14,77,100 -csv ../outputs/fhe/query_cost.csv` | The same path at d = 768 and five label-space sizes | `results/fhe/query_cost.csv` | CPU, 48 GB | 1.6 h of timed operations |
| `-smudge-noise -smudge-log2 25 -smudge-reps 3 -smudge-trials 10` | Noise of the label ciphertext before the key switch on the bootstrapped serving path, and agreement of the served label after the switch, at smudging 8 times `rlwe.DefaultNoise` and 2^25, C in {4, 6, 14, 77, 100} | `results/fhe/smudge_noise.csv` | CPU, 64 GB | 4.4 h of timed operations |
| `-serve-real -logn 15 -real-parties 10 -real-export ../outputs/export_head/ag_news_s42_A.json`, then the same with `_B` | Sixteen real AG-News queries per arrangement on a trained head: encrypted head application, tracked argmax under collective refresh, a mask that keeps the label slot alone, key switch to the querier | `results/fhe/real_query.csv` (standard output) and `results/fhe/real_query/ag_news_s42_{A,B}_answers.json` | CPU, 64 GB | 10.4 min of timed queries per arrangement |
| no mode flag | The depth-one aggregation check, d in {5130, 7700}, N in {5, 10}, ring degree 2^14 | none | CPU | under 1 s on a laptop |

Byte counts, levels, and refresh and bootstrap counts depend only on the parameters and the seeded plaintexts, and a rerun reproduces them exactly. Errors and decoded values change in their last digits, because secret keys and encryption noise are sampled afresh in every run. Timings depend on the machine.

`-btp-eval-logn` below 15 is for quick checks only, since the bootstrapped chain then falls outside the 128-bit security budget. `-serve-real` refuses `-logn` below 15 for the same reason.
