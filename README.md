# HE-OFT

Code and records for *HE-OFT: Privacy-Preserving One-Shot Federated Fine-Tuning
under Homomorphic Encryption*, by Halil İbrahim Kanpak, Sinem Sav and Alptekin
Küpçü. The paper is under review. The technical report will appear as
arXiv:XXXX.XXXXX.

## The protocol

Every client fine-tunes the same frozen public backbone. It trains a low-rank
adapter whose down-projection is frozen at a shared initialization, and a
classifier head. The adapter never leaves the client. Each client encrypts its
head displacement under multiparty CKKS and uploads it once.

The server merges the displacements with coverage weights. Row c of the shared
head is the average of the row-c displacements of the clients that hold class
c, weighted by their example counts. The server multiplies by an encrypted
reciprocal of the per-class totals, and the shared head is never decrypted.

To answer a test-time query, the querier computes its features with its own
backbone and adapter and encrypts them. The server applies the encrypted head
and takes the argmax under encryption. A quorum of clients then key-switches
the label to the querier, who receives only the label. Each client may query
up to a fixed allowance.

The federation chooses between two arrangements: the shared head over the
bare backbone, and the shared head over each client's own adapter. A
global-prior estimator makes the choice and decrypts one value.

## Layout

| path | contents |
|---|---|
| `heoft/` | Python package, the plaintext side: data and partition, the client model, local training, the head merge, the selection estimator, attack helpers |
| `experiments/` | one entry point per experiment of the paper |
| `fhe/` | Go code on Lattigo v6: the encrypted merge, encrypted serving with the tournament argmax and bootstrapping, key switching, and every cost measurement |
| `results/` | the records behind every number in the paper, CSV and JSON |
| `figures/` | the scripts that draw the paper's plots from `results/` |
| `slurm/` | batch templates for a Slurm cluster |
| `tests/` | unit tests of the merge and the estimator |

## Requirements

- Python 3.9 or later with PyTorch, transformers, peft and datasets.
  `requirements.txt` pins the environment of the paper's runs (Python 3.9.25,
  PyTorch 2.3.0 with CUDA 12.1). They used one NVIDIA A40 (48 GB) per job.
- Go 1.24 or later. `go build` fetches Lattigo v6.1.0.

```bash
pip install -r requirements.txt
pip install -e .
cd fhe && go build -o heoft-fhe .
```

## Reproducing the paper

`REPRODUCE.md` gives, for every table and figure, the command that produces it
and the record it is compared against. Runs write to `outputs/<experiment>/`,
which git ignores, so a rerun never overwrites a record in `results/`.

The attack experiments read the per-client states that the accuracy runs save,
so `experiments/accuracy_text.py` and `experiments/accuracy_vision.py` run
first.

## Labels in the records

The records keep the labels the code prints. Their names in the paper:

| label | paper |
|---|---|
| `A_headonly`, arrangement `A` | the shared head |
| `B_personal`, arrangement `B` | the personal arrangement |
| `sel_globalprior` | selected, by the global-prior estimator |
| `local` | a client alone |
| `current` | the disclosed model, a plaintext reference the protocol never builds |
| `matched_total` | pooled, the same recipe on the union of the clients' data |
| `sel_federated`, `sel_fed_balanced`, `sel_perclient`, `sel_gp_rarefill` | the alternative selection rules the paper compares against |

## Citation

Please cite the technical report. The identifier is a placeholder until the
report is posted.

```bibtex
@article{kanpak2026heoft,
  title   = {{HE-OFT}: Privacy-Preserving One-Shot Federated Fine-Tuning under Homomorphic Encryption (Technical Report)},
  author  = {Kanpak, Halil \.{I}brahim and Sav, Sinem and K\"up\c{c}\"u, Alptekin},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

Apache License 2.0, see `LICENSE`.
