"""HE-OFT: one-shot federated fine-tuning under multiparty CKKS, plaintext side.

This package holds the client-side training, the head merge and the selection
estimator, computed in plaintext. The encrypted protocol (merge, serving and
cost measurements) is the Go code in fhe/.
"""
