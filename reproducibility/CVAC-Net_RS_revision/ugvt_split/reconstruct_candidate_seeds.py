#!/usr/bin/env python3
"""Reconstruct canonical UGVT membership for the remembered split seeds."""

import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_names(name):
    return [x.strip() for x in (ROOT / name).read_text().splitlines() if x.strip()]


def digest(names):
    return hashlib.sha256(("\n".join(sorted(names)) + "\n").encode()).hexdigest()


def main():
    train = read_names("train_sequences_165.txt")
    test = read_names("test_sequences_45.txt")
    universe = sorted(set(train) | set(test))
    if len(train) != 165 or len(test) != 45 or len(universe) != 210:
        raise RuntimeError("expected disjoint 165/45 membership over 210 pairs")
    if set(train) & set(test):
        raise RuntimeError("train/test overlap")

    candidates = {}
    for seed in (0, 1, 42):
        shuffled = list(universe)
        random.Random(seed).shuffle(shuffled)
        candidates[str(seed)] = {
            "train_membership_exact": set(shuffled[:165]) == set(train),
            "test_membership_exact": set(shuffled[165:]) == set(test),
        }
    report = {
        "procedure": "sort 210 canonical names; Python random.Random(seed).shuffle; first 165 train, final 45 test",
        "train_count": len(train),
        "test_count": len(test),
        "universe_count": len(universe),
        "train_membership_sha256": digest(train),
        "test_membership_sha256": digest(test),
        "candidates": candidates,
        "unique_matching_seed": [
            int(seed) for seed, value in candidates.items()
            if value["train_membership_exact"] and value["test_membership_exact"]
        ],
        "scope_note": "Canonical reconstruction establishes a portable generator for the released membership. The discovered historical script used unsorted os.listdir input, so the exact lists remain the authoritative split definition.",
    }
    (ROOT / "candidate_seed_reconstruction.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
