#!/usr/bin/env python3
"""Sequence-cluster bootstrap confidence intervals for Stage-2 OOF predictions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def auroc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    positives, negatives = labels == 1, labels == 0
    if not positives.any() or not negatives.any():
        return None
    comparisons = scores[positives, None] - scores[negatives][None, :]
    return float((np.sum(comparisons > 0) + .5*np.sum(comparisons == 0)) /
                 comparisons.size)


def load(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {(row["object"], row["sequence"], row["seed"], row["frame_index"]): row
            for row in rows}


def interval(values):
    values = np.asarray(values)
    return {"mean": float(values.mean()), "lower_95": float(np.quantile(values, .025)),
            "upper_95": float(np.quantile(values, .975))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference-name", default="candidate_only")
    parser.add_argument("--candidate-name", default="candidate_projection")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, candidate = load(args.reference), load(args.candidate)
    if set(reference) != set(candidate):
        raise ValueError("Prediction rows do not match")
    clusters = {}
    for key in reference:
        clusters.setdefault(key[:2], []).append(key)
    cluster_names = sorted(clusters)
    rng = np.random.default_rng(args.seed)
    ref_values, cand_values, differences = [], [], []
    for _ in range(args.samples):
        sampled = rng.choice(len(cluster_names), len(cluster_names), replace=True)
        keys = [key for index in sampled for key in clusters[cluster_names[index]]]
        y = [int(reference[key]["failure_label"]) for key in keys]
        ref = auroc(y, [float(reference[key]["failure_probability"]) for key in keys])
        cand = auroc(y, [float(candidate[key]["failure_probability"]) for key in keys])
        if ref is not None and cand is not None:
            ref_values.append(ref); cand_values.append(cand); differences.append(cand-ref)
    result = {"schema_version": 1, "resampling_unit": "object/sequence",
              "requested_samples": args.samples, "valid_samples": len(differences),
              "seed": args.seed, args.reference_name: interval(ref_values),
              args.candidate_name: interval(cand_values),
              f"{args.candidate_name}_minus_{args.reference_name}": interval(differences),
              "probability_candidate_better": float(np.mean(np.asarray(differences) > 0))}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
