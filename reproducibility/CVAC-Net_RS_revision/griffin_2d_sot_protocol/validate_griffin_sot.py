#!/usr/bin/env python3
"""Validate alignment, annotations, and image availability of Griffin SOT."""

import argparse
import json
from pathlib import Path


def load_manifest(root: Path, name: str):
    with (root / name / f"{name}.json").open() as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "sot_test")
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()
    names = ("GRIFFIN_V", "GRIFFIN_D")
    manifests = {name: load_manifest(args.root, name) for name in names}
    assert list(manifests[names[0]]) == list(manifests[names[1]]), "sequence order/names differ"
    total = 0
    for sequence in manifests[names[0]]:
        expected = None
        for name in names:
            info = manifests[name][sequence]
            count = len(info["img_names"])
            assert count == len(info["gt_rect"]) > 0, f"{name}/{sequence}: length mismatch"
            assert info["init_rect"] == info["gt_rect"][0], f"{name}/{sequence}: bad init_rect"
            assert all(len(box) == 4 and box[2] > 0 and box[3] > 0 for box in info["gt_rect"])
            if expected is None:
                expected = count
            assert count == expected, f"{sequence}: paired views differ in length"
            if not args.skip_images:
                missing = [path for path in info["img_names"] if not (args.root / name / path).is_file()]
                assert not missing, f"{name}/{sequence}: {len(missing)} images missing"
        total += expected
    print(f"validated {len(manifests[names[0]])} aligned sequences, {total} frames/view, {2 * total} total")


if __name__ == "__main__":
    main()
