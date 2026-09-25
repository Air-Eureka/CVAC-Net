# Derived Griffin 2D single-object tracking protocol

This package documents the paired 2D single-object tracking protocol derived for the CVAC-Net study from the CARLA–AirSim Griffin co-simulation. It does not redefine the official Griffin tasks, distribute the original images or annotations, or represent real-world aerial–ground deployment data.

The conversion script requires Python 3.8 or newer and NumPy. The validation script uses the Python standard library.

## Contents

- `convert_griffin_sot.py`: projects synchronized 3D cuboids into paired 2D tracking sequences.
- `validate_griffin_sot.py`: validates the generated sequence structure, boxes, images, and initialization.
- `protocol.json`: frozen camera, projection, filtering, split, view-mapping, and evaluation rules.
- `train_sequences.txt`, `val_sequences.txt`, and `difficult26_sequences.txt`: exact sequence-pair lists.
- `sequence_manifest.jsonl`: source scene, object identity, source-frame, camera, visibility, and box-area mapping for all 370 derived pairs.
- `derivation_audit.json`: independently checked counts and source hashes.
- `SHA256SUMS`: hashes of every distributed file except itself.

The required upstream split is `data/split_datas/griffin_100scenes_random.json` from the Griffin release. Its expected SHA-256 is `24e0ec8c30518b53a2b5be034a840951932e3db13f4df8501ae0164819b627ea`. Obtain Griffin under its original access conditions.

`GRIFFIN_V` is the vehicle-side ground view and `GRIFFIN_D` is the drone-side aerial view. The converter uses these names directly.

## Conversion

Training example:

```bash
python convert_griffin_sot.py \
  --source-root /path/to/griffin-release \
  --split-file /path/to/griffin-release/data/split_datas/griffin_100scenes_random.json \
  --split train \
  --output-root /path/to/derived/train \
  --targets-per-scene 0 --seed 2026 --top-k 3 \
  --min-frames 30 --fallback-min-frames 20 --min-run-ratio 0.2 \
  --min-visibility 0.05 --init-min-visibility 0.5 --min-box-side 4 \
  --materialize symlink
```

Use `--split val` and a separate output directory for validation. `--dry-run` checks selection and counts without writing derived images. Validate a materialized split with:

```bash
python validate_griffin_sot.py /path/to/derived/train
python validate_griffin_sot.py /path/to/derived/val
```

The expected output contains 303 training pairs from 73 scenes and 67 validation pairs from 16 disjoint scenes. Each view contains 22,706 training frames and 4,863 validation frames.

## Evaluation

Each tracker is initialized with the ground-truth box in frame zero. Later frames provide images only. Success rate is the mean of the IoU-threshold success curve over thresholds 0 to 1 in steps of 0.05. Precision rate uses a 20-pixel centre-error threshold. Normalized precision uses a 0.20 threshold after dividing horizontal and vertical centre errors by the ground-truth width and height. Scores average sequences equally within each physical view and then average the aerial and ground views equally.

`Difficult26` is fixed from annotations before tracker evaluation. A pair enters the subset if either view has visibility below 0.60 in at least 10% of frames, or either view has mean projected box area no greater than 400 pixels squared. Its scores are selected from the same full-set trajectories; no second inference run or score-based selection is used.

This package defines the data derivation and evaluation protocol; it does not distribute model predictions, score tables, or checkpoint provenance. The article describes the adaptation policy and interpretation of its Griffin results.
