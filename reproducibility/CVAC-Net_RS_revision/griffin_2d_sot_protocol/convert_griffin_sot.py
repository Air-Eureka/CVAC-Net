#!/usr/bin/env python3
"""Convert a Griffin front-camera pair into aligned single-object benchmarks.

The Griffin release stores 3D boxes in each agent's ego frame.  This script
projects each cuboid into the corresponding front camera, finds track IDs that
are continuously valid in *both* views, and selects one reproducible difficult
track per original scene.  The two output manifests have identical sequence
names and frame counts, which is required by CVAF-Net's two-stream evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
SIDES = ("vehicle-side", "drone-side")
OUTPUT_NAMES = {"vehicle-side": "GRIFFIN_V", "drone-side": "GRIFFIN_D"}
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


@dataclass(frozen=True)
class ObjectLabel:
    category: str
    center: np.ndarray
    dimensions: np.ndarray
    rotation_deg: np.ndarray
    track_id: int
    visibility: float


@dataclass(frozen=True)
class Observation:
    frame: str
    category: str
    bbox: Tuple[float, float, float, float]
    visibility: float


@dataclass
class Candidate:
    track_id: int
    category: str
    frame_indices: List[int]
    observations: Dict[str, List[Observation]]
    score: float = 0.0
    difficulty: Optional[dict] = None


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_source = here / "data/griffin_100scenes_random/griffin-release"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=default_source)
    parser.add_argument("--output-root", type=Path, default=here / "sot_test")
    parser.add_argument("--split-file", type=Path, default=None,
                        help="Official Griffin split JSON with batch_split train/val lists.")
    parser.add_argument("--split", choices=("train", "val"), default=None,
                        help="Restrict conversion to one official scene split.")
    parser.add_argument("--targets-per-scene", type=int, default=1,
                        help="Number of distinct tracks per scene; 0 keeps every eligible track.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--top-k", type=int, default=3,
                        help="Seeded choice among the top-k difficult candidates.")
    parser.add_argument("--min-frames", type=int, default=60)
    parser.add_argument("--fallback-min-frames", type=int, default=30,
                        help="Never emit a sequence shorter than this fallback limit.")
    parser.add_argument("--min-run-ratio", type=float, default=0.60)
    parser.add_argument("--min-visibility", type=float, default=0.05)
    parser.add_argument("--init-min-visibility", type=float, default=0.50)
    parser.add_argument("--min-box-side", type=float, default=4.0)
    parser.add_argument("--materialize", choices=("hardlink", "symlink", "copy", "none"),
                        default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Select tracks and print statistics without writing output.")
    return parser.parse_args()


def euler_xyz_matrix(angles_deg: Sequence[float]) -> np.ndarray:
    """Match scipy Rotation.from_euler('xyz', angles).as_matrix()."""
    roll, pitch, yaw = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float64)
    ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ ry @ rx


def load_calibration(side_root: Path, camera: str = "front") -> Tuple[np.ndarray, np.ndarray]:
    with (side_root / "calib" / f"{camera}.json").open() as handle:
        calib = json.load(handle)
    intrinsic = np.asarray(calib["intrinsic"], dtype=np.float64)
    sensor_to_ego = np.asarray(calib["extrinsic"], dtype=np.float64)
    return intrinsic, np.linalg.inv(sensor_to_ego)


def load_labels(path: Path) -> List[ObjectLabel]:
    labels: List[ObjectLabel] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) not in (9, 11, 12):
                raise ValueError(f"{path}:{line_number}: expected 9, 11, or 12 fields, got {len(fields)}")
            if len(fields) == 12:
                category = fields[0]
                center = fields[1:4]
                dimensions = fields[4:7]
                rotation = fields[7:10]
                track_id = fields[10]
                visibility = fields[11]
            elif len(fields) == 11:
                category = fields[0]
                center = fields[1:4]
                dimensions = fields[4:7]
                rotation = fields[7:10]
                track_id = fields[10]
                visibility = 1.0
            else:
                category = fields[0]
                center = fields[1:4]
                dimensions = fields[4:7]
                rotation = (0.0, 0.0, fields[7])
                track_id = fields[8]
                visibility = 1.0
            labels.append(ObjectLabel(
                category=category.lower(),
                center=np.asarray(center, dtype=np.float64),
                dimensions=np.asarray(dimensions, dtype=np.float64),
                rotation_deg=np.asarray(rotation, dtype=np.float64),
                track_id=int(track_id),
                visibility=float(visibility),
            ))
    return labels


def cuboid_corners(label: ObjectLabel) -> np.ndarray:
    half_l, half_w, half_h = label.dimensions / 2.0
    local = np.array((
        (half_l, half_w, half_h), (half_l, -half_w, half_h),
        (-half_l, -half_w, half_h), (-half_l, half_w, half_h),
        (half_l, half_w, -half_h), (half_l, -half_w, -half_h),
        (-half_l, -half_w, -half_h), (-half_l, half_w, -half_h),
    ), dtype=np.float64)
    return local @ euler_xyz_matrix(label.rotation_deg).T + label.center


def clip_cuboid_to_near_plane(corners_camera: np.ndarray, near: float = 1e-3) -> np.ndarray:
    points: List[np.ndarray] = []
    for first, second in BOX_EDGES:
        p0, p1 = corners_camera[first], corners_camera[second]
        in0, in1 = p0[2] >= near, p1[2] >= near
        if in0:
            points.append(p0)
        if in1:
            points.append(p1)
        if in0 != in1:
            fraction = (near - p0[2]) / (p1[2] - p0[2])
            points.append(p0 + fraction * (p1 - p0))
    return np.asarray(points, dtype=np.float64)


def project_bbox(label: ObjectLabel, intrinsic: np.ndarray, ego_to_camera: np.ndarray,
                 min_box_side: float) -> Optional[Tuple[float, float, float, float]]:
    corners_ego = cuboid_corners(label)
    homogeneous = np.column_stack((corners_ego, np.ones(len(corners_ego))))
    corners_camera = (ego_to_camera @ homogeneous.T).T[:, :3]
    visible = clip_cuboid_to_near_plane(corners_camera)
    if len(visible) == 0:
        return None
    pixels_h = (intrinsic @ visible.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    x0 = float(np.clip(pixels[:, 0].min(), 0.0, IMAGE_WIDTH - 1.0))
    y0 = float(np.clip(pixels[:, 1].min(), 0.0, IMAGE_HEIGHT - 1.0))
    x1 = float(np.clip(pixels[:, 0].max(), 0.0, IMAGE_WIDTH - 1.0))
    y1 = float(np.clip(pixels[:, 1].max(), 0.0, IMAGE_HEIGHT - 1.0))
    width, height = x1 - x0, y1 - y0
    if width < min_box_side or height < min_box_side:
        return None
    return x0, y0, width, height


def projected_observations(side_root: Path, frames: Sequence[str], min_visibility: float,
                           min_box_side: float, camera: str = "front") -> Dict[str, Dict[int, Observation]]:
    intrinsic, ego_to_camera = load_calibration(side_root, camera)
    result: Dict[str, Dict[int, Observation]] = {}
    for frame in frames:
        frame_result: Dict[int, Observation] = {}
        label_path = side_root / "label" / f"{frame}.txt"
        if label_path.is_file():
            for label in load_labels(label_path):
                if label.visibility < min_visibility:
                    continue
                bbox = project_bbox(label, intrinsic, ego_to_camera, min_box_side)
                if bbox is not None:
                    frame_result[label.track_id] = Observation(
                        frame=frame, category=label.category, bbox=bbox,
                        visibility=label.visibility,
                    )
        result[frame] = frame_result
    return result


def contiguous_runs(indices: Iterable[int]) -> List[List[int]]:
    runs: List[List[int]] = []
    for index in sorted(indices):
        if not runs or index != runs[-1][-1] + 1:
            runs.append([index])
        else:
            runs[-1].append(index)
    return runs


def difficulty(candidate: Candidate) -> Tuple[float, dict]:
    view_stats = {}
    for side, observations in candidate.observations.items():
        boxes = np.asarray([obs.bbox for obs in observations], dtype=np.float64)
        visibility = np.asarray([obs.visibility for obs in observations], dtype=np.float64)
        areas = boxes[:, 2] * boxes[:, 3]
        centers = boxes[:, :2] + boxes[:, 2:] / 2.0
        relative_area = areas / float(IMAGE_WIDTH * IMAGE_HEIGHT)
        scale_variation = min(float(np.std(np.log(np.maximum(areas, 1.0)))) / 1.5, 1.0)
        if len(centers) > 1:
            motion = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            motion = min(float(np.mean(motion)) / 25.0, 1.0)
        else:
            motion = 0.0
        border_dist = np.minimum.reduce((
            boxes[:, 0], boxes[:, 1],
            IMAGE_WIDTH - boxes[:, 0] - boxes[:, 2],
            IMAGE_HEIGHT - boxes[:, 1] - boxes[:, 3],
        ))
        border = float(np.mean(border_dist < 0.02 * min(IMAGE_WIDTH, IMAGE_HEIGHT)))
        small = 1.0 - min(float(np.sqrt(np.mean(relative_area))) / 0.10, 1.0)
        view_stats[side] = {
            "mean_visibility": float(np.mean(visibility)),
            "min_visibility": float(np.min(visibility)),
            "scale_variation": scale_variation,
            "motion": motion,
            "border_fraction": border,
            "small_target": small,
            "mean_area_ratio": float(np.mean(relative_area)),
        }
    aggregate = {
        key: float(np.mean([stats[key] for stats in view_stats.values()]))
        for key in ("mean_visibility", "scale_variation", "motion", "border_fraction", "small_target")
    }
    score = (
        0.25 * (1.0 - aggregate["mean_visibility"])
        + 0.25 * aggregate["scale_variation"]
        + 0.20 * aggregate["motion"]
        + 0.15 * aggregate["border_fraction"]
        + 0.15 * aggregate["small_target"]
    )
    return score, {"score": score, "aggregate": aggregate, "views": view_stats}


def make_candidates(frames: Sequence[str], per_side: Dict[str, Dict[str, Dict[int, Observation]]],
                    excluded_ids: set, min_frames: int, fallback_min_frames: int,
                    min_run_ratio: float,
                    init_min_visibility: float) -> Tuple[List[Candidate], str]:
    common_indices: Dict[int, List[int]] = defaultdict(list)
    for index, frame in enumerate(frames):
        common = set(per_side[SIDES[0]][frame]) & set(per_side[SIDES[1]][frame])
        for track_id in common - excluded_ids:
            common_indices[track_id].append(index)

    all_candidates: List[Candidate] = []
    for track_id, indices in common_indices.items():
        for run in contiguous_runs(indices):
            # A clearly visible first frame is important for one-pass SOT initialization.
            while run:
                first = frames[run[0]]
                initial_visibility = min(per_side[side][first][track_id].visibility for side in SIDES)
                if initial_visibility >= init_min_visibility:
                    break
                run = run[1:]
            if not run:
                continue
            observations = {
                side: [per_side[side][frames[index]][track_id] for index in run]
                for side in SIDES
            }
            categories = [obs.category for obs in observations[SIDES[0]] + observations[SIDES[1]]]
            category = max(set(categories), key=categories.count)
            candidate = Candidate(track_id, category, run, observations)
            candidate.score, candidate.difficulty = difficulty(candidate)
            all_candidates.append(candidate)

    preferred_length = max(min_frames, int(math.ceil(len(frames) * min_run_ratio)))
    preferred = [candidate for candidate in all_candidates if len(candidate.frame_indices) >= preferred_length]
    if preferred:
        return preferred, "preferred"
    fallback = [candidate for candidate in all_candidates if len(candidate.frame_indices) >= min_frames]
    if fallback:
        return fallback, "min_frames_fallback"
    if all_candidates:
        longest = max(len(candidate.frame_indices) for candidate in all_candidates)
        if longest >= fallback_min_frames:
            return [candidate for candidate in all_candidates if len(candidate.frame_indices) == longest], "longest_fallback"
        return [], "track_too_short"
    return [], "no_common_target"


def stable_rng(seed: int, scene_name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{scene_name}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def select_candidate(candidates: List[Candidate], top_k: int, seed: int,
                     scene_name: str) -> Candidate:
    ranked = sorted(candidates, key=lambda item: (-item.score, -len(item.frame_indices), item.track_id))
    return ranked[stable_rng(seed, scene_name).randrange(min(top_k, len(ranked)))]


def select_candidates(candidates: List[Candidate], targets_per_scene: int, top_k: int,
                      seed: int, scene_name: str) -> List[Candidate]:
    if targets_per_scene == 1:
        return [select_candidate(candidates, top_k, seed, scene_name)]
    ranked = sorted(candidates, key=lambda item: (-item.score, -len(item.frame_indices), item.track_id))
    selected = []
    used_track_ids = set()
    for candidate in ranked:
        if candidate.track_id in used_track_ids:
            continue
        selected.append(candidate)
        used_track_ids.add(candidate.track_id)
        if targets_per_scene > 0 and len(selected) >= targets_per_scene:
            break
    return selected


def split_scene_names(split_file: Path, split: str) -> set:
    with split_file.open() as handle:
        raw_names = json.load(handle)["batch_split"][split]
    # Official entries are e.g. scene-0000-Town03-000, while scene_infos uses Town03-000.
    return {"-".join(name.split("-")[2:]) for name in raw_names}


def aligned_scenes(source_root: Path) -> List[dict]:
    by_side = {}
    for side in SIDES:
        with (source_root / side / "scene_infos.json").open() as handle:
            by_side[side] = {scene["name"]: scene for scene in json.load(handle)}
    names = list(by_side[SIDES[0]])
    if set(names) != set(by_side[SIDES[1]]):
        raise ValueError("Vehicle and drone scene names do not match")
    scenes = []
    for name in names:
        first = by_side[SIDES[0]][name]
        second = by_side[SIDES[1]][name]
        second_frames = set(second["info"]["frames"])
        frames = [frame for frame in first["info"]["frames"] if frame in second_frames]
        scenes.append({"name": name, "frames": frames, "by_side": {SIDES[0]: first, SIDES[1]: second}})
    return scenes


def materialize_image(source: Path, destination: Path, mode: str) -> None:
    if mode == "none":
        return
    if not source.is_file():
        raise FileNotFoundError(f"Missing source image: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, destination)


def write_sequence(output_root: Path, source_root: Path, dataset_name: str, side: str,
                   sequence_name: str, candidate: Candidate, scene: dict, mode: str,
                   camera: str = "front") -> dict:
    sequence_dir = output_root / dataset_name / sequence_name
    sequence_dir.mkdir(parents=True)
    observations = candidate.observations[side]
    image_names = []
    for output_index, observation in enumerate(observations, 1):
        relative = f"{sequence_name}/img/{output_index:08d}.png"
        source = source_root / side / "camera" / camera / f"{observation.frame}.png"
        materialize_image(source, output_root / dataset_name / relative, mode)
        image_names.append(relative)
    boxes = [list(observation.bbox) for observation in observations]
    with (sequence_dir / "groundtruth.txt").open("w") as handle:
        for box in boxes:
            handle.write(",".join(f"{value:.6f}" for value in box) + "\n")
    with (sequence_dir / "full_occlusion.txt").open("w") as handle:
        handle.write("\n".join("0" for _ in boxes) + "\n")
    sequence_metadata = {
        "source_scene": scene["name"],
        "source_side": side,
        "source_camera": camera,
        "source_frames": [observation.frame for observation in observations],
        "target_id": candidate.track_id,
        "category": candidate.category,
        "visibility": [observation.visibility for observation in observations],
        "difficulty": candidate.difficulty,
    }
    with (sequence_dir / "metadata.json").open("w") as handle:
        json.dump(sequence_metadata, handle, indent=2)
    return {
        "video_dir": sequence_name,
        "init_rect": boxes[0],
        "img_names": image_names,
        "gt_rect": boxes,
        "attr": [candidate.category],
    }


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    scenes = aligned_scenes(source_root)
    if (args.split_file is None) != (args.split is None):
        raise ValueError("--split-file and --split must be provided together")
    if args.split_file is not None:
        allowed_scenes = split_scene_names(args.split_file.resolve(), args.split)
        scenes = [scene for scene in scenes if scene["name"] in allowed_scenes]
    selected = []
    skipped = []

    for scene_index, scene in enumerate(scenes, 1):
        frames = scene["frames"]
        per_side = {
            side: projected_observations(
                source_root / side, frames, args.min_visibility, args.min_box_side
            ) for side in SIDES
        }
        excluded = set()
        for side in SIDES:
            info = scene["by_side"][side]["info"]
            excluded.update(int(info[key]) for key in ("ego_vehicle_id", "ego_drone_id") if key in info)
        candidates, tier = make_candidates(
            frames, per_side, excluded, args.min_frames, args.fallback_min_frames,
            args.min_run_ratio,
            args.init_min_visibility,
        )
        if not candidates:
            skipped.append({"scene": scene["name"], "reason": tier})
            print(f"[{scene_index:03d}/{len(scenes)}] skip {scene['name']}: {tier}")
            continue
        scene_candidates = select_candidates(
            candidates, args.targets_per_scene, args.top_k, args.seed, scene["name"]
        )
        selected.extend((scene, candidate, tier) for candidate in scene_candidates)
        description = ", ".join(
            f"id={candidate.track_id}/{candidate.category}/{len(candidate.frame_indices)}f"
            for candidate in scene_candidates
        )
        print(f"[{scene_index:03d}/{len(scenes)}] {scene['name']}: {description} tier={tier}")

    total_frames = sum(len(candidate.frame_indices) for _, candidate, _ in selected)
    summary = {
        "source_root": str(source_root),
        "seed": args.seed,
        "split_file": str(args.split_file.resolve()) if args.split_file else None,
        "split": args.split,
        "scene_count": len(scenes),
        "selected_sequences": len(selected),
        "frames_per_view": total_frames,
        "combined_frames": 2 * total_frames,
        "skipped": skipped,
        "selection": {
            "top_k": args.top_k,
            "min_frames": args.min_frames,
            "fallback_min_frames": args.fallback_min_frames,
            "min_run_ratio": args.min_run_ratio,
            "min_visibility": args.min_visibility,
            "init_min_visibility": args.init_min_visibility,
            "min_box_side": args.min_box_side,
            "targets_per_scene": args.targets_per_scene,
        },
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_root}; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    manifests = {dataset_name: {} for dataset_name in OUTPUT_NAMES.values()}
    selections = []
    for scene, candidate, tier in selected:
        sequence_name = f"{scene['name']}-id{candidate.track_id}"
        selections.append({
            "sequence": sequence_name, "scene": scene["name"],
            "target_id": candidate.track_id, "category": candidate.category,
            "frames": len(candidate.frame_indices), "tier": tier,
            "difficulty": candidate.difficulty,
        })
        for side in SIDES:
            dataset_name = OUTPUT_NAMES[side]
            manifests[dataset_name][sequence_name] = write_sequence(
                output_root, source_root, dataset_name, side, sequence_name,
                candidate, scene, args.materialize,
            )
    for dataset_name, manifest in manifests.items():
        dataset_dir = output_root / dataset_name
        with (dataset_dir / f"{dataset_name}.json").open("w") as handle:
            json.dump(manifest, handle, indent=2)
        with (dataset_dir / "list.txt").open("w") as handle:
            handle.write("\n".join(manifest) + "\n")
    summary["materialize"] = args.materialize
    summary["sequences"] = selections
    with (output_root / "conversion_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
