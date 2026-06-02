"""
Relabel episode files in a LeRobot-format dataset directory.

Single-dir mode (in-place relabel):
    python relabel_episodes.py <dataset_dir>
    python relabel_episodes.py <dataset_dir> --task "pick up the tripod"

Merge mode (copy multiple dirs into one output):
    python relabel_episodes.py --output <out_dir> <dir1> <dir2> ...
    python relabel_episodes.py --output <out_dir> --task "pick up the tripod" <dir1> <dir2> ...

In merge mode the script copies all valid episodes from each input directory
into <out_dir>, renumbering them 0..N-1 in the order the inputs are given.
The original directories are not modified.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def get_valid_episodes(videos_dir: Path) -> list[int]:
    """Return sorted list of episode indices present in the first video subdir."""
    cam_dirs = sorted(d for d in videos_dir.iterdir() if d.is_dir())
    if not cam_dirs:
        raise RuntimeError(f"No camera subdirectories found in {videos_dir}")
    ref_dir = cam_dirs[0]
    pattern = re.compile(r"episode_(\d+)\.mp4$")
    indices = []
    for f in ref_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            indices.append(int(m.group(1)))
    return sorted(indices)


def rename_with_tmp(paths: list[tuple[Path, Path]]):
    """Two-phase rename: old→tmp then tmp→new, avoiding name collisions."""
    tmp_paths = [(old, old.with_name(old.name + ".relabel_tmp"), new)
                 for old, new in paths]
    for old, tmp, _ in tmp_paths:
        old.rename(tmp)
    for _, tmp, new in tmp_paths:
        tmp.rename(new)


def relabel_videos(videos_dir: Path, old_to_new: dict[int, int]):
    for cam_dir in sorted(videos_dir.iterdir()):
        if not cam_dir.is_dir():
            continue
        renames = []
        for old_idx, new_idx in old_to_new.items():
            old_file = cam_dir / f"episode_{old_idx:06d}.mp4"
            new_file = cam_dir / f"episode_{new_idx:06d}.mp4"
            if old_file.exists():
                renames.append((old_file, new_file))
        rename_with_tmp(renames)
        print(f"  [videos] {cam_dir.name}: renamed {len(renames)} files")


def relabel_data(data_dir: Path, old_to_new: dict[int, int], episode_lengths: dict[int, int]):
    """Rename parquets and update episode_index + global index columns."""
    sorted_new = sorted(old_to_new.items(), key=lambda x: x[1])
    offset = 0
    new_offsets: dict[int, int] = {}
    for old_idx, new_idx in sorted_new:
        new_offsets[new_idx] = offset
        offset += episode_lengths[old_idx]

    renames = []
    for old_idx, new_idx in old_to_new.items():
        old_file = data_dir / f"episode_{old_idx:06d}.parquet"
        new_file = data_dir / f"episode_{new_idx:06d}.parquet"
        if not old_file.exists():
            print(f"  [data] WARNING: {old_file.name} not found, skipping")
            continue

        table = pq.read_table(old_file)
        n_rows = len(table)

        new_ep_col = pa.array([new_idx] * n_rows, type=pa.int64())
        table = table.set_column(
            table.schema.get_field_index("episode_index"),
            "episode_index",
            new_ep_col,
        )

        start = new_offsets[new_idx]
        new_index_col = pa.array(range(start, start + n_rows), type=pa.int64())
        table = table.set_column(
            table.schema.get_field_index("index"),
            "index",
            new_index_col,
        )

        pq.write_table(table, old_file)
        renames.append((old_file, new_file))

    rename_with_tmp(renames)
    print(f"  [data] renamed and updated {len(renames)} parquet files")

    pattern = re.compile(r"episode_(\d+)\.parquet$")
    deleted = 0
    for f in data_dir.iterdir():
        m = pattern.match(f.name)
        if m and int(m.group(1)) not in {v for v in old_to_new.values()}:
            f.unlink()
            deleted += 1
    if deleted:
        print(f"  [data] deleted {deleted} orphaned parquet files")


def rewrite_tasks(meta_dir: Path, task: str):
    """Overwrite tasks.jsonl with a single entry using the given task string."""
    tasks_file = meta_dir / "tasks.jsonl"
    with open(tasks_file, "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task}) + "\n")
    print(f"  [meta] tasks.jsonl: set task to '{task}'")


def relabel_meta(meta_dir: Path, old_to_new: dict[int, int], total_frames: int, task: str | None = None):
    ep_file = meta_dir / "episodes.jsonl"
    valid_old = set(old_to_new.keys())
    new_episodes = []
    with open(ep_file) as f:
        for line in f:
            ep = json.loads(line)
            old_idx = ep["episode_index"]
            if old_idx in valid_old:
                ep["episode_index"] = old_to_new[old_idx]
                if task is not None:
                    ep["tasks"] = [task]
                new_episodes.append(ep)
    new_episodes.sort(key=lambda e: e["episode_index"])
    with open(ep_file, "w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")
    print(f"  [meta] episodes.jsonl: kept {len(new_episodes)} entries")

    stats_file = meta_dir / "episodes_stats.jsonl"
    new_stats = []
    with open(stats_file) as f:
        for line in f:
            s = json.loads(line)
            old_idx = s["episode_index"]
            if old_idx in valid_old:
                s["episode_index"] = old_to_new[old_idx]
                new_stats.append(s)
    new_stats.sort(key=lambda s: s["episode_index"])
    with open(stats_file, "w") as f:
        for s in new_stats:
            f.write(json.dumps(s) + "\n")
    print(f"  [meta] episodes_stats.jsonl: kept {len(new_stats)} entries")

    info_file = meta_dir / "info.json"
    with open(info_file) as f:
        info = json.load(f)
    n = len(old_to_new)
    info["total_episodes"] = n
    info["total_videos"] = n
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{n}"}
    info["discarded_episode_indices"] = []
    with open(info_file, "w") as f:
        json.dump(info, f, indent=4)
    print(f"  [meta] info.json: updated to {n} episodes, {total_frames} frames")


def relabel_dataset(dataset_dir: str, task: str | None = None):
    root = Path(dataset_dir)
    videos_dir = root / "videos" / "chunk-000"
    data_dir = root / "data" / "chunk-000"
    meta_dir = root / "meta"

    for d in [videos_dir, data_dir, meta_dir]:
        if not d.exists():
            raise RuntimeError(f"Expected directory not found: {d}")

    print(f"Dataset: {root}")

    valid_old = get_valid_episodes(videos_dir)
    print(f"Valid episodes in videos/: {len(valid_old)} ({valid_old[0]}..{valid_old[-1]})")

    old_to_new = {old: new for new, old in enumerate(valid_old)}

    ep_file = meta_dir / "episodes.jsonl"
    episode_lengths: dict[int, int] = {}
    with open(ep_file) as f:
        for line in f:
            ep = json.loads(line)
            episode_lengths[ep["episode_index"]] = ep["length"]

    total_frames = sum(episode_lengths[old] for old in valid_old)

    print(f"Relabeling {len(valid_old)} episodes → 0..{len(valid_old)-1}")
    print(f"Total frames after relabel: {total_frames}")

    relabel_videos(videos_dir, old_to_new)
    relabel_data(data_dir, old_to_new, episode_lengths)
    relabel_meta(meta_dir, old_to_new, total_frames, task=task)
    if task is not None:
        rewrite_tasks(meta_dir, task)

    print("Done.")


def _load_source(root: Path) -> dict:
    """Load all metadata from one input dataset directory."""
    videos_dir = root / "videos" / "chunk-000"
    data_dir = root / "data" / "chunk-000"
    meta_dir = root / "meta"

    for d in [videos_dir, data_dir, meta_dir]:
        if not d.exists():
            raise RuntimeError(f"Expected directory not found: {d}")

    valid_episodes = get_valid_episodes(videos_dir)

    episode_lengths: dict[int, int] = {}
    episodes_meta: dict[int, dict] = {}
    with open(meta_dir / "episodes.jsonl") as f:
        for line in f:
            ep = json.loads(line)
            episode_lengths[ep["episode_index"]] = ep["length"]
            episodes_meta[ep["episode_index"]] = ep

    episodes_stats: dict[int, dict] = {}
    with open(meta_dir / "episodes_stats.jsonl") as f:
        for line in f:
            s = json.loads(line)
            episodes_stats[s["episode_index"]] = s

    with open(meta_dir / "info.json") as f:
        info = json.load(f)

    cam_dirs = sorted(d.name for d in videos_dir.iterdir() if d.is_dir())

    return {
        "root": root,
        "valid_episodes": valid_episodes,
        "episode_lengths": episode_lengths,
        "episodes_meta": episodes_meta,
        "episodes_stats": episodes_stats,
        "info": info,
        "cam_dirs": cam_dirs,
    }


def merge_datasets(input_dirs: list[str], output_dir: str, task: str | None = None):
    """Copy multiple input datasets into output_dir, renumbering episodes 0..N-1."""
    out_root = Path(output_dir)
    if out_root.exists():
        raise RuntimeError(f"Output directory already exists: {out_root}. Remove it first or choose a different name.")

    print(f"Loading metadata from {len(input_dirs)} input dataset(s)...")
    sources = []
    for d in input_dirs:
        src = _load_source(Path(d))
        print(f"  {d}: {len(src['valid_episodes'])} valid episodes")
        sources.append(src)

    # Use camera layout from first source; warn if others differ
    cam_dirs = sources[0]["cam_dirs"]
    for src in sources[1:]:
        if src["cam_dirs"] != cam_dirs:
            print(f"  WARNING: camera dirs differ in {src['root']} — using first source's layout")

    # Build flat assignment list: (source, old_idx, new_idx)
    assignments: list[tuple[dict, int, int]] = []
    new_idx = 0
    for src in sources:
        for old_idx in src["valid_episodes"]:
            assignments.append((src, old_idx, new_idx))
            new_idx += 1

    total_episodes = new_idx
    total_frames = sum(src["episode_lengths"].get(old, 0) for src, old, _ in assignments)
    print(f"Total: {total_episodes} episodes, {total_frames} frames → {out_root}")

    # Create output directory structure
    out_videos = out_root / "videos" / "chunk-000"
    out_data = out_root / "data" / "chunk-000"
    out_meta = out_root / "meta"
    for cam in cam_dirs:
        (out_videos / cam).mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    # Copy videos
    print("Copying videos...")
    for src, old_idx, new_idx in assignments:
        src_videos = src["root"] / "videos" / "chunk-000"
        for cam in cam_dirs:
            src_file = src_videos / cam / f"episode_{old_idx:06d}.mp4"
            dst_file = out_videos / cam / f"episode_{new_idx:06d}.mp4"
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
            else:
                print(f"  WARNING: {src_file} not found, skipping")
    print(f"  Copied {total_episodes} episodes × {len(cam_dirs)} camera(s)")

    # Copy and update parquet files
    print("Processing parquet files...")
    frame_offset = 0
    for src, old_idx, new_idx in assignments:
        src_file = src["root"] / "data" / "chunk-000" / f"episode_{old_idx:06d}.parquet"
        dst_file = out_data / f"episode_{new_idx:06d}.parquet"
        ep_len = src["episode_lengths"].get(old_idx, 0)

        if not src_file.exists():
            print(f"  WARNING: {src_file} not found, skipping")
            frame_offset += ep_len
            continue

        table = pq.read_table(src_file)
        n_rows = len(table)

        table = table.set_column(
            table.schema.get_field_index("episode_index"),
            "episode_index",
            pa.array([new_idx] * n_rows, type=pa.int64()),
        )
        table = table.set_column(
            table.schema.get_field_index("index"),
            "index",
            pa.array(range(frame_offset, frame_offset + n_rows), type=pa.int64()),
        )

        pq.write_table(table, dst_file)
        frame_offset += n_rows
    print(f"  Processed {total_episodes} parquet files")

    # Write merged meta files
    print("Writing meta files...")

    with open(out_meta / "episodes.jsonl", "w") as f:
        for src, old_idx, new_idx in assignments:
            ep = dict(src["episodes_meta"][old_idx])
            ep["episode_index"] = new_idx
            if task is not None:
                ep["tasks"] = [task]
            f.write(json.dumps(ep) + "\n")
    print(f"  [meta] episodes.jsonl: {total_episodes} entries")

    with open(out_meta / "episodes_stats.jsonl", "w") as f:
        for src, old_idx, new_idx in assignments:
            if old_idx in src["episodes_stats"]:
                s = dict(src["episodes_stats"][old_idx])
                s["episode_index"] = new_idx
                f.write(json.dumps(s) + "\n")
    print(f"  [meta] episodes_stats.jsonl: {total_episodes} entries")

    info = dict(sources[0]["info"])
    info["total_episodes"] = total_episodes
    info["total_videos"] = total_episodes
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{total_episodes}"}
    info["discarded_episode_indices"] = []
    with open(out_meta / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    print(f"  [meta] info.json: {total_episodes} episodes, {total_frames} frames")

    if task is not None:
        rewrite_tasks(out_meta, task)
    else:
        src_tasks = sources[0]["root"] / "meta" / "tasks.jsonl"
        if src_tasks.exists():
            shutil.copy2(src_tasks, out_meta / "tasks.jsonl")

    # Copy static meta files from first source (same across all sessions)
    for fname in ("modality.json",):
        src_file = sources[0]["root"] / "meta" / fname
        if src_file.exists():
            shutil.copy2(src_file, out_meta / fname)
            print(f"  [meta] {fname}: copied from first source")

    print(f"Done. Merged dataset at: {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Relabel or merge LeRobot-format dataset directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # In-place relabel a single directory
  python relabel_episodes.py outputs/2026-05-09-19-17-36

  # Merge three directories into one output
  python relabel_episodes.py --output outputs/merged \\
      outputs/2026-05-09-19-17-36 \\
      outputs/2026-05-09-19-46-35 \\
      outputs/2026-05-09-20-24-20
""",
    )
    parser.add_argument(
        "input_dirs",
        nargs="+",
        help="One or more input dataset directories",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="OUTPUT_DIR",
        help="Output directory for merged dataset (required when giving multiple inputs)",
    )
    parser.add_argument(
        "--task", "-t",
        metavar="TASK",
        help="Override the task instruction string in tasks.jsonl (e.g. 'pick up the tripod')",
    )
    args = parser.parse_args()

    if len(args.input_dirs) == 1 and args.output is None:
        relabel_dataset(args.input_dirs[0], task=args.task)
    else:
        if args.output is None:
            parser.error("--output / -o is required when merging multiple input directories")
        merge_datasets(args.input_dirs, args.output, task=args.task)
