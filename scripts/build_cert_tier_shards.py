#!/usr/bin/env python3
"""Build weighted key shards from the MAP1 certifier rank array."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, TextIO

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - this host normally has numpy.
    raise SystemExit("build_cert_tier_shards.py requires numpy for the 4 GiB rank scan") from exc


KEYSPACE = 1 << 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan rank_array.bin and emit weighted key shards for keys whose "
            "certified shed-bit count is at least --min-bits, optionally bounded "
            "above by --max-bits-exclusive."
        )
    )
    parser.add_argument(
        "--rank-array",
        default="common/results/certify_sweep/rank_array.bin",
        help="Flat uint8 rank array indexed by key id.",
    )
    parser.add_argument(
        "--out-dir",
        default="common/results/tier_sweep",
        help="Output directory for shards and manifest.",
    )
    parser.add_argument("--min-bits", type=int, default=12, help="Minimum certified shed bits to include.")
    parser.add_argument(
        "--max-bits-exclusive",
        type=int,
        default=0,
        help="Optional exclusive upper bound for certified shed bits; 0 means no upper bound.",
    )
    parser.add_argument("--shards", type=int, default=2, help="Number of balanced output shards.")
    parser.add_argument(
        "--shard-weights",
        default="",
        help=(
            "Optional comma-separated positive weights, one per shard. "
            "Example: --shard-weights 55,45 gives shard00 55%% of keys."
        ),
    )
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=256 * 1024 * 1024,
        help="Rank-array scan chunk size.",
    )
    parser.add_argument(
        "--max-keys",
        type=int,
        default=0,
        help="Optional cap for smoke builds; 0 means include all matching keys.",
    )
    parser.add_argument(
        "--format",
        choices=("hex", "dec", "bin32", "bin32u8"),
        default="hex",
        help="Shard key encoding. Text and bin32u8 formats include cert_bits; bin32 stores keys only.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing shard files.")
    return parser.parse_args()


def parse_weights(value: str, shards: int) -> list[float]:
    if not value:
        return [1.0 for _ in range(shards)]
    try:
        weights = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit("--shard-weights must be comma-separated numbers") from exc
    if len(weights) != shards:
        raise SystemExit("--shard-weights must have exactly one value per shard")
    if any(weight <= 0.0 for weight in weights):
        raise SystemExit("--shard-weights must be positive")
    return weights


def choose_shard(shard_counts: list[int], weights: list[float]) -> int:
    # Keep each shard close to its target ratio without needing a pre-count pass.
    return min(range(len(weights)), key=lambda i: ((shard_counts[i] + 1.0) / weights[i], i))


def integer_weights(weights: list[float]) -> list[int] | None:
    out = []
    for weight in weights:
        rounded = round(weight)
        if abs(weight - rounded) > 1e-9 or rounded <= 0:
            return None
        out.append(int(rounded))
    return out


def open_shard(path: Path, fmt: str) -> TextIO | BinaryIO:
    if fmt in {"bin32", "bin32u8"}:
        return path.open("wb")
    fh = path.open("w", encoding="ascii")
    fh.write("# key\tcert_bits\n")
    return fh


def write_key(fh: TextIO | BinaryIO, key: int, bits: int, fmt: str) -> None:
    if fmt == "hex":
        fh.write(f"0x{key:08x}\t{bits}\n")
    elif fmt == "dec":
        fh.write(f"{key}\t{bits}\n")
    elif fmt == "bin32u8":
        fh.write(key.to_bytes(4, "little"))
        fh.write(bytes([bits & 0xFF]))
    else:
        fh.write(key.to_bytes(4, "little"))


def main() -> int:
    args = parse_args()
    if args.min_bits < 0 or args.min_bits > 32:
        raise SystemExit("--min-bits must be in [0,32]")
    if args.max_bits_exclusive and (args.max_bits_exclusive <= args.min_bits or args.max_bits_exclusive > 33):
        raise SystemExit("--max-bits-exclusive must be in (min_bits,33]")
    if args.shards <= 0:
        raise SystemExit("--shards must be positive")
    if args.chunk_bytes <= 0:
        raise SystemExit("--chunk-bytes must be positive")
    shard_weights = parse_weights(args.shard_weights, args.shards)

    rank_path = Path(args.rank_array)
    out_dir = Path(args.out_dir)
    if not rank_path.exists():
        raise SystemExit(f"rank array not found: {rank_path}")
    if rank_path.stat().st_size != KEYSPACE:
        raise SystemExit(f"rank array size is {rank_path.stat().st_size}, expected {KEYSPACE}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.format == "bin32":
        suffix = "bin"
    elif args.format == "bin32u8":
        suffix = "bin5"
    else:
        suffix = "tsv"
    shard_paths = [
        out_dir / f"keys_cert_ge{args.min_bits}_shard{i:02d}.{suffix}" for i in range(args.shards)
    ]
    existing = [str(p) for p in shard_paths if p.exists()]
    if existing and not args.force:
        raise SystemExit("refusing to overwrite existing shard files; pass --force")

    handles = [open_shard(path, args.format) for path in shard_paths]
    shard_counts = [0 for _ in shard_paths]
    tier_counts: dict[int, int] = {}
    emitted = 0
    started = datetime.now(timezone.utc)
    fast_bin32u8_weights = integer_weights(shard_weights) if args.format == "bin32u8" else None
    fast_bin32u8_period = sum(fast_bin32u8_weights) if fast_bin32u8_weights else 0
    fast_bin32u8_edges: list[tuple[int, int]] = []
    if fast_bin32u8_weights:
        edge = 0
        for weight in fast_bin32u8_weights:
            fast_bin32u8_edges.append((edge, edge + weight))
            edge += weight

    try:
        ranks = np.memmap(rank_path, dtype=np.uint8, mode="r", shape=(KEYSPACE,))
        chunk = max(1, int(args.chunk_bytes))
        for base in range(0, KEYSPACE, chunk):
            stop = min(KEYSPACE, base + chunk)
            view = ranks[base:stop]
            if args.max_bits_exclusive:
                rel = np.flatnonzero((view >= args.min_bits) & (view < args.max_bits_exclusive))
            else:
                rel = np.flatnonzero(view >= args.min_bits)
            if rel.size == 0:
                continue
            bits_view = view[rel]
            if args.max_keys and emitted + int(rel.size) > args.max_keys:
                keep = args.max_keys - emitted
                rel = rel[:keep]
                bits_view = bits_view[:keep]
            if fast_bin32u8_weights:
                counts = np.bincount(bits_view, minlength=33)
                for bits, count in enumerate(counts):
                    if count:
                        tier_counts[bits] = tier_counts.get(bits, 0) + int(count)
                keys_view = (base + rel).astype("<u4", copy=False)
                positions = np.arange(emitted, emitted + int(rel.size), dtype=np.uint64) % fast_bin32u8_period
                bits_u8 = bits_view.astype(np.uint8, copy=False)
                for shard, (lo, hi) in enumerate(fast_bin32u8_edges):
                    mask = (positions >= lo) & (positions < hi)
                    selected = int(np.count_nonzero(mask))
                    if selected == 0:
                        continue
                    records = np.empty((selected, 5), dtype=np.uint8)
                    records[:, :4] = keys_view[mask].view(np.uint8).reshape(-1, 4)
                    records[:, 4] = bits_u8[mask]
                    handles[shard].write(records.tobytes())
                    shard_counts[shard] += selected
                emitted += int(rel.size)
                if args.max_keys and emitted >= args.max_keys:
                    raise StopIteration
                continue
            for rel_key, bits_value in zip(rel, bits_view, strict=False):
                key = base + int(rel_key)
                bits = int(bits_value)
                tier_counts[bits] = tier_counts.get(bits, 0) + 1
                shard = choose_shard(shard_counts, shard_weights)
                write_key(handles[shard], key, bits, args.format)
                shard_counts[shard] += 1
                emitted += 1
                if args.max_keys and emitted >= args.max_keys:
                    raise StopIteration
    except StopIteration:
        pass
    finally:
        for fh in handles:
            fh.close()

    manifest = {
        "created_utc": started.isoformat(),
        "rank_array": str(rank_path),
        "rank_array_size": rank_path.stat().st_size,
        "min_bits": args.min_bits,
        "max_bits_exclusive": args.max_bits_exclusive,
        "format": args.format,
        "max_keys": args.max_keys,
        "shard_weights": shard_weights,
        "total_keys": emitted,
        "tier_counts": {str(k): tier_counts[k] for k in sorted(tier_counts)},
        "shards": [
            {
                "index": i,
                "path": str(path),
                "count": shard_counts[i],
                "bytes": path.stat().st_size,
            }
            for i, path in enumerate(shard_paths)
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")

    with (out_dir / "tier_counts.tsv").open("w", encoding="ascii") as fh:
        fh.write("cert_bits\tkeys\n")
        for bits in sorted(tier_counts):
            fh.write(f"{bits}\t{tier_counts[bits]}\n")

    if args.max_bits_exclusive:
        print(
            f"wrote {emitted:,} keys with {args.min_bits} <= cert bits < "
            f"{args.max_bits_exclusive} into {args.shards} shard(s)"
        )
    else:
        print(f"wrote {emitted:,} keys >= {args.min_bits} cert bits into {args.shards} shard(s)")
    for i, path in enumerate(shard_paths):
        print(f"  shard {i:02d}: {shard_counts[i]:,} keys  {path}")
    print(f"  manifest: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
