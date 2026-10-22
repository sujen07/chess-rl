#!/usr/bin/env python3
"""Convert Lichess open-database PGNs into packed supervised warm-start shards.

Downloads (optional) monthly dumps from https://database.lichess.org/, streams
games, filters by rating / time control / result on the raw header text (so
rejected games are never parsed), encodes accepted games across all CPU cores,
and writes sl_dataset.py shards for train_supervised.py.

Value targets: about 6-10% of Lichess games carry Stockfish "[%eval ...]"
annotations. With --require-eval only those games are kept, and each
position's value target blends the engine eval with the game result
(--eval-weight), which is a far less noisy signal than the result alone.

Example:
  python prepare_lichess_dataset.py \\
    --month 2024-01 \\
    --min-elo 2000 \\
    --require-eval \\
    --max-positions 20000000 \\
    --out-dir data/lichess_sl
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import chess.pgn

from sl_dataset import concat_packed, encode_game, save_packed_shard, slice_packed

LICHESS_DB_URL = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
)

# Approximate base seconds for Lichess TimeControl tags like "180+2".
TIME_CONTROL_BUCKETS = {
    "bullet": (0, 179),       # < 3 min
    "blitz": (180, 479),      # 3–8 min
    "rapid": (480, 1499),     # 8–25 min
    "classical": (1500, 10**9),
}

VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}
# Keep games that finished on the board or on the clock; drop abandons / rules issues.
ALLOWED_TERMINATIONS = {
    "Normal",
    "Time forfeit",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pgn",
        type=Path,
        help="Local .pgn or .pgn.zst file (or plain .zst of a PGN dump)",
    )
    source.add_argument(
        "--month",
        help="YYYY-MM dump to download from database.lichess.org into --download-dir",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/lichess_raw"),
        help="Where to store downloaded monthly dumps (default: data/lichess_raw)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/lichess_sl"),
        help="Directory for packed shard_*.npz files (default: data/lichess_sl)",
    )
    parser.add_argument("--min-elo", type=int, default=2000)
    parser.add_argument(
        "--require-both-rated",
        action="store_true",
        help="Require both players to meet --min-elo (default: either side)",
    )
    parser.add_argument(
        "--time-controls",
        default="blitz,rapid,classical",
        help="Comma-separated buckets: bullet,blitz,rapid,classical (or 'all')",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Stop after this many accepted games (default: no limit)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Stop after this many positions written (default: no limit)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1_000_000,
        help="Positions per output shard (default: 1000000)",
    )
    parser.add_argument(
        "--require-eval",
        action="store_true",
        help="Keep only games with Lichess Stockfish [%%eval] annotations",
    )
    parser.add_argument(
        "--eval-weight",
        type=float,
        default=0.5,
        help="Value target = w * engine eval + (1 - w) * game result "
        "for annotated positions (default: 0.5)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Game-encoding processes (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=500,
        help="Skip games longer than this many half-moves",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N scanned games",
    )
    return parser.parse_args()


def open_pgn_stream(path: Path):
    """Yield a text stream over a .pgn or zstd-compressed PGN dump."""
    path = Path(path)
    if path.suffix == ".zst" or path.name.endswith(".pgn.zst"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise SystemExit(
                "Reading .zst dumps requires the 'zstandard' package.\n"
                "  pip install zstandard"
            ) from exc
        fh = path.open("rb")
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        reader = dctx.stream_reader(fh)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        return text, fh
    return path.open("r", encoding="utf-8", errors="replace"), None


def _ssl_context() -> ssl.SSLContext:
    """HTTPS context that avoids the broken Windows cert-store load path.

    On some Windows installs, ssl.create_default_context() fails while loading
    the system store with ASN1: NOT_ENOUGH_DATA. Prefer certifi's CA bundle.
    """
    try:
        import certifi
    except ImportError as exc:
        raise SystemExit(
            "Downloading monthly dumps requires the 'certifi' package "
            "(needed for a reliable HTTPS CA bundle on Windows).\n"
            "  pip install certifi\n"
            "Or download the dump yourself and pass --pgn <path>."
        ) from exc

    cafile = certifi.where()
    try:
        ctx = ssl.create_default_context(cafile=cafile)
    except ssl.SSLError:
        # Default construction can still touch the Windows store on some builds;
        # build a clean context and load only certifi.
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = True
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_verify_locations(cafile=cafile)
    return ctx


def _download_url(url: str, dest: Path, chunk_size: int = 8 * 1024 * 1024) -> None:
    """Stream url to dest with MB progress, using a certifi-backed SSL context."""
    req = urllib.request.Request(url, headers={"User-Agent": "chess-rl/1.0"})
    ctx = _ssl_context()
    try:
        with urllib.request.urlopen(req, context=ctx) as resp, dest.open("wb") as out:
            total = resp.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else None
            downloaded = 0
            last_report = time.monotonic()
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 2.0:
                    mb = downloaded / (1024 * 1024)
                    if total_bytes:
                        pct = 100.0 * downloaded / total_bytes
                        total_mb = total_bytes / (1024 * 1024)
                        print(f"  … {mb:.0f} / {total_mb:.0f} MB ({pct:.1f}%)", flush=True)
                    else:
                        print(f"  … {mb:.0f} MB", flush=True)
                    last_report = now
    except urllib.error.URLError as exc:
        cause = exc.reason if isinstance(exc.reason, BaseException) else exc
        if isinstance(cause, ssl.SSLError) or "SSL" in str(exc):
            raise SystemExit(
                f"HTTPS download failed ({exc}).\n"
                "Try: pip install --upgrade certifi\n"
                "Or download the dump yourself and pass --pgn <path>."
            ) from exc
        raise


def download_month(month: str, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    filename = f"lichess_db_standard_rated_{month}.pgn.zst"
    dest = download_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using cached dump: {dest}")
        return dest

    url = LICHESS_DB_URL.format(month=month)
    print(f"Downloading {url}")
    print("Note: standard monthly dumps are large (often 15–30 GB compressed).")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        _download_url(url, tmp)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    print(f"Saved to {dest}")
    return dest


def parse_elo(headers, key: str):
    raw = headers.get(key, "")
    if not raw or raw == "?":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def base_time_seconds(time_control: str):
    """Parse PGN TimeControl 'base+inc' → base seconds, or None if unknown."""
    if not time_control or time_control in {"-", "?"}:
        return None
    # Correspondence / unlimited
    if time_control == "0+0":
        return None
    base = time_control.split("+", 1)[0]
    try:
        return int(base)
    except ValueError:
        return None


def time_control_allowed(time_control: str, allowed_buckets: set[str] | None) -> bool:
    if allowed_buckets is None:
        return True
    base = base_time_seconds(time_control)
    if base is None:
        return False
    for name in allowed_buckets:
        lo, hi = TIME_CONTROL_BUCKETS[name]
        if lo <= base <= hi:
            return True
    return False


def game_passes_filters(game, args, allowed_buckets) -> bool:
    return headers_pass_filters(game.headers, args, allowed_buckets)


def headers_pass_filters(headers, args, allowed_buckets) -> bool:
    result = headers.get("Result", "*")
    if result not in VALID_RESULTS:
        return False

    termination = headers.get("Termination", "Normal")
    if termination not in ALLOWED_TERMINATIONS:
        return False

    white_elo = parse_elo(headers, "WhiteElo")
    black_elo = parse_elo(headers, "BlackElo")
    if white_elo is None and black_elo is None:
        return False
    if args.require_both_rated:
        if white_elo is None or black_elo is None:
            return False
        if white_elo < args.min_elo or black_elo < args.min_elo:
            return False
    else:
        best = max(e for e in (white_elo, black_elo) if e is not None)
        if best < args.min_elo:
            return False

    if not time_control_allowed(headers.get("TimeControl", ""), allowed_buckets):
        return False

    return True


_HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')


def iter_raw_games(stream):
    """Yield (headers dict, raw PGN text) without invoking the PGN parser."""
    header_lines, move_lines = [], []
    for line in stream:
        if line.startswith("["):
            if move_lines:
                yield header_lines, move_lines
                header_lines, move_lines = [], []
            header_lines.append(line)
        elif line.strip():
            move_lines.append(line)
        elif move_lines:
            yield header_lines, move_lines
            header_lines, move_lines = [], []
    if move_lines:
        yield header_lines, move_lines


def parse_raw_headers(header_lines):
    headers = {}
    for line in header_lines:
        match = _HEADER_RE.match(line)
        if match:
            headers[match.group(1)] = match.group(2)
    return headers


_WORKER_OPTIONS = {}


def _init_worker(eval_weight, max_plies):
    _WORKER_OPTIONS.update(eval_weight=eval_weight, max_plies=max_plies)


def _encode_raw_game(text):
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None or game.errors:
        return None
    return encode_game(game, **_WORKER_OPTIONS)


def resolve_buckets(spec: str):
    spec = spec.strip().lower()
    if spec in {"all", "*"}:
        return None
    buckets = {part.strip() for part in spec.split(",") if part.strip()}
    unknown = buckets - set(TIME_CONTROL_BUCKETS)
    if unknown:
        raise SystemExit(f"Unknown time-control buckets: {sorted(unknown)}")
    return buckets


def main():
    args = parse_args()
    allowed_buckets = resolve_buckets(args.time_controls)
    if not 0.0 <= args.eval_weight <= 1.0:
        raise SystemExit("--eval-weight must be between 0 and 1")

    if args.month:
        pgn_path = download_month(args.month, args.download_dir)
    else:
        pgn_path = args.pgn
        if not pgn_path.exists():
            raise SystemExit(f"PGN not found: {pgn_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.glob("shard_*.npz")):
        raise SystemExit(
            f"{args.out_dir} already has shard_*.npz files; "
            "remove them or pass a different --out-dir"
        )

    stats = {"scanned": 0, "filtered": 0}
    stream, raw_handle = open_pgn_stream(pgn_path)

    def accepted_texts():
        for header_lines, move_lines in iter_raw_games(stream):
            stats["scanned"] += 1
            headers = parse_raw_headers(header_lines)
            if not headers_pass_filters(headers, args, allowed_buckets) or (
                args.require_eval and not any("[%eval" in l for l in move_lines)
            ):
                stats["filtered"] += 1
                continue
            yield "".join(header_lines) + "\n" + "".join(move_lines)

    pending, pending_positions = [], 0
    shard_idx = accepted = rejected = positions = 0
    written_shards = []
    started = time.monotonic()

    def flush(final=False):
        nonlocal pending, pending_positions, shard_idx
        if not pending:
            return
        packed = concat_packed(pending)
        total = len(packed["move"])
        start = 0
        while total - start >= args.shard_size or (final and start < total):
            stop = min(start + args.shard_size, total)
            path = args.out_dir / f"shard_{shard_idx:05d}.npz"
            save_packed_shard(path, slice_packed(packed, start, stop))
            written_shards.append(path)
            print(f"wrote {path} ({stop - start:,} positions)", flush=True)
            shard_idx += 1
            start = stop
        pending = [slice_packed(packed, start, total)] if start < total else []
        pending_positions = total - start

    try:
        with mp.Pool(
            args.workers,
            initializer=_init_worker,
            initargs=(args.eval_weight, args.max_plies),
        ) as pool:
            for packed in pool.imap(_encode_raw_game, accepted_texts(), chunksize=16):
                if packed is None:
                    rejected += 1
                    continue
                n = len(packed["move"])
                if args.max_positions is not None:
                    n = min(n, args.max_positions - positions)
                    packed = slice_packed(packed, 0, n)
                accepted += 1
                positions += n
                pending.append(packed)
                pending_positions += n
                if pending_positions >= args.shard_size:
                    flush()

                if accepted % args.progress_every == 0:
                    rate = positions / max(time.monotonic() - started, 1e-9)
                    print(
                        f"scanned={stats['scanned']:,} accepted={accepted:,} "
                        f"positions={positions:,} ({rate:,.0f}/s)",
                        file=sys.stderr,
                        flush=True,
                    )
                if args.max_games is not None and accepted >= args.max_games:
                    break
                if args.max_positions is not None and positions >= args.max_positions:
                    break
    finally:
        stream.close()
        if raw_handle is not None:
            raw_handle.close()

    flush(final=True)
    if not written_shards:
        raise SystemExit("No games passed the filters; relax them or check the PGN.")

    manifest = {
        "format": "sl_dataset packed npz",
        "source_pgn": str(pgn_path),
        "min_elo": args.min_elo,
        "require_both_rated": args.require_both_rated,
        "time_controls": args.time_controls,
        "require_eval": args.require_eval,
        "eval_weight": args.eval_weight,
        "games_scanned": stats["scanned"],
        "games_accepted": accepted,
        "games_rejected": rejected + stats["filtered"],
        "positions": positions,
        "shards": [str(p) for p in written_shards],
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    print(f"Done. accepted_games={accepted:,} positions={positions:,}")
    print(f"Shards: {len(written_shards)} under {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    print("Next: python train_supervised.py, then multiprocess_train_v2.py.")


if __name__ == "__main__":
    main()
