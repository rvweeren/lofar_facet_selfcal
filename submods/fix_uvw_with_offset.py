#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from casacore.tables import taql
from casacore.tables import table as casacore_table


DEFAULT_OFFSET_SECONDS = 17723.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shift TIME by a constant offset, recompute UVW with "
            "mscal.UVWJ2000(), then restore TIME. By default the input MS "
            "is updated in place; provide output_ms to write to a copied MS."
        )
    )
    parser.add_argument("input_ms", help="Input Measurement Set path")
    parser.add_argument(
        "output_ms",
        nargs="?",
        help="Optional output Measurement Set path; if omitted, update input_ms in place",
    )
    parser.add_argument(
        "offset_seconds",
        nargs="?",
        type=float,
        default=DEFAULT_OFFSET_SECONDS,
        help=(
            "Constant time offset in seconds to apply during UVW recomputation "
            f"(default: {DEFAULT_OFFSET_SECONDS:g})"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output Measurement Set if it already exists",
    )
    parser.add_argument(
        "--flagged-only",
        action="store_true",
        help=(
            "Only replace UVW on flagged rows (FLAG_ROW=true or rows where all FLAG "
            "values are true); leave unflagged rows unchanged"
        ),
    )
    return parser.parse_args()


def quote_table(path: Path) -> str:
    return f'"{path}"'


def run_taql(command: str) -> None:
    taql(command)


def build_flagged_row_mask(ms_path: Path) -> np.ndarray:
    ms = casacore_table(str(ms_path), readonly=True)
    try:
        flag = ms.getcol("FLAG")
        flag_row = ms.getcol("FLAG_ROW").astype(bool)
    finally:
        ms.close()

    if flag.ndim == 3:
        row_all_flagged = np.all(flag, axis=(1, 2))
    elif flag.ndim == 2:
        row_all_flagged = np.all(flag, axis=1)
    else:
        row_all_flagged = flag.astype(bool)

    return flag_row | row_all_flagged


def recompute_uvw_in_place(ms_path: Path, offset: float) -> None:
    ms_ref = quote_table(ms_path)
    run_taql(f"update {ms_ref} set TIME=TIME+({offset})s")
    run_taql(f"update {ms_ref} set UVW=mscal.UVWJ2000()")
    run_taql(f"update {ms_ref} set TIME=TIME-({offset})s")


def recompute_flagged_only(ms_path: Path, offset: float) -> int:
    flagged_mask = build_flagged_row_mask(ms_path)
    flagged_count = int(flagged_mask.sum())
    if flagged_count == 0:
        return 0

    temp_ms = ms_path.parent / f"{ms_path.name}.uvwtmp"
    if temp_ms.exists():
        shutil.rmtree(temp_ms)

    try:
        shutil.copytree(ms_path, temp_ms)
        recompute_uvw_in_place(temp_ms, offset)

        target = casacore_table(str(ms_path), readonly=False)
        temp = casacore_table(str(temp_ms), readonly=True)
        try:
            uvw = target.getcol("UVW")
            uvw_new = temp.getcol("UVW")
            uvw[flagged_mask] = uvw_new[flagged_mask]
            target.putcol("UVW", uvw)
        finally:
            temp.close()
            target.close()
    finally:
        if temp_ms.exists():
            shutil.rmtree(temp_ms)

    return flagged_count


def main() -> int:
    args = parse_args()

    input_ms = Path(args.input_ms).expanduser().resolve()
    output_ms = None if args.output_ms is None else Path(args.output_ms).expanduser().resolve()

    if not input_ms.exists():
        raise FileNotFoundError(f"Input Measurement Set does not exist: {input_ms}")
    if not input_ms.is_dir():
        raise NotADirectoryError(f"Input path is not a Measurement Set directory: {input_ms}")

    if output_ms is None:
        target_ms = input_ms
        action = "Updated"
    else:
        if output_ms.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output Measurement Set already exists: {output_ms}. "
                    "Use --overwrite to replace it."
                )
            shutil.rmtree(output_ms)

        shutil.copytree(input_ms, output_ms)
        target_ms = output_ms
        action = "Created"

    output_ref = quote_table(target_ms)
    offset = args.offset_seconds

    if args.flagged_only:
        flagged_count = recompute_flagged_only(target_ms, offset)
    else:
        recompute_uvw_in_place(target_ms, offset)
        flagged_count = None

    print(f"{action} {target_ms}")
    print(f"Applied temporary TIME offset of {offset} s for UVW recomputation")
    if flagged_count is not None:
        print(f"Updated UVW on {flagged_count} flagged rows only")
    else:
        print("Updated UVW on all rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())