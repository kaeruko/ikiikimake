"""Render summary.html for an existing Lab analysis directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.analyze_cheek_lab import write_summary_html


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lab_output", type=Path, help="Existing lab_selected_pair directory")
    args = parser.parse_args(argv)

    lab_output = args.lab_output.resolve()
    summary_path = lab_output / "analysis_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"analysis_summary.json does not exist: {summary_path}")

    output = lab_output / "summary.html"
    if output.exists():
        raise FileExistsError(f"summary.html already exists; not overwriting: {output}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = write_summary_html(lab_output, summary)
    print(result)
    return result


if __name__ == "__main__":
    main()
