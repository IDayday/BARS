#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gate", required=True)
    p.add_argument("--decision-report", required=True)
    args = p.parse_args()
    text = Path(args.decision_report).read_text() if Path(args.decision_report).exists() else ""
    if args.gate in text:
        print(f"{args.gate}: PASS")
        return
    print(f"{args.gate}: HOLD")
    sys.exit(1)


if __name__ == "__main__":
    main()
