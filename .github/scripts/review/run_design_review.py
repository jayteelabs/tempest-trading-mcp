# ruff: noqa: E402,I001

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from review.producer_common import execute_review


if __name__ == "__main__":
    raise SystemExit(execute_review(agent="shuna", stage="review/design"))
