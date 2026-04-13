from __future__ import annotations

from review.producer_common import execute_review

if __name__ == "__main__":
    raise SystemExit(execute_review(agent="shion", stage="review/logic_audit"))
