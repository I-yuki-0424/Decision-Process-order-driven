"""`python -m integrations.botsystem --root exchange --run-id r1 --policy random --spec min`"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .action_spec import proposal_full, proposal_min, proposal_min_pitch
from .serve import NoopPolicy, PolicyServer, UniformRandomPolicy

SPECS = {"full": proposal_full, "min": proposal_min, "min_pitch": proposal_min_pitch}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("exchange"))
    p.add_argument("--run-id", required=True)
    p.add_argument("--policy", choices=("noop", "random"), default="noop")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--expect-spec", choices=sorted(SPECS), help="refuse a hello whose action spec differs from this proposal")
    p.add_argument("--idle-timeout-s", type=float, default=300.0)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    policy = UniformRandomPolicy(a.seed) if a.policy == "random" else NoopPolicy()
    server = PolicyServer(a.root, a.run_id, policy, expected_spec=SPECS[a.expect_spec]() if a.expect_spec else None)
    try:
        server.run(idle_timeout_s=a.idle_timeout_s)
    finally:
        server.close()


if __name__ == "__main__":
    main()
