"""Single argparse CLI dispatching to the stage modules.

    python experiment/cli.py --config ... <readiness|collect|warm|pairs|
        train|geometry|correction|evaluate|lock|analyze>

--protocol / --allow-unlocked gating: development stages run unlocked only
when explicit (--allow-unlocked); final stages require the lock.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import contract as C
from contract import (
    ProtocolError, sha256_file, write_meta, atomic_savez, reserve_outputs,
    complete_output, verify_completed, source_hashes, package_versions,
    stable_seed, scene_range, validate_config, verify_history_cache,
    verify_pair_bank, load_protocol, verify_current_protocol,
    check_binary_success, require_complete, require_eval_metadata,
    canonical_design, design_id, SCHEMA_VERSION,
)

STAGES = ("readiness", "collect", "warm", "pairs", "train", "geometry",
          "correction", "evaluate", "lock", "analyze")
DEVELOPMENT_STAGES = ("readiness", "collect", "warm", "pairs", "geometry",
                      "lock")
FINAL_STAGES = ("train", "correction", "evaluate", "analyze")


def protocol_for(args, config):
    # Item 21: locking requirements follow the explicit run role, not a
    # blanket stage list. Development evaluation/training with
    # --allow-unlocked --development stays unlocked; everything else final
    # needs the lock.
    path = getattr(args, "protocol", None)
    stage = getattr(args, "stage", None)
    rest = list(getattr(args, "rest", []) or [])
    development = "--development" in rest
    if stage == "lock":
        return None
    if stage in FINAL_STAGES and not (
            getattr(args, "allow_unlocked", False) and development):
        if not path:
            raise ProtocolError(
                f"{stage} is a final stage: pass --protocol "
                "runs/protocol_locked.json")
        protocol = load_protocol(path)
        verify_current_protocol(protocol, config, args.config, stage=stage)
        return protocol
    if not path and not getattr(args, "allow_unlocked", False):
        raise ProtocolError(
            f"{stage}: pass --protocol runs/protocol_locked.json, or "
            "--allow-unlocked for a development stage")
    if path:
        protocol = load_protocol(path)
        verify_current_protocol(protocol, config, args.config, stage=stage)
        return protocol
    return None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment/config.json")
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--allow-unlocked", action="store_true")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    with open(args.config) as handle:
        import json
        config = json.load(handle)
    validate_config(config, args.config)
    protocol_for(args, config)
    # Item 21: forward globals before the subcommand so stage parsers see
    # the same config/protocol/unlocked options (argparse optionals are
    # position-independent).
    glob = ["--config", args.config]
    if getattr(args, "protocol", None):
        glob += ["--protocol", args.protocol]
    elif getattr(args, "allow_unlocked", False):
        glob += ["--allow-unlocked"]
    if args.stage == "readiness":
        import readiness
        sys.argv = ["readiness.py"] + glob + args.rest
        return readiness.main()
    if args.stage == "collect":
        import collect
        # collect.py requires the "collect" subcommand; globals go first.
        sys.argv = ["collect.py"] + glob + ["collect"] + args.rest
        return collect.main()
    if args.stage == "warm":
        import train
        sys.argv = ["train.py"] + glob + ["warm"] + args.rest
        return train.main()
    if args.stage == "pairs":
        import pairs
        sys.argv = ["pairs.py"] + glob + ["pairs"] + args.rest
        return pairs.main()
    if args.stage == "train":
        import train
        sys.argv = ["train.py"] + glob + ["train"] + args.rest
        return train.main()
    if args.stage == "geometry":
        import geometry
        sys.argv = ["geometry.py"] + glob + ["geometry"] + args.rest
        return geometry.main()
    if args.stage == "correction":
        import correction
        sys.argv = ["correction.py"] + glob + args.rest
        return correction.main()
    if args.stage == "evaluate":
        import evaluate
        sys.argv = ["evaluate.py"] + glob + args.rest
        return evaluate.main()
    if args.stage == "lock":
        import lock
        sys.argv = ["lock.py"] + glob + args.rest
        return lock.main()
    if args.stage == "analyze":
        import analyze
        sys.argv = ["analyze.py"] + args.rest
        return analyze.main()
    raise ProtocolError(f"stage {args.stage!r} has no module yet")


if __name__ == "__main__":
    raise SystemExit(main())
