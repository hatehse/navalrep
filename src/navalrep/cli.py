import argparse
import json
import logging
from pathlib import Path
import sys

from .config import load_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Local Spanish cliffhanger clips; sequential CUDA and NVENC")
    parser.add_argument("--config", type=Path, help="JSON configuration; relative paths use the current working directory")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check dependencies, cached model and actual NVENC hardware")
    doctor.add_argument("--no-encode-test", action="store_true", help="Only inspect FFmpeg features; does not prove GPU readiness")
    run = sub.add_parser("run", help="Process all input files strictly sequentially")
    run.add_argument("--input", type=Path, action="append", default=[], help="Explicit file; repeat for several episodes")
    run.add_argument("--skip-preflight", action="store_true", help="Skip doctor (runtime failures are still logged)")
    download = sub.add_parser("download-model", help="Explicit online preparation; later runs stay offline")
    download.add_argument("--destination", type=Path, default=Path("models/large-v3"))
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        if args.command == "download-model":
            from huggingface_hub import snapshot_download
            destination = args.destination.resolve()
            snapshot_download(repo_id="Systran/faster-whisper-large-v3", local_dir=str(destination))
            print(f"Model prepared in {destination}; set model_dir to this directory in your config.")
            return 0
        if args.command == "doctor":
            from .doctor import check_environment
            checks = check_environment(config, encode_test=not args.no_encode_test)
            print(json.dumps(checks, ensure_ascii=False, indent=2))
            return 0 if all(c["ok"] for c in checks) else 1
        config.output_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(config.output_dir / "pipeline.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logging.getLogger().addHandler(handler)
        try:
            if not args.skip_preflight:
                from .doctor import check_environment
                failed = [c for c in check_environment(config) if not c["ok"]]
                if failed:
                    for check in failed:
                        logging.error("Preflight %s: %s", check["name"], check["detail"])
                    return 1
            from .pipeline import run_pipeline
            reports = run_pipeline(config, args.input)
            summary = {"episodes": len(reports), "clips": sum(len(r["clips"]) for r in reports), "statuses": [r["status"] for r in reports]}
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            # No candidate is an explicit non-success, rather than a misleading zero-clip pass.
            return 0 if all(r["status"] == "ok" for r in reports) else 1
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()
    except KeyboardInterrupt:
        logging.warning("Interrupted; owned temporary files will be cleaned")
        return 130
    except Exception:
        logging.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
