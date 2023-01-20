from __future__ import annotations
import argparse
import subprocess
import time
import yaml
from pathlib import Path
from cobaya.yaml import yaml_dump_file
from cobaya.yaml import yaml_load_file


def reformat_minimize(file: Path, outdir: Path | None = None):
    info = yaml_load_file(str(file))
    if "minimize" not in info["sampler"]:
        info["sampler"] = {"minimize": None}
    if outdir:
        info["output"] = str(outdir / Path(info["output"]).name)
    yaml_dump_file(str(file), info, error_if_exists=False)


def reformat_evaluate(file: Path, outdir: Path | None = None):
    info = yaml_load_file(str(file))
    if "evaluate" not in info["sampler"]:
        info["sampler"] = {"evaluate": None}
    if outdir:
        info["output"] = str(outdir / Path(info["output"]).name)
    yaml_dump_file(str(file), info, error_if_exists=False)


def reformat_mcmc(
    file: Path,
    outdir: Path | None = None,
    Rminus1_stop: float = 0.005,
    max_tries: int = 100000,
):
    info = yaml_load_file(str(file))
    if "mcmc" not in info["sampler"]:
        info["sampler"] = {
            "mcmc": {"Rminus1_stop": Rminus1_stop, "max_tries": max_tries}
        }
    if outdir:
        info["output"] = str(outdir / Path(info["output"]).name)
    yaml_dump_file(str(file), info, error_if_exists=False)


def collect_jobs(files: list[str]) -> set[Path]:
    jobs: set[Path] = set()
    for file in files:
        path = Path(file)
        if path.exists():
            jobs.add(path)
        else:
            jobs.union(Path(".").glob(file))
    print(f"collected {len(jobs)} jobs")
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["mcmc", "minimize", "evaluate"])
    parser.add_argument("files", nargs="*", help="yaml files, support glob")
    parser.add_argument("--outdir", type=Path, default=None, help="output folder")
    parser.add_argument("--local", action="store_true", help="do not submit job")
    parser.add_argument(
        "--wait", type=int, default=0, help="seconds to wait between different jobs"
    )
    parser.add_argument("--Rminus1_stop", type=float, default=0.005)
    parser.add_argument("--max_tries", type=int, default=100000)
    parser.add_argument(
        "-lf", "--last_failed", action="store_true", help="resume last failed jobs"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", action="store_true", help="resume")
    group.add_argument("-f", action="store_true", help="force overwrite")
    args = parser.parse_args()

    jobqueue = collect_jobs(args.files)
    if args.last_failed and (cache_path := Path(".subjob.cache")).exists():
        with cache_path.open() as f:
            last_failed = yaml.safe_load(f)
        jobqueue.update(last_failed)
    remains = jobqueue.copy()
    try:
        for job in jobqueue:
            if args.mode == "minimize":
                reformat_minimize(job, args.outdir)
            elif args.mode == "evaluate":
                reformat_evaluate(job, args.outdir)
            else:
                reformat_mcmc(
                    job,
                    args.outdir,
                    Rminus1_stop=args.Rminus1_stop,
                    max_tries=args.max_tries,
                )

            command = ["cobaya-run" if args.local else "cobaya-run-job", str(job)]
            if args.r:
                command += ["-r"]
            elif args.f:
                command += ["-f"]

            print(" ".join(command))
            subprocess.run(command)
            remains.remove(job)
            time.sleep(args.wait)
    finally:
        if remains:
            with Path(".subjob.cache").open("w") as f:
                yaml.safe_dump(list(remains), f)


if __name__ == "__main__":
    main()
