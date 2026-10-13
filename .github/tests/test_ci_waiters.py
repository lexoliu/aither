import os
import pathlib
import subprocess
import tempfile

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[1] / "workflows"
PREFIX = r"""
gh() {
  printf '%s' "${FIXTURE_CHECKS}"
  return "${FIXTURE_GH_STATUS}"
}
sleep() {
  printf 'CHECK_WAITING\n' >&2
  return 43
}
"""


def main():
    cases = {
        "success": ("Test\tsuccess\n", 0, 0),
        "pending": ("Test\tpending\n", 0, 43),
        "empty": ("", 0, 43),
        "api_error": ("", 2, 2),
    }
    for conclusion in [
        "failure", "cancelled", "timed_out", "action_required", "startup_failure", "stale",
    ]:
        cases[conclusion] = (f"Test\t{conclusion}\n", 0, 1)
    for filename in ["release.yml", "pr-base-gate.yml"]:
        workflow = yaml.safe_load((ROOT / filename).read_text())
        steps = next(iter(workflow["jobs"].values()))["steps"]
        step = next(
            step for step in steps
            if step.get("name", "").lower().startswith("wait for")
        )
        script = PREFIX + step["run"]
        for name, (checks, api_status, expected_status) in cases.items():
            env = os.environ.copy()
            env.update({
                "GITHUB_REPOSITORY": "fixture/repo",
                "GITHUB_SHA": "fixture-head",
                "HEAD_SHA": "fixture-head",
                "MERGE_SHA": "",
                "FIXTURE_CHECKS": checks,
                "FIXTURE_GH_STATUS": str(api_status),
            })
            with tempfile.TemporaryDirectory() as directory:
                env["RUNNER_TEMP"] = directory
                result = subprocess.run(
                    ["/bin/bash", "-c", script],
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            assert result.returncode == expected_status, (
                filename, name, result.returncode, result.stdout, result.stderr,
            )
            if expected_status == 43:
                assert "CHECK_WAITING" in result.stderr
                assert "passed" not in result.stdout
            print(f"{filename}: {name}: expected exit {expected_status}")


if __name__ == "__main__":
    main()
