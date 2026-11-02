#!/usr/bin/env python3
"""Check token and full-logit bitwise parity between two Qwen probe runs."""
import argparse
from array import array
import json
from pathlib import Path


def compare_turn(old, new):
    result = {"token_ids_equal": old["token_ids"] == new["token_ids"],
              "logits_bitwise_equal": len(old["steps"]) == len(new["steps"]),
              "steps": len(new["steps"]), "max_abs_logit_difference": 0.0}
    for before, after in zip(old["steps"], new["steps"]):
        a = Path(before["logits_file"]).read_bytes()
        b = Path(after["logits_file"]).read_bytes()
        if (len(a) != before["vocab_size"] * 4 or
                len(b) != after["vocab_size"] * 4):
            raise ValueError("A recorded logits file is truncated or has invalid geometry")
        result["logits_bitwise_equal"] &= a == b
        if a != b:
            x, y = array("f"), array("f")
            x.frombytes(a)
            y.frombytes(b)
            result["max_abs_logit_difference"] = max(
                result["max_abs_logit_difference"],
                max((abs(i - j) for i, j in zip(x, y)), default=0.0))
    result["passed"] = result["token_ids_equal"] and result["logits_bitwise_equal"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_manifest = json.loads((args.baseline / "report.json").read_text())
    new_manifest = json.loads((args.current / "report.json").read_text())
    old_keys = {(r["model"], r["prompt"]) for r in old_manifest["runs"]}
    new_keys = {(r["model"], r["prompt"]) for r in new_manifest["runs"]}
    if old_keys != new_keys:
        parser.error("The baseline and current prompt/model sets differ")
    report = {"baseline": str(args.baseline), "current": str(args.current), "runs": []}
    for model, prompt in sorted(new_keys):
        name = f"{model}-{prompt}.json"
        old = json.loads((args.baseline / name).read_text())
        new = json.loads((args.current / name).read_text())
        item = {"model": model, "prompt": prompt, **compare_turn(old, new)}
        report["runs"].append(item)
    old_sessions = {r["model"] for r in old_manifest.get("sessions", [])}
    new_sessions = {r["model"] for r in new_manifest.get("sessions", [])}
    if old_sessions != new_sessions:
        parser.error("The baseline and current session sets differ")
    report["sessions"] = []
    for model in sorted(new_sessions):
        name = f"{model}-session.json"
        old = json.loads((args.baseline / name).read_text())
        new = json.loads((args.current / name).read_text())
        item = {"model": model, "first_turn": compare_turn(old, new), "continuations": []}
        if len(old["continuations"]) != len(new["continuations"]):
            raise RuntimeError("The baseline and current continuation sets differ")
        for before, after in zip(old["continuations"], new["continuations"]):
            item["continuations"].append(compare_turn(before, after))
        item["passed"] = item["first_turn"]["passed"] and all(
            turn["passed"] for turn in item["continuations"])
        report["sessions"].append(item)
    report["passed"] = all(item["passed"] for item in report["runs"] + report["sessions"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
