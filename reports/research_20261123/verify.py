"""Recalculate report metrics; verify the published log snapshot."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median

HERE = Path(__file__).resolve().parent
EVAL = HERE / "logs/evaluate_results"


def read(path):
    return json.loads(path.read_text())


def demo_metrics(run):
    root = EVAL / run
    samples = read(root / "manifest.json")["samples"]
    expected = {(s["task"], s["demo"], s["t"], s["seed"]) for s in samples}
    completed = []
    for marker in (root / "results").glob("*/*/complete.json"):
        if read(marker)["complete"]:
            completed.append(read(marker.with_name("metrics.json")))
    actual = {(s["task"], s["demo"], s["t"], s["seed"]) for s in completed}
    assert len(expected) == len(samples) == len(completed) and actual == expected, run
    return completed


def verify_demo_logs():
    old = demo_metrics("demo_fivearm_collect_20261117")
    new = demo_metrics("demo_sixarm_diag_20261118")
    assert (len(old), len(new)) == (600, 100)
    assert [round(mean(s["branches"][a]["eef_target_error_mm_8"] for s in old), 2)
            for a in ("A", "final025", "final1", "old789")] == [3.86, 3.75, 5.27, 4.99]
    assert sum(s["branches"]["final025"]["actual_target_mse"] <
               s["branches"]["reverse025"]["actual_target_mse"] for s in new) == 81
    front = [s for s in new if s["phase"] in {"start", "middle"}]
    assert len(front) == 68
    for metric, expected in (("eef_target_error_mm_4", 7.8), ("eef_target_error_mm_8", 7.0),
                             ("actual_target_mse", 4.7)):
        a, b = (mean(s["branches"][arm][metric] for s in front) for arm in ("A", "final025"))
        assert round(100 * (1 - b / a), 1) == expected, (metric, a, b)
    old_by_key = {(s["task"], s["demo"], s["t"], s["seed"]): s for s in old}
    matched_old = [old_by_key[(s["task"], s["demo"], s["t"], s["seed"])] for s in new]
    for group, expected_mean, expected_count in ((matched_old, 1.24, 22), (new, .30, 4)):
        errors = [s["branches"]["demo"]["eef_target_error_mm_8"] for s in group]
        assert (round(mean(errors), 2), sum(e > 1 for e in errors)) == (expected_mean, expected_count)


def verify_closed_loop_logs():
    run = EVAL / "final025_closedloop_resume_gpu0_20261118"
    rows = read(run / "completed.json")["rows"]
    by_arm = {arm: {(r["seed"], r["task"]): r for r in rows if r["arm"] == arm}
              for arm in ("A", "final025")}
    a, b = by_arm["A"], by_arm["final025"]
    assert len(rows) == 28 and len(a) == len(b) == 14 and a.keys() == b.keys()
    assert Counter((a[k]["success"], b[k]["success"]) for k in a) == {(True, True): 8, (False, False): 6}
    assert [mean(r["steps"] for r in arm.values() if r["success"]) for arm in (a, b)] == [243.5, 278.25]
    assert median(b[k]["steps"] - a[k]["steps"] for k in a if a[k]["success"]) == 1.5

    updates = Counter()
    for row in b.values():
        source = EVAL / Path(row["reused_from"]).name if row.get("reused_from") else run
        path = source / f"task{row['task']}_seed{row['seed']}/final025/updates.jsonl"
        with path.open() as f:
            for update in map(json.loads, f):
                change = update["predicted_after"] - update["predicted_before"]
                updates[(change > 0) - (change < 0)] += 1
    assert updates == {-1: 808, 1: 1}, updates


def report_counts():
    clean = read(EVAL / "three_tasks_cleanprompt_42_43_20261111/final_results_20261113.json")
    assert tuple(clean["totals"][k] for k in ("episodes","baseline","mse","rescued","harmed")) == (244,212,207,6,11)
    gradient = read(EVAL / "input_gradient_two_tasks_20261116/paired_report.json")
    assert [sum(r[k] for r in gradient["rows"]) for k in ("n","control","new","unweighted")] == [162,133,133,128]
    assert [len(gradient["paired"]["control"][k]) for k in ("rescues","harms")] == [4,4]
    rollout = read(EVAL / "input_gradient_two_tasks_20261116/combined_162_results.json")
    rows = rollout["rollouts"]
    assert len(rows) == len({(r["task_id"], r["seed"]) for r in rows}) == 162
    assert sum(r["successes"] for r in rows) == rollout["successes"] == 133

    snapshot=read(HERE/"server_verified_summary.json")
    old=snapshot["demo_fivearm_collect_20261117"]
    new=snapshot["demo_sixarm_diag_20261118"]
    assert (old["completed"],new["completed"])==(600,100)
    assert [round(old["mean_t8_eef_error_mm"][a],2) for a in ("A","final025","final1","old789")]==[3.86,3.75,5.27,4.99]
    assert new["positive_beats_reverse_on_real_target_mse"]==81
    assert len(read(EVAL/"demo_sixarm_diag_20261118/manifest.json")["samples"])==new["completed"]
    real = "actual_target_mse"
    early=new["early_and_middle"]
    assert early["n"]==68
    for metric,percentage in (("mean_t4_eef_error_mm",7.8),("mean_t8_eef_error_mm",7.0),("mean_real_target_mse",4.7)):
        a,b = [early[metric][arm] for arm in ("A","final025")]
        assert round(100*(1-b/a),1)==percentage, (metric,a,b)
    replay=new["matched_demo_replay_t8"]
    assert [round(replay[k]["mean_mm"],2) for k in ("before","after")]==[1.24,.30]
    assert [replay[k]["over_1mm"] for k in ("before","after")]==[22,4]

    small = read(EVAL / "real_ab_v2_20261117/aggregate.json")
    assert (small["states"],small["predicted_improved"],small["actual_improved"]) == (12,12,4)
    for metric, expected in ((real,[.72976,.73288]),("prediction_mse",[.37197,.37222]),
                             ("predicted_target_mse",[.76275,.75764])):
        assert [round(small["means"][a][metric],5) for a in ("A","B")] == expected

    closed=snapshot["final025_closedloop_resume_gpu0_20261118"]
    paired=[dict(zip(closed["fields"],r)) for r in closed["rows"]]
    assert len({(r["seed"],r["task"]) for r in paired})==len(paired)==closed["pairs"]==14
    assert Counter((r["baseline_success"],r["final025_success"]) for r in paired)=={(True,True):8,(False,False):6}
    successes=[r for r in paired if r["baseline_success"] and r["final025_success"]]
    assert [mean(r[k] for r in successes) for k in ("baseline_steps","final025_steps")] == [243.5,278.25]
    assert median(r["final025_steps"]-r["baseline_steps"] for r in successes)==1.5
    assert sum(r["final025_updates"] for r in paired)==809
    assert sum(r["final025_updates_with_higher_loss"] for r in paired)==1
    assert closed["final025_updates"]==dict(total=809,lower_loss=808,higher_loss=1)

    audit = EVAL / "latent_action_response_audit_20261118"
    groups=read(audit/"summary.json")["groups"]
    for name,n,ratio,cos in (("demo_front_middle",68,28,.11),("ood_all",12,37,.07)):
        r=groups[f"{name}/A_to_positive025/t+8"]
        assert (r["n"],round(r["norm_ratio"]*100),round(r["cosine"],2))==(n,ratio,cos)
    action=read(audit/"action_audit.json")
    assert len(action["records"])==100
    windows=[w for r in action["training_cache"] for w in r["observations"]]
    assert len(windows)==45 and all(w["action_t"] for w in windows)
    times=read(audit/"training_frame_state_timing.json")
    assert len(times)==20 and all(min(r["angles_rms_by_state_offset"],key=lambda a:a[1])==[1,0.] for r in times)
    print("Verified main report counts, paired effects, and training offset evidence.")


def manifest(deep):
    log_manifest=read(HERE/"log_manifest.json")
    records=log_manifest["records"]
    paths={r["path"] for r in records}
    assert len(paths)==len(records)
    assert paths=={f.relative_to(HERE).as_posix() for f in (HERE/"logs").rglob("*") if f.is_file()}
    for row in records:
        f=HERE/row["path"]
        data=f.read_bytes()
        assert len(data)==row["bytes"] and hashlib.sha256(data).hexdigest()==row["sha256"], f
        if deep and f.name.endswith((".jsonl",".jsonl.gz")):
            file_open=gzip.open if f.suffix==".gz" else open
            with file_open(f,"rt",encoding="utf-8") as stream:
                for i,line in enumerate(stream,1):
                    if line.strip():
                        try:json.loads(line)
                        except ValueError as exc:raise ValueError(f"{f}:{i}: {exc}") from exc
    for dest in re.findall(r"\]\(([^)]+)\)", (HERE/"README.md").read_text()):
        if not dest.startswith(("http://","https://")):
            assert (HERE/dest).exists(),dest
    print(f"Verified {len(records)} archived file hashes; {len(log_manifest['exclusions'])} skipped entries.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true", help="Also parse every JSONL record")
    opts=parser.parse_args()
    manifest(opts.deep)
    report_counts()
    verify_demo_logs()
    verify_closed_loop_logs()
    print("Recomputed demonstration and closed-loop metrics from archived source logs.")
