import json

import pytest

from beamroot import experiment


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def fake_run(run_dir, beam1=0.90, soft=0.95, minutes=2.5, merge_kls=(0.03,), candidates=None):
    """A finished run directory in the format train.run + the harness write."""
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"dataset": "synthetic", "seed": 0}))
    (run_dir / "model.pt").write_bytes(b"")
    events = [{"epoch": 1, "merge": None, "threshold": 0.05, "candidates": {}},
              {"epoch": 1, "split": "leaf1", "new_nodes": ["leaf2", "leaf3"], "num_nodes": 5}]
    events += [{"epoch": 2, "merge": ["leaf2", "leaf4"], "kl": kl, "removed": ["leaf4"],
                "num_nodes": 6, "threshold": 0.05} for kl in merge_kls]
    events.append({"epoch": 3, "merge": None, "threshold": 0.05,
                   "candidates": candidates or {"leaf2+leaf5": 0.4, "leaf3+leaf5": 0.2}})
    write_jsonl(run_dir / "growth.jsonl", events)
    (run_dir / "experiment.json").write_text(json.dumps({
        "train_minutes": minutes,
        "router_entropy": {"root": 0.69, "leaf1": 0.1, "leaf4": 0.3},
        "leaf_purity": {"leaf2": 0.8, "leaf3": 0.6, "leaf5": 1.0}}))
    (run_dir / "eval.json").write_text(json.dumps([
        {"mode": "beam 1", "accuracy": beam1, "mean_nodes": 3.0},
        {"mode": "soft", "accuracy": soft, "mean_nodes": 7.0}]))
    return run_dir


def test_run_metrics_from_a_finished_run_directory(tmp_path):
    m = experiment.run_metrics(fake_run(tmp_path / "run"))
    assert m["beam1_acc"] == pytest.approx(0.90)
    assert m["soft_acc"] == pytest.approx(0.95)
    assert m["gap"] == pytest.approx(0.05)
    assert m["split_router_entropy"] == pytest.approx(0.2)  # root excluded
    assert m["leaf_purity"] == pytest.approx(0.8)
    assert m["merges"] == 1
    assert m["closest_merge_kl"] == pytest.approx(0.03)  # fired merges count as candidates
    assert m["train_minutes"] == pytest.approx(2.5)


GRID = ["default", "--datasets", "synthetic", "--seeds", "0", "1", "--epochs", "2",
        "--device", "cpu"]


def test_grid_has_no_loss_or_init_axes():
    for axis in ("--losses", "--inits"):
        with pytest.raises(SystemExit):
            experiment.parse_args(["default", axis, "path"])
    with pytest.raises(SystemExit):
        experiment.parse_args(["routing"])  # its runs/summary are experiment 0001's record


def test_two_seed_grid_runs_end_to_end_and_writes_the_table(tmp_path):
    out = experiment.main(GRID + ["--runs-dir", str(tmp_path)])
    exp_dir = tmp_path / "default"
    assert out == exp_dir
    runs = sorted(p.name for p in exp_dir.iterdir() if p.is_dir())
    assert len(runs) == 2
    assert runs[0].endswith("-synthetic-seed0")
    assert runs[1].endswith("-synthetic-seed1")
    config = json.loads((exp_dir / runs[1] / "config.json").read_text())
    assert (config["model"], config["merge"], config["seed"]) == ("dag", True, 1)
    assert config["no_report"] is True
    measured = json.loads((exp_dir / runs[1] / "experiment.json").read_text())
    assert "root" in measured["router_entropy"]
    assert measured["leaf_purity"] and all(0 < p <= 1 for p in measured["leaf_purity"].values())
    modes = [r["mode"] for r in json.loads((exp_dir / runs[1] / "eval.json").read_text())]
    assert modes == ["beam 1", "soft"]
    summary = json.loads((exp_dir / "summary.json").read_text())
    assert [(r["dataset"], r["n"]) for r in summary["rows"]] == [("synthetic", 2)]
    assert len(summary["runs"]) == 2
    table = (exp_dir / "summary.md").read_text()
    for col in ["beam-1 acc", "soft acc", "soft − beam-1", "split-router entropy",
                "leaf purity", "merges", "closest merge KL", "train min"]:
        assert col in table
    assert "| synthetic | 2 |" in table


def test_rerun_skips_finished_runs_and_redoes_unfinished_ones(tmp_path, monkeypatch):
    experiment.main(GRID + ["--runs-dir", str(tmp_path)])
    calls = []
    real_run = experiment.train.run

    def spy(args):
        calls.append(args.seed)
        return real_run(args)

    monkeypatch.setattr(experiment.train, "run", spy)
    experiment.main(GRID + ["--runs-dir", str(tmp_path)])
    assert calls == []
    (next((tmp_path / "default").glob("*-synthetic-seed1")) / "eval.json").unlink()
    experiment.main(GRID + ["--runs-dir", str(tmp_path)])
    assert calls == [1]
    summary = json.loads((tmp_path / "default" / "summary.json").read_text())
    assert len(summary["runs"]) == 2


def test_rows_follow_dataset_order_with_mean_and_std(tmp_path):
    runs = []
    for i, (dataset, seed, beam1) in enumerate([
            ("cifar10", 0, 0.70), ("mnist", 0, 0.90), ("mnist", 1, 0.92)]):
        m = experiment.run_metrics(fake_run(tmp_path / f"r{i}", beam1=beam1))
        runs.append({"dataset": dataset, "seed": seed, **m})
    rows = experiment.summarise(runs, ["mnist", "cifar10"])
    assert [r["dataset"] for r in rows] == ["mnist", "cifar10"]
    assert rows[0]["beam1_acc"]["mean"] == pytest.approx(0.91)
    assert rows[0]["beam1_acc"]["std"] == pytest.approx(0.0141421, abs=1e-6)
    lines = experiment.format_table(rows).splitlines()
    assert lines[2].startswith("| mnist | 2 | 0.9100 ± 0.0141 |")
