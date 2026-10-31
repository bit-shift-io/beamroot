import json

import pytest

from beamroot import experiment


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def fake_run(run_dir, beam1=0.90, soft=0.95, minutes=2.5, merge_kls=(0.03,), candidates=None,
             split_extra=None):
    """A finished run directory in the format train.run + the harness write."""
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"dataset": "synthetic", "seed": 0}))
    (run_dir / "model.pt").write_bytes(b"")
    events = [{"epoch": 1, "merge": None, "threshold": 0.05, "candidates": {}},
              {"epoch": 1, "split": "leaf1", "new_nodes": ["leaf2", "leaf3"], "num_nodes": 5,
               **(split_extra or {})}]
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
    assert m["beam1_nodes"] == pytest.approx(3.0)
    assert m["beam3_acc"] is None  # old run: no beam-3 row
    assert m["leaves"] == 3  # old run: counted from leaf_purity


def test_run_metrics_defaults_old_runs_to_binary_and_link_free(tmp_path):
    m = experiment.run_metrics(fake_run(tmp_path / "run"))
    assert (m["mean_k"], m["links"]) == (2, 0)


def test_run_metrics_reads_k_links_leaves_and_beam3(tmp_path):
    run = fake_run(tmp_path / "run", split_extra={"k": 4, "links": [["leaf2", "leaf9"]]})
    rows = json.loads((run / "eval.json").read_text())
    rows.insert(1, {"mode": "beam 3", "accuracy": 0.93, "mean_nodes": 5.0})
    (run / "eval.json").write_text(json.dumps(rows))
    extra = json.loads((run / "experiment.json").read_text())
    (run / "experiment.json").write_text(json.dumps({**extra, "leaves": 7}))
    m = experiment.run_metrics(run)
    assert (m["mean_k"], m["links"], m["leaves"], m["beam3_acc"]) == (4, 1, 7, 0.93)


def test_variant_flags_reach_train_and_summary_groups_by_variant(tmp_path, monkeypatch):
    monkeypatch.setitem(experiment.GRIDS, "vgrid", {
        "datasets": ["synthetic"], "seeds": [0],
        "variants": {"small": ["--max-nodes", "3"], "big": ["--max-nodes", "19"]}})
    seen = []
    real_run = experiment.train.run

    def spy(args):
        seen.append(args.max_nodes)
        return real_run(args)

    monkeypatch.setattr(experiment.train, "run", spy)
    experiment.main(["vgrid", "--datasets", "synthetic", "--epochs", "2", "--device", "cpu",
                     "--limit-batches", "2", "--runs-dir", str(tmp_path)])
    assert seen == [3, 19]
    summary = json.loads((tmp_path / "vgrid" / "summary.json").read_text())
    assert sorted(r["variant"] for r in summary["rows"]) == ["big", "small"]
    table = (tmp_path / "vgrid" / "summary.md").read_text()
    assert "| dataset | variant | n |" in table


def test_grid_epochs_per_dataset_unless_overridden(tmp_path, monkeypatch):
    monkeypatch.setitem(experiment.GRIDS, "egrid", {
        "datasets": ["synthetic"], "seeds": [0], "epochs": {"synthetic": 1}})
    seen = []
    real_run = experiment.train.run

    def spy(args):
        seen.append(args.epochs)
        return real_run(args)

    monkeypatch.setattr(experiment.train, "run", spy)
    common = ["--device", "cpu", "--limit-batches", "2"]
    experiment.main(["egrid", "--runs-dir", str(tmp_path / "a"), *common])
    experiment.main(["egrid", "--epochs", "2", "--runs-dir", str(tmp_path / "b"), *common])
    assert seen == [1, 2]


def test_variant_rows_sort_by_beam1_within_dataset(tmp_path):
    runs = []
    for i, (dataset, variant, beam1) in enumerate([
            ("mnist", "a", 0.80), ("cifar10", "a", 0.50), ("mnist", "b", 0.90)]):
        m = experiment.run_metrics(fake_run(tmp_path / f"r{i}", beam1=beam1))
        runs.append({"dataset": dataset, "variant": variant, "seed": 0, **m})
    rows = experiment.summarise(runs, ["mnist", "cifar10"])
    assert [(r["dataset"], r["variant"]) for r in rows] == [
        ("mnist", "b"), ("mnist", "a"), ("cifar10", "a")]


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
    assert modes == ["beam 1", "beam 3", "soft"]
    assert (exp_dir / runs[1] / "flow.txt").read_text().strip()
    summary = json.loads((exp_dir / "summary.json").read_text())
    assert [(r["dataset"], r["n"]) for r in summary["rows"]] == [("synthetic", 2)]
    assert len(summary["runs"]) == 2
    table = (exp_dir / "summary.md").read_text()
    for col in ["beam-1 acc", "beam-3 acc", "soft acc", "soft − beam-1", "leaves", "mean k",
                "links", "beam-1 nodes", "split-router entropy",
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


def test_run_metrics_reads_cost_bump_and_step_fields(tmp_path):
    run = fake_run(tmp_path / "run", split_extra={"bump": {"check": "silhouette", "from": 1,
                                                          "to": 2, "value": 0.05}})
    with open(run / "growth.jsonl", "a") as f:
        f.write(json.dumps({"epoch": 3, "bump": "r1", "check": "entropy", "from": 2, "to": 3,
                            "value": 0.9}) + "\n")
        f.write(json.dumps({"epoch": 3, "bump": "r2", "check": "spread", "at_max": True,
                            "value": 0.0}) + "\n")
    write_jsonl(run / "metrics.jsonl", [{"epoch": 1, "node_evals": 4.0},
                                        {"epoch": 2, "node_evals": 2.0}])
    rows = json.loads((run / "eval.json").read_text())
    rows[0].update(mean_macs=1000.0, ms_per_image=1.5)
    (run / "eval.json").write_text(json.dumps(rows))
    extra = json.loads((run / "experiment.json").read_text())
    (run / "experiment.json").write_text(json.dumps({**extra, "mean_step": 2.5}))
    m = experiment.run_metrics(run)
    assert (m["beam1_macs"], m["beam1_ms"], m["bumps"], m["mean_step"], m["train_evals"]) == (
        1000.0, 1.5, 2, 2.5, 3.0)


def test_run_metrics_old_run_without_cost_fields(tmp_path):
    m = experiment.run_metrics(fake_run(tmp_path / "run"))
    assert (m["beam1_macs"], m["beam1_ms"], m["bumps"], m["mean_step"], m["train_evals"]) == (
        None, None, 0, None, None)


def test_cascade_grid_variants_build_per_variant_flags():
    variants = experiment.GRIDS["cascade"]["variants"]
    assert len(variants) == 7
    assert "--merge" not in sum((v for k, v in variants.items() if k != "dag"), [])
    assert variants["dag"][:3] == ["--model", "dag", "--merge"]
    flags = variants["hidden-label"]
    assert flags[flags.index("--model") + 1] == "cascade"
    assert flags[flags.index("--start-step") + 1] == str(experiment.START_STEP)
    assert experiment.START_STEP < 4


def test_cascade_grid_smoke_runs_all_variants(tmp_path):
    exp = experiment.main(["cascade", "--datasets", "synthetic", "--epochs", "2", "--device", "cpu",
                           "--limit-batches", "2", "--runs-dir", str(tmp_path)])
    summary = json.loads((exp / "summary.json").read_text())
    assert {r["variant"] for r in summary["runs"]} == set(experiment.GRIDS["cascade"]["variants"])


def test_cascade29_grid_gives_every_variant_29_nodes():
    variants = experiment.GRIDS["cascade29"]["variants"]
    assert set(variants) == {"dag-29", "resolution-label-29", "hidden-label-29", "both-label-29"}
    for flags in variants.values():
        assert flags[flags.index("--max-nodes") + 1] == "29"
    assert all("--merge" not in f for n, f in variants.items() if n != "dag-29")


def test_cascadeinit_grid_variants():
    variants = experiment.GRIDS["cascadeinit"]["variants"]
    assert len(variants) == 6
    for flags in variants.values():
        assert flags[flags.index("--max-nodes") + 1] == "19"
        assert "--leaf" in flags and flags[flags.index("--leaf") + 1] == "classifier"
    copy = variants["both-classifier-copy-settle"]
    assert "--leaf-init" in copy and "--settle-epochs" in copy and "--bump-cooldown" in copy
