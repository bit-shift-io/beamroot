import re

from beamroot.models.graph import RouterGraph
from beamroot.report import draw_graph, layout, main, node_labels
from beamroot.train import parse_args, run


def tiny_run(tmp_path, *extra):
    return run(parse_args(["--dataset", "synthetic", "--model", "dag", "--device", "cpu",
                           "--epochs", "2", "--limit-batches", "2", "--runs-dir", str(tmp_path),
                           *extra]))["run_dir"]


def balanced_tree(depth):
    g = RouterGraph()
    g.add_node("n0", "router")
    level, count = ["n0"], 1
    for d in range(depth):
        nxt = []
        for p in level:
            kind = "leaf" if d == depth - 1 else "router"
            for _ in range(2):
                name = f"n{count}"
                count += 1
                g.add_node(name, kind, parents=[p])
                nxt.append(name)
        level = nxt
    return g


def test_report_cli_writes_graph_png_and_self_contained_html(tmp_path):
    run_dir = tiny_run(tmp_path, "--no-report")
    assert not (run_dir / "report.html").exists()
    main([str(run_dir), "--device", "cpu", "--samples", "3", "--limit-batches", "1"])
    assert (run_dir / "graph.png").read_bytes().startswith(b"\x89PNG")
    html = (run_dir / "report.html").read_text()
    for section in ("Accuracy", "Router graph", "Sample paths", "Per-node class histogram"):
        assert section in html
    assert "beam 3" in html and "soft" in html
    assert html.count('alt="sample ') == 3
    assert "-&gt;" in html
    srcs = re.findall(r'src="([^"]*)"', html)
    assert srcs and all(s.startswith("data:image/png;base64,") for s in srcs)
    assert "http" not in html


def test_train_writes_report_automatically_unless_no_report(tmp_path):
    run_dir = tiny_run(tmp_path)
    assert (run_dir / "report.html").exists() and (run_dir / "graph.png").exists()


def test_baseline_run_skips_report(tmp_path):
    res = run(parse_args(["--dataset", "synthetic", "--device", "cpu", "--epochs", "1",
                          "--limit-batches", "1", "--runs-dir", str(tmp_path)]))
    assert not (res["run_dir"] / "report.html").exists()


def test_layout_puts_rows_by_depth_and_spreads_within_row():
    g = balanced_tree(2)
    pos = layout(g)
    assert [pos[n][1] for n in ("n0", "n1", "n3")] == [0, -1, -2]
    xs = sorted(pos[n][0] for n in g.leaves())
    assert len(set(xs)) == 4 and min(xs) > 0 and max(xs) < 1


def test_node_labels_show_dominant_class_and_purity():
    g = balanced_tree(1)
    mass = {"n0": [6.0, 2.0], "n1": [3.0, 1.0], "n2": [0.0, 0.0]}
    labels = node_labels(g, mass)
    assert "n1" in labels["n1"] and "0 75%" in labels["n1"]
    assert "- 0%" in labels["n2"]


def test_diagram_renders_31_nodes(tmp_path):
    g = balanced_tree(4)
    assert len(g.nodes) == 31
    mass = {n: [1.0, 3.0] for n in g.nodes}
    out = tmp_path / "g31.png"
    data = draw_graph(g, {e: 0.5 for e in g.edges}, node_labels(g, mass), out)
    assert data.startswith(b"\x89PNG") and out.exists()


def test_report_shows_baseline_row_and_group_purity(tmp_path):
    common = ["--dataset", "synthetic_cifar", "--device", "cpu", "--epochs", "1",
              "--limit-batches", "2", "--runs-dir", str(tmp_path)]
    dag = run(parse_args(["--model", "dag", *common, "--no-report"]))["run_dir"]
    base = run(parse_args(["--model", "baseline", *common]))
    main([str(dag), "--device", "cpu", "--limit-batches", "1", "--baseline", str(base["run_dir"])])
    html = (dag / "report.html").read_text()
    assert "baseline (" in html and f"{base['test_acc']:.4f}" in html
    assert "group purity" in html and ("animal" in html or "vehicle" in html)
