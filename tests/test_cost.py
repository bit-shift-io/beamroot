import json

import torch
from torch import nn

from beamroot.cost import model_costs, module_macs
from beamroot.evaluate import evaluate_modes, main
from beamroot.models.cascade import CascadeModel
from beamroot.models.dag import DAGRouterModel
from beamroot.train import parse_args, run


def test_conv_and_linear_macs_match_hand_computation():
    conv = nn.Conv2d(2, 4, 3, padding=1)   # 8x8 out: 4*8*8 outputs * (2*3*3)
    assert module_macs(conv, torch.zeros(1, 2, 8, 8)) == 4 * 8 * 8 * 18
    assert module_macs(nn.Linear(5, 7), torch.zeros(3, 5)) == 35


def test_dag_stem_is_about_3_8m_and_counted_once_per_image():
    model = DAGRouterModel().eval()
    stem, nodes = model_costs(model)
    assert 3.5e6 < stem < 4.1e6
    batches = [(torch.randn(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))]
    rows = evaluate_modes(model, batches, "cpu", beams=(1,))
    beam1 = rows[0]
    assert beam1["mean_macs"] > stem
    soft = rows[-1]
    assert soft["mean_macs"] == stem + sum(nodes.values())


def test_cascade_macs_fall_at_lower_steps():
    batches = [(torch.randn(4, 1, 28, 28), torch.zeros(4, dtype=torch.long))]
    macs = {}
    for step in (4, 1):
        m = CascadeModel(start_step=step).eval()
        rows = evaluate_modes(m, batches, "cpu", beams=(1,))
        macs[step] = rows[0]["mean_macs"]
        assert model_costs(m)[0] == 0
    assert macs[1] < macs[4]


def test_timing_present_and_eval_json(tmp_path):
    run(parse_args(["--dataset", "synthetic", "--device", "cpu", "--limit-batches", "2",
                    "--runs-dir", str(tmp_path), "--no-report"]))
    rows = main([str(next(tmp_path.iterdir())), "--device", "cpu", "--limit-batches", "1"])
    on_disk = json.loads((next(tmp_path.iterdir()) / "eval.json").read_text())
    for r in on_disk:
        assert r["ms_per_image"] > 0 and r["mean_macs"] > 0
    assert len(rows) == len(on_disk)
