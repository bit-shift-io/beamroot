Status: pending
Complexity: 5 (design 1, scope 2, tests 1, hazards 1)
MCP: none

# Graph diagram and HTML report

## What to build
- `python -m beamroot.report <run-dir>` writes `graph.png` and `report.html`.
- Diagram: nodes laid out by depth, edges labelled with mean routing probability, nodes labelled with dominant class and purity.
- Report: accuracy table from evaluate, sample images with their top-N paths, per-node class histogram.

## Files to create/modify
- beamroot/report.py
- beamroot/analysis.py (per-node class mix, purity)
- beamroot/templates/report.html
- tests/test_analysis.py
- tests/test_report.py

## Test approach
- Purity and class mix are correct on a hand-built graph with synthetic data.
- Report generation on a tiny fake run writes both files; HTML contains expected sections.

## Acceptance criteria
- [ ] Report is a single self-contained HTML file (images embedded as base64).
- [ ] Train CLI writes the report automatically at the end of a run (`--no-report` skips).
- [ ] Diagram is legible for up to 31 nodes.

## Blocked by
03, 05

## Gotchas
- Use networkx + matplotlib for the diagram; do not require the Graphviz system binary.
- Set matplotlib to the `Agg` backend so it runs headless.
