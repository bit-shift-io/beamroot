Status: done
Complexity: 5 (design 1, scope 2, tests 1, hazards 1)
MCP: none

# CIFAR-10 and baseline comparison

## What to build
- `--dataset cifar10` works for both models, with a stem suited to 32×32 colour images.
- `--model baseline --match-params <run-dir>` sizes the CNN within 10% of that DAG run's parameter count.
- Report adds class-group purity: animals vs vehicles per node, and a baseline comparison row.

## Files to create/modify
- beamroot/data.py
- beamroot/models/dag.py (stem per dataset)
- beamroot/models/baseline.py (width search for parameter matching)
- beamroot/analysis.py (class groups)
- beamroot/report.py
- tests/test_param_match.py
- tests/test_analysis.py

## Test approach
- Param matching lands within 10% for several target sizes.
- Group purity is correct on synthetic labels.
- Smoke run on CIFAR-shaped synthetic tensors passes for both models.

## Acceptance criteria
- [ ] CIFAR-10 DAG run completes in ~20 minutes on MPS with default settings.
- [ ] Report shows group purity per node and the baseline accuracy alongside the DAG.
- [ ] Basic augmentation (random crop, flip) for CIFAR-10 training only.

## Blocked by
06

## Gotchas
- Class groups: vehicles = airplane, automobile, ship, truck; animals = the other six.
