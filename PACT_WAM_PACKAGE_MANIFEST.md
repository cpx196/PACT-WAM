# PACT-WAM package manifest

This directory is a local GitHub-ready package assembled from the current
`FastWAM` working tree. It contains the current uncommitted source changes,
configuration files, experiment scripts, project notes, and lightweight
historical evaluation metadata.

## Included

- `src/fastwam/`: original FastWAM model and scheduler code, including the
  current JEPA-guided/action-flow changes.
- `configs/`: training, LIBERO, RoboTwin, and model configuration files.
- `experiments/`, `scripts/`, and `tests/`: rollout, experiment, deployment,
  and verification code.
- Root and `experiments/libero/` Markdown reports documenting the project
  history and current settings.
- `artifacts/evaluate_results/`: historical JSON, CSV, YAML, Markdown, and log
  metadata copied from local evaluation runs. The archive excludes rollout MP4s
  and transient worker-pool files.

## Deliberately excluded

- `checkpoints/`: model weights and split checkpoint parts.
- Rollout and predicted-video MP4 files.
- Python caches, virtual environments, local credentials, and temporary files.

The excluded files remain in the original `/data/chenpengxu/FastWAM` workspace;
the source project was not modified or deleted during packaging.

## Current package identity

- Project name: `pact-wam`
- Display name: `PACT-WAM`
- Source lineage: FastWAM + current action-centric/JEPA-guided experiments
- GitHub upload status: not pushed yet; this is the local packaging stage.
