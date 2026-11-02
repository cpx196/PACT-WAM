# Hosting the evaluated predictor on Hugging Face

Keep source and result reports on GitHub, and host checkpoint binaries in a Hugging Face model repository. A private repository is the default for the upload helper. The helper does not change weights or convert precision, and does not upload WAM or JEPA encoder checkpoints.

Evaluated checkpoint on the training server:

```
/data/pxchen/PACT-WAM/artifacts/fastwam_predictor21_ar2_full10epoch_bs16_20261014/step19690.pt
```

Expected SHA256: `dec6eef937335a04c9fd8e23d7065163d2348919041e548749d8fe132313289e`.

The predictor has approximately 304M parameters (roughly 1.2 GB of FP32 parameters alone); original checkpoint byte size depends on its saved payload and must be measured. The helper retains the original checkpoint to preserve the hash expected by existing offline calibration tables.

On the training server, authenticate interactively with write access to your own namespace:

```bash
hf auth login
```

After placing the current `scripts/publish_predictor_hf.py` in the server checkout, validate without uploading:

```bash
/data/pxchen/envs/jepa-wam/bin/python scripts/publish_predictor_hf.py \
  --repo-id YOUR_HF_ACCOUNT/PACT-WAM-AR2-Predictor \
  --include-weights --dry-run
```

Upload to a private model repository:

```bash
/data/pxchen/envs/jepa-wam/bin/python scripts/publish_predictor_hf.py \
  --repo-id YOUR_HF_ACCOUNT/PACT-WAM-AR2-Predictor \
  --include-weights
```

This uploads the exact predictor, training-only normalization, latent input contract, predictor source, a model card and SHA256 manifest. `--include-weights` also uploads all ten task weight files. Existing repository visibility must match the requested visibility. No HF repository is created merely by preparing this code.

This is a custom action-conditioned JEPA predictor, not a standalone Transformers AutoModel. Use the supplied architecture and the PACT-WAM environment to load the downloaded checkpoint. The architecture source retains project-specific paths; integration with a portable Hub-native loader is separate work.

Official references: [authentication](https://huggingface.co/docs/huggingface_hub/en/quick-start), [uploading files](https://huggingface.co/docs/huggingface_hub/en/guides/upload).
