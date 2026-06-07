Latent prediction folder
========================

Case folders:

- `cylinder/`: 10-to-10 non-autoregressive latent prediction scripts,
  checkpoints, latest result summaries, and sample-folder figures.
- `kdv/`: copied KDV latent scripts, selected latent checkpoints, and seven-row
  figure outputs.
- `wave/`: copied wave latent scripts, selected latent checkpoints, figure
  outputs, and inference timing helper.
- `ns/`: NS latent scripts, selected d150 latent checkpoints, and copied
  seven-row sample folders.

The selected latent checkpoints are named:

- `FOAE_d{dim}_{case}_latent.pt`
- `OAE_d{dim}_{case}_latent.pt`
- `AE_d{dim}_{case}_latent.pt`

Current caveat
--------------

KDV and wave result folders were copied as working artifacts from the previous
locations. Their plotting scripts may still reference legacy folder names for
model-class imports. The consolidated data/checkpoints are present here, but
full script cleanup should be done case by case before rerunning them.
