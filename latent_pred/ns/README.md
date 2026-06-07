NS case
=======

The NS case is consolidated under the common OAE layout.

Contents:

- `figures/generate_ns2d_sample_folders.py`
- `sample_folders/`
- `models/FOAE_d150_ns_latent.pt`
- `models/OAE_d150_ns_latent.pt`
- `models/AE_d150_ns_latent.pt`
- `latent.py`, `latent_test.py`, and AE-side latent scripts

The reduction checkpoints are already mapped in `../../reduction/models`:

- `FOAE_d150_ns.pth`
- `OAE_d150_ns.pth`
- `AE_d150_ns.pth`

Data-dependent scripts use `../../reduction/data/data_ns.h5df`.
