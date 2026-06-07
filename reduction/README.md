Reduction folder
================

Layout:

- `data/`: case HDF5 files, named `data_*.h5df`.
- `nn_*.py`: case-specific reduction model definitions.
- `models/`: trained reduction checkpoints, load-test scripts, and
  latent-output scripts.

Data files
----------

- `data_cylinder.h5df`: cylinder wake data and cylinder latent/trunk datasets.
- `data_kdv.h5df`: KDV full-field data plus OAE/FOAE/AE latent outputs.
- `data_wave.h5df`: wave full-field data plus OAE/FOAE/AE latent outputs.

Selected reduction checkpoints
------------------------------

- Cylinder: `FOAE_d140_cylinder.pth`, `OAE_d140_cylinder.pth`,
  `AE_d140_cylinder.pth`.
- KDV: `FOAE_d50_kdv.pth`, `OAE_d50_kdv.pth`, `AE_d50_kdv.pth`.
- Wave: `FOAE_d30_wave.pth`, `OAE_d30_wave.pth`, `AE_d30_wave.pth`.
- NS: `FOAE_d150_ns.pth`, `OAE_d150_ns.pth`, `AE_d150_ns.pth`.
  Optional AE checkpoints `AE_d100_ns.pth` and `AE_d120_ns.pth` are also kept
  because they were present in the cleaned NS folder.

NS data-dependent scripts use `data_ns.h5df` as the case data entry.
