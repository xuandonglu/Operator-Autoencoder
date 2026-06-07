OAE consolidated case workspace
================================

This folder collects the current OAE / FOAE / AE case assets under one
project-level layout:

- `reduction/`: full-field data, reduction network definitions, trained
  reduction checkpoints, load-test scripts, and latent-output scripts.
- `latent_pred/`: case-specific latent prediction scripts, latent checkpoints,
  result summaries, and figure-generation folders.

Large HDF5 data files were moved into `reduction/data` rather than copied.
The original legacy data paths therefore no longer contain those files.

Current case status
-------------------

| case | status | notes |
| --- | --- | --- |
| cylinder | available | data, OAE/FOAE/AE reduction models, latent models, and sample folders are consolidated. |
| kdv | available | data and selected d50 reduction/latent models are consolidated; figure folders were copied. |
| wave | available | data and selected d30 reduction/latent models are consolidated; figure folders and timing script were copied. |
| ns | available | reduction/latent models and figure folders are consolidated. |

Naming convention
-----------------

Reduction checkpoints use:

`{METHOD}_d{TOTAL_LATENT_DIM}_{CASE}.pth`

Examples:

- `FOAE_d140_cylinder.pth`
- `OAE_d140_cylinder.pth`
- `AE_d140_cylinder.pth`
- `FOAE_d50_kdv.pth`
- `OAE_d30_wave.pth`
- `FOAE_d150_ns.pth`
- `OAE_d150_ns.pth`
- `AE_d150_ns.pth`

Latent-prediction checkpoints use the same prefix plus `_latent.pt`, for
example `FOAE_d140_cylinder_latent.pt`.

HDF5 dataset keys
-----------------

The HDF5 files still keep the legacy dataset keys such as
`latent_DPN_d80_fno_cylinder`, `trunk_DPN_d80_cylinder`, or
`latent_fno_d50_m20_w1`. This preserves compatibility with the already trained
latent-prediction scripts and plotting code.

Known compatibility note
------------------------

Some copied KDV/Wave plotting scripts still contain legacy import assumptions
from their original folders. The data and selected checkpoints are now in this
folder, but if those scripts are run again they may need a small path/import
cleanup for the pickled model classes. Cylinder has been adjusted first because
it is the most recent active 2D case.
