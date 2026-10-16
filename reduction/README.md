# Reduction

`nn_<case>.py` defines OAE and F-OAE networks, and `nn_ae_<case>.py` defines
the conventional AE baseline.

`main.py`, `configs.py`, `runners.py`, and `train.py` implement reduction
training. The current training entry point selects the wave case.

The Python scripts in `models/` perform reconstruction evaluation and
latent extraction. `checkpoint_utils.py` supports historical checkpoint
class names.
