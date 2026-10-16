# Operator Autoencoder

Code accompanying **Decoder-free operator autoencoder for reduced-order
modeling of dynamical systems**, by Xuandong Lu and Yongming Liu.

Machine Learning for Computational Science and Engineering, 2, 37 (2026).
[Paper](https://doi.org/10.1007/s44379-026-00087-7).

The implementation includes conventional autoencoders (AE), operator
autoencoders (OAE), and Fourier-enhanced operator autoencoders (F-OAE) for
the wave, KdV, Navier-Stokes, and cylinder wake cases.

## Installation

Use Python 3.10 or later:

```bash
pip install -r requirements.txt
```

The original training scripts target a CUDA-enabled PyTorch installation.

## Code Layout

- `reduction/nn_*.py`: reduction network architectures.
- `reduction/main.py`: reduction training entry point.
- `reduction/configs.py`: reduction training configuration.
- `reduction/models/*.py`: reconstruction evaluation and latent extraction.
- `latent_pred/<case>/latent*.py`: latent dynamics training and evaluation.
- `latent_pred/cylinder/train_latent_hist10_to10_2dfno.py`: cylinder dynamics training.
- `latent_pred/<case>/*timing*.py`: training and inference timing utilities.
- `latent_pred/<case>/figures/*.py`: plotting utilities.

Run scripts from their respective case folders. Experiment settings remain
at the top of each script.

## Citation

```bibtex
@article{lu2026decoder,
  title={Decoder-free operator autoencoder for reduced-order modeling of dynamical systems},
  author={Lu, Xuandong and Liu, Yongming},
  journal={Machine Learning for Computational Science and Engineering},
  volume={2},
  pages={37},
  year={2026},
  doi={10.1007/s44379-026-00087-7}
}
```
