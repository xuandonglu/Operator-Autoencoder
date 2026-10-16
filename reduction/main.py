import time
import torch
import argparse
from pathlib import Path
from runners import aeRunner

parser = argparse.ArgumentParser()
args = parser.parse_args()
device = ('cuda' if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    datafile = Path(__file__).resolve().parent / 'data' / 'data_wave.h5df'
    ae = aeRunner(device, datafile)
    # time.sleep(1100)
    ae.train()
