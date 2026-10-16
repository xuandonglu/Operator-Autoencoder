"""
Runners for the ae and temporal-dynamic prediction in latent space 
@yuningw
"""

import os 
import time
from pathlib import Path
import h5py
import numpy as np
import torch 
from torch import nn
from train import train_epoch, test_epoch, printProgress
from datas import loadData, get_ae_DataLoader

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

####################################################
### RUNNER for AE
####################################################
# torch.manual_seed(0)
# np.random.seed(0)

class aeRunner(nn.Module):
    def __init__(self, device, datafile) -> None:

        from configs import AE_config as cfg
        from nomenclature import Name_AE

        super(aeRunner, self).__init__()

        self.config = cfg
        self.filename = Name_AE(self.config)
        self.datafile = datafile
        import importlib
        model = getattr(importlib.import_module('nn_wave'), self.config.model)
        self.input_dim = loadData(self.datafile).shape[-1]
        self.model = model(self.input_dim, self.config.latent_dim, self.config.pattern)

        self.device = device
        self.model.to(device)
        self.fmat = '.pth'

        print(f"INIT ae, device: {device}")
        print(f"Case Name:\n {self.filename}")

#-------------------------------------------------
    def train(self):
        print("#"*30)
        print("INFO: Start Training ")
        self.get_data()
        self.compile()
        self.fit()
        self.train_dl   = None
        self.val_dl     = None
        print(f"INFO: Training finished, cleaned the data loader")
        print("#"*30)

#-------------------------------------------------

    def get_data(self):

        u_scaled = loadData(self.datafile)
        u_scaled = u_scaled[::self.config.downsample]
        n_total = u_scaled.shape[0]
        self.n_train = n_total - self.config.n_test
        print(f"INFO: Data Summary: N train: {self.n_train:d}," + \
                f"N test: {self.config.n_test:d},"+\
                f"N total {n_total:d}")

        self.train_dl, self.val_dl = get_ae_DataLoader(    d_train=u_scaled,
                                                            n_train=self.n_train,
                                                            device= 'cpu',
                                                            batch_size= self.config.batch_size)
        print( f"INFO: Dataloader generated, Num train batch = {len(self.train_dl)} \n" +\
                f"Num val batch = {len(self.val_dl)}")

#-------------------------------------------------
    def compile(self):

        from torch.optim import lr_scheduler

        print("#"*30)
        print(f"INFO: Start Compiling")

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=0)

        self.opt_sch = lr_scheduler.OneCycleLR(self.opt,
                                            max_lr=self.config.lr,
                                            total_steps=self.config.epochs,
                                            div_factor=2,
                                            final_div_factor=self.config.lr/self.config.lr_end,
                                            pct_start=0.2)

        print(f"INFO: Compiling Finished!")

#-------------------------------------------------
    def fit(self):

        print(f"Training {self.filename}")
        bestloss = 1e6
        loss = 1e6
        model_dir = Path(__file__).resolve().parent / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        ckp_file = model_dir / (self.filename + self.fmat)
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            loss, elapsed = train_epoch(model=self.model,
                                        data=self.train_dl,
                                        optimizer=self.opt,
                                        device=self.device)
            self.model.eval()
            loss_test, elapsed_test = test_epoch(model=self.model,
                                                 data=self.val_dl,
                                                 device=self.device)

            self.opt_sch.step()

            printProgress(epoch=epoch,
                          epochs=self.config.epochs,
                          loss=loss,
                          loss_test=loss_test,
                          elapsed=elapsed,
                          elapsed_test=elapsed_test)

            if (loss_test < bestloss and epoch > 100):
                bestloss = loss_test
                torch.save(self.model, ckp_file)
                print(f'## Checkpoint. Epoch: {epoch}, test loss: {loss_test}, saving {ckp_file}')
        torch.save(self.model, ckp_file)
        print(f'Checkpoint. Final epoch, loss: {loss}, test loss: {loss_test}, saving {ckp_file}')
