import torch
import torch.nn as nn
import time
import numpy as np
from torch import fft

##############################################
# Functions used for training ae
################################################
MSELoss = nn.MSELoss(reduction='mean')
def loss_function(rec, data):
    loss = MSELoss(rec, data)
    return loss

def train_epoch(model, data, optimizer, device):
    start_epoch_time = time.time()

    loss_batch = []

    for batch in data:
        if not batch.is_cuda:
            batch = batch.to(device, non_blocking=True)

        rec = model(batch)
        loss = loss_function(rec, batch)

        loss_batch.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    return sum(loss_batch) / len(loss_batch),\
            time.time() - start_epoch_time,

def test_epoch(model, data, device):
    start_epoch_time = time.time()

    with torch.no_grad():
        loss_batch = []

        for batch in data:
            if not batch.is_cuda:
                batch = batch.to(device, non_blocking=True)

            rec = model(batch)
            loss = loss_function(rec, batch)

            loss_batch.append(loss.item())

    return  sum(loss_batch) / len(loss_batch),\
            time.time() - start_epoch_time


def printProgress(epoch, epochs, loss, loss_test, elapsed, elapsed_test):
    print(f"Epoch: {epoch:3d}/{epochs:d}, Loss: {loss:2.4e}, Loss_test: {loss_test:2.4e}, time train: {elapsed:2.3f}, time test: {elapsed_test:2.3f}")
