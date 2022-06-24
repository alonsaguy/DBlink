# Imports
from DataHandlers import *
from torch.utils.data import DataLoader
from NN_model import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

class LSTM_Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, batch_size, patience, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.patience = patience
        self.device = device

        self.tv_loss = TVLoss(1e-3)
        self.lam = 1e-2 #ratio between MSE loss and optical flow loss

    def fit(self, dl_train, dl_test, num_epochs, early_stopping=10, print_every=1, **kw):

        train_loss, val_loss = [], []
        best_loss = None
        epochs_without_improvement = 0

        for epoch in range(1, num_epochs + 1):
            print('--- EPOCH {}/{} ---'.format(epoch, num_epochs))

            loss = self.train_epoch(dl_train, **kw)
            train_loss.append(loss)

            loss = self.test_epoch(dl_test, **kw)
            val_loss.append(loss)

            self.scheduler.step(loss)

            if (epoch == 1):
                best_loss = loss
            else:
                if (loss >= best_loss):
                    epochs_without_improvement += 1
                    if (epochs_without_improvement > early_stopping):
                        print("Reached early stopping criterion")
                        self.model.load_state_dict(torch.load('best_model'))
                        break
                else:
                    epochs_without_improvement = 0
                    best_loss = loss
                    torch.save(self.model.state_dict(), 'best_model')

            if epoch % print_every == 0 or epoch == num_epochs - 1:
                print("Train loss =", train_loss[-1])
                print("Validation loss =", val_loss[-1])

    def train_epoch(self, dl_train, **kw):
        self.model.train()
        total_loss = 0
        cnt = 0
        for X_train, y_train in tqdm(dl_train):
            X_train = X_train.to(self.device)
            y_train = y_train.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            out, _ = self.model(X_train, torch.flip(X_train, dims=[1]))

            # Compute Loss
            loss = self.loss_fn(out, y_train) + self.lam * consistency_reg(out) + self.tv_loss(out)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            cnt += 1

        return total_loss / cnt

    def test_epoch(self, dl_test, **kw):
        self.model.eval()
        total_loss = 0
        cnt = 0

        for X_test, y_test in tqdm(dl_test):
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            out, _ = self.model(X_test, torch.flip(X_test, dims=[1]))

            # Compute Loss
            loss = self.loss_fn(out, y_test) + self.lam * consistency_reg(out) + self.tv_loss(out)

            total_loss += loss.item()
            cnt += 1

        return total_loss / cnt
