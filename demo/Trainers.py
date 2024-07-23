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

            out = self.model(X_train, torch.flip(X_train, dims=[1]))
            if len(out) == 2:
              out = out[0]
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

        with torch.no_grad():
            for X_test, y_test in tqdm(dl_test):
                X_test = X_test.to(self.device)
                y_test = y_test.to(self.device)

                out = self.model(X_test, torch.flip(X_test, dims=[1]))

                # Compute Loss
                loss = self.loss_fn(out, y_test) + self.lam * consistency_reg(out) + self.tv_loss(out)

                total_loss += loss.item()
                cnt += 1

        return total_loss / cnt

class LSTM_overlap_Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, batch_size, window_size, vid_length, patience, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.window_size = window_size
        self.patience = patience
        self.device = device
        self.down = torch.zeros(vid_length, requires_grad=False, dtype=torch.int).to(device)
        self.up = torch.zeros(vid_length, requires_grad=False, dtype=torch.int).to(device)
        self.out_ind = torch.zeros(vid_length, requires_grad=False, dtype=torch.int).to(device)
        for i in range(vid_length):
            self.down[i] = torch.max(torch.IntTensor([0, i-window_size]))
            self.up[i] = torch.min(torch.IntTensor([vid_length, i+window_size]))
            self.out_ind[i] = i-self.down[i]
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

            prev_out = self.model(X_train[:, self.down[0]:self.up[0]],
                                      torch.flip(X_train[:, self.down[0]:self.up[0]], dims=[1]),
                                  out_ind=self.out_ind[0])

            for j in range(X_train.shape[1]):
                # Forward pass
                self.optimizer.zero_grad()

                out = self.model(X_train[:, self.down[j]:self.up[j]],
                                      torch.flip(X_train[:, self.down[j]:self.up[j]], dims=[1]),
                                 out_ind=self.out_ind[j])

                # Compute Loss
                consistency_loss = torch.sum(torch.abs(out - prev_out.detach()))
                loss = self.loss_fn(out, y_train[:, j]) + self.lam * consistency_loss + self.tv_loss(out[None, :, :, :, :])

                # Backward pass
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                prev_out = out

            cnt += 1

        return total_loss / (cnt * self.batch_size * X_train.shape[1])

    def test_epoch(self, dl_test, **kw):
        self.model.eval()
        total_loss = 0
        cnt = 0
        with torch.no_grad():
            for X_test, y_test in tqdm(dl_test):
                X_test = X_test.to(self.device)
                y_test = y_test.to(self.device)

                prev_out = self.model(X_test[:, self.down[0]:self.up[0]],
                                      torch.flip(X_test[:, self.down[0]:self.up[0]], dims=[1]),
                                      out_ind=self.out_ind[0])

                for j in range(X_test.shape[1]):
                    out = self.model(X_test[:, self.down[j]:self.up[j]],
                                     torch.flip(X_test[:, self.down[j]:self.up[j]], dims=[1]),
                                     out_ind=self.out_ind[j])

                    # Compute Loss
                    consistency_loss = torch.sum(torch.abs(out - prev_out.detach()))
                    loss = self.loss_fn(out, y_test[:, j]) + self.lam * consistency_loss + self.tv_loss(out[None, :, :, :, :])

                    total_loss += loss.item()
                    prev_out = out
                cnt += 1

        return total_loss / (cnt * self.batch_size * X_test.shape[1])

class LSTM_One_Directional_Trainer:
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

            out = self.model(X_train)

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

            out = self.model(X_test)

            # Compute Loss
            loss = self.loss_fn(out, y_test) + self.lam * consistency_reg(out) + self.tv_loss(out)

            total_loss += loss.item()
            cnt += 1

        return total_loss / cnt
