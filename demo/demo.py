# imports
import os

import torch
from DataHandlers import *
from NN_model import *
from Trainers import *
from Utils import *
from demo_exp_params import *
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device", device)

####### Step I - Parameter Initialization #######
# Run flags
GenerateTrainData = False
GenerateTestData = False
TrainNetFlag = False
TestOnRealData = False

path = r'./' # Path to model
model_name = 'LSTM_model' # Model name
scale = 4 # Scale factor, the size of the reconstructed image pixels
sum_factor = 10 # The temporal window size DBlink uses to sum localizations
pixel_size = 160 # Camera pixel size - relevant for experimental data
simulated_video_length = 3000 # Length of simulated video - relevant for simulated data generation
density = 0.002 # Blinking density (percentage out of the number of non-zero pixels in the simulated structure)
img_size = 32 # Simulated image size - relevant for simulated data generation

####### Step II - Training data generation #######
trainset_size = 1024
valset_size = 256
if(TrainNetFlag):
    if(GenerateTrainData):
        [X_train, y_train] = Simulate_Train_Data_060622(obs_size=img_size, dataset_size=trainset_size,
                                                        video_length=simulated_video_length, emitters_density=density,
                                                         scale=scale, sum_range=sum_factor, datatype='tubules')
        [X_val, y_val] = Simulate_Train_Data_060622(obs_size=img_size, dataset_size=valset_size,
                                                    video_length=simulated_video_length, emitters_density=density,
                                                    scale=scale, sum_range=sum_factor, datatype='tubules')
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)

        torch.save(X_train, 'X_train')
        torch.save(y_train, 'y_train')
        torch.save(X_val, 'X_val')
        torch.save(y_val, 'y_val')
    else:
        X_train = torch.load('X_train')
        y_train = torch.load('y_train')
        X_val = torch.load('X_val')
        y_val = torch.load('y_val')

####### Step III - Build Model, loss and optimizer #######
num_layers = 2 # The number of LSTM layers
hidden_channels = 4 # The hidden layer number of channels
lr = 1e-4 # Training learning rate
window_size = 25 # The number of used windows (in each direction) for the inference of each reconstructed frame
betas = (0.99, 0.999) # Parameters of Adam optimizer
batch_size = 1
epochs = 150
patience = 3

model = ConvOverlapBLSTM(input_size=(img_size, img_size), input_channels=1, hidden_channels=hidden_channels, num_layers=num_layers, device=device).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, min_lr=1e-9, verbose=True)

####### Step IV - Training the model #######
if(TrainNetFlag):
    dl_train = CreateDataLoader(X_train, y_train, batch_size=batch_size)
    dl_val = CreateDataLoader(X_val, y_val, batch_size=batch_size)

    trainer = LSTM_overlap_Trainer(model, criterion, optimizer, scheduler, batch_size, window_size=window_size,
                                   vid_length=X_train.shape[1], patience=patience, device=device)
    trainer.fit(dl_train, dl_val, num_epochs=epochs)
    torch.save(model.state_dict(), model_name)
else:
    model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))

####### Step V - Testing the model #######
if(TestOnRealData):
    analyze_storm_exp_overlap(path_to_model='./{}'.format(model_name),
                              exp_class=demo_params(),
                              hidden_channels=hidden_channels,
                              num_layers=num_layers,
                              scale=scale,
                              device=device)
    post_process_results(r'./tmp_results', 1)
else:
    testset_size = 4
    if(GenerateTestData):
        [X_test, y_test] = Simulate_Train_Data_060622(obs_size=img_size, dataset_size=testset_size,
                                                      video_length=simulated_video_length, emitters_density=density,
                                                      scale=scale, sum_range=sum_factor, datatype='tubules')
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        torch.save(X_test, 'X_test')
        torch.save(y_test, 'y_test')
    else:
        X_test = torch.load('X_test')
        y_test = torch.load('y_test')

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    N, T, C, H, W = X_test.shape

    model = ConvOverlapBLSTM(input_size=(img_size, img_size), input_channels=1, hidden_channels=hidden_channels,
                                num_layers=num_layers, device=device).to(device)
    model.load_state_dict(torch.load(os.path.join(path, model_name), map_location=torch.device(device)))

    down = torch.zeros(X_test.size(1), requires_grad=False, dtype=torch.int)
    up = torch.zeros(X_test.size(1), requires_grad=False, dtype=torch.int)
    out_ind = torch.zeros(X_test.size(1), requires_grad=False, dtype=torch.int)
    for i in range(X_test.size(1)):
        down[i] = torch.max(torch.IntTensor([0, i - sum_factor * window_size]))
        up[i] = torch.min(torch.IntTensor([X_test.size(1), i + sum_factor * window_size]))
        out_ind[i] = i - down[i]

    for i in range(X_test.size(0)):
        out = []
        print('Forward pass through the network')
        with torch.no_grad():
            for j in tqdm(range(X_test.shape[1])):
                curr_out = model(X_test[i:i + 1, down[j]:up[j]:sum_factor],
                                 torch.flip(X_test[i:i + 1, down[j]:up[j]:sum_factor], dims=[1]))
                curr_out = curr_out.detach().cpu()[0, int(out_ind[j] / sum_factor)]
                out.append(curr_out)

        out = torch.stack(out, dim=1)

        curr_vid = np.zeros([1, X_test.size(1), C, H, W])
        for j in tqdm(range(X_test.size(1))):
            curr_vid[0, j] = 255 * normalize_input_01(out[0, j].numpy())

        np.save('tmp_results/np_vid_{}'.format(i + 1), curr_vid[0, :-2*window_size])
        np.save('tmp_results/gt_vid_{}'.format(i + 1), y_test[i, :-2*window_size].detach().cpu().numpy())

        print("-I- Completed vid", i + 1)

        # Post process reconstruction and generate output video
        post_process_results(r'./tmp_results', i + 1)
