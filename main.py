# imports
import torch
from DataHandlers import *
from NN_model import *
from Trainers import *
from Utils import *
from exp_params import *
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Using device", device)
# Run flags
GenerateTrainData = False
GenerateTestData = False
TrainNetFlag = False
TestOnRealData = False

# Parameter Initialization
exp_class = drift_params()
model_name = exp_class.model_name
scale = 4
simulated_video_length = 1000
density = 0.002
input_size = 32 #exp_class.img_size

# Training data generation
if(GenerateTrainData):
    [X_train, y_train] = Simulate_Train_Data_060622(obs_size=input_size, dataset_size=1024,
                                                    video_length=simulated_video_length, emitters_density=density,
                                                     scale=scale, sum_range=10, datatype='tubules')
    [X_val, y_val] = Simulate_Train_Data_060622(obs_size=input_size, dataset_size=256,
                                                video_length=simulated_video_length, emitters_density=density,
                                                scale=scale, sum_range=10, datatype='tubules')
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

# Build Model, loss and optimizer
num_layers = 2
hidden_channels = 4
lr = 1e-4
betas = (0.99, 0.999)
batch_size = 1
epochs = 150
patience = 3

model = ConvBLSTM(input_size=(input_size, input_size), input_channels=1, hidden_channels=hidden_channels, num_layers=num_layers, device=device).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, min_lr=1e-9, verbose=True)

# Train the model
if(TrainNetFlag):
    dl_train = CreateDataLoader(X_train, y_train, batch_size=batch_size)
    dl_val = CreateDataLoader(X_val, y_val, batch_size=batch_size)

    trainer = LSTM_Trainer(model, criterion, optimizer, scheduler, batch_size, patience, device)
    trainer.fit(dl_train, dl_val, num_epochs=epochs)
    torch.save(model.state_dict(), model_name)
else:
    model.load_state_dict(torch.load(model_name, map_location=torch.device(device)))

if(TestOnRealData):
    # Test on experimental data
    analyze_storm_exp(path_to_model='./{}'.format(model_name),
                      exp_class=exp_class,
                      hidden_channels=hidden_channels,
                      num_layers=num_layers,
                      scale=scale,
                      device=device)
    # Generate output video
    post_process_results(r'./tmp_results', 1)
else:
    # Testing data generation
    if(GenerateTestData):
        [X_test, y_test] = Simulate_Train_Data_060622(obs_size=input_size, dataset_size=4,
                                                      video_length=simulated_video_length, emitters_density=density,
                                                      scale=scale, sum_range=10, datatype='tubules')
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        torch.save(X_test, 'X_test')
        torch.save(y_test, 'y_test')
    else:
        X_test = torch.load('X_test')
        y_test = torch.load('y_test')

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Generate video and calculate accuracy
    N, T, C, H, W = X_test.shape
    for i in range(N):
        # Inference
        out, _ = model(X_test[i:i+1], torch.flip(X_test[i:i+1], dims=[1]))

        # Accuracy calculation
        curr_vid = np.zeros([1, T, C, H, W])
        for j in range(T):
            curr_vid[0, j] = 255 * normalize_input_01(out[0, j].detach().cpu().numpy())
            calc_acc(curr_vid[0, j, 0], y_test[i, j, 0])

        # Save npy files -> these files are used for video generation
        np.save('tmp_results/np_vid_{}'.format(i + 1), curr_vid[0])
        np.save('tmp_results/gt_vid_{}'.format(i + 1), y_test[i].cpu().data.numpy())

        print("-I- Completed vid", i + 1)

        # Post process reconstruction and generate output video
        post_process_results(r'./tmp_results', i + 1)
