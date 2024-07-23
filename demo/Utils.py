# Imports
from NN_model import *
from DataHandlers import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
import matplotlib.animation as animation
import torch
from PIL import Image
from csv import reader

# Useful Functions
def gauss2d(xy, sigma):
    '''
    :param xy: meshgrid of x and y positions in a patch
    :param sigma: the sigma of 2D gaussian distribution
    :return: the intensity of a symmetric 2D gaussian distribution centered at the center of the patch defined by xy
    '''
    # Fit patch to gaussian
    x, y = xy
    return np.exp(-((x - (np.sqrt(x.shape[0]) - 1) / 2) ** 2 + (y - (np.sqrt(y.shape[0]) - 1) / 2) ** 2) / (2 * sigma ** 2))

def post_process_results(path, index):
    '''
    :param path: path to npy files
    :param index: index of generated video
    :return: create a mp4 video representing the data in the npy files
    '''
    gt = np.load(os.path.join(path, 'gt_vid_{}.npy'.format(index)))
    pred = np.load(os.path.join(path, 'np_vid_{}.npy'.format(index)))

    # Create a meshgrid of xy positions in a patch
    tmp = []
    patch_size = 7
    mid = int(np.floor(patch_size / 2))
    xy = np.zeros([2, int(patch_size ** 2)])
    for i1 in range(patch_size):
        for j1 in range(patch_size):
            xy[:, int(i1 + patch_size * j1)] = [i1, j1]

    # Binarize reconstruction movie
    binary_movie = np.zeros_like(pred)
    recon_movie = np.zeros_like(pred)
    thresh = 0.5 * np.mean(pred)
    binary_movie[np.where(pred > thresh)] = 1

    # Post-process the reconstruction by multiplying patches around every pixel by a gaussian with varying sigma
    # sigma indicate the uncertainty in out prediction, therefore low intensity pixels get relatively high sigma.
    print('Post processing')
    for i in tqdm(range(gt.shape[0])):
        # convolve with gaussian based on uncertainty
        for row in range(mid, recon_movie.shape[2] - mid):
            for col in range(mid, recon_movie.shape[3] - mid):
                if (binary_movie[i, 0, row, col] == 1):
                    if (pred[i, 0, row, col] == 1):
                        uncertainty = 0.1
                    else:
                        uncertainty = np.min([0.1 / (pred[i, 0, row, col] + 1e-3), 100])
                    recon_movie[i, 0, row-mid:row+mid+1, col-mid:col+mid+1] += (1 / uncertainty) ** 2 * gauss2d(
                        xy, sigma=uncertainty).reshape([patch_size, patch_size])

        # Create normalized network reconstructions
        if (np.max(recon_movie[i, 0, :, :]) != 0):
            norm_recon = 255 * normalize_input_01(recon_movie[i, 0, :, :])
        else:
            norm_recon = recon_movie[i, 0] + 1e-9

        # Create normalized ground truth images
        if (np.max(gt[i, 0]) != 0):
            norm_gt = 255 * normalize_input_01(gt[i, 0, :, :])
        else:
            norm_gt = gt[i, 0] + 1e-9

        concat_img = np.concatenate([norm_recon, norm_gt], axis=1)
        tmp.append(concat_img)

    tmp = np.array(tmp)[None, :, None, :, :]
    create_example_vid('tmp_results/out_vid_{}'.format(index), tmp)

    print("-I- Completed vid", index)

def no_post_process_results(path, index):
    '''
    :param path: path to npy files
    :param index: index of generated video
    :return: create a mp4 video representing the data in the npy files
    '''
    gt = np.load(os.path.join(path, 'gt_vid_{}.npy'.format(index)))
    pred = np.load(os.path.join(path, 'np_vid_{}.npy'.format(index)))

    tmp = []
    # Create normalized network reconstructions
    for i in tqdm(range(gt.shape[0])):
        if (np.max(pred[i, 0, :, :]) != 0):
            norm_recon = 255 * normalize_input_01(pred[i, 0, :, :])
        else:
            norm_recon = pred[i, 0] + 1e-9

        # Create normalized ground truth images
        if (np.max(gt[i, 0]) != 0):
            norm_gt = 255 * normalize_input_01(gt[i, 0, :, :])
        else:
            norm_gt = gt[i, 0] + 1e-9

        concat_img = np.concatenate([norm_recon, norm_gt], axis=1)
        tmp.append(concat_img)

    tmp = np.array(tmp)[None, :, None, :, :]
    create_example_vid('tmp_results/out_vid_{}'.format(index), tmp)

    print("-I- Completed vid", index)

def analyze_storm_exp(path_to_model, exp_class, hidden_channels, num_layers, scale, device):
    '''
    :param path_to_model: path to neural network model
    :param exp_class: a class that contains all the relevant parameter for a specific experiment, e.g. pixel size, etc.
    :param hidden_channels: the number of model hidden channels
    :param num_layers: the number of model hidden layers
    :param scale: the scale ratio between the input image and the reconstruction
    :param device: the currently used device
    :return: None, this function generate npy files of the reconstruction and Deep-STORM localizations
    '''
    # Initialize params from exp_params class
    img_size = exp_class.img_size
    crop_loc = exp_class.crop_loc #[y, x]
    pixel_size = exp_class.pixel_size
    DS_sum = exp_class.DS_sum # how many DeepSTORM frames to sum?
    sum_factor = exp_class.sum_factor
    T = exp_class.vid_length # vid length
    dir_path = exp_class.path
    obs_TIF = exp_class.filename

    # Calculate the ratio between the frame rate of Deep-STORM and DBlink
    num_of_DS_frames = T*sum_factor/DS_sum
    # Define reconstruction intensity threshold (normalized values in [0, 1])
    output_threshold = 0.2

    # Load localization file and transfer to image data
    X_test = np.zeros([T, img_size * scale, img_size * scale])
    i = 0
    with open(os.path.join(dir_path, '{}.csv'.format(obs_TIF[:-4])), 'r') as read_obj:
        csv_reader = reader(read_obj)
        for my_data in csv_reader:
            if(i == 0 or int(float(my_data[0])) < sum_factor or int(float(my_data[0])) > T*sum_factor):
                i += 1
                continue
            if(int(float(my_data[2])*scale/pixel_size) < crop_loc[0]*scale or
                    int(float(my_data[1])*scale/pixel_size) < crop_loc[1]*scale or
                    int(float(my_data[2])*scale/pixel_size) >= (crop_loc[0]+img_size)*scale or
                    int(float(my_data[1])*scale/pixel_size) >= (crop_loc[1]+img_size)*scale):
                continue
            X_test[int((float(my_data[0]) - 1)/sum_factor),
                   int(float(my_data[2])*scale/pixel_size - crop_loc[0]*scale),
                   int(float(my_data[1])*scale/pixel_size - crop_loc[1]*scale)] += 1
            i += 1

    # Convert input to GPU and tensor
    X_test = torch.from_numpy(X_test)
    X_test = X_test.unsqueeze(1).unsqueeze(0)
    X_test = X_test.type(torch.FloatTensor)
    X_test = X_test.to(device)

    # Initialize model
    model = ConvBLSTM(input_size=(img_size, img_size), input_channels=1, hidden_channels=hidden_channels, num_layers=num_layers, device=device).to(device)
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device(device)))

    N, T, C, H, W = X_test.shape

    for i in range(X_test.size(0)):
        out = model(X_test[i:i + 1], torch.flip(X_test[i:i + 1], dims=[1]))

        curr_vid = np.zeros([1, T, C, H, W])
        for j in tqdm(range(T)):
            curr_vid[0, j] = 255 * normalize_input_01(out[0, j].detach().cpu().numpy())
            curr_vid[0, j][np.where(curr_vid[0, j] < output_threshold * np.max(curr_vid[0, j]))] = 0

        # Create DeepSTORM windows reconstruction
        deepSTORM_windows = np.zeros_like(curr_vid)
        for frame in range(int(num_of_DS_frames)):
            deepSTORM_windows[0, int(frame*DS_sum/sum_factor):int((frame+1)*DS_sum/sum_factor), 0] = \
                torch.sum(X_test[i, int(frame*DS_sum/sum_factor):int((frame+1)*DS_sum/sum_factor), 0].cpu().detach(), axis=0)
            deepSTORM_windows[np.where(deepSTORM_windows > 2)] = 2

        np.save('tmp_results/np_vid_{}'.format(i + 1), curr_vid[0])
        np.save('tmp_results/gt_vid_{}'.format(i + 1), deepSTORM_windows[i, :T])

        print("-I- Completed vid", i + 1)

def analyze_storm_exp_overlap(path_to_model, exp_class, hidden_channels, num_layers, scale, device, use_overlap=False):
    ''' In development '''
    # Init params from exp_params class
    img_size = exp_class.img_size
    crop_loc = exp_class.crop_loc #[y, x]
    pixel_size = exp_class.pixel_size
    sum_factor = exp_class.sum_factor
    T = exp_class.vid_length # vid length
    dir_path = exp_class.path
    obs_TIF = exp_class.filename

    output_threshold = 0.0

    if(use_overlap):
        print("Reading csv file")
        # Load localization file and transfer to image data
        X_test = np.zeros([T * sum_factor, img_size * scale, img_size * scale])
        for shift in tqdm(range(sum_factor)):
            with open(os.path.join(dir_path, '{}.csv'.format(obs_TIF[:-4])), 'r') as read_obj:
                csv_reader = reader(read_obj)
                next(csv_reader) # Skip first line containing the column names
                for my_data in csv_reader:
                    if(int(float(my_data[0])) + shift < 0):
                        continue
                    if(sum_factor * int((float(my_data[0]) - 1)/sum_factor) + shift >= T*sum_factor):
                        break
                    if(int(float(my_data[2])*scale/pixel_size) < crop_loc[0]*scale or
                            int(float(my_data[1])*scale/pixel_size) < crop_loc[1]*scale or
                            int(float(my_data[2])*scale/pixel_size) >= (crop_loc[0]+img_size)*scale or
                            int(float(my_data[1])*scale/pixel_size) >= (crop_loc[1]+img_size)*scale):
                        continue
                    X_test[sum_factor * int((float(my_data[0]) - 1)/sum_factor) + shift,
                          int(float(my_data[2])*scale/pixel_size - crop_loc[0]*scale),
                          int(float(my_data[1])*scale/pixel_size - crop_loc[1]*scale)] += 1

        X_test = torch.from_numpy(X_test)
        X_test = X_test.unsqueeze(1).unsqueeze(0)
        X_test = X_test.type(torch.FloatTensor)
        X_test = X_test.to(device)

        N, T, C, H, W = X_test.shape

        model = ConvOverlapBLSTM(input_size=(img_size, img_size), input_channels=1, hidden_channels=hidden_channels, num_layers=num_layers, device=device).to(device)
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device(device)))

        down = torch.zeros(X_test.size(1), requires_grad=False, dtype=torch.int)
        up = torch.zeros(X_test.size(1),  requires_grad=False, dtype=torch.int)
        out_ind = torch.zeros(X_test.size(1),  requires_grad=False, dtype=torch.int)
        for i in range(X_test.size(1)):
            down[i] = torch.max(torch.IntTensor([0, i - sum_factor*exp_class.window_size]))
            up[i] = torch.min(torch.IntTensor([X_test.size(1), i + sum_factor*exp_class.window_size]))
            out_ind[i] = i - down[i]

        for i in range(X_test.size(0)):
            out = []
            print('Feeding the input to the model')
            with torch.no_grad():
                for j in tqdm(range(X_test.shape[1])):
                    curr_out = model(X_test[i:i + 1, down[j]:up[j]:sum_factor], torch.flip(X_test[i:i + 1, down[j]:up[j]:sum_factor], dims=[1]))
                    curr_out = curr_out.detach().cpu()[0, int(out_ind[j]/sum_factor)]
                    out.append(curr_out)

            out = torch.stack(out, dim=1)

            curr_vid = np.zeros([1, X_test.size(1), C, H, W])
            for j in tqdm(range(X_test.size(1))):
                curr_vid[0, j] = 255 * normalize_input_01(out[0, j].numpy())
                curr_vid[0, j][np.where(curr_vid[0, j] < output_threshold * np.max(curr_vid[0, j]))] = 0
    else:
        # Load localization file and transfer to image data
        X_test = np.zeros([int(np.ceil(T/sum_factor)), img_size * scale, img_size * scale])
        with open(os.path.join(dir_path, '{}.csv'.format(obs_TIF[:-4])), 'r') as read_obj:
            csv_reader = reader(read_obj)
            next(csv_reader) # Skip first line containing the column names
            for my_data in csv_reader:
                if(int(float(my_data[0])) < 0):
                    continue
                if(int((float(my_data[0]) - 1)) >= T):
                    break
                if(int(float(my_data[2])*scale/pixel_size) < crop_loc[0]*scale or
                        int(float(my_data[1])*scale/pixel_size) < crop_loc[1]*scale or
                        int(float(my_data[2])*scale/pixel_size) >= (crop_loc[0]+img_size)*scale or
                        int(float(my_data[1])*scale/pixel_size) >= (crop_loc[1]+img_size)*scale):
                    continue
                X_test[int((float(my_data[0]) - 1)//sum_factor),
                      int(float(my_data[2])*scale/pixel_size - crop_loc[0]*scale),
                      int(float(my_data[1])*scale/pixel_size - crop_loc[1]*scale)] += 1

        X_test = torch.from_numpy(X_test)
        X_test = X_test.unsqueeze(1).unsqueeze(0)
        X_test = X_test.type(torch.FloatTensor)
        X_test = X_test.to(device)

        N, T, C, H, W = X_test.shape
        model = ConvBLSTM(input_size=(img_size, img_size), input_channels=1, hidden_channels=hidden_channels, num_layers=num_layers, device=device).to(device)
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device(device)))

        print('Feeding the input to the model')
        with torch.no_grad():
            curr_out, _ = model(X_test[:1], torch.flip(X_test[:1], dims=[1]))
            curr_out = curr_out.detach().cpu()

            curr_vid = np.zeros([1, X_test.size(1), C, H, W])
            for j in tqdm(range(X_test.size(1))):
                curr_vid[0, j] = 255 * normalize_input_01(curr_out[0, j].numpy())
                curr_vid[0, j][np.where(curr_vid[0, j] < output_threshold * np.max(curr_vid[0, j]))] = 0

        np.save('tmp_results/np_vid_1', curr_vid[0])
        np.save('tmp_results/gt_vid_1', X_test[0, :].detach().cpu().numpy())

def analyze_storm_exp_one_directional(path_to_model, exp_class, hidden_channels, num_layers, scale, device):
    ''' In development '''
    # Init params from exp_params class
    img_size = exp_class.img_size
    crop_loc = exp_class.crop_loc #[y, x]
    pixel_size = exp_class.pixel_size
    DS_sum = exp_class.DS_sum # how many DeepSTORM frames to sum?
    sum_factor = exp_class.sum_factor
    T = exp_class.vid_length # vid length
    dir_path = exp_class.path
    obs_TIF = exp_class.filename

    num_of_DS_frames = T*sum_factor/DS_sum
    output_threshold = 0.2

    # Load localization file and transfer to image data
    X_test = np.zeros([T, img_size * scale, img_size * scale])
    i = 0
    with open(os.path.join(dir_path, '{}.csv'.format(obs_TIF[:-4])), 'r') as read_obj:
        csv_reader = reader(read_obj)
        for my_data in csv_reader:
            if(i == 0 or int(float(my_data[0])) < sum_factor or int(float(my_data[0])) > T*sum_factor):
                i += 1
                continue
            if(int(float(my_data[2])*scale/pixel_size) < crop_loc[0]*scale or
                    int(float(my_data[1])*scale/pixel_size) < crop_loc[1]*scale or
                    int(float(my_data[2])*scale/pixel_size) >= (crop_loc[0]+img_size)*scale or
                    int(float(my_data[1])*scale/pixel_size) >= (crop_loc[1]+img_size)*scale):
                continue
            X_test[int((float(my_data[0]) - 1)/sum_factor),
                   int(float(my_data[2])*scale/pixel_size - crop_loc[0]*scale),
                   int(float(my_data[1])*scale/pixel_size - crop_loc[1]*scale)] += 1
            i += 1

    X_test = torch.from_numpy(X_test)
    X_test = X_test.unsqueeze(1).unsqueeze(0)
    X_test = X_test.type(torch.FloatTensor)
    X_test = X_test.to(device)

    N, T, C, H, W = X_test.shape

    model = ConvOneDirectionalLSTM(input_size=(img_size, img_size), input_channels=1, hidden_channels=hidden_channels, num_layers=num_layers, device=device).to(device)
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device(device)))

    for i in range(X_test.size(0)):
        hidden = None
        curr_vid = np.zeros([1, T, C, H, W])
        for j in range(int(T/exp_class.window_size) + 1):
            out, hidden = model(X_test[i:i + 1, j*exp_class.window_size:(j+1)*exp_class.window_size], hidden)

            for k in tqdm(range(j*exp_class.window_size, (j+1)*exp_class.window_size)):
                if(k >= T):
                    break
                curr_vid[0, k] = 255 * normalize_input_01(out[0, k - j * exp_class.window_size].detach().cpu().numpy())
                curr_vid[0, k][np.where(curr_vid[0, k] < output_threshold * np.max(curr_vid[0, k]))] = 0

        # Create DeepSTORM windows reconstruction
        deepSTORM_windows = np.zeros_like(curr_vid)
        for frame in range(int(num_of_DS_frames)):
            deepSTORM_windows[0, int(frame*DS_sum/sum_factor):int((frame+1)*DS_sum/sum_factor), 0] = \
                torch.sum(X_test[i, int(frame*DS_sum/sum_factor):int((frame+1)*DS_sum/sum_factor), 0].cpu().detach(), axis=0)
            deepSTORM_windows[np.where(deepSTORM_windows > 2)] = 2

        np.save('tmp_results/np_vid_{}'.format(i + 1), curr_vid[0])
        np.save('tmp_results/gt_vid_{}'.format(i + 1), deepSTORM_windows[i, :T])

        print("-I- Completed vid", i + 1)

def normalize_input_01(input):
    '''
    :param input: an image
    :return: normalized image in the range [0, 1]
    '''
    min_val = np.min(input)
    max_val = np.max(input)
    norm_input = (input - min_val) / (max_val - min_val)
    return norm_input

def LoadTIFF(path, file, crop_loc, img_size):
    '''
    :param path: path to TIFF file
    :param file: name of TIFF file
    :param crop_loc: bottom left position of crop
    :param img_size: crop size
    :return: a numpy 3D vector of size [TIFF length, img size, img size]
    '''
    tiff = Image.open(os.path.join(path, file))
    data = []
    frame = 0
    while(True):
        try:
            tiff.seek(frame)
        except:
            break

        data.append(np.array(tiff)[crop_loc[0]:crop_loc[0]+img_size, crop_loc[1]:crop_loc[1]+img_size])
        frame += 1
    return np.array(data)

def calc_acc(curr_vid, y_test):
    '''
    :param curr_vid: reconstructed video
    :param y_test: ground truth video
    :return: None, print the fidelity measure and the hallucination measure (see DBlink SI for more details)
    '''
    binary_curr_vid = np.zeros_like(curr_vid)
    binary_curr_vid[np.where(curr_vid > 100)] = 1
    binary_test = np.zeros_like(binary_curr_vid)
    binary_test[np.where(y_test.cpu().numpy() > 100)] = 1
    one_mask = np.where(binary_curr_vid == 1)
    if (len(np.where(binary_test == 1)[0]) == 0):
        fidelity = 1
    else:
        fidelity = np.sum(binary_test[one_mask] == binary_curr_vid[one_mask]) / len(np.where(binary_test == 1)[0])
    hallucination = np.sum(1 + binary_test[one_mask] == binary_curr_vid[one_mask]) / len(np.where(binary_test == 0)[0])

    print("fidelity = {}%, hallucination = {}%".format(fidelity, hallucination))
