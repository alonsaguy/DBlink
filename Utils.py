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
    # Fit patch to gaussian
    x, y = xy
    return np.exp(-((x - (np.sqrt(x.shape[0]) - 1) / 2) ** 2 + (y - (np.sqrt(y.shape[0]) - 1) / 2) ** 2) / (2 * sigma ** 2))

def show_sim_results(path, index, scale):
    gt = np.load(os.path.join(path, 'gt_vid_{}.npy'.format(index)))
    pred = np.load(os.path.join(path, 'np_vid_{}.npy'.format(index)))

    tmp = []

    patch_size = 7
    mid = int(np.floor(patch_size / 2))
    xy = np.zeros([2, int(patch_size ** 2)])
    for i1 in range(patch_size):
        for j1 in range(patch_size):
            xy[:, int(i1 + patch_size * j1)] = [i1, j1]

    binary_movie = np.zeros_like(pred)
    recon_movie = np.zeros_like(pred)
    thresh = 0.5 * np.mean(pred)
    binary_movie[np.where(pred > thresh)] = 1

    for i in range(gt.shape[0]):
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

        concat_img = np.concatenate([norm_recon,
                                     norm_gt], axis=1)
        tmp.append(concat_img)

    tmp = np.array(tmp)[None, :, None, :, :]
    create_example_vid('tmp_results/out_vid_{}'.format(index), tmp)

def show_exp_results(path, index, scale):
    ds_windows = np.load(os.path.join(path, 'gt_vid_{}.npy'.format(index)))
    pred = np.load(os.path.join(path, 'np_vid_{}.npy'.format(index)))

    # convolve with gaussian based on uncertainty
    patch_size = 7
    mid = int(np.floor(patch_size / 2))
    xy = np.zeros([2, int(patch_size ** 2)])
    for i1 in range(patch_size):
        for j1 in range(patch_size):
            xy[:, int(i1 + patch_size * j1)] = [i1, j1]

    binary_movie = np.zeros_like(pred)
    recon_movie = np.zeros_like(pred)
    thresh = 0.5 * np.mean(pred)
    binary_movie[np.where(pred > thresh)] = 1
    for frame in tqdm(range(recon_movie.shape[0])):
        for row in range(mid, recon_movie.shape[2] - mid):
            for col in range(mid, recon_movie.shape[3] - mid):
                if (binary_movie[frame, 0, row, col] == 1):
                    if (pred[frame, 0, row, col] == 1):
                        uncertainty = 0.1
                    else:
                        uncertainty = np.min([0.1 / (pred[frame, 0, row, col] + 1e-3), 100])
                    recon_movie[frame, 0, row-mid:row+mid+1, col-mid:col+mid+1] += (1/uncertainty)**2 * gauss2d(xy, sigma=uncertainty).reshape([patch_size, patch_size])

    tmp = []
    for i in range(pred.shape[0]):
        if(np.max(ds_windows[i, 0]) > 0):
            norm_ds_windows = 255 * normalize_input_01(ds_windows[i, 0, :, :])
        else:
            norm_ds_windows = ds_windows[i, 0] + 1e-9

        if (np.max(recon_movie[i, 0]) > 0):
            norm_recon = 255 * normalize_input_01(recon_movie[i, 0, :, :])
        else:
            norm_recon = recon_movie[i, 0] + 1e-9

        concat_img = np.concatenate([norm_ds_windows,
                                     norm_recon], axis=1)
        tmp.append(concat_img)

    tmp = np.array(tmp)[None, :, None, :, :]
    create_example_vid('tmp_results/out_vid_{}'.format(index), tmp)

def compare_recon_to_japan(path, index, crop_loc, img_size, sum_factor, scale):
    pred = np.load(os.path.join(path, 'np_vid_{}.npy'.format(index)))
    tmp = LoadTIFF(r'./data/japan', 'japan_slow_movie.tif', crop_loc * scale * 10, img_size * scale * 10)

    japan_recon = np.zeros([tmp.shape[0], img_size * scale, img_size * scale])
    for i in range(japan_recon.shape[0]):
        japan_recon[i] = cv2.resize(tmp[i], (img_size * scale, img_size * scale))

    # convolve with gaussian based on uncertainty
    patch_size = 7
    mid = int(np.floor(patch_size / 2))
    xy = np.zeros([2, int(patch_size ** 2)])
    for i1 in range(patch_size):
        for j1 in range(patch_size):
            xy[:, int(i1 + patch_size * j1)] = [i1, j1]

    binary_movie = np.zeros_like(pred)
    recon_movie = np.zeros_like(pred)
    thresh = 0.5 * np.mean(pred)
    binary_movie[np.where(pred > thresh)] = 1
    for frame in tqdm(range(recon_movie.shape[0])):
        for row in range(mid, recon_movie.shape[2] - mid):
            for col in range(mid, recon_movie.shape[3] - mid):
                if (binary_movie[frame, 0, row, col] == 1):
                    if (pred[frame, 0, row, col] == 1):
                        uncertainty = 0.1
                    else:
                        uncertainty = np.min([0.1 / (pred[frame, 0, row, col] + 1e-3), 100])
                    recon_movie[frame, 0, row-mid:row+mid+1, col-mid:col+mid+1] += (1/uncertainty)**2 * gauss2d(xy, sigma=uncertainty).reshape([patch_size, patch_size])

    conversion = japan_recon.shape[0]/recon_movie.shape[0]
    frames = []  # for storing the generated images
    tmp = []
    fig = plt.figure()
    for i in range(pred.shape[0]):
        japan_frame = int(i * conversion)
        if(np.max(japan_recon[japan_frame]) > 0):
            norm_japan = 255 * normalize_input_01(japan_recon[japan_frame, :, :])
        else:
            norm_japan = japan_recon[japan_frame] + 1e-9

        if (np.max(recon_movie[i, 0]) > 0):
            norm_recon = 255 * normalize_input_01(recon_movie[i, 0, :, :])
        else:
            norm_recon = recon_movie[i, 0] + 1e-9

        concat_img = np.concatenate([norm_japan,
                                     norm_recon], axis=1)
        tmp.append(concat_img)
        frames.append([plt.imshow(concat_img, cmap=cm.Greys_r, animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=500, blit=True, repeat_delay=500)

    # ani.save('movie.mp4')
    plt.show()

    tmp = np.array(tmp)[None, :, None, :, :]
    create_example_vid('tmp_results/out_vid_{}'.format(index), tmp)

def reduce_mem_usage(vid):
    norm_vid = torch.zeros(vid.shape, dtype=torch.uint8)
    for i in range(vid.shape[0]):
        norm_vid[i] = torch.tensor(255 * ((vid[i] - vid[i].min()) / (vid[i].max() - vid[i].min())), dtype=torch.uint8)
    return norm_vid

def analyze_storm_exp(path_to_model, exp_class, hidden_channels, num_layers, scale, device):
    # Init params from exp_params class
    img_size = exp_class.img_size
    crop_loc = exp_class.crop_loc #[y, x]
    pixel_size = exp_class.pixel_size
    DS_sum = exp_class.DS_sum # how many DeepSTORM frames to sum?
    sum_factor = exp_class.sum_factor
    T = exp_class.vid_length # vid length
    dir_path = exp_class.path
    obs_TIF = exp_class.filename
    input_size = exp_class.input_size

    num_of_DS_frames = T*sum_factor/DS_sum
    output_threshold = 0.2
    detection_threshold = 0.0

    # Load localization file and transfer to image data
    X_test = np.zeros([T, img_size * scale, img_size * scale])
    i = 0
    with open(os.path.join(dir_path, '{}.csv'.format(obs_TIF[:-4])), 'r') as read_obj:
        csv_reader = reader(read_obj)
        for my_data in csv_reader:
            if(i == 0 or int(my_data[0]) > T*sum_factor):
                i += 1
                continue
            if(float(my_data[-1]) > detection_threshold):
                if(int(float(my_data[2])*scale/pixel_size) < crop_loc[0]*scale or
                        int(float(my_data[1])*scale/pixel_size) < crop_loc[1]*scale or
                        int(float(my_data[2])*scale/pixel_size) >= (crop_loc[0]+img_size)*scale or
                        int(float(my_data[1])*scale/pixel_size) >= (crop_loc[1]+img_size)*scale):
                    continue
                X_test[int((float(my_data[0])-1)/sum_factor),
                       int(float(my_data[2])*scale/pixel_size - crop_loc[0]*scale),
                       int(float(my_data[1])*scale/pixel_size - crop_loc[1]*scale)] += 1
            i += 1

    X_test = torch.from_numpy(X_test)
    X_test = X_test.unsqueeze(1).unsqueeze(0)
    X_test = X_test.type(torch.FloatTensor)
    X_test = X_test.to(device)

    N, T, C, H, W = X_test.shape

    model = ConvBLSTM(input_size=(img_size, img_size), input_channels=1, hidden_channels=hidden_channels, num_layers=num_layers, device=device).to(device)
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device(device)))

    for i in range(X_test.size(0)):
        prev_hidden_fwd = None
        if (T > input_size):
            out = []
            curr_frame = 0
            while (curr_frame < T):
                end_frame = np.min([curr_frame + input_size, T])
                curr_out, prev_hidden_fwd = model(X_test[i:i + 1, curr_frame:end_frame],
                                                  torch.flip(X_test[i:i + 1, curr_frame:end_frame], dims=[1]),
                                                  prev_hidden_fwd,
                                                  prop_hidden=True)
                out.append(reduce_mem_usage(curr_out))
                curr_frame = end_frame

            out = torch.cat(out, dim=1)
        else:
            out, _ = model(X_test[i:i + 1], torch.flip(X_test[i:i + 1], dims=[1]))

        curr_vid = np.zeros([1, T, C, H, W])
        for j in tqdm(range(T)):
            curr_vid[0, j] = 255 * normalize_input_01(out[0, j, 0].detach().cpu().numpy())
            curr_vid[0, j][np.where(curr_vid[0, j] < output_threshold * np.max(curr_vid[0, j]))] = 0

        curr_vid = normalize_input_01(curr_vid)

        # Create DeepSTORM windows reconstruction
        deepSTORM_windows = np.zeros_like(curr_vid)
        for frame in range(int(num_of_DS_frames)):
            deepSTORM_windows[0, int(frame*DS_sum/sum_factor):int((frame+1)*DS_sum/sum_factor), 0] = \
                torch.sum(X_test[i, int(frame*DS_sum/sum_factor):int((frame+1)*DS_sum/sum_factor), 0].cpu().detach(), axis=0)
            deepSTORM_windows[np.where(deepSTORM_windows > 2)] = 2

        np.save('tmp_results/np_vid_{}'.format(i + 1), curr_vid[0])
        np.save('tmp_results/obs_vid_{}'.format(i + 1), X_test[i, :T].cpu().data.numpy())
        np.save('tmp_results/gt_vid_{}'.format(i + 1), deepSTORM_windows[i, :T])

        print("-I- Completed vid", i + 1)

def analyze_storm_exp_multiGPU(path_to_model, exp_class, hidden_channels, num_layers, scale, device_list):
    img_size = exp_class.img_size
    crop_loc = exp_class.crop_loc  # [y, x]
    pixel_size = exp_class.pixel_size
    DS_sum = exp_class.DS_sum  # how many DeepSTORM frames to sum?
    sum_factor = exp_class.sum_factor
    T = exp_class.vid_length  # vid length
    dir_path = exp_class.path
    obs_TIF = exp_class.filename

    num_of_DS_frames = T * sum_factor / DS_sum
    detection_threshold = 0.0
    output_threshold = 0.0

    # Load localization file and transfer to image data
    X_test = np.zeros([T, img_size * scale, img_size * scale])
    i = 0
    with open(os.path.join(dir_path, '{}.csv'.format(obs_TIF[:-4])), 'r') as read_obj:
        csv_reader = reader(read_obj)
        for my_data in csv_reader:
            if (i == 0 or int(my_data[0]) > T * sum_factor):
                i += 1
                continue
            if (float(my_data[-1]) > detection_threshold):
                if (int(float(my_data[2]) * scale / pixel_size) < crop_loc[0] * scale or
                        int(float(my_data[1]) * scale / pixel_size) < crop_loc[1] * scale or
                        int(float(my_data[2]) * scale / pixel_size) >= (crop_loc[0] + img_size) * scale or
                        int(float(my_data[1]) * scale / pixel_size) >= (crop_loc[1] + img_size) * scale):
                    continue
                X_test[int((float(my_data[0]) - 1) / sum_factor),
                       int(float(my_data[2]) * scale / pixel_size - crop_loc[0] * scale),
                       int(float(my_data[1]) * scale / pixel_size - crop_loc[1] * scale)] += 1
            i += 1

    X_test = torch.from_numpy(X_test)
    X_test = X_test.unsqueeze(1).unsqueeze(0)
    X_test = X_test.type(torch.FloatTensor)

    N, T, C, H, W = X_test.shape

    model = ConvBLSTM(input_size=(img_size, img_size), input_channels=1, hidden_channels=hidden_channels,
                      num_layers=num_layers, device=device_list[0]).to(device_list[0])
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device(device_list[0])))

    num_cycles = exp_class.num_cycles
    for i in range(X_test.size(0)):
        input_size = int(T / (num_cycles * len(device_list)))
        prev_hidden_fwd = None
        out = []
        for cycle in range(num_cycles):
            for dev_num in range(len(device_list)):
                # Send inputs and model to current device
                curr_input = X_test[i:i+1, (cycle*input_size*len(device_list)+input_size*(dev_num)):(cycle*input_size*len(device_list)+input_size*(dev_num+1))].to(device_list[dev_num])
                model = model.to(device_list[dev_num])
                if(prev_hidden_fwd != None):
                    for layer in range(len(prev_hidden_fwd)):
                        for element in range(len(prev_hidden_fwd[layer])):
                            prev_hidden_fwd[layer][element] = prev_hidden_fwd[layer][element].to(device_list[dev_num])

                # Inference
                curr_out, prev_hidden_fwd = model(curr_input,
                                                  torch.flip(curr_input, dims=[1]),
                                                  prev_hidden_fwd,
                                                  prop_hidden=True)
                out.append(reduce_mem_usage(curr_out))

        out = torch.cat(out, dim=1)

        curr_vid = np.zeros([1, T, C, H, W])

        for j in tqdm(range(T)):
            curr_vid[0, j] = 255 * normalize_input_01(out[0, j, 0].detach().cpu().numpy())
            curr_vid[0, j][np.where(curr_vid[0, j] < output_threshold * np.max(curr_vid[0, j]))] = 0

        curr_vid = normalize_input_01(curr_vid)

        # Create DeepSTORM windows reconstruction
        deepSTORM_windows = np.zeros_like(curr_vid)
        for frame in range(int(num_of_DS_frames)):
            deepSTORM_windows[0, int(frame * DS_sum / sum_factor):int((frame + 1) * DS_sum / sum_factor), 0] = \
                torch.sum(X_test[i, int(frame * DS_sum / sum_factor):int((frame + 1) * DS_sum / sum_factor), 0].detach(), axis=0)
            deepSTORM_windows[np.where(deepSTORM_windows > 2)] = 2

        np.save('tmp_results/np_vid_{}'.format(i + 1), curr_vid[0])
        np.save('tmp_results/obs_vid_{}'.format(i + 1), X_test[i, :T].data.numpy())
        np.save('tmp_results/gt_vid_{}'.format(i + 1), deepSTORM_windows[i, :T])

        print("-I- Completed vid", i + 1)

        #compare_recon_to_japan(r'./tmp_results', i + 1, crop_loc, img_size, sum_factor, scale)

def normalize_input(input):
    per5 = np.percentile(input, 5)
    per95 = np.percentile(input, 95)
    norm_input = (input - per5) / (per95 - per5)
    return norm_input

def normalize_input_01(input):
    min_val = np.min(input)
    max_val = np.max(input)
    norm_input = (input - min_val) / (max_val - min_val)
    return norm_input

def LoadTIFF(path, file, crop_loc, img_size):
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