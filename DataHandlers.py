# Imports
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import cv2
from tqdm import tqdm
from scipy.ndimage import shift
from scipy.ndimage import rotate

# Create Data Sets

def Simulate_Train_Data_060622(obs_size, dataset_size, video_length, emitters_density,
                               scale, sum_range, datatype):
    Velocities = np.random.uniform(-0.0025*scale, 0.0025*scale, [dataset_size, 2])
    rot_speed = np.random.uniform(0.00015*scale, 0.00015*scale, dataset_size)

    # Create TrainSet
    Observations = np.zeros([dataset_size, int(video_length/sum_range), obs_size*scale, obs_size*scale], dtype=np.uint8)
    tmp_obs = np.zeros([dataset_size, video_length, obs_size * scale, obs_size * scale], dtype=np.uint8)
    ScaledGroundTruths = np.zeros([dataset_size, int(video_length/sum_range), obs_size*scale, obs_size*scale], dtype=np.uint8)
    GroundTruths = np.zeros([dataset_size, video_length, obs_size*scale, obs_size*scale], dtype=np.uint8)

    print("-I- Generating training data")
    for img in tqdm(range(dataset_size)):
        if datatype == 'tubules':
            # Generate First frame
            GroundTruths[img, 0, :, :] = 255 * generate_microtubules_sim(obs_size * scale, scale)

            # Apply changes to images in time
            for frame in range(1, video_length):
                tmp = apply_change(GroundTruths[img, 0], Velocities[img]*frame, rot_speed[img]*frame)[:obs_size*scale, :obs_size*scale]
                tmp[tmp < 0.1] = 0
                GroundTruths[img, frame] = tmp

        elif datatype == 'mito':
            GroundTruths[img, :, :, :] = 255 * generate_mitochondria_sim(obs_size * scale, video_length)

        for frame in range(0, video_length, sum_range):
            ScaledGroundTruths[img, int(frame/sum_range)] = GroundTruths[img, frame]

        # Add light emissions at random spots
        for frame in range(video_length):
            tmp_obs[img, frame] = add_emissions_deepSTORM(GroundTruths[img, frame], emitters_density)

        for frame in range(0, video_length, sum_range):
            Observations[img, int(frame/sum_range)] = np.sum(tmp_obs[img, frame:frame+sum_range], axis=0)

    return Observations[:, :, None, :, :], ScaledGroundTruths[:, :, None, :, :]

def Simulate_Train_Data_200422(obs_size, dataset_size, video_length, emitters_density,
                               scale, sum_range, photobleaching_time):
    Velocities = np.random.uniform(-0.01, 0.01, [dataset_size, 2])
    rot_speed = np.random.uniform(0.005, 0.005, dataset_size)

    # Create TrainSet
    Observations = np.zeros([dataset_size, int(video_length/sum_range), obs_size*scale, obs_size*scale], dtype=np.uint8)
    tmp_obs = np.zeros([dataset_size, video_length, obs_size * scale, obs_size * scale], dtype=np.uint8)
    ScaledGroundTruths = np.zeros([dataset_size, int(video_length/sum_range), obs_size*scale, obs_size*scale], dtype=np.uint8)
    GroundTruths = np.zeros([dataset_size, video_length, obs_size*scale, obs_size*scale], dtype=np.uint8)

    print("-I- Generating training data")
    for img in tqdm(range(dataset_size)):
        # Generate First frame
        GroundTruths[img, 0, :, :] = 255 * generate_microtubules_sim(obs_size * scale, scale) #generate_random_lines(obs_size * scale)

        # Apply changes to images in time
        for frame in range(1, video_length):
            tmp = apply_change(GroundTruths[img, 0], Velocities[img]*frame, rot_speed[img]*frame)[:obs_size*scale, :obs_size*scale]
            tmp[tmp < 0.1] = 0
            GroundTruths[img, frame] = tmp

        for frame in range(0, video_length, sum_range):
            ScaledGroundTruths[img, int(frame/sum_range)] = GroundTruths[img, frame]

        # Add light emissions at random spots
        photobleaching_mask = np.ones_like(GroundTruths[0, 0])
        for frame in range(video_length):
            if(frame == int((np.random.uniform(-0.1, 0.1) + photobleaching_time)*video_length)):
                photobleaching_mask = np.random.randint(0, 2, photobleaching_mask.shape)
            tmp = apply_change(GroundTruths[img, 0] * photobleaching_mask,
                               Velocities[img] * frame, rot_speed[img] * frame)[:obs_size * scale, :obs_size * scale]
            tmp[tmp < 0.1] = 0
            # Add photobleaching after a certain time
            tmp_obs[img, frame] = add_emissions_deepSTORM(tmp, emitters_density)

        for frame in range(0, video_length, sum_range):
            Observations[img, int(frame/sum_range)] = np.sum(tmp_obs[img, frame:frame+sum_range], axis=0)

    return Observations[:, :, None, :, :], ScaledGroundTruths[:, :, None, :, :]

def add_emissions_deepSTORM(gt, emitters_density):
    # Initialize observation image
    obs = np.zeros_like(gt)

    # Decide on number of emitters for the given gt image
    if(np.any(gt > 15)):
        num_of_emitters = int(np.ceil(len(np.where(gt > 15)[0]) * emitters_density))
    else:
        num_of_emitters = 0

    # Add random blinking events
    num_of_noise = np.random.randint(0, 5)
    for i in range(num_of_noise):
        xy = np.random.randint(0, gt.shape[-1], 2)
        obs[xy[0], xy[1]] += 1

    # Find the mask of possible places for emitters in the image
    mask = np.zeros_like(gt, dtype=np.int)
    mask[np.where(gt > 15)] = 1

    if(np.sum(mask) == 0):
        return obs
    # Add the PSF where the emitter is located
    for i in range(num_of_emitters):
        possible_locs_size = len(np.where(mask == 1)[0])
        probability = np.repeat(1 / possible_locs_size, possible_locs_size)
        if(possible_locs_size == 0):
            break
        # Make sure the PSF is inside the image bounds
        loc_ind = np.random.choice(np.arange(possible_locs_size), size=1, p=probability)
        xy = [np.where(mask == 1)[0][loc_ind], np.where(mask == 1)[1][loc_ind]]
        if(xy[0] > 0 and xy[1] > 0 and xy[0] < obs.shape[0] - 1 and xy[1] < obs.shape[1] - 1):
            # Add offset to localization with some probability
            offset = np.random.randint(-1, 2, 2)
            obs[xy[0] + offset[0], xy[1] + offset[1]] += 1

        # Add cluster of localizations
        if(np.random.uniform(0, 1) < 0.3):
            if(xy[0] > 5 and xy[0] < obs.shape[0] - 5 and xy[1] > 5 and xy[1] < obs.shape[1] - 5):
                num_of_locs_in_cluster = np.random.randint(1, 8)
                for j in range(num_of_locs_in_cluster):
                    obs[xy[0] + np.random.randint(-5, 6), xy[1] + np.random.randint(-5, 6)] += 1

    return obs

def generate_random_lines(img_size):
    img = np.zeros([img_size, img_size])

    num_of_lines = np.random.randint(0, int(img_size/12))
    for i in range(num_of_lines):
        x, y = np.random.randint(0, img_size, 2)
        img[y, x] = 1
        dir = np.random.choice([-1, 1], 2)
        pref_dir = dir
        while(y+dir[0] >= 0 and y+dir[0] < img_size and x+dir[1] >= 0 and x+dir[1] < img_size):
            y += dir[0]
            x += dir[1]
            img[y, x] = 1
            diry = np.random.choice([0, pref_dir[0]])
            dirx = np.random.choice([0, pref_dir[1]])
            dir = [diry, dirx]

    return img

def generate_mitochondria_sim(img_size, vid_length):
    from skimage.draw import polygon
    # Initialize parameters
    mitochondrias = np.zeros([vid_length, img_size, img_size], dtype=np.uint8)
    min_polygon_pts, max_polygon_pts = 30, 50
    R = 20
    # Randomize the number of mitochondria in the FOV
    num_of_mito = np.random.randint(1, int(img_size/10))
    for i in range(num_of_mito):
        # Generate new polygon
        curr_polygon = np.zeros([2 * R + 1, 2 * R + 1])
        num_of_polygon_pts = np.random.randint(min_polygon_pts, max_polygon_pts)
        random_phase = np.arange(num_of_polygon_pts) * 2 * np.pi / (num_of_polygon_pts) + 2 * np.pi * np.random.uniform(0, 0.5)
        random_range = np.random.randint(6, 12)
        random_radius = np.random.randint(0, 3) + np.abs(random_range * np.sin(random_phase)**10)
        angle = np.arange(num_of_polygon_pts) * 2 * np.pi / num_of_polygon_pts
        polygon_pts = R + np.array(random_radius * [np.sin(angle), np.cos(angle)]).astype(int)
        rr, cc = polygon(polygon_pts[0, :], polygon_pts[1, :])
        curr_polygon[rr, cc] = 1
        # Cut the polygon patch to fit the FOV
        bot_left_corner = np.random.randint(0, img_size - 1, 2)
        cut_curr_patch = cut_edges(curr_polygon, bot_left_corner, img_size)
        # Simulate first mitochondria frame
        mitochondrias[0, bot_left_corner[0]:bot_left_corner[0] + cut_curr_patch.shape[0],
        bot_left_corner[1]:bot_left_corner[1] + cut_curr_patch.shape[1]] += cut_curr_patch
        # Randomize movement
        velocity = np.random.uniform(-0.05, 0.05, 2)
        num_of_moving_pts = np.random.randint(0, polygon_pts.shape[1]/2)
        elon_ind = np.random.randint(0, polygon_pts.shape[1], num_of_moving_pts)
        original_pt = np.copy(polygon_pts[:, elon_ind])
        elon_velocity = np.random.uniform(-0.01, 0.01, [2, num_of_moving_pts])
        # Apply tranformation
        for frame in range(vid_length):
            new_bot_left = bot_left_corner + (velocity*frame).astype(int)
            curr_polygon = np.zeros([2 * R + 1, 2 * R + 1])
            polygon_pts[:, elon_ind] = original_pt + (3 * np.abs(np.sin(frame * 2*np.pi*elon_velocity))).astype(int)
            if (np.any(cut_curr_patch.shape == 0) or np.any(new_bot_left < 0) or np.any(new_bot_left >= img_size) or
                    np.any(polygon_pts[:, elon_ind] >= 2 * R) or np.any(polygon_pts[:, elon_ind] < 0)):
                new_bot_left = bot_left_corner + (velocity * (frame - 1)).astype(int)
                polygon_pts[:, elon_ind] = original_pt + (3 * np.abs(np.sin((frame-1) * 2*np.pi*elon_velocity))).astype(int)
                rr, cc = polygon(polygon_pts[0, :], polygon_pts[1, :])
                curr_polygon[rr, cc] = 1
                cut_curr_patch = cut_edges(curr_polygon, new_bot_left, img_size)
                mitochondrias[frame:, new_bot_left[0]:new_bot_left[0] + cut_curr_patch.shape[0],
                new_bot_left[1]:new_bot_left[1] + cut_curr_patch.shape[1]] += cut_curr_patch
                break
            rr, cc = polygon(polygon_pts[0, :], polygon_pts[1, :])
            curr_polygon[rr, cc] = 1
            cut_curr_patch = cut_edges(curr_polygon, new_bot_left, img_size)
            mitochondrias[frame, new_bot_left[0]:new_bot_left[0] + cut_curr_patch.shape[0], new_bot_left[1]:new_bot_left[1] + cut_curr_patch.shape[1]] += cut_curr_patch

    return mitochondrias

def generate_microtubules_sim(img_size, scale):
    # Model based on the paper: Shariff A, Murphy RF, Rohde GK.A generative model of microtubule distributions,
    # and indirect estimation of its parameters from fluorescence microscopy images.Cytometry A. 2010; 77(5): 457 - 66.
    # Matlab Version:
    # Elias Nehme, 11 / 11 / 2018
    # Python Version
    # Alon Saguy, 07 / 12 / 2021

    # === setup constants ===
    # pixel_size and image size
    obs_size = img_size / scale# [pixels]
    pixel_size = 0.16# [um]
    FOV_size = obs_size * pixel_size# [um]

    # step size
    gamma = 0.75# [um]

    # range of number of microtubules
    N = np.random.randint(1, int(img_size))
    if(N == 0):
        return np.zeros([img_size, img_size])

    # collinearity parameter
    cosa = 0.85

    # mean microtubule length
    mean_length = 10# [um]

    # length std
    std_length = 5# [um]

    # initialize the paths list
    paths = []

    # maximal and minimal microtubule length
    max_length = 5 * std_length
    min_length = 2 * std_length

    # loop over microtubules
    for i in range(N):
        Xi = []
        # random starting point and shift
        X0i = np.random.uniform(0, 1, [2, 1]) * FOV_size
        if (np.any(X0i >= FOV_size - pixel_size) or np.any(X0i < pixel_size)):
            Xi.append([0, 0])
            break
        # sample a random orientation for elongation
        alpha0 = np.random.uniform(0, 1) * 2 * np.pi
        # sample a length for the current microtubule
        lengthi = max(min(mean_length + std_length * np.random.uniform(0, 1), max_length), min_length)

        # elongate till length is achieved
        Xni = X0i
        Xi.append(X0i)
        vni = np.array([[np.cos(alpha0)], [np.sin(alpha0)]])
        lengthni = 0

        while lengthni <= lengthi:
            # sample a direction satisfying the stiffness constraint
            alphai = -np.arccos(cosa) + 2 * np.arccos(cosa) * np.random.uniform(0, 1)

            # generate a new direction vector
            vni = np.matmul(np.array([[np.cos(alphai), np.sin(alphai)], [-np.sin(alphai), np.cos(alphai)]]), vni)

            # take a step in this direction
            step_size = np.random.uniform(0.5, gamma)
            Xni = Xni + vni * step_size

            # length so far
            lengthni = lengthni + step_size

            if(np.any(Xni >= FOV_size) or np.any(Xni < pixel_size)):
                break

            # save path so far
            Xi.append(Xni)

        Xi = np.array(Xi)

        # plot resulting microtubules for visualization
        # plt.plot(Xi[:, 0], Xi[:, 1])
        # plt.title(i)
        # plt.xlim([0, FOV_size])
        # plt.ylim([0, FOV_size])

        # save current microtubule
        paths.append(Xi)

        # randomly add parallel microtubule
        probabilities = [0.05, 0.05, 0.05, 0.1, 0.25, 0, 0.25, 0.1, 0.05, 0.05, 0.05]
        if(np.random.uniform(0, 1) < 0.3):
            X2i_list = []
            shift = np.random.choice(np.arange(-5, 6), [2, 1], p=probabilities) * pixel_size / scale
            for j in range(Xi.shape[0]):
                if(Xi[j, 0] + shift[0] > FOV_size or Xi[j, 0] + shift[0] < 0 or
                        Xi[j, 1] + shift[1] > FOV_size or Xi[j, 1] + shift[1] < 0):
                    break
                X2i_list.append(Xi[j] + shift)
            if(len(X2i_list) > 0):
                X2i = np.array(X2i_list)
                paths.append(X2i)

    # plt.xlabel('x[um]')
    # plt.ylabel('y[um]')
    # plt.show()
    # decide on a refined pixel size
    refine_factor = 8
    paths = np.array(paths)

    # generate resulting image
    imtubes = np.zeros([img_size, img_size], dtype=np.uint8)
    for i in range(paths.shape[0]):
        # current microtubule path
        xy_ref = refine_points(paths[i], refine_factor)
        xy_tube = (xy_ref / (pixel_size / scale)).astype(int)

        # project locations on the grid and shift the center
        for j in range(xy_tube.shape[0]):
            imtubes[xy_tube[j, 0], xy_tube[j, 1]] = 1

    # plot resulting microtubules for visualization
    # plt.imshow(imtubes.transpose())
    # plt.gca().invert_yaxis()
    # plt.show()

    return imtubes

def refine_points(xy, Nr):
    # function refines a set of points Nr times using iterative mean interpolation refine Nr times
    xy_prev = xy[:, :, 0]

    for i in range(Nr):
        xy_refined = []
        xy_refined.append(xy_prev[0])
        for j in range(xy_prev.shape[0]-1):
            # extrapolated set of pts
            xy_extrap = (xy_prev[j, :] + xy_prev[j+1, :]) / 2

            # add the extrapolated points to the refined set of pts
            xy_refined.append(xy_extrap)
            xy_refined.append(xy_prev[j+1])

        # update previous set of pts
        xy_prev = np.array(xy_refined)

    return np.array(xy_refined)

def draw_PSF(PSF_size, loc, amp=1, sigma=1):
    x = np.arange(PSF_size[1])
    y = np.arange(PSF_size[0])

    random_shift = np.random.uniform(-0.25, 0.25, 2)
    random_amp = amp + 0.1*amp*np.random.uniform(-1, 1)
    x, y = np.meshgrid(x, y)
    PSF = (random_amp / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(
        -((x - loc[1] - random_shift[1]) ** 2 / (2 * sigma ** 2)
          + (y - loc[0] - random_shift[0]) ** 2 / (2 * sigma ** 2))))
    return PSF

def CreateDataLoader(X, y, batch_size):
    data_loader = torch.utils.data.DataLoader(TensorDataset(X, y), batch_size=batch_size)
    return data_loader

def apply_change(image, velocity, rot_speed):
    # Apply changes to an image according to velocity
    shifted = shift(image, velocity, order=3, mode='constant')
    return rotate(shifted, rot_speed)

def create_example_vid(name, obs):
    video_length = obs.shape[1]
    row = obs.shape[3]
    col = obs.shape[4]

    fps = 3.33

    out = cv2.VideoWriter('{}.mp4'.format(name), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (col, row))

    for i in range(video_length):
        img = np.array(np.repeat(obs[0][i].reshape([row, col, 1]), 3, axis=2), dtype=np.uint8)
        img[np.where(img < 0)] = 0
        out.write(img)

    out.release()

def cut_edges(patch, bot_left_corner, img_size):
    up = patch.shape[0] - np.max([0, bot_left_corner[0] + patch.shape[0] - img_size])
    right = patch.shape[1] - np.max([0, bot_left_corner[1] + patch.shape[1] - img_size])
    return patch[:up, :right].astype(np.uint)
