import os
from time import time

import numpy as np
import matplotlib.pyplot as plt

# My libraries
import src.bop_dataloader as data_loader
import src.grasp_dataloader as grasp_data_loader
import src.segmentation as segmentation
import src.util.utilities as util_
import src.util.flowlib as flowlib

# Run the network on a single batch, and plot the results
# dataloader
data_loading_params = {

    # Camera/Frustum parameters
    'img_width': 671,
    'img_height': 502,
    'fx': 1122.375,
    'fy': 1122.375,
    'x_offset': 334.4445,
    'y_offset': 264.443075,

    'use_data_augmentation': True,

    # Multiplicative noise
    'gamma_shape': 1000.,
    'gamma_scale': 0.001,

    # Additive noise
    'gaussian_scale_range': [0., 0.003],  # up to 2.5mm standard dev
    'gp_rescale_factor_range': [12, 20],  # [low, high (exclusive)]

    # Random ellipse dropout
    'ellipse_dropout_mean': 10,
    'ellipse_gamma_shape': 5.0,
    'ellipse_gamma_scale': 1.0,

    # Random high gradient dropout
    'gradient_dropout_left_mean': 15,
    'gradient_dropout_alpha': 2.,
    'gradient_dropout_beta': 5.,

    # Random pixel dropout
    'pixel_dropout_alpha': 0.2,
    'pixel_dropout_beta': 10.,

}
test_list_path = "dataset/BOP/train_pbr/valid_list_1010.txt"
test_grasp_path = "dataset/BOP/train_pbr/test_grasp_1010.txt"
# dl = data_loader.get_BOP_test_dataloader(test_list_path, data_loading_params, batch_size=1)
dl = grasp_data_loader.get_grasp_test_dataloader(test_list_path, test_grasp_path, data_loading_params, batch_size=1)

# DSN model
dsn_config = {

    # Sizes
    'feature_dim': 64,  # 32 would be normal

    # Mean Shift parameters (for 3D voting)
    'max_GMS_iters': 10,
    'num_seeds': 200,  # Used for MeanShift, but not BlurringMeanShift
    'epsilon': 0.05,  # Connected Components parameter
    'sigma': 0.02,  # Gaussian bandwidth parameter
    'subsample_factor': 5,
    'min_pixels_thresh': 500,

    # Differentiable backtracing params
    'tau': 15.,
    'M_threshold': 0.3,

    # Robustness stuff
    'angle_discretization': 100,
    'discretization_threshold': 0.,

}
checkpoint_dir = './checkpoints/models_3d/' # TODO: change this to directory of downloaded models
dsn_filename = '/home/yumi/Project/uois3d/checkpoints/models_grasp/DSNWrapper_iter143841_64c_checkpoint.pth'
dsn = segmentation.DSNWrapper(dsn_config)
dsn.load(dsn_filename)
dsn.eval_mode()

# test
test_num = dl.__len__()
for i, batch in enumerate(dl):
    ### Compute segmentation masks ###
    rgb_imgs = util_.torch_to_numpy(batch['rgb'], is_standardized_image=True)
    xyz_imgs = util_.torch_to_numpy(batch['xyz'])  # Shape: [N x H x W x 3]
    foreground_labels = util_.torch_to_numpy(batch['foreground_labels'])  # Shape: [N x H x W]
    center_offset_labels = util_.torch_to_numpy(batch['center_offset_labels'])  # Shape: [N x 2 x H x W]
    # obj_num = util_.torch_to_numpy(batch['view_num'])
    obj_num = 2
    N, H, W = foreground_labels.shape[:3]

    st_time = time()
    fg_masks, center_offsets, object_centers, initial_masks = dsn.run_on_batch(batch)
    total_time = time() - st_time
    print('Total time taken for Segmentation: {0} seconds'.format(round(total_time, 3)))
    print('FPS: {0}'.format(round(N / total_time, 3)))

    # Get segmentation masks in numpy
    fg_masks = fg_masks.cpu().numpy()
    center_offsets = center_offsets.cpu().numpy().transpose(0, 2, 3, 1)
    initial_masks = initial_masks.cpu().numpy()
    for j in range(len(object_centers)):
        object_centers[j] = object_centers[j].cpu().numpy()

    fig_index = 1
    fig = plt.figure(fig_index);
    fig_index += 1
    fig.set_size_inches(20, 5)

    # Plot image
    plt.subplot(2, 3, 1)
    plt.imshow(rgb_imgs[0, ...].astype(np.uint8))
    plt.title('Image {0}/{1}'.format(i + 1, test_num))

    # Plot Depth
    plt.subplot(2, 3, 2)
    plt.imshow(xyz_imgs[0, ..., 2])
    plt.title('Depth')

    # Plot labels
    plt.subplot(2, 3, 3)
    plt.imshow(util_.get_color_mask(foreground_labels[0, ...]))
    plt.title("ground truth, object number:{}".format(obj_num))

    # Plot prediction
    plt.subplot(2, 3, 4)
    plt.imshow(util_.get_color_mask(fg_masks[0, ...]))
    plt.title("Predicted foreground")

    # Plot Center Direction Predictions
    plt.subplot(2, 3, 5)
    fg_mask = np.expand_dims(fg_masks[0, ...] == 2, axis=-1)
    plt.imshow(flowlib.flow_to_image(center_offsets[0, ...] * fg_mask))
    plt.title("Center Direction Predictions")

    # Plot Initial Masks
    plt.subplot(2, 3, 6)
    plt.imshow(util_.get_color_mask(initial_masks[0, ...]))
    plt.title(f"Initial Masks. #objects: {np.unique(initial_masks[0, ...]).shape[0] - 1}")

    plt.waitforbuttonpress()

