import os
from time import time

import numpy as np
import matplotlib.pyplot as plt

# My libraries
import src.bop_dataloader as data_loader
import src.segmentation as segmentation
import src.train as train
import src.util.utilities as util_
import src.util.flowlib as flowlib

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # TODO: Change this if you have more than 1 GPU

# BOP data set
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

train_list_path = "dataset/BOP/train_pbr/trainList_1010.txt"
dl = data_loader.get_BOP_train_dataloader(train_list_path, data_loading_params,
                                          batch_size=1, num_workers=10, shuffle=True)

# Train Depth Seeding Network
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

tb_dir = "log/bop_log/1010_1"  # TODO: change this to desired tensorboard directory
dsn_training_config = {

    # Training parameters
    'lr': 1e-4,  # learning rate
    'iter_collect': 20,  # Collect results every _ iterations
    'max_iters': 150000,

    # Loss function stuff
    'lambda_fg': 3.,
    'lambda_co': 5.,
    'lambda_sep': 1.,
    'lambda_cl': 1.,
    'num_seeds_training': 50,
    'delta': 0.1,  # for clustering loss. 2*eps
    'max_GMS_iters': 10,

    # Tensorboard stuff
    'tb_directory': os.path.join(tb_dir, 'train_DSN/'),
    'flush_secs': 10,  # Write tensorboard results every _ seconds
}

iter_num = 0
dsn_training_config.update({
    # Starting optimization from previous checkpoint
    'load' : False,
    'opt_filename' : os.path.join(dsn_training_config['tb_directory'],
                                  f'DSNTrainer_DSNWrapper_{iter_num}_checkpoint.pth'),
    'model_filename' : os.path.join(dsn_training_config['tb_directory'],
                                    f'DSNTrainer_DSNWrapper_{iter_num}_checkpoint.pth'),
})

dsn = segmentation.DSNWrapper(dsn_config)
dsn_trainer = train.DSNTrainer(dsn, dsn_training_config)

# Train the network for 1 epoch
num_epochs = 10
dsn_trainer.train(num_epochs, dl)
save_dir = "checkpoints/models_bop/1010_2/"
dsn_trainer.save(save_dir=save_dir)

# Plot some losses
fig = plt.figure(1, figsize=(15,3))
total_subplots = 5
starting_epoch = 0
info_items = {k:v for (k,v) in dsn_trainer.infos.items() if k > starting_epoch}

plt.subplot(1,total_subplots,1)
losses = [x['loss'] for (k,x) in info_items.items()]
plt.plot(info_items.keys(), losses)
plt.xlabel('Iteration')
plt.title('Losses. {0}'.format(losses[-1]))

plt.subplot(1,total_subplots,2)
fg_losses = [x['FG loss'] for (k,x) in info_items.items()]
plt.plot(info_items.keys(), fg_losses)
plt.xlabel('Iteration')
plt.title('Foreground Losses. {0}'.format(fg_losses[-1]))

plt.subplot(1,total_subplots,3)
co_losses = [x['Center Offset loss'] for (k,x) in info_items.items()]
plt.plot(info_items.keys(), co_losses)
plt.xlabel('Iteration')
plt.title('Center Offset Losses. {0}'.format(co_losses[-1]))

plt.subplot(1,total_subplots,4)
sep_losses = [x['Separation loss'] for (k,x) in info_items.items()]
plt.plot(info_items.keys(), sep_losses)
plt.xlabel('Iteration')
plt.title('Separation Losses. {0}'.format(sep_losses[-1]))

plt.subplot(1,total_subplots,5)
cl_losses = [x['Cluster loss'] for (k,x) in info_items.items()]
plt.plot(info_items.keys(), cl_losses)
plt.xlabel('Iteration')
plt.title('Cluster Losses. {0}'.format(cl_losses[-1]))

print("Number of iterations: {0}".format(dsn_trainer.iter_num))

