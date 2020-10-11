import os
import matplotlib.pyplot as plt

# My libraries
import src.grasp_dataloader as data_loader
import src.segmentation as segmentation
import src.train as train

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # TODO: Change this if you have more than 1 GPU

# grasp data set
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
data_list_path = "dataset/BOP/train_pbr/train_list_1010.txt"
grasp_list_path = "dataset/BOP/train_pbr/train_grasp_1010.txt"
dl = data_loader.get_grasp_train_dataloader(data_list_path, grasp_list_path, data_loading_params,
                                            batch_size=1, num_workers=10)

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

tb_dir = "log/bop_log/1011_4/"  # TODO: change this to desired tensorboard directory
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
    'load' : True,
    'opt_filename' : "/home/yumi/Project/uois3d/checkpoints/models_grasp/DSNTrainer_DSNWrapper_iter53941_checkpoint.pth",
    'model_filename' : "/home/yumi/Project/uois3d/checkpoints/models_grasp/DSNWrapper_iter53941_64c_checkpoint.pth",
})

dsn = segmentation.DSNWrapper(dsn_config)
dsn_trainer = train.DSNTrainer(dsn, dsn_training_config)

# Train the network for 1 epoch
num_epochs = 100
dsn_trainer.train(num_epochs, dl)
save_dir = "checkpoints/models_grasp/"
dsn_trainer.save(save_dir=save_dir)
