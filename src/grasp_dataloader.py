import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

cv2.setNumThreads(0)
import json
import matplotlib.pyplot as plt

from src.util import utilities as util_
from src import data_augmentation
import src.util.flowlib as flowlib

BACKGROUND_LABEL = 0
OTHER_LABEL = 1
TARGET_LABEL = 2  # objects of best suction and grasping


###### Some utilities #####
def get_mask_idx(mask_name:str):
    mask_id = mask_name[7:]
    mask_id = mask_id.lstrip('0')
    mask_id = mask_id.rsplit('.png')[0]
    if mask_id == '':
        mask_id = '0'
    return int(mask_id)

def worker_init_fn(worker_id):
    """ Use this to bypass issue with PyTorch dataloaders using deterministic RNG for Numpy
        https://github.com/pytorch/pytorch/issues/5059
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_scene_data(folder_path, scene_id):
    scene_info_path = os.path.join(folder_path, "scene_gt.json")
    with open(scene_info_path, 'r') as f2:
        scene_info = json.load(f2)
        scene_id = scene_id.lstrip('0')
        if scene_id == '':
            scene_id = '0'
        scene_date = scene_info[scene_id]
        return scene_date


def check_data_align(data_list, grasp_list):
    data_len = min(len(data_list), len(grasp_list))
    for i in range(data_len):
        data_item = data_list[i]
        grasp_item = grasp_list[i]
        data_words = data_item.split()
        grasp_words = grasp_item.split()
        data_scene_id = data_words[1]
        grasp_scene_id = grasp_words[0]

        if data_scene_id == grasp_scene_id:
            # print("{0}={1}".format(data_sence_id, grasp_sence_id))
            continue
        else:
            print("{0}!={1}".format(data_scene_id, grasp_scene_id))
            return False

    return True


class GraspDataloader(Dataset):
    """
    Data loader for best grasp data
    """

    def __init__(self, data_list_path, grasp_list_path, config):
        self.data_list = util_.read_lines(data_list_path)
        self.grasp_list = util_.read_lines(grasp_list_path)
        self.data_dir = os.path.dirname(data_list_path)
        self.config = config
        check_data_align(self.data_list, self.grasp_list)

    def process_label_3D(self, foreground_labels, xyz_img, scene_description, suction_name, grasp_name):
        """ Process foreground_labels

            @param foreground_labels: a [H x W] numpy array of labels
            @param xyz_img: a [H x W x 3] numpy array of xyz coordinates (in left-hand coordinate system)
            @param scene_description: a Python dictionary describing scene

            @return: foreground_labels
                     offsets: a [H x W x 2] numpy array of 2D directions. The i,j^th element has (y,x) direction to object center
        """

        # Any zero depth value will have foreground label set to background
        foreground_labels = foreground_labels.copy()
        foreground_labels[xyz_img[..., 2] == 0] = 0


        # Compute object centers and directions
        H, W = foreground_labels.shape
        offsets = np.zeros((H, W, 3), dtype=np.float32)
        cf_3D_centers = np.zeros((2, 3), dtype=np.float32)  # 2 max object centers, cf:camera frame
        obj_list = np.unique(foreground_labels)
        for i, k in enumerate(np.unique(foreground_labels)):

            # Get mask
            mask = foreground_labels == k
            mask_num = mask.nonzero()[0].size

            # For background/table, prediction direction should point towards origin
            if k in [BACKGROUND_LABEL, OTHER_LABEL]:
                offsets[mask, ...] = 0
                continue

            if k == 2:
                # Compute 3D object centers in camera frame
                mask_id = get_mask_idx(suction_name)
                object_pose = scene_description[mask_id]
                center_in_camera = object_pose['cam_t_m2c']
                for j in range(len(center_in_camera)):
                    center_in_camera[j] = center_in_camera[j] / 1000.

                # If center isn't contained within the object, use point cloud average
                if center_in_camera[0] < xyz_img[mask, 0].min() or \
                        center_in_camera[0] > xyz_img[mask, 0].max() or \
                        center_in_camera[1] < xyz_img[mask, 1].min() or \
                        center_in_camera[1] > xyz_img[mask, 1].max():
                    center_in_camera = xyz_img[mask, ...].mean(axis=0)

                # Get directions
                cf_3D_centers[k - 2] = center_in_camera
                object_center_offsets = (center_in_camera - xyz_img).astype(np.float32)  # Shape: [H x W x 3]

                # Add it to the labels
                offsets[mask, ...] = object_center_offsets[mask, ...]

            if k==3:
                # Compute 3D object centers in camera frame
                mask_id = get_mask_idx(grasp_name)
                object_pose = scene_description[mask_id]
                center_in_camera = object_pose['cam_t_m2c']
                for j in range(len(center_in_camera)):
                    center_in_camera[j] = center_in_camera[j] / 1000.

                # If center isn't contained within the object, use point cloud average
                if center_in_camera[0] < xyz_img[mask, 0].min() or \
                        center_in_camera[0] > xyz_img[mask, 0].max() or \
                        center_in_camera[1] < xyz_img[mask, 1].min() or \
                        center_in_camera[1] > xyz_img[mask, 1].max():
                    center_in_camera = xyz_img[mask, ...].mean(axis=0)

                # Get directions
                cf_3D_centers[k - 2] = center_in_camera
                object_center_offsets = (center_in_camera - xyz_img).astype(np.float32)  # Shape: [H x W x 3]

                # Add it to the labels
                offsets[mask, ...] = object_center_offsets[mask, ...]

        return offsets, cf_3D_centers

    def process_rgb(self, rgb_img):
        """ Process RGB image
                - random color warping
        """
        rgb_img = rgb_img.astype(np.float32)

        if self.config['use_data_augmentation']:
            # rgb_img = data_augmentation.random_color_warp(rgb_img)
            pass
        rgb_img = data_augmentation.standardize_image(rgb_img)

        return rgb_img

    def process_depth(self, depth_img):
        """ Process depth channel
                TODO: CHANGE THIS
                - change from millimeters to meters
                - cast to float32 data type
                - add random noise
                - compute xyz ordered point cloud
        """

        # 0.1millimeters -> meters
        depth_mask = depth_img > 20000
        depth_img[depth_mask, ...] = 0
        depth_img = (depth_img / 10000.).astype(np.float32)

        # add random noise to depth
        if self.config['use_data_augmentation']:
            depth_img = data_augmentation.add_noise_to_depth(depth_img, self.config)
            # depth_img = data_augmentation.dropout_random_ellipses(depth_img, self.config)

        # Compute xyz ordered point cloud
        xyz_img = util_.compute_xyz(depth_img, self.config)
        if self.config['use_data_augmentation']:
            xyz_img = data_augmentation.add_noise_to_xyz(xyz_img, depth_img, self.config)

        return xyz_img

    def get_item(self, item_name, grasp_item_name):
        data_words = item_name.split()
        grasp_words = grasp_item_name.split()
        folder_name = data_words[0]
        data_scene_id = data_words[1]
        object_num = int(data_words[2])
        grasp_scene_id = grasp_words[0]
        suction_name = grasp_words[1]
        grasp_name = grasp_words[2]
        # print("scene id:{}".format(data_scene_id))

        if data_scene_id != grasp_scene_id:
            print("{0}!={1}".format(data_scene_id, grasp_scene_id))
            return None

        # RGB image
        data_dir = self.data_dir
        data_dir_abs = os.path.abspath(data_dir)
        folder_name = os.path.join(data_dir_abs, folder_name)
        rgb_path = os.path.join(folder_name, "rgb/{}.jpg".format(data_scene_id))
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = self.process_rgb(rgb_img)  # random color warping
        rgb_img = rgb_img[:480, :640, ...]

        # Depth image
        depth_path = os.path.join(folder_name, "depth/{}.png".format(data_scene_id))
        depth_img = cv2.imread(depth_path,
                               cv2.IMREAD_ANYDEPTH)  # This reads a 16-bit single-channel image. Shape: [H x W]
        xyz_img = self.process_depth(depth_img)[:480, :640, ...]  # cover depth to xyz image

        # labels
        image_shape = depth_img.shape
        target_labels = np.zeros(image_shape).astype("uint8")
        for i in range(object_num):
            mask_name = "{}_{:0>6d}.png".format(data_scene_id, int(i))
            mask_path = os.path.join(folder_name, "mask_visib/{}".format(mask_name))
            mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
            mask = (np.asarray(mask > 0)).astype("uint8")
            mask_idx = mask > 0
            mask_idx = mask_idx.nonzero()
            if (mask_name != grasp_name) & (mask_name != suction_name):
                target_labels[mask_idx] = 1  # 1:other objects
            elif mask_name == grasp_name:
                target_labels[mask_idx] = 2  # 2:suction object
            elif mask_name == suction_name:
                target_labels[mask_idx] = 3  # 3:grasping object
            else:
                print("target labels error")
        scene_description = get_scene_data(folder_name, data_scene_id)
        foreground_labels = target_labels[:480, :640, ...]
        center_offset_labels, object_centers = self.process_label_3D(
            foreground_labels, xyz_img, scene_description, suction_name, grasp_name
        )

        view_num = len(scene_description)
        # Turn these all into torch tensors
        rgb_img = data_augmentation.array_to_tensor(rgb_img)  # Shape: [3 x H x W]
        xyz_img = data_augmentation.array_to_tensor(xyz_img)  # Shape: [3 x H x W]
        foreground_labels = data_augmentation.array_to_tensor(foreground_labels)  # Shape: [H x W]
        center_offset_labels = data_augmentation.array_to_tensor(center_offset_labels)  # Shape: [2 x H x W]
        object_centers = data_augmentation.array_to_tensor(object_centers)  # Shape: [100 x 3]
        num_3D_centers = torch.tensor(np.count_nonzero(np.unique(foreground_labels) >= TARGET_LABEL))

        return {'rgb': rgb_img,
                'xyz': xyz_img,
                'foreground_labels': foreground_labels,
                'center_offset_labels': center_offset_labels,
                'object_centers': object_centers,
                # This is gonna bug out because the dimensions will be different per frame
                'num_3D_centers': num_3D_centers,
                'scene_dir': "",
                'view_num': "",
                'label_abs_path': "",
                }


    def __len__(self):
        return min(len(self.data_list), len(self.grasp_list))

    def __getitem__(self, idx):
        item_name = self.data_list[idx]
        grasp_item_name = self.grasp_list[idx]
        data = self.get_item(item_name, grasp_item_name)

        while data is None:
            print("to few points:{}".format(idx))
            idx = np.random.randint(0, len(self.data_list))
            item_name = self.data_list[idx]
            grasp_item_name = self.grasp_list[idx]
            print("replaced by :{}".format(idx))
            data = self.get_item(item_name, grasp_item_name)

        return data

def get_grasp_train_dataloader(data_list_path, grasp_list_path, config, batch_size=8, num_workers=10, shuffle=True):
    config = config.copy()
    dataset = GraspDataloader(data_list_path, grasp_list_path, config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)

def get_grasp_test_dataloader(data_list_path, grasp_list_path, config, batch_size=8, num_workers=10, shuffle=False):
    config = config.copy()
    dataset = GraspDataloader(data_list_path, grasp_list_path, config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)


def main():
    plot = False

    data_list_path = "../dataset/BOP/train_pbr/train_list_1010.txt"
    grasp_list_path = "../dataset/BOP/train_pbr/train_grasp_1010.txt"
    config = {
        # Camera/Frustum parameters
        'img_width': 671,
        'img_height': 502,
        'fx': 1122.375,
        'fy': 1122.375,
        'x_offset': 334.4445,
        'y_offset': 264.443075,

        'use_data_augmentation': False,

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
    ds = GraspDataloader(data_list_path, grasp_list_path, config)
    dl = get_grasp_test_dataloader(data_list_path, grasp_list_path, config, batch_size=1)
    test_num = dl.__len__()

    for i in range(4):
        data = ds.__getitem__(i)

    for i, batch in enumerate(dl):
        ### Compute segmentation masks ###
        rgb_imgs = util_.torch_to_numpy(batch['rgb'], is_standardized_image=True)
        xyz_imgs = util_.torch_to_numpy(batch['xyz'])  # Shape: [N x H x W x 3]
        foreground_labels = util_.torch_to_numpy(batch['foreground_labels'])  # Shape: [N x H x W]
        center_offset_labels = util_.torch_to_numpy(batch['center_offset_labels'])  # Shape: [N x 2 x H x W]
        obj_num = util_.torch_to_numpy(batch['view_num'])
        N, H, W = foreground_labels.shape[:3]

        if plot:
            fig_index = 1
            fig = plt.figure(fig_index);
            fig_index += 1
            fig.set_size_inches(20, 5)

            # Plot image
            plt.subplot(1, 4, 1)
            plt.imshow(rgb_imgs[0, ...].astype(np.uint8))
            plt.title('Image {0}/{1}'.format(i + 1, test_num))

            # Plot Depth
            plt.subplot(1, 4, 2)
            plt.imshow(xyz_imgs[0, ..., 2])
            plt.title('Depth')

            # Plot labels
            plt.subplot(1, 4, 3)
            plt.imshow(util_.get_color_mask(foreground_labels[0, ...]))
            plt.title("ground truth, object number:{}".format(obj_num))

            # Plot Center Direction Predictions
            plt.subplot(1, 4, 4)
            plt.imshow(flowlib.flow_to_image(center_offset_labels[0, ...]))
            plt.title("Center Direction Predictions")

            plt.waitforbuttonpress()


if __name__ == '__main__':
    main()