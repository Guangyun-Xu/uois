import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import cv2
cv2.setNumThreads(0)
import json
from PIL import Image
import pybullet as p

import sys

# sys.path.append("./")
# print(sys.path)

from src.util import utilities as util_
from src import data_augmentation

BACKGROUND_LABEL = 0
TABLE_LABEL = 1
OBJECTS_LABEL = 2


###### Some utilities #####
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


class BOPDataset(Dataset):
    """
    Data loader for BOP dataset.
    """

    def __init__(self, data_list_path, config):
        self.data_list = util_.read_lines(data_list_path)
        self.data_dir = os.path.dirname(data_list_path)
        self.config = config

    def process_label_3D(self, foreground_labels, xyz_img, scene_description):
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

        # # Get inverse of camera extrinsics matrix. This is called "view_matrix" in OpenGL jargon
        # view_num = len(scene_description)
        # if view_num == 0: # bg-only image
        #     camera_dict = scene_description['views']['background']
        # elif view_num == 1: # bg+table image
        #     key = 'background+tabletop' if 'v6' in self.base_dir else 'background+table'
        #     camera_dict = scene_description['views'][key] # table for TODv5, tabletop for TODv6
        # else: # bg+table+objects image
        #     key = 'background+tabletop' if 'v6' in self.base_dir else 'background+table'
        #     camera_dict = scene_description['views'][key + '+objects'][view_num-2]
        # view_matrix = p.computeViewMatrix(camera_dict['camera_pos'],
        #                                   camera_dict['lookat_pos'],
        #                                   camera_dict['camera_up_vector']
        #                                  )
        # view_matrix = np.array(view_matrix).reshape(4,4, order='F')
        #
        # Compute object centers and directions
        H, W = foreground_labels.shape
        offsets = np.zeros((H, W, 3), dtype=np.float32)
        cf_3D_centers = np.zeros((100, 3), dtype=np.float32)  # 100 max object centers, cf:camera frame
        obj_list = np.unique(foreground_labels)
        for i, k in enumerate(np.unique(foreground_labels)):

            # Get mask
            mask = foreground_labels == k
            mask_num = mask.nonzero()[0].size

            # For background/table, prediction direction should point towards origin
            if k in [BACKGROUND_LABEL, TABLE_LABEL]:
                offsets[mask, ...] = 0
                continue

            # Compute 3D object centers in camera frame
            idx = k - OBJECTS_LABEL
            object_pose = scene_description[idx]
            center_in_camera = object_pose['cam_t_m2c']
            for j in range(len(center_in_camera)):
                center_in_camera[j] = center_in_camera[j] / 1000

            # wf_3D_center = np.array(scene_description['object_descriptions'][idx]['axis_aligned_bbox3D_center'])
            # cf_3D_center = view_matrix.dot(np.append(wf_3D_center, 1.))[:3] # in OpenGL camera frame (right-hand system, z-axis pointing backward)
            # cf_3D_center[2] = -1 * cf_3D_center[2] # negate z to get the left-hand system, z-axis pointing forward

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

    def get_item(self, item_name):
        words = item_name.split()
        folder_name = words[0]
        scene_id = words[1]
        object_num = int(words[2])


        cv2.setNumThreads(0)  # some hack to make sure pyTorch doesn't deadlock. Found at
        # https://github.com/pytorch/pytorch/issues/1355. Seems to work for me

        # RGB image
        data_dir = self.data_dir
        data_dir_abs = os.path.abspath(data_dir)

        folder_name = os.path.join(data_dir_abs, folder_name)
        rgb_path = os.path.join(folder_name, "rgb/{}.jpg".format(scene_id))

        with Image.open(rgb_path) as di:
            rgb_img = np.array(di)
        # rgb_img = cv2.imread(rgb_path, -1)
        # print(rgb_img)
        # rgb_img = rgb_img[:, :, ::-1].copy()
        # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = self.process_rgb(rgb_img)  # random color warping

        # Depth image
        depth_path = os.path.join(folder_name, "depth/{}.png".format(scene_id))
        depth_img = cv2.imread(depth_path,
                               cv2.IMREAD_ANYDEPTH)  # This reads a 16-bit single-channel image. Shape: [H x W]
        xyz_img = self.process_depth(depth_img)  # cover depth to xyz image

        # labels
        image_shape = depth_img.shape
        foreground_labels = np.zeros(image_shape).astype("uint8")
        for i in range(object_num):
            mask_name = "{}_{:0>6d}".format(scene_id, int(i))
            mask_path = os.path.join(folder_name, "mask_visib/{}.png".format(mask_name))
            mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
            mask = (np.asarray(mask > 0)).astype("uint8")
            mask_idx = mask > 0
            mask_idx = mask_idx.nonzero()
            # mask_size = mask_idx[0].size
            foreground_labels[mask_idx] = i + 2  # 0:background, 1:box, other:foreground
        scene_description = get_scene_data(folder_name, scene_id)
        center_offset_labels, object_centers = self.process_label_3D(
            foreground_labels, xyz_img, scene_description
        )

        view_num = len(scene_description)
        # Turn these all into torch tensors
        rgb_img = data_augmentation.array_to_tensor(rgb_img)  # Shape: [3 x H x W]
        xyz_img = data_augmentation.array_to_tensor(xyz_img)  # Shape: [3 x H x W]
        foreground_labels = data_augmentation.array_to_tensor(foreground_labels)  # Shape: [H x W]
        center_offset_labels = data_augmentation.array_to_tensor(center_offset_labels)  # Shape: [2 x H x W]
        object_centers = data_augmentation.array_to_tensor(object_centers)  # Shape: [100 x 3]
        num_3D_centers = torch.tensor(np.count_nonzero(np.unique(foreground_labels) >= OBJECTS_LABEL))

        return {'rgb': rgb_img,
                'xyz': xyz_img,
                'foreground_labels': foreground_labels,
                'center_offset_labels': center_offset_labels,
                'object_centers': object_centers,
                # This is gonna bug out because the dimensions will be different per frame
                'num_3D_centers': num_3D_centers,
                'scene_dir': "",
                'view_num': view_num,
                'label_abs_path': "",
                }

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item_name = self.data_list[idx]
        data = self.get_item(item_name)

        while data is None:
            print("to few points:{}".format(idx))
            idx = np.random.randint(0, len(self.data_list))
            item_name = self.data_list[idx]
            print("replaced by :{}".format(idx))
            data = self.get_item(item_name)

        return data


def get_BOP_train_dataloader(data_list_path, config, batch_size=8, num_workers=10, shuffle=True):
    config = config.copy()
    dataset = BOPDataset(data_list_path, config)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)


def main():
    # save .npy in example
    data_list = "../dataset/BOP/train_pbr/trainList_1010.txt"
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
    data_set = BOPDataset(data_list, config)

    for i in range(4):
        data = data_set.__getitem__(i)
        data_d = {
            'rgb': data['rgb'].cpu().numpy().transpose(1, 2, 0),
            'xyz': data['xyz'].cpu().numpy().transpose(1, 2, 0),
            'label': data['foreground_labels'].cpu().numpy()
        }
        np.save("../example_images/bop_data/bop_image_{}.npy".format(i), data_d)

        rgb_img = data['rgb'].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        xyz_img = data['xyz'].cpu().numpy().transpose(1, 2, 0)
        foreground_img = util_.get_color_mask(data['foreground_labels'].cpu().numpy()).astype(np.uint8)
        center_offset_labels = data['center_offset_labels'].cpu().numpy().transpose(1, 2, 0)

        xyz_img_ = xyz_img * 100
        # cv2.imshow("xyz_img", xyz_img_)
        # cv2.waitKey(0)
        xyz_img_ = xyz_img_.astype(np.uint8)
        b, g, r = cv2.split(xyz_img_)
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        xyz_img_ = cv2.merge([bH, gH, rH])
        # cv2.imshow("dst", result)
        # cv2.waitKey(0)

        center_offset_labels_ = np.sum(np.absolute(center_offset_labels), axis=2)
        offset_mask = (center_offset_labels_ < 5e-2) & (-5e-2 < center_offset_labels_)
        center_offset_labels[offset_mask, ...] = 255
        center_offset_labels_ = np.absolute(center_offset_labels) * 1500
        offset_mask = center_offset_labels_[..., 1] > 250
        center_offset_labels_[offset_mask, ...] = 255
        center_offset_labels_ = center_offset_labels_.astype(np.uint8)
        # cv2.imshow("center_offset_labels", center_offset_labels_)
        # cv2.waitKey(0)

        images = [rgb_img, xyz_img_, foreground_img, center_offset_labels_]
        titles = [f'Image {i + 1}', 'xyz',
                  f"foreground. #objects: {data['view_num']}",
                  f"center_offset"
                  ]
        util_.subplotter(images, titles, fig_num=i + 1)


if __name__ == '__main__':
    # cv2.setNumThreads(0)
    main()
