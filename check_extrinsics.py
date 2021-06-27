import os
import json
import numpy as np
import matplotlib as plt
from util.camera_pose_visualizer import CameraPoseVisualizer
from util.camera_parameter_loader import CameraParameterLoader


def getExtrinsicsFromJson(read_name):
    """get extrinsic parameters from json file"""
    with open(read_name) as json_file:
        data = json.load(json_file)
    ext_params = []
    for frame in data['frames']:
        ext_params.append(np.array(frame['transform_matrix']))
    return ext_params


def getExtrinsicsFromNumpy(read_name):
    """load extrinsics from numpy file"""
    data = np.load(read_name)
    print(data.shape)


def plot_framewise(ext_params):
    visualizer = CameraPoseVisualizer([-10, 10], [-10, 10], [0, 20])
    n_params = len(ext_params)
    for i, param in enumerate(ext_params):
        visualizer.extrinsic2pyramid(param, plt.cm.rainbow(i/n_params), 2)
    visualizer.colorbar(n_params)
    visualizer.show()


if __name__ == '__main__':    
    ext_params = getExtrinsicsFromJson(os.path.join(
        'Z:\\Github\\nerf-pytorch\\data\\nerf_synthetic\\lego', 'transforms_val.json'))
    plot_framewise(ext_params)

    # getExtrinsicsFromNumpy(os.path.join(
    #     'Z:\\Github\\nerf-pytorch\\data\\nerf_llff_data\\fern', 'poses_bounds.npy'))


