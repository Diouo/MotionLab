import argparse
import os
from visualize import vis_utils
import shutil
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='stick figure mp4 file to be rendered.')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    params = parser.parse_args()

    npy_path = params.input_path
    results_dir = params.input_path.replace('.npy', '_obj')
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)
    motions = np.load(npy_path, allow_pickle=True)
    batch = motions.shape[1]

    for batch_i in range(batch):  
        result_i_path = results_dir + '/result_{}/'.format(batch_i) 
        os.makedirs(result_i_path)
        npy2obj = vis_utils.npy2obj_mld(npy_path,batch_i,device=params.device, cuda=params.cuda)
        for frame_i in range(npy2obj.real_num_frames):
            npy2obj.save_obj(os.path.join(result_i_path, 'human_obj_{}.obj'.format(frame_i)), frame_i)
