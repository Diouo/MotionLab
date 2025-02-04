import os
import random
import pickle
import trimesh

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from pathlib import Path

from visualize import vis_utils
from rfmotion.config import parse_args
from rfmotion.data.get_data import get_datasets
from rfmotion.models.get_model import get_model
from rfmotion.utils.logger import create_logger
from rfmotion.render.mesh_viz import render_motion
from rfmotion.models.tools.tools import pack_to_render
from rfmotion.render.video import get_offscreen_renderer

def rotate_2D_z(theta,x):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    return np.matmul(x,R.transpose())

def rotate_3D_z(theta,x):
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return np.matmul(x, R.transpose())

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

def transform_smpl(trans,orient):
    trans_start = trans[0,:].clone()
    trans -= trans_start
    trans[:,1:] = rotate_2D_z(np.pi/2, trans[:,1:])

    orient_ = np.zeros((orient.shape[0],6),dtype=np.float32)
    R1 = np.array([
        [1, 0, 0],
        [0,np.cos(np.pi/2), -np.sin(np.pi/2)],
        [0,np.sin(np.pi/2), np.cos(np.pi/2)],
    ], dtype=np.float32)

    for i in range(len(orient)):
        orient_i = orient[i]
        orient_i = rotation_6d_to_matrix(orient_i)
        orient_i = np.matmul(R1, orient_i)
        orient_i = matrix_to_rotation_6d(orient_i)
        orient_[i,:] = orient_i.numpy().astype(np.float32)

    trans[:,0] +=  trans_start[0]
    trans[:,1] -=  trans_start[2]
    trans[:,2] +=  trans_start[1]

    return trans, torch.Tensor(orient_)

def transform_motion(motion):
    origin = motion[0,0,:3].clone().unsqueeze(0).unsqueeze(0)
    motion -= origin
    motion[:,:,1:] = rotate_2D_z(np.pi/2,motion[:,:,1:])

    motion[:,:,0] += origin[0,0,0] 
    motion[:,:,1] -= origin[0,0,2]
    motion[:,:,2] += origin[0,0,1]

    return  motion

def render_video(r, motion, filename, text=None, hint=None, hint_masks=None, color=(160 / 255, 160 / 255, 160 / 255, 1.0)):
    if not isinstance(motion, list):
        smpl_params = pack_to_render(trans=motion[..., :3], rots=motion[..., 3:], pose_repr='6d')
    else:
        smpl_params = []
        for m in motion:
            smpl_params.append(pack_to_render(trans=m[..., :3], rots=m[..., 3:], pose_repr='6d'))
        color=[(0/ 255, 0 / 255, 255 / 255, 1.0), (255 / 255, 0 / 255, 0 / 255, 1.0)]

    render_motion(r, smpl_params, pose_repr='aa',filename=filename, text_for_vid=text, hint=hint, hint_masks=hint_masks, color=color)

def demo_text_based_gen(cfg, output_dir, r, model, dataset):
    test_dataloader = dataset.test_dataloader()
    for i, data in enumerate(test_dataloader):
        text = data['text'][0]
        length = data['length'][0]

        if length < 90:
            continue

        for j in range(cfg.DEMO.REPLICATION):
            result = model.demo_text(data)
            generated_motion = result['generated_motion']
            seq_len = generated_motion.shape[0]
            xyz2smpl = vis_utils.npy2obj_mld(generated_motion, device=model.device)

            ## output obj
            for k in range(seq_len):
                smpl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'obj')
                if not os.path.exists(smpl_output_path):
                    os.makedirs(smpl_output_path, exist_ok=True)
                xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)

            body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
            body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
            body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
            body_transl, body_orient = transform_smpl(body_transl, body_orient)

            # ## output pkl 
            # pkl = {}
            # pkl["smpl_trans"] = body_transl
            # pkl["smpl_poses"] = torch.cat([body_orient, body_pose, torch.zeros(seq_len, 6)], dim=1)
            # pkl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'motion.pkl')
            # with open(pkl_output_path, 'wb') as f:
            #     pickle.dump(pkl, f)
            
            ## output video
            motion = torch.cat([body_transl, body_orient, body_pose], dim=1)
            render_video(r, motion, os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'text'), text=text, color=(0.863158, 0.345263, 0.36263, 1.0))  

def demo_hint_based_gen(cfg, output_dir, r, model, dataset):
    test_dataloader = dataset.test_dataloader()
    for i, data in enumerate(test_dataloader):
        text = data['text'][0]
        length = data['length'][0]

        if length < 60 or i < 600:
            continue
        print(text)

        for j in range(cfg.DEMO.REPLICATION):
            result = model.demo_hint(data)
            target_motion = result['target_motion']
            generated_motion = result['generated_motion']
            hint_masks = result['hint_masks']
            seq_len = generated_motion.shape[0]
            
            distance = np.linalg.norm((target_motion - generated_motion) * hint_masks, axis=-1).sum() / hint_masks.sum()
            print(distance)
            if distance > 0.01:
                continue

            # load xyz2smpl for visualization
            xyz2smpl = vis_utils.npy2obj_mld(generated_motion, device=model.device)

            ## output obj
            for k in range(seq_len):
                smpl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'obj')
                if not os.path.exists(smpl_output_path):
                    os.makedirs(smpl_output_path, exist_ok=True)
                xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)

            body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
            body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
            body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
            body_transl, body_orient = transform_smpl(body_transl, body_orient)
            motion = torch.cat([body_transl, body_orient, body_pose], dim=1)
            hint = transform_motion(result['target_motion']) 
            hint_masks=result['hint_masks']

            ball_list = []
            for k in range(hint_masks.shape[0]):
                for l in range(hint_masks.shape[1]):
                    if hint_masks[k, l, 0] == 1:
                        hint_mesh = trimesh.primitives.Sphere(radius=0.02, center=hint[k,l,:])
                        hint_mesh.visual.vertex_colors = [1, 0, 0, 1]
                        ball_list.append(hint_mesh)
            merged_mesh = trimesh.util.concatenate(ball_list)
            hint_export_path = os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'hint')
            if not os.path.exists(hint_export_path):
                os.makedirs(hint_export_path, exist_ok=True)
            merged_mesh.export(hint_export_path+"/hint.obj")

            render_video(r, motion, os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'hint'), text=text, hint=hint, hint_masks=result['hint_masks'])

def demo_text_hint_based_gen(cfg, output_dir, r, model, dataset):
    test_dataloader = dataset.test_dataloader()
    for i, data in enumerate(test_dataloader):
        text = data['text'][0]
        length = data['length'][0]

        if length < 60 or i < 600:
            continue
        print(text)

        for j in range(cfg.DEMO.REPLICATION):
            result = model.demo_text_hint(data)
            target_motion = result['target_motion']
            generated_motion = result['generated_motion']
            hint_masks = result['hint_masks']
            seq_len = generated_motion.shape[0]
            
            distance = np.linalg.norm((target_motion - generated_motion) * hint_masks, axis=-1).sum() / hint_masks.sum()
            print(distance)
            if distance > 0.01:
                continue

            # load xyz2smpl for visualization
            xyz2smpl = vis_utils.npy2obj_mld(generated_motion, device=model.device)

            ## output obj
            for k in range(seq_len):
                smpl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'obj')
                if not os.path.exists(smpl_output_path):
                    os.makedirs(smpl_output_path, exist_ok=True)
                xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)

            body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
            body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
            body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
            body_transl, body_orient = transform_smpl(body_transl, body_orient)
            motion = torch.cat([body_transl, body_orient, body_pose], dim=1)
            hint = transform_motion(result['target_motion']) 
            hint_masks=result['hint_masks']

            ball_list = []
            for k in range(hint_masks.shape[0]):
                for l in range(hint_masks.shape[1]):
                    if hint_masks[k, l, 0] == 1:
                        hint_mesh = trimesh.primitives.Sphere(radius=0.02, center=hint[k,l,:])
                        hint_mesh.visual.vertex_colors = [1, 0, 0, 1]
                        ball_list.append(hint_mesh)
            merged_mesh = trimesh.util.concatenate(ball_list)
            hint_export_path = os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'hint')
            if not os.path.exists(hint_export_path):
                os.makedirs(hint_export_path, exist_ok=True)
            merged_mesh.export(hint_export_path+"/hint.obj")

            render_video(r, motion, os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'text_hint'), text=text, hint=hint, hint_masks=result['hint_masks'])

def demo_inbetween_based_gen(cfg, output_dir, r, model, dataset):
    test_dataloader = dataset.test_dataloader()
    for i, data in enumerate(test_dataloader):
        text = data['text'][0]
        length = data['length'][0]

        # if length > 90:
        #     print(i, text)
        # else:
        #     continue

        # if i != 1223:
        #     continue
        # print(text)

        
        for j in range(cfg.DEMO.REPLICATION): # cfg.DEMO.REPLICATION
            result = model.demo_inbetween(data)
            target_motion = result['target_motion']
            inbetween_motion = result['inbetween_motion']
            hint_masks=result['hint_masks']

            distance = np.linalg.norm((target_motion - inbetween_motion) * hint_masks, axis=-1).sum() / hint_masks.sum()
            print(distance)

            # load xyz2smpl for visualization
            xyz2smpl = vis_utils.npy2obj_mld(inbetween_motion, device=model.device)
            seq_len = inbetween_motion.shape[0]
            ## output obj
            for k in range(seq_len):
                smpl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'generated_obj')
                if not os.path.exists(smpl_output_path):
                    os.makedirs(smpl_output_path, exist_ok=True)
                xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)
            body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
            body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
            body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
            body_transl, body_orient = transform_smpl(body_transl, body_orient)
            motion2 = torch.cat([body_transl, body_orient, body_pose], dim=1)

            # load xyz2smpl for visualization
            xyz2smpl = vis_utils.npy2obj_mld(target_motion, device=model.device)
            seq_len = target_motion.shape[0]
            ## output obj
            for k in range(seq_len):
                smpl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'target_obj')
                if not os.path.exists(smpl_output_path):
                    os.makedirs(smpl_output_path, exist_ok=True)
                if hint_masks[k, 0, 0] == torch.tensor(0).to(torch.bool):
                    xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k, delta=100)
                else:
                    xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)
            body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
            body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
            body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
            body_transl, body_orient = transform_smpl(body_transl, body_orient)
            # print(hint_masks.shape, body_transl.shape) # torch.Size([40, 22, 3]) torch.Size([40, 3])
            for k in range(seq_len):
                if hint_masks[k, 0, 0] == torch.tensor(0).to(torch.bool):
                    body_transl[k, 2] += 100
            motion1 = torch.cat([body_transl, body_orient, body_pose], dim=1)

            motion = [motion2, motion1]
            render_video(r, motion, os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'inbetween'), text=text) 

def demo_text_inbetween_based_gen(cfg, output_dir, r, model, dataset):
    test_dataloader = dataset.test_dataloader()
    for i, data in enumerate(test_dataloader):
        text = data['text'][0]
        length = data['length'][0]

        # if length > 90:
        #     print(i, text)
        # else:
        #     continue

        # if i != 1223:
        #     continue
        # print(text)

        
        for j in range(cfg.DEMO.REPLICATION): # cfg.DEMO.REPLICATION
            result = model.demo_text_inbetween(data)
            target_motion = result['target_motion']
            inbetween_motion = result['inbetween_motion']
            hint_masks=result['hint_masks']

            distance = np.linalg.norm((target_motion - inbetween_motion) * hint_masks, axis=-1).sum() / hint_masks.sum()
            print(distance)

            # load xyz2smpl for visualization
            xyz2smpl = vis_utils.npy2obj_mld(inbetween_motion, device=model.device)
            seq_len = inbetween_motion.shape[0]
            ## output obj
            for k in range(seq_len):
                smpl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'generated_obj')
                if not os.path.exists(smpl_output_path):
                    os.makedirs(smpl_output_path, exist_ok=True)
                xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)
            body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
            body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
            body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
            body_transl, body_orient = transform_smpl(body_transl, body_orient)
            motion2 = torch.cat([body_transl, body_orient, body_pose], dim=1)

            # load xyz2smpl for visualization
            xyz2smpl = vis_utils.npy2obj_mld(target_motion, device=model.device)
            seq_len = target_motion.shape[0]
            ## output obj
            for k in range(seq_len):
                smpl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'target_obj')
                if not os.path.exists(smpl_output_path):
                    os.makedirs(smpl_output_path, exist_ok=True)
                if hint_masks[k, 0, 0] == torch.tensor(0).to(torch.bool):
                    xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k, delta=100)
                else:
                    xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)
            body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
            body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
            body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
            body_transl, body_orient = transform_smpl(body_transl, body_orient)
            # print(hint_masks.shape, body_transl.shape) # torch.Size([40, 22, 3]) torch.Size([40, 3])
            for k in range(seq_len):
                if hint_masks[k, 0, 0] == torch.tensor(0).to(torch.bool):
                    body_transl[k, 2] += 100
            motion1 = torch.cat([body_transl, body_orient, body_pose], dim=1)

            motion = [motion2, motion1]
            render_video(r, motion, os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'inbetween'), text=text) 

def demo_style_transfer(cfg, output_dir, r, model, dataset):
    for i in range(10,11):
        for j in range(9,14):
            result = model.demo_style(i,j)
            generated_motion = result['generated_motion']
            # npy_path = os.path.join(output_dir, "{}".format(i), "{}".format(j))
            # os.makedirs(npy_path, exist_ok=True)
            # np.save(os.path.join(output_dir, "{}".format(i), "{}".format(j),'generated.npy'), generated_motion)

            seq_len = generated_motion.shape[0]
            xyz2smpl = vis_utils.npy2obj_mld(generated_motion, device=model.device)
            body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
            body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
            body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
            body_transl, body_orient = transform_smpl(body_transl, body_orient)
            motion = torch.cat([body_transl, body_orient, body_pose], dim=1)

            ## output obj
            for k in range(seq_len):
                smpl_output_path = os.path.join(output_dir, "{}".format(i), "{}".format(j),'generated')
                if not os.path.exists(smpl_output_path):
                    os.makedirs(smpl_output_path, exist_ok=True)
                xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)

            render_video(r, motion, os.path.join(output_dir, "{}".format(i), "{}".format(j),'generated'))

            # generated_motion = result['content_motion']
            # seq_len = generated_motion.shape[0]         
            # xyz2smpl = vis_utils.npy2obj_mld(generated_motion, device=model.device)
            # body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
            # body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
            # body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
            # body_transl, body_orient = transform_smpl(body_transl, body_orient)
            # motion = torch.cat([body_transl, body_orient, body_pose], dim=1)

            # ## output obj
            # for k in range(seq_len):
            #     smpl_output_path = os.path.join(output_dir, "{}".format(i), "{}".format(j),'content')
            #     if not os.path.exists(smpl_output_path):
            #         os.makedirs(smpl_output_path, exist_ok=True)
            #     xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)

            # render_video(r, motion, os.path.join(output_dir, "{}".format(i), "{}".format(j),'content'))

            # generated_motion = result['style_motion']
            # seq_len = generated_motion.shape[0]         
            # xyz2smpl = vis_utils.npy2obj_mld(generated_motion, device=model.device)
            # body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
            # body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
            # body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
            # body_transl, body_orient = transform_smpl(body_transl, body_orient)
            # motion = torch.cat([body_transl, body_orient, body_pose], dim=1)

            # ## output obj
            # for k in range(seq_len):
            #     smpl_output_path = os.path.join(output_dir, "{}".format(i), "{}".format(j),'style')
            #     if not os.path.exists(smpl_output_path):
            #         os.makedirs(smpl_output_path, exist_ok=True)
            #     xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)

            # render_video(r, motion, os.path.join(output_dir, "{}".format(i), "{}".format(j),'style'))

def demo_text_based_editing(cfg, output_dir, r, model, dataset):
    test_dataloader = dataset.test_dataloader()
    for i, data in enumerate(test_dataloader):
        if i <60:
            continue
        text = data['text'][0]
        length = data['length_target'][0]
        print(text)

        for j in range(cfg.DEMO.REPLICATION):
            result = model.demo_source_text(data)
            source_motion = result['source_motion']
            target_motion = result['target_motion']
            generated_motion = result['edited_motion']

            if j == 0:
                # load xyz2smpl for visualization
                seq_len = source_motion.shape[0]
                xyz2smpl = vis_utils.npy2obj_mld(source_motion, device=model.device)
                body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
                body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
                body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
                body_transl, body_orient = transform_smpl(body_transl, body_orient)

                ## output obj
                for k in range(seq_len):
                    smpl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "source",'obj')
                    if not os.path.exists(smpl_output_path):
                        os.makedirs(smpl_output_path, exist_ok=True)
                    xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)

                motion = torch.cat([body_transl, body_orient, body_pose], dim=1)
                render_video(r, motion, os.path.join(output_dir, "{}_".format(i)+text, "source", 'source.mp4')) 

                # load xyz2smpl for visualization
                seq_len = target_motion.shape[0]
                xyz2smpl = vis_utils.npy2obj_mld(target_motion, device=model.device)
                body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
                body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
                body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
                body_transl, body_orient = transform_smpl(body_transl, body_orient)

                ## output obj
                for k in range(seq_len):
                    smpl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "target",'obj')
                    if not os.path.exists(smpl_output_path):
                        os.makedirs(smpl_output_path, exist_ok=True)
                    xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)

                motion = torch.cat([body_transl, body_orient, body_pose], dim=1)
                render_video(r, motion, os.path.join(output_dir, "{}_".format(i)+text, "target", 'target.mp4')) 

            # load xyz2smpl for visualization
            seq_len = generated_motion.shape[0]
            xyz2smpl = vis_utils.npy2obj_mld(target_motion, device=model.device)
            body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
            body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
            body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
            body_transl, body_orient = transform_smpl(body_transl, body_orient)

            ## output obj
            for k in range(seq_len):
                smpl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'obj')
                if not os.path.exists(smpl_output_path):
                    os.makedirs(smpl_output_path, exist_ok=True)
                xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)

            motion = torch.cat([body_transl, body_orient, body_pose], dim=1)
            render_video(r, motion, os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j), 'edit')) 

def demo_hint_based_editing(cfg, output_dir, r, model, dataset):
    test_dataloader = dataset.test_dataloader()
    for i, data in enumerate(test_dataloader):
        text = data['text'][0]
        if i != 441:
            continue
        print(i, text)
        # continue

        for j in range(10):
            result = model.demo_source_hint(data)
            source_motion = result['source_motion']
            target_motion = result['target_motion']
            generated_motion = result['edited_motion']
            hint_masks = result['hint_masks']

            distance = np.linalg.norm((target_motion - generated_motion) * hint_masks, axis=-1).sum() / hint_masks.sum()
            print(distance)

            if j == 0:
                # load xyz2smpl for visualization
                seq_len = source_motion.shape[0]
                xyz2smpl = vis_utils.npy2obj_mld(source_motion, device=model.device)
                body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
                body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
                body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
                body_transl, body_orient = transform_smpl(body_transl, body_orient)

                ## output obj
                for k in range(seq_len):
                    smpl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "source",'obj')
                    if not os.path.exists(smpl_output_path):
                        os.makedirs(smpl_output_path, exist_ok=True)
                    xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)

                motion = torch.cat([body_transl, body_orient, body_pose], dim=1)
                render_video(r, motion, os.path.join(output_dir, "{}_".format(i)+text, "source", 'source.mp4')) 

                seq_len = target_motion.shape[0]
                xyz2smpl = vis_utils.npy2obj_mld(target_motion, device=model.device)
                body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
                body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
                body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
                body_transl, body_orient = transform_smpl(body_transl, body_orient)

                ## output obj
                for k in range(seq_len):
                    smpl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "target",'obj')
                    if not os.path.exists(smpl_output_path):
                        os.makedirs(smpl_output_path, exist_ok=True)
                    xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)

                motion = torch.cat([body_transl, body_orient, body_pose], dim=1)
                render_video(r, motion, os.path.join(output_dir, "{}_".format(i)+text, "target", 'target.mp4')) 




            # load xyz2smpl for visualization
            seq_len = generated_motion.shape[0]
            xyz2smpl = vis_utils.npy2obj_mld(generated_motion, device=model.device)
            body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
            body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
            body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
            body_transl, body_orient = transform_smpl(body_transl, body_orient)

            ## output obj
            for k in range(seq_len):
                smpl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'obj')
                if not os.path.exists(smpl_output_path):
                    os.makedirs(smpl_output_path, exist_ok=True)
                xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)

            ## output hint
            hint = transform_motion(result['target_motion']) 
            hint_masks=result['hint_masks']
            ball_list = []
            for k in range(hint_masks.shape[0]):
                for l in range(hint_masks.shape[1]):
                    if hint_masks[k, l, 0] == 1:
                        hint_mesh = trimesh.primitives.Sphere(radius=0.02, center=hint[k,l,:])
                        hint_mesh.visual.vertex_colors = [1, 0, 0, 1]
                        ball_list.append(hint_mesh)
            merged_mesh = trimesh.util.concatenate(ball_list)
            hint_export_path = os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'hint')
            if not os.path.exists(hint_export_path):
                os.makedirs(hint_export_path, exist_ok=True)
            merged_mesh.export(hint_export_path+"/hint.obj")

            motion = torch.cat([body_transl, body_orient, body_pose], dim=1)
            render_video(r, motion, os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j), 'edit'), hint=target_motion, hint_masks=result['hint_masks']) 

def demo_text_hint_based_editing(cfg, output_dir, r, model, dataset):
    test_dataloader = dataset.test_dataloader()
    for i, data in enumerate(test_dataloader):
        text = data['text'][0]
        if i != 441:
            continue
        print(i, text)
        # continue

        for j in range(10):
            result = model.demo_source_text_hint(data)
            source_motion = result['source_motion']
            target_motion = result['target_motion']
            generated_motion = result['edited_motion']
            hint_masks = result['hint_masks']

            distance = np.linalg.norm((target_motion - generated_motion) * hint_masks, axis=-1).sum() / hint_masks.sum()
            print(distance)

            if j == 0:
                # load xyz2smpl for visualization
                seq_len = source_motion.shape[0]
                xyz2smpl = vis_utils.npy2obj_mld(source_motion, device=model.device)
                body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
                body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
                body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
                body_transl, body_orient = transform_smpl(body_transl, body_orient)

                ## output obj
                for k in range(seq_len):
                    smpl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "source",'obj')
                    if not os.path.exists(smpl_output_path):
                        os.makedirs(smpl_output_path, exist_ok=True)
                    xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)

                motion = torch.cat([body_transl, body_orient, body_pose], dim=1)
                render_video(r, motion, os.path.join(output_dir, "{}_".format(i)+text, "source", 'source.mp4')) 

                seq_len = target_motion.shape[0]
                xyz2smpl = vis_utils.npy2obj_mld(target_motion, device=model.device)
                body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
                body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
                body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
                body_transl, body_orient = transform_smpl(body_transl, body_orient)

                ## output obj
                for k in range(seq_len):
                    smpl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "target",'obj')
                    if not os.path.exists(smpl_output_path):
                        os.makedirs(smpl_output_path, exist_ok=True)
                    xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)

                motion = torch.cat([body_transl, body_orient, body_pose], dim=1)
                render_video(r, motion, os.path.join(output_dir, "{}_".format(i)+text, "target", 'target.mp4')) 




            # load xyz2smpl for visualization
            seq_len = generated_motion.shape[0]
            xyz2smpl = vis_utils.npy2obj_mld(generated_motion, device=model.device)
            body_transl = xyz2smpl.motions[0, -1, :3, :].permute(1, 0).cpu()
            body_orient = xyz2smpl.motions[0, :1, :, :].permute(2, 0, 1).cpu()
            body_pose = xyz2smpl.motions[0, 1:-3, :, :].permute(2, 0, 1).reshape(seq_len, -1).cpu()
            body_transl, body_orient = transform_smpl(body_transl, body_orient)

            ## output obj
            for k in range(seq_len):
                smpl_output_path = os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'obj')
                if not os.path.exists(smpl_output_path):
                    os.makedirs(smpl_output_path, exist_ok=True)
                xyz2smpl.save_obj(smpl_output_path+'/smpl_{}.obj'.format(k), k)

            ## output hint
            hint = transform_motion(result['target_motion']) 
            hint_masks=result['hint_masks']
            ball_list = []
            for k in range(hint_masks.shape[0]):
                for l in range(hint_masks.shape[1]):
                    if hint_masks[k, l, 0] == 1:
                        hint_mesh = trimesh.primitives.Sphere(radius=0.02, center=hint[k,l,:])
                        hint_mesh.visual.vertex_colors = [1, 0, 0, 1]
                        ball_list.append(hint_mesh)
            merged_mesh = trimesh.util.concatenate(ball_list)
            hint_export_path = os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j),'hint')
            if not os.path.exists(hint_export_path):
                os.makedirs(hint_export_path, exist_ok=True)
            merged_mesh.export(hint_export_path+"/hint.obj")

            motion = torch.cat([body_transl, body_orient, body_pose], dim=1)
            render_video(r, motion, os.path.join(output_dir, "{}_".format(i)+text, "{}".format(j), 'edit'), hint=target_motion, hint_masks=result['hint_masks']) 

def main(cfg, output_dir):
    # load dataset to extract nfeats dim of model
    dataset = get_datasets(cfg, phase="demo")[0]
    model = get_model(cfg, dataset)
    r = get_offscreen_renderer('./checkpoints') # this path if you followed my setup should be data/body_models

    # loading checkpoints
    state_dict = torch.load(cfg.DEMO.CHECKPOINTS,map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)
    model.to('cuda')
    model.eval()

    if cfg.model.condition_type == "text" or cfg.DEMO.TYPE == "text":
        demo_text_based_gen(cfg, output_dir, r, model, dataset)

    elif cfg.model.condition_type == "hint" or cfg.DEMO.TYPE == "hint":
        demo_hint_based_gen(cfg, output_dir, r, model, dataset)

    elif cfg.model.condition_type == "text_hint" or cfg.DEMO.TYPE == "text_hint":
        demo_text_hint_based_gen(cfg, output_dir, r, model, dataset)

    elif cfg.model.condition_type  == "inbetween" or cfg.DEMO.TYPE == "inbetween":
        demo_inbetween_based_gen(cfg, output_dir, r, model, dataset)

    elif cfg.model.condition_type  == "text_inbetween" or cfg.DEMO.TYPE == "text_inbetween":
        demo_text_inbetween_based_gen(cfg, output_dir, r, model, dataset)

    elif cfg.model.condition_type == "style" or cfg.DEMO.TYPE == "style":
        demo_style_transfer(cfg, output_dir, r, model, dataset)

    elif cfg.model.condition_type == "source_text" or cfg.DEMO.TYPE == "source_text":
        demo_text_based_editing(cfg, output_dir, r, model, dataset)

    elif cfg.model.condition_type == "source_hint" or cfg.DEMO.TYPE == "source_hint":
        demo_hint_based_editing(cfg, output_dir, r, model, dataset)

    elif cfg.model.condition_type == "source_text_hint" or cfg.DEMO.TYPE == "source_text_hint":
        demo_text_hint_based_editing(cfg, output_dir, r, model, dataset)
    
if __name__ == "__main__":
    # for video
    os.system("Xvfb :12 -screen 1 640x480x24 &")
    os.environ['DISPLAY'] = ":12"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # parse options
    cfg = parse_args(phase="demo")
    create_logger(cfg, phase="demo")

    output_dir = Path(cfg.TEST.FOLDER, str(cfg.model.model_type), str(cfg.NAME), "samples_" + cfg.TIME)
    output_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(cfg.SEED_VALUE)
    # cuda options
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in cfg.DEVICE)
        torch.set_float32_matmul_precision('high')

    main(cfg, output_dir)
