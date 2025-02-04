import os
import numpy as np
import torch

from rfmotion.utils.smpl_body_utils import get_smpl_skeleton

from einops import rearrange
from scipy.spatial.transform import Rotation as R
from rfmotion.tools.transforms3d import transform_body_pose
import subprocess
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.meshes import Meshes

def render_skeleton(renderer: HeadlessRenderer, positions: torch.Tensor, 
                    filename: str, text_for_vid=None,
                    color=(1/255, 1 / 255, 1.0, 1.0)) -> None:
    """
    Function to render a video of a motion sequence
    renderer: aitviewer renderer
    datum: dictionary containing sequence of poses, body translations and body orientations
        data could be numpy or pytorch tensors
    filename: the absolute path you want the video to be saved at

    """
    # assert {'body_transl', 'body_orient', 'body_pose'}.issubset(set(datum.keys()))
    import trimesh

    from aitviewer.headless import HeadlessRenderer
    from aitviewer.models.smpl import SMPLLayer
    from aitviewer.renderables.smpl import SMPLSequence
    from aitviewer.renderables.skeletons import Skeletons

    skeletons_seq = Skeletons(joint_positions=positions, 
                              joint_connections=get_smpl_skeleton(),
                              color=color,
                              radius=0.03)

    #  if 'gender' in datum.keys():
    #  gender = datum['gender']
    import sys
    sys.stdout.flush()
    old = os.dup(1)
    os.close(1)
    os.open(os.devnull, os.O_WRONLY)

    renderer.scene.add(skeletons_seq)
    # camera follows smpl sequence
    camera = renderer.lock_to_node(skeletons_seq, (2, 2, 2), smooth_sigma=5.0)
    
    renderer.save_video(video_dir=str(filename), output_fps=30)
    # aitviewer adds a counter to the filename, we remove it
    # filename.split('_')[-1].replace('.mp4', '')
    # os.rename(filename + '_0.mp4', filename[:-4] + '.mp4')
    os.rename(filename + '_0.mp4', filename + '.mp4')

    # empty scene for the next rendering
    renderer.scene.remove(skeletons_seq)
    renderer.scene.remove(camera)

    sys.stdout.flush()
    os.close(1)
    os.dup(old)
    os.close(old)

    if text_for_vid is not None:
        fname = put_text(text_for_vid, f'{filename}.mp4', f'{filename}_ts.mp4')
        os.remove(f'{filename}.mp4')
    else:
        fname = f'{filename}.mp4'

    return fname

def render_motion(renderer: HeadlessRenderer, datum: dict, 
                  filename: str, text_for_vid=None, hint=None, hint_masks=None, pose_repr='6d',
                  color=(160 / 255, 160 / 255, 160 / 255, 1.0),
                  return_verts=False, smpl_layer=None) -> None:
    """
    Function to render a video of a motion sequence
    renderer: aitviewer renderer
    datum: dictionary containing sequence of poses, body translations and body orientations
        data could be numpy or pytorch tensors
    filename: the absolute path you want the video to be saved at

    """
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.smpl import SMPLSequence
    import trimesh
    from rfmotion.render.video import put_text
    from rfmotion.tools.transforms3d import transform_body_pose

    if isinstance(datum, dict): datum = [datum]
    if not isinstance(color, list): 
        colors = [color] 
    else:
        colors = color
    # assert {'body_transl', 'body_orient', 'body_pose'}.issubset(set(datum[0].keys()))
    # os.environ['DISPLAY'] = ":11"
    gender = 'neutral'
    only_skel = False
    import sys
    seqs_of_human_motions = []
    if smpl_layer is None:
        from aitviewer.models.smpl import SMPLLayer
        smpl_layer = SMPLLayer(model_type='smplh', 
                                ext='npz',
                                gender=gender)
    
    for iid, mesh_seq in enumerate(datum):
        if pose_repr != 'aa':
            global_orient = transform_body_pose(mesh_seq['body_orient'],
                                                f"{pose_repr}->aa")
            body_pose = transform_body_pose(mesh_seq['body_pose'],
                                            f"{pose_repr}->aa")
        else:
            global_orient = mesh_seq['body_orient']
            body_pose = mesh_seq['body_pose']

        body_transl = mesh_seq['body_transl']
        sys.stdout.flush()

        old = os.dup(1)
        os.close(1)
        os.open(os.devnull, os.O_WRONLY)
        smpl_template = SMPLSequence(body_pose,
                                     smpl_layer,
                                     poses_root=global_orient,
                                     trans=body_transl,
                                     color=colors[iid],
                                     z_up=True)
        if only_skel:
            smpl_template.remove(smpl_template.mesh_seq)

        seqs_of_human_motions.append(smpl_template)
        renderer.scene.add(smpl_template)

        ball_list = []
        if hint is not None:
            for i in range(hint_masks.shape[0]):
                for j in range(hint_masks.shape[1]):
                    if hint_masks[i, j, 0] == 1:
                        hint_mesh = trimesh.primitives.Sphere(radius=0.01)
                        hint_mesh_ait = Meshes(
                                            hint_mesh.vertices,
                                            hint_mesh.faces,
                                            name="ball_{}_{}".format(i,j),
                                            position=hint[i,j,:],
                                            color=(0.5, 0, 0, 1),
                                            flat_shading=True,
                                        )
                        ball_list.append(hint_mesh_ait)
                        renderer.scene.add(hint_mesh_ait)

        
    # camera follows smpl sequence
    # FIX CAMERA
    from rfmotion.tools.transforms3d import get_z_rot
    R_z = get_z_rot(global_orient[0], in_format='aa')
    heading = -R_z[:, 1]
    xy_facing = body_transl[0] + heading*2.5
    camera = renderer.lock_to_node(seqs_of_human_motions[0],
                                    (xy_facing[0], xy_facing[1], 1.5), smooth_sigma=5.0)

    # /FIX CAMERA
    if len(mesh_seq['body_pose']) == 1:
        renderer.save_frame(file_path=str(filename) + '.png')
        sfx = 'png'
    else:
        renderer.save_video(video_dir=str(filename), output_fps=20)
        sfx = 'mp4'


    # empty scene for the next rendering
    for mesh in seqs_of_human_motions:
        renderer.scene.remove(mesh)
    for mesh in ball_list:
        renderer.scene.remove(mesh)
    renderer.scene.remove(camera)

    sys.stdout.flush()
    os.close(1)
    os.dup(old)
    os.close(old)

    if text_for_vid is not None:
        fname = put_text(text_for_vid, f'{filename}.{sfx}', f'{filename}_.{sfx}')
    else:
        fname = f'{filename}.{sfx}'
    return fname



# import numpy as np
# import torch
# import trimesh
# import math
# import torch.nn.functional as F
# from rfmotion.utils.mesh_utils import MeshViewer
# from rfmotion.utils.smpl_body_utils import colors,marker2bodypart,bodypart2color
# from rfmotion.utils.smpl_body_utils import c2rgba, get_body_model
# from hydra.utils import get_original_cwd
# from rfmotion.render.video import save_video_samples

# import os 
# from PIL import Image
# import subprocess
# import contextlib

# def visualize_meshes(vertices, pcd=None, multi_col=None, text=None,
#                      multi_angle=False, h=720, w=720, bg_color='white',
#                      save_path=None, fig_label=None, use_hydra_path=True):
#     import pyrender

#     """[summary]

#     Args:
#         rec (torch.tensor): [description]
#         inp (torch.tensor, optional): [description]. Defaults to None.
#         multi_angle (bool, optional): Whether to use different angles. Defaults to False.

#     Returns:
#         np.array :   Shape of output (view_angles, seqlen, 3, im_width, im_height, )
 
#     """
#     import os
#     os.environ['PYOPENGL_PLATFORM'] = 'egl'

#     im_height: int = h
#     im_width: int = w
#     vis_mar= True if pcd is not None else False

#     # if multi_angle:
#     #     view_angles = [0, 180, 90, -90]
#     # else:
#     #     view_angles = [0]
#     seqlen = vertices.shape[0]
#     kfs_viz = np.zeros((seqlen))
#     if multi_col is not None:
#         # assert len(set(multi_col)) == len(multi_col)
#         multi_col = multi_col.detach().cpu().numpy()
#         for i, kf_ids in enumerate(multi_col):
#             kfs_viz[i, :].put(np.round_(kf_ids).astype(int), 1, mode='clip')
    

#     if isinstance(pcd, np.ndarray):
#         pcd = torch.from_numpy(pcd)

#     if vis_mar:
#         pcd = pcd.reshape(seqlen, 67, 3).to('cpu')
#         if len(pcd.shape) == 3:
#             pcd = pcd.unsqueeze(0)
#     mesh_rec = vertices
#     if use_hydra_path:
#         with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
#             smpl = get_body_model(path=f'{get_original_cwd()}/data/smpl_models',
#                                 model_type='smpl', gender='neutral',
#                                 batch_size=1, device='cpu')
#     else:
#         with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
#             smpl = get_body_model(path=f'data/smpl_models',
#                                 model_type='smpl', gender='neutral',
#                                 batch_size=1, device='cpu')
#     minx, miny, _ = mesh_rec.min(axis=(0, 1))
#     maxx, maxy, _ = mesh_rec.max(axis=(0, 1))
#     minsxy = (minx, maxx, miny, maxy)
#     height_offset = np.min(mesh_rec[:, :, 2])  # Min height

#     mesh_rec = mesh_rec.copy()
#     mesh_rec[:, :, 2] -= height_offset

#     mv = MeshViewer(width=im_width, height=im_height,
#                     add_ground_plane=True, plane_mins=minsxy, 
#                     use_offscreen=True,
#                     bg_color=bg_color)
#                 #ground_height=(mesh_rec.detach().cpu().numpy()[0, 0, 6633, 2] - 0.01))
#     # ground plane has a bug if we want batch_size to work
#     mv.render_wireframe = False
    
#     video = np.zeros([seqlen, im_height, im_width, 3])
#     for i in range(seqlen):
#         Rx = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])
#         Ry = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])

#         if vis_mar:
#             m_pcd = trimesh.points.PointCloud(pcd[i])

#         if kfs_viz[i]:
#             mesh_color = np.tile(c2rgba(colors['light_grey']), (6890, 1))
#         else:
#             mesh_color = np.tile(c2rgba(colors['yellow_pale']), (6890, 1))
#         m_rec = trimesh.Trimesh(vertices=mesh_rec[i],
#                                 faces=smpl.faces,
#                                 vertex_colors=mesh_color)

#         m_rec.apply_transform(Rx)
#         m_rec.apply_transform(Ry)

#         all_meshes = []
#         if vis_mar:
#             m_pcd.apply_transform(Rx)
#             m_pcd.apply_transform(Ry)

#             m_pcd = np.array(m_pcd.vertices)
#             # after trnaofrming poincloud visualize for each body part separately
#             pcd_bodyparts = dict()
#             for bp, ids in marker2bodypart.items():
#                 points = m_pcd[ids]
#                 tfs = np.tile(np.eye(4), (points.shape[0], 1, 1))
#                 tfs[:, :3, 3] = points
#                 col_sp = trimesh.creation.uv_sphere(radius=0.01)

#                 # debug markers, maybe delete it
#                 # if bp == 'special':
#                 #     col_sp = trimesh.creation.uv_sphere(radius=0.03)

#                 if kfs_viz[i]:
#                     col_sp.visual.vertex_colors = c2rgba(colors["black"])
#                 else:
#                     col_sp.visual.vertex_colors = c2rgba(colors[bodypart2color[bp]])

#                 pcd_bodyparts[bp] = pyrender.Mesh.from_trimesh(col_sp, poses=tfs)

#             for bp, m_bp in pcd_bodyparts.items():
#                 all_meshes.append(m_bp)


#         all_meshes.append(m_rec)
#         mv.set_meshes(all_meshes, group_name='static')
#         video[i] = mv.render()

#     # if save_path is not None:
#     #     save_video_samples(np.transpose(np.squeeze(video),
#     #                                     (0, 3, 1, 2)).astype(np.uint8), 
#     #                                     save_path, write_gif=True)
#     #     return
#     if multi_angle:
#         return np.transpose(video, (0, 1, 4, 2, 3)).astype(np.uint8)

#     if save_path is not None:
#         from rfmotion.render.video import Video
#         vid = Video(list(np.squeeze(video).astype(np.uint8)))
#         if text is not None:
#             vid.add_text(text)
#         return vid.save(save_path)
#         # return save_video_samples(np.transpose(np.squeeze(video),
#         #                                        (0, 3, 1, 2)).astype(np.uint8),
#         #                           save_path, text)
#     return np.transpose(np.squeeze(video), (0, 3, 1, 2)).astype(np.uint8)
