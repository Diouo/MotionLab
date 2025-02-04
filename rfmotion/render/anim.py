# Inspired by
# - https://github.com/anindita127/Complextext2animation/blob/main/src/utils/visualization.py
# - https://github.com/facebookresearch/QuaterNet/blob/main/common/visualization.py

from typing import List, Tuple
import numpy as np

mmm_joints = [
    "root", "BP", "BT", "BLN", "BUN", "LS", "LE", "LW", "RS", "RE", "RW", "LH",
    "LK", "LA", "LMrot", "LF", "RH", "RK", "RA", "RMrot", "RF"
]

number_of_joints = {
    "smplh": 22,
}

smplh_joints = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_index1", "left_index2", "left_index3", "left_middle1",
    "left_middle2", "left_middle3", "left_pinky1", "left_pinky2",
    "left_pinky3", "left_ring1", "left_ring2", "left_ring3", "left_thumb1",
    "left_thumb2", "left_thumb3", "right_index1", "right_index2",
    "right_index3", "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1",
    "right_ring2", "right_ring3", "right_thumb1", "right_thumb2",
    "right_thumb3", "nose", "right_eye", "left_eye", "right_ear", "left_ear",
    "left_big_toe", "left_small_toe", "left_heel", "right_big_toe",
    "right_small_toe", "right_heel", "left_thumb", "left_index", "left_middle",
    "left_ring", "left_pinky", "right_thumb", "right_index", "right_middle",
    "right_ring", "right_pinky"
]

smpl_bps = {
    'global': ['pelvis'],
    'torso': ['spine1', 'spine2', 'spine3', 'neck', 'head'],
    'left arm': ['left_collar', 'left_shoulder', 'left_elbow', 'left_wrist'],
    'right arm':
    ['right_collar', 'right_shoulder', 'right_elbow', 'right_wrist'],
    'left leg': ['left_hip', 'left_knee', 'left_ankle', 'left_foot'],
    'right leg': ['right_hip', 'right_knee', 'right_ankle', 'right_foot']
}

bp2ids = {
    bp_name: [smplh_joints.index(j) for j in jts_names]
    for bp_name, jts_names in smpl_bps.items()
    }

mmm2smplh_correspondence = {
    "root": "pelvis",
    "BP": "spine1",
    "BT": "spine3",
    "BLN": "neck",
    "BUN": "head",
    "LS": "left_shoulder",
    "LE": "left_elbow",
    "LW": "left_wrist",
    "RS": "right_shoulder",
    "RE": "right_elbow",
    "RW": "right_wrist",
    "LH": "left_hip",
    "LK": "left_knee",
    "LA": "left_ankle",
    "LMrot": "left_heel",
    "LF": "left_foot",
    "RH": "right_hip",
    "RK": "right_knee",
    "RA": "right_ankle",
    "RMrot": "right_heel",
    "RF": "right_foot"
}
smplh2mmm_correspondence = {
    val: key
    for key, val in mmm2smplh_correspondence.items()
}

smplh2mmm_indexes = [
    smplh_joints.index(mmm2smplh_correspondence[x]) for x in mmm_joints
]

smpl2gpt = {
    'global': [
        'spine', 'butt', 'buttocks', 'buttock', 'crotch', 'pelvis', 'groin',
        'bottom', 'waist',
    ],
    'torso': [
        'spine', 'body', 'head', 'neck', 'torso', 'trunk', 'jaw', 'nose',
        'breast', 'chest', 'belly', 'mouth', 'throat',
        'chin', 'chest', 'back', 'face',
        'jaws', 'side', 'teeth'
    ],
    'left arm': [
        'arms', 'hands', 'shoulders', 'elbows', 'arm', 'wrists', 'bicep',
        'palm',   'wrist', 'shoulder', 'hand', 'arm', 'elbow',
        'tricep',  'biceps', 'thumb', 'fists', 'finger', 'fingers',
        'deltoid', 'trapezius', 
    ],
    'right arm': [
        'arms', 'hands', 'shoulders', 'elbows', 'arm', 'wrists', 'bicep',
        'palm',  'wrist', 'shoulder', 'hand', 'arm', 'elbow', 'tricep',  'biceps',
        'thumb', 'fists', 'finger', 'fingers', 'deltoid', 
    ],
    'left leg': [
        'legs', 'feet', 'hips', 'knee', 'ankle', 'leg', 'hip', 'calf',
        'thigh', 'thighs', 'foot', 'knees', 'ankles', 'heel', 
         'toe', 'toes', 'calves'
    ],
    'right leg': [
        'legs', 'feet', 'hips', 'knee', 'ankle', 'leg', 'hip', 'calf',
        'thigh', 'thighs', 'foot', 'knees', 'ankles', 'heel',
         'toe', 'toes',  'calves'
    ]
}

body_parts = ['left arm', 'right arm', 'left leg', 'global orientation',
'right leg', 'torso', 'left hand', 'right hand', 'left ankle', 'right ankle', 'left foot',
'right foot', 'head', 'neck', 'right shoulder', 'left shoulder', 'pelvis', 'spine']

body_parts_coarse = ['arm', 'arms', 'leg', 'legs' 'right arm', 'left leg', 'global orientation',
'right leg', 'torso', 'left hand', 'right hand', 'left ankle', 'right ankle', 'left foot',
'right foot', 'head', 'neck', 'right shoulder', 'left shoulder', 'pelvis', 'spine']

mmm_kinematic_tree = [[0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10],
                      [0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20]]

smplh_kinematic_tree = [[0, 3, 6, 9, 12, 15], [9, 13, 16, 18, 20],
                        [9, 14, 17, 19, 21], [0, 1, 4, 7, 10],
                        [0, 2, 5, 8, 11]]

mmm_colors = ['black', 'magenta', 'red', 'green', 'blue']


def init_axis(fig, title, radius=1.5, dist=10):
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20., azim=-60)

    fact = 2
    ax.set_xlim3d([-radius / fact, radius / fact])
    ax.set_ylim3d([-radius / fact, radius / fact])
    ax.set_zlim3d([0, radius])

    ax.set_aspect('auto')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_axis_off()

    ax.dist = dist
    ax.grid(b=False)

    ax.set_title(title, loc='center', wrap=True)
    return ax


def plot_floor(ax, minx, maxx, miny, maxy, minz):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # Plot a plane XZ
    verts = [
        [minx, miny, minz],
        [minx, maxy, minz],
        [maxx, maxy, minz],
        [maxx, miny, minz]
    ]
    xz_plane = Poly3DCollection([verts], zorder=1)
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 1))
    ax.add_collection3d(xz_plane)

    # Plot a bigger square plane XZ
    radius = max((maxx - minx), (maxy - miny))

    # center +- radius
    minx_all = (maxx + minx) / 2 - radius
    maxx_all = (maxx + minx) / 2 + radius

    miny_all = (maxy + miny) / 2 - radius
    maxy_all = (maxy + miny) / 2 + radius

    verts = [
        [minx_all, miny_all, minz],
        [minx_all, maxy_all, minz],
        [maxx_all, maxy_all, minz],
        [maxx_all, miny_all, minz]
    ]
    xz_plane = Poly3DCollection([verts], zorder=1)
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
    ax.add_collection3d(xz_plane)
    return ax


def update_camera(ax, root, radius=1.5):
    fact = 2
    ax.set_xlim3d([-radius / fact + root[0], radius / fact + root[0]])
    ax.set_ylim3d([-radius / fact + root[1], radius / fact + root[1]])


def render_animation(joints: np.ndarray, output: str = "notebook", title: str = "",
                     fps: float = 12.5,
                     kinematic_tree: List[List[int]] = smplh_kinematic_tree,
                     colors: List[str] = mmm_colors,
                     figsize: Tuple[int] = (4, 4),
                     fontsize: int = 7):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.patheffects as pe
    plt.rcParams.update({'font.size': fontsize})

    # Z is gravity here
    x, y, z = 0, 1, 2

    # Convert mmm joints for visualization
    # into smpl-h "scale" and axis
    # joints = joints.copy()[..., [2, 0, 1]] * mmm_to_smplh_scaling_factor
    # Create a figure and initialize 3d plot
    fig = plt.figure(figsize=figsize)
    ax = init_axis(fig, title)

    # Create spline line
    trajectory = joints[:, 0, [x, y]]
    avg_segment_length = np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) + 1e-3
    # draw_offset = int(25 / avg_segment_length)
    # spline_line, = ax.plot(*trajectory.T, zorder=10, color="white")

    # Create a floor
    minx, miny, _ = joints.min(axis=(0, 1))
    maxx, maxy, _ = joints.max(axis=(0, 1))
    plot_floor(ax, minx, maxx, miny, maxy, 0)

    # Put the character on the floor
    height_offset = np.min(joints[:, :, z])  # Min height
    joints = joints.copy()
    joints[:, :, z] -= height_offset

    # Initialization for redrawing
    lines = []
    initialized = False

    def update(frame):
        nonlocal initialized
        skeleton = joints[frame]

        root = skeleton[0]
        update_camera(ax, root)

        for index, (chain, color) in enumerate(zip(reversed(kinematic_tree), reversed(colors))):
            if not initialized:
                lines.append(ax.plot(skeleton[chain, x],
                                     skeleton[chain, y],
                                     skeleton[chain, z], linewidth=8.0, color=color, zorder=20,
                                     path_effects=[pe.SimpleLineShadow(), pe.Normal()]))

            else:
                lines[index][0].set_xdata(skeleton[chain, x])
                lines[index][0].set_ydata(skeleton[chain, y])
                lines[index][0].set_3d_properties(skeleton[chain, z])

        # left = max(frame - draw_offset, 0)
        # right = min(frame + draw_offset, trajectory.shape[0])

        # spline_line.set_xdata(trajectory[left:right, 0])
        # spline_line.set_ydata(trajectory[left:right, 1])
        # spline_line.set_3d_properties(np.zeros_like(trajectory[left:right, 0]))
        initialized = True

    fig.tight_layout()
    frames = joints.shape[0]
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=False)

    if output == "notebook":
        from IPython.display import HTML
        HTML(anim.to_jshtml())
    else:
        anim.save(output, writer='ffmpeg', fps=fps)

    plt.close()

def render_animation_parallel(args):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.patheffects as pe
    joints, output, title, fps = args
    kinematic_tree = mmm_kinematic_tree
    colors = mmm_colors
    fontsize = 7
    figsize = (4, 4)
    plt.rcParams.update({'font.size': fontsize})

    # Z is gravity here
    x, y, z = 0, 1, 2

    # Convert mmm joints for visualization
    # into smpl-h "scale" and axis
    joints = joints.copy()[..., [2, 0, 1]] * mmm_to_smplh_scaling_factor

    # Create a figure and initialize 3d plot
    fig = plt.figure(figsize=figsize)
    ax = init_axis(fig, title)

    # Create spline line
    trajectory = joints[:, 0, [x, y]]
    avg_segment_length = np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) + 1e-3
    draw_offset = int(25 / avg_segment_length)
    spline_line, = ax.plot(*trajectory.T, zorder=10, color="white")

    # Create a floor
    minx, miny, _ = joints.min(axis=(0, 1))
    maxx, maxy, _ = joints.max(axis=(0, 1))
    plot_floor(ax, minx, maxx, miny, maxy, 0)

    # Put the character on the floor
    height_offset = np.min(joints[:, :, z])  # Min height
    joints = joints.copy()
    joints[:, :, z] -= height_offset

    # Initialization for redrawing
    lines = []
    initialized = False

    def update(frame):
        nonlocal initialized
        skeleton = joints[frame]

        root = skeleton[0]
        update_camera(ax, root)

        for index, (chain, color) in enumerate(zip(reversed(kinematic_tree), reversed(colors))):
            if not initialized:
                lines.append(ax.plot(skeleton[chain, x],
                                     skeleton[chain, y],
                                     skeleton[chain, z], linewidth=8.0, color=color, zorder=20,
                                     path_effects=[pe.SimpleLineShadow(), pe.Normal()]))

            else:
                lines[index][0].set_xdata(skeleton[chain, x])
                lines[index][0].set_ydata(skeleton[chain, y])
                lines[index][0].set_3d_properties(skeleton[chain, z])

        left = max(frame - draw_offset, 0)
        right = min(frame + draw_offset, trajectory.shape[0])

        spline_line.set_xdata(trajectory[left:right, 0])
        spline_line.set_ydata(trajectory[left:right, 1])
        spline_line.set_3d_properties(np.zeros_like(trajectory[left:right, 0]))
        initialized = True

    fig.tight_layout()
    frames = joints.shape[0]
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=False)

    if output == "notebook":
        from IPython.display import HTML
        HTML(anim.to_jshtml())
    else:
        anim.save(output, writer='ffmpeg', fps=fps)

    plt.close()
    return output, title
