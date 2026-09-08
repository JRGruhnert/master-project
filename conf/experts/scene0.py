from heca.experts.tapas import TapasExpert
from heca.scenes.ogbench.scene0 import OGScene0

button0_s0_s1 = TapasExpert.Config(
    label="scene0",
    tag="button0_s0_s1",
    scene=OGScene0.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button0_s1_s0 = TapasExpert.Config(
    label="scene0",
    tag="button0_s1_s0",
    scene=OGScene0.Config(),
    gt_frames=[["ee_init", "button0"], ["button0", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    snap_ee_actions=False,
)

button1_s1_s0 = TapasExpert.Config(
    label="scene0",
    tag="button1_s1_s0",
    scene=OGScene0.Config(),
    gt_frames=[["ee_init", "button1"], ["button1", "ee_target"]],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    snap_ee_actions=False,
)


button1_s0_s1 = TapasExpert.Config(
    label="scene0",
    tag="button1_s0_s1",
    scene=OGScene0.Config(),
    gt_frames=[["ee_init", "button1"], ["button1", "ee_target"]],
    segment_ids=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    snap_ee_actions=False,
)


cube0_base_base = TapasExpert.Config(
    label="scene0",
    tag="cube0_base_base",
    scene=OGScene0.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0"],
        ["cube0"],
        ["cube0", "cube0_target"],
        ["cube0_target"],
        ["cube0_target"],
        ["cube0_target", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    fix_bimodal=True,
    pos_only=False,
)

cube0_base_drawer0 = TapasExpert.Config(
    label="scene0",
    tag="cube0_base_drawer0",
    scene=OGScene0.Config(),
    gt_frames=[
        ["ee_init", "cube0"],
        ["cube0"],
        ["cube0"],
        ["cube0", "drawer0"],
        ["drawer0"],
        ["drawer0"],
        ["drawer0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23],
    fix_bimodal=True,
    pos_only=False,
)

drawer0_a_b = TapasExpert.Config(
    label="scene0",
    tag="drawer0_a_b",
    scene=OGScene0.Config(),
    gt_frames=[
        ["ee_init", "drawer0"],
        ["drawer0", "button0"],
        ["drawer0", "button0"],
        ["drawer0", "button0"],
        ["drawer0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    fix_bimodal=True,
)
drawer0_b_a = TapasExpert.Config(
    label="scene0",
    tag="drawer0_b_a",
    scene=OGScene0.Config(),
    gt_frames=[
        ["ee_init", "drawer0"],
        ["drawer0", "button0"],
        ["drawer0", "button0"],
        ["drawer0", "button0"],
        ["drawer0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 4, 5, 7, 9, 10, 11, 12, 14, 15, 17, 18, 19, 20, 22, 23, 24],
    # fix_bimodal=True,
    # use_ee_frames=True,
    snap_ee_actions=False,
)

window0_a_b = TapasExpert.Config(
    label="scene0",
    tag="window0_a_b",
    scene=OGScene0.Config(),
    gt_frames=[
        ["ee_init", "window0"],
        ["window0", "button1"],
        ["window0", "button1"],
        ["window0", "button1"],
        ["window0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    fix_bimodal=True,
    snap_ee_actions=False,
)

window0_b_a = TapasExpert.Config(
    label="scene0",
    tag="window0_b_a",
    scene=OGScene0.Config(),
    gt_frames=[
        ["ee_init", "window0"],
        ["window0", "button1"],
        ["window0", "button1"],
        ["window0", "button1"],
        ["window0", "ee_target"],
    ],
    segment_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    fix_bimodal=True,
)

agents = [
    window0_b_a,
    window0_a_b,
    cube0_base_base,
    button0_s1_s0,
    button0_s0_s1,
    button1_s1_s0,
    button1_s0_s1,
    drawer0_a_b,
    cube0_base_drawer0,
    drawer0_b_a,
]
