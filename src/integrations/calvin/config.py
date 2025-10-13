from tapas_gmm.env.calvin import CalvinConfig

calvin_config = CalvinConfig(
    task="Undefined",
    cameras=("wrist", "front"),
    camera_pose={},
    image_size=(256, 256),
    static=False,
    headless=False,
    scale_action=False,
    delay_gripper=False,
    gripper_plot=False,
    postprocess_actions=False,
    eval_mode=False,
    real_time=True,
    pybullet_vis=False,
)
