import setuptools

with open("requirements.txt") as fh:
    requirements = [line.strip() for line in fh.readlines()]


franka_requires = [
    "spatialmath-rospy",
    "robot_io",
    "pyrealsense2",
    "roboticstoolbox-python",
]

diffusion_requires = [
    "einops==0.4.1",
    "diffusers==0.11.1",
]

maniskill_requires = [
    "maniskill2",
]

rlbench_requires = [
    "rlbench",
]

calvin_requires = [
    "calvin_env",
]


setuptools.setup(
    name="master_project",
    version="0.0.1",
    author="Jan Gruhnert",
    description="Implementation of my master project",
    # long_description=read("README.md"),
    # url="http://tapas-gmm.cs.uni-freiburg.de",
    install_requires=requirements,
    extras_require={
        "franka": franka_requires,
        "diffusion": diffusion_requires,
        "maniskill": maniskill_requires,
        "rlbench": rlbench_requires,
        "calvin": calvin_requires,
    },
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "master-train = master_project.master_train:entry_point",
            "master-retrain = master_project.master_retrain:entry_point",
            "master-debug = master_project.master_debug:entry_point",
            "master-plot = master_project.master_plot:entry_point",
            "master-eval = master_project.master_eval:entry_point",
        ]
    },
)
