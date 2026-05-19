from setuptools import setup, find_packages
setup(
    name="bars-experiment",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["numpy>=1.21","pandas>=1.3","scipy>=1.7","scikit-learn>=1.0","tqdm>=4.62","torch>=1.12","gym>=0.21"],
    extras_require={
        "stage22-gas": [
            "absl-py",
            "ml-collections",
            "jax",
            "jaxlib",
            "flax",
            "optax",
            "distrax",
            "gymnasium",
            "ogbench",
            "mujoco",
            "dm-control",
            "tensorboard",
            "opencv-python",
            "networkx",
        ],
    },
    entry_points={"console_scripts": ["bars=bars.cli:main", "barsctl=bars.sched.jobctl:main"]},
)
