import setuptools

setuptools.setup(
    name="jax_trainer",
    version="0.1",
    author="Phillip Lippe",
    author_email="phillip.lippe@googlemail.com",
    description="Lightning-like framework for JAX",
    url="https://github.com/phlippe/jax-trainer",
    packages=setuptools.find_packages(),
    install_requires=[
        "jax>=0.4.13",
        "jaxlib>=0.4.13+cuda12.cudnn89",
        "torchvision==0.15.2+cpu",
        "torchaudio==2.0.2+cpu",
        "torch==2.0.1+cpu",
        "numpy",
        "seaborn",
        "matplotlib",
        "pytorch-lightning>=2.0.5",
        "tensorboard>=2.13.0",
        "optax>=0.1.5",
        "orbax>=0.1.7",
        "flax>=0.7.0" "absl-py",
        "ml-collections",
    ],
)
