"""Setup configuration for ai-sim-rl-cartpole package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ai-sim-rl-cartpole",
    version="0.1.0",
    author="AI Sim RL Team",
    description="A minimal reinforcement learning demo with experiment tracking and evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-sim-rl-cartpole",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rl-train=src.train:main",
            "rl-evaluate=src.evaluate:main",
            "rl-compare=src.compare:main",
            "rl-reproduce=src.reproduce:main",
            "rl-video=src.record_video:main",
        ],
    },
)
