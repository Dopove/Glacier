from setuptools import setup, find_packages
import os

# Function to read the requirements.txt file
def read_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
    name="ice-lite",
    version="0.1.0",
    author="Saran",
    author_email="saran@dopove.com",
    description="ICE-Lite: A virtual memory manager for State Space Models like Mamba, preventing context rot.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dopove/ice", # Replace with actual URL
    packages=find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
)
