from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the contents of requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read().splitlines()

setup(
    name="glacier-ice-lite",  # Renamed to reflect its core component
    version="0.1.0",
    author="Saran S",
    author_email="saran@dopove.com",
    description="GLACIER: Mamba with Infinite Memory (ICE-Lite + Temporal-RAG)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dopove/Glacier",
    packages=find_packages(where="src"), # Look for packages in the 'src' directory
    package_dir={"": "src"}, # Tell setuptools that packages are under 'src'
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
)