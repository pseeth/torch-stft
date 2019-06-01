from setuptools import find_packages, setup

NAME = 'torch_stft'
REQUIREMENTS = [
    'numpy',
    'scipy',
    'librosa',
]

with open("README.md", "r") as fh:
    long_description = fh.read()

EXTRAS={
        'tests': ['pytest']
    }

setup(
    name="torch_stft",
    version="0.1",
    description="An STFT/iSTFT for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pseeth/torch-stft",
    author="Prem Seetharaman",
    author_email="prem@u.northwestern.edu",
    classifiers=[
        "Environment :: Plugins",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    # Exclude the build files.
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    extras_requires=EXTRAS,
)