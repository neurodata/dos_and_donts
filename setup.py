from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    "networkx=2.3",
    "numpy>=1.8.1",
    "scikit-learn>=0.19.1",
    "scipy>=1.1.0",
    "seaborn>=0.9.0",
    "matplotlib>=3.0.0",
]

setup(
    name="src",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    version="0.1.0",
    description="Experiments showing the 'Do's and Don'ts' of connectome analysis",
    author="Neurodata",
    license="BSD-3",
)
