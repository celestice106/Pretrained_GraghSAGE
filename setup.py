from setuptools import setup, find_packages

setup(
    name="graphsage-pretraining",
    version="0.1.0",
    description="Structural pre-training for Memory R1 Bank",
    author="Trung Hieu",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "networkx>=3.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.9",
)