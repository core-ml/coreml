import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="coreml",
    version="0.0.15",
    author="Aman Dalmia",
    author_email="amandalmia18@gmail.com",
    description="Generic Framework for ML projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dalmia/coreml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'matplotlib==3.2.1',
        'natsort==7.0.1',
        'opencv-python',
        'pandas==1.0.3',
        'Pillow==7.1.1',
        'PyYAML>=5.1',
        'scikit-learn==0.22.2.post1',
        'termcolor==1.1.0',
        'torch==1.6.0',
        'torchsummary==1.5.1',
        'torchtext==0.4.0',
        'torchvision==0.7.0',
        'torchaudio==0.6.0',
        'tqdm==4.45.0',
        'wandb==0.9.4',
        'xgboost==1.1.1',
        'timm==0.2.1',
        'pytorch-lightning==0.9.0',
        'kornia==0.4.0',
    ],
    python_requires='>=3.6',
)
