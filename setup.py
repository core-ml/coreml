import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="coreml",
    version="0.0.3",
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
    python_requires='>=3.6',
)
