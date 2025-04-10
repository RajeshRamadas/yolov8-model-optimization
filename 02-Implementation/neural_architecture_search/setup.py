from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="yolov8-nas",
    version="0.1.0",
    author="Rajesh kumar Ramadas",
    author_email="Rajeshkumar1988r@gmail.com",
    description="Neural Architecture Search for YOLOv8 models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RajeshRamadas/yolov8-model-optimization",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "yolov8-nas=main:main",
        ],
    },
)
