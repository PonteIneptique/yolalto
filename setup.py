import os
from setuptools import setup

setup(
    name="yolalto",
    version="0.1.0",
    py_modules=["yolalto"],
    install_requires=[
        "ultralytics",
        "lxml",
        "click",
        "scikit-learn"
    ],
    entry_points={
        "console_scripts": [
            "yolalto=yolalto:cli",
        ],
    },
    author="Thibault Clérice",
    author_email="thibault.clerice@inria.fr",
    description="Convert YOLO predictions using SegmOnto ontology into ALTO XML format.",
    url="https://github.com/ponteineptique/yolalto",  # Update with your actual URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Change if needed
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)