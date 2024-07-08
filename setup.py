import os
from setuptools import setup, find_packages
abs_path = os.path.dirname(os.path.abspath(__file__))

setup(
    name="onnx2tflite",
    version="2.0",
    author="MPolaris",
    description="onnx to keras/tensorflow lite",
    long_description=open(os.path.join(abs_path, "readme.md")).read(),
    long_description_content_type='text/markdown',
    packages=find_packages(include=['onnx2tflite']),
    license="Apache-2.0",
    platforms=["Windows", "linux"],
    install_requires=open(os.path.join(abs_path, "requirements.txt")).read().splitlines()
)