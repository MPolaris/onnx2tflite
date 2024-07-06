from setuptools import setup, find_packages

setup(
    name="onnx2tflite",
    version="2.0",
    author="MPolaris",
    description="onnx to keras/tensorflow lite",
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    packages=find_packages(include=['onnx2tflite']),
    license="Apache-2.0",
    platforms=["Windows", "linux"],
    install_requires=open('requirements.txt').read().splitlines()
)