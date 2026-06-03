import os
import re
from setuptools import setup, find_packages

_here = os.path.dirname(os.path.abspath(__file__))


def _read(path):
    with open(os.path.join(_here, path), encoding="utf-8") as f:
        return f.read()


def _read_version():
    content = _read("onnx2tflite/_version.py")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    if not m:
        raise RuntimeError("Could not find __version__ in _version.py")
    return m.group(1)


setup(
    name="onnx2tflite",
    version=_read_version(),
    author="MPolaris",
    description="ONNX to TensorFlow Lite converter",
    long_description=_read("readme.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["onnx2tflite", "onnx2tflite.*"]),
    python_requires=">=3.8, <3.11",
    install_requires=_read("requirements.txt").splitlines(),
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
