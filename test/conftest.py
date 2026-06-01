"""
Pytest configuration - makes helper module and onnx2tflite importable without install.
"""
import os
import sys

# Add project root to sys.path so onnx2tflite can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Add test/ directory before stdlib 'test' so helper can be imported
_test_dir = os.path.dirname(os.path.abspath(__file__))
if _test_dir not in sys.path:
    sys.path.insert(0, _test_dir)
