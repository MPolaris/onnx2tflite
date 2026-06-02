import os
import cv2
import logging
import numpy as np

LOG = logging.getLogger("Quantization DataLoder :")

class RandomLoader(object):
    def __init__(self, target_size):
        self.target_size = target_size
        LOG.warning(f"Generate quantization data from random, it's will lead to accuracy problem!")
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index > 5:
            raise StopIteration()
        self.index += 1
        return [np.random.randn(*self.target_size).astype(np.float32)]
    
class ImageLoader(object):
    '''
        generate data for quantization from image datas.
        img_quan_data = (img - mean)/std, it's important for accuracy of model.
    '''
    VALID_FORMAT = ['.jpg', '.png', '.jpeg']
    
    def __init__(self, img_root, target_size, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) -> None:
        assert os.path.exists(img_root), F"{img_root} is not exists, please check!"
        self.fns = os.listdir(img_root)
        self.fns = list(filter(lambda fn: os.path.splitext(fn)[-1].lower() in self.VALID_FORMAT, self.fns))
        self.nums = len(self.fns)
        assert self.nums > 0, f"No images detected in {img_root}."
        if self.nums > 100:
            LOG.warning(f"{self.nums} images detected, the number of recommended images is less than 100.")
        else:
            LOG.info(f"{self.nums} images detected.")
        self.fns = [os.path.join(img_root, fn) for fn in self.fns]

        self.batch, self.size = target_size[0], target_size[1:-1]
        if isinstance(mean, list):
            mean = np.array(mean, dtype=np.float32)
        if isinstance(std, list):
            std = np.array(std, dtype=np.float32)
        self.mean, self.std = mean, std

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.nums:
            raise StopIteration()
    
        _input = cv2.imread(self.fns[self.index])
        _input = cv2.resize(_input, self.size)[:, :, ::-1]#BGR->RGB
        _input = _input.astype(np.float32)

        if self.mean is not None:
            _input = (_input - self.mean)
        if self.std is not None:
            _input = _input/self.std

        _input = np.expand_dims(_input, axis=0)
        if self.batch > 1:
            _input = np.repeat(_input, self.batch, axis=0).astype(np.float32)
        
        self.index += 1
        return [_input]
    