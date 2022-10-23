import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img



def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)




def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)

def facez_augment_pool():
    augs = [(AutoContrast, None, None),
#             (Brightness, 2, 8),
#             (Color, 1.8, 4),
#             (Contrast, 2, 10),
            (Rotate, 10, 0),
            (Identity, None, None),
            (Equalize, None, None),
    ]
    return augs


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (Solarize, 256, 0)]
    return augs


def my_augment_pool():
    # Test
    augs = [(AutoContrast, None, None),
            (Brightness, 1.8, 0.1),
            (Color, 1.8, 0.1),
            (Contrast, 1.8, 0.1),
            (Cutout, 0.2, 0),
            (Equalize, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 1.8, 0.1),
            (Solarize, 256, 0),
            (SolarizeAdd, 110, 0)]
    return augs


class RandAugmentPC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = my_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        print(ops)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(16*0.5))
        return img

class RandAugmentFaceZ(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = facez_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, int(32*0.5))
        return img
    
class Cutout(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = CutoutAbs(img, self.size)
        return img
    
    
face_aug1 = transforms.Compose([
#     transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(5),
    Cutout(16),
    transforms.ToTensor()
])
face_aug2 = transforms.Compose([
#     transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ColorJitter(contrast=0.5),
    transforms.RandomRotation(5),
    Cutout(16),
    transforms.ToTensor()
])
face_aug3 = transforms.Compose([
#     transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.ColorJitter(saturation=0.5),
    transforms.RandomRotation(5),
    Cutout(16),
    transforms.ToTensor()
])
face_aug4 = transforms.Compose([
#     transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomResizedCrop(size=160, scale=(0.8,1), ratio=(0.833333, 1.2)),
    transforms.RandomRotation(5),
    Cutout(16),
    transforms.ToTensor()
])

face_aug5 = transforms.Compose([
#     transforms.ToPILImage(),
    transforms.Resize(100),
    transforms.Resize(160),
    transforms.ToTensor()
])

face_aug6 = transforms.Compose([
#     transforms.ToPILImage(),
    transforms.Resize(70),
    transforms.Resize(160),
    transforms.ToTensor()
])

face_aug7 = transforms.Compose([
#     transforms.ToPILImage(),
    transforms.Resize(50),
    transforms.Resize(160),
    transforms.ToTensor()
])

face_aug8 = transforms.Compose([
#     transforms.ToPILImage(),
    transforms.Resize(45),
    transforms.Resize(160),
    transforms.ToTensor()
])

def face_augmentations():
    return [transforms.ToTensor(), face_aug1, face_aug2, face_aug3, face_aug4, face_aug5, face_aug6, face_aug7]

def face_augmentations_simple():
    return [transforms.ToTensor(), face_aug6, face_aug7, face_aug8]