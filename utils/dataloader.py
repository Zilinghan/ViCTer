# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""
import cv2
import numpy as np
from pathlib import Path


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
            
class LoadVideos:
    # YOLOv5 video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True, batchsize=32, sample_stride=1):
        p = str(Path(path).resolve())               # os-agnostic absolute path
        self.video = p                              # path to the input video
        self.img_size = img_size                    # size of the video image
        self.stride = stride        
        self.mode = 'video'         
        self.auto = auto        
        self.batchsize = batchsize                  # number of images to return 
        self.sample_stride = sample_stride          # stride for the sampling
        self.frame = 0                              # current frame index
        self.cap = cv2.VideoCapture(self.video)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame == self.frames:
            raise StopIteration

        # Read video
        imgs = []
        for i in range(self.batchsize):
            ret_val, img0 = self.cap.read()
            if not ret_val:
                if self.frame != 0: # handle the special case: first frame cannot be read
                    self.cap.release()
                continue
            self.frame += 1

            # skipping (sample_stride-1) frames
            for j in range(self.sample_stride-1):
                ret_val_new, _ = self.cap.read()
                if not ret_val_new:
                    if i == (self.batchsize-1):
                        self.cap.release()
                    break
                self.frame += 1

            # Padded resize image
            img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

            # Convert
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.ascontiguousarray(img)
            imgs.append(img)
        try:
            imgs = np.stack(imgs)
        except:
            imgs = imgs[0]
        s = f'Frame: {self.frame}/{self.frames} '
        return self.video, imgs, self.cap, s

    def __len__(self):
        return 1  # number of files            

