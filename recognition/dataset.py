import os
import cv2
import torch
import random
from torch import nn
from glob import glob
from facenet_pytorch import MTCNN
from collections import defaultdict
from torchvision import transforms
from . import augmentation as aug

# TrainingSetLabeled:
# Labeled Training Dataset
class TrainingSetLabeled(torch.utils.data.Dataset):
    def __init__(self, files_dir, h=160, w=160,transform=True, class_idx=None):
        self.imgs = []
        self.labels = []
        self.height = h
        self.width = w
        self.num_classes = 0
        self.class_idx = class_idx
        self.image_loader(files_dir, h, w)
        if transform:
            self.transformers = aug.face_augmentations()
        else:
            self.transformers = [transforms.ToTensor()]
        
    def image_loader(self,files_dir, h, w):
        # image_loader:
        # load all the labeled images from the folder
        self.mtcnn = MTCNN(
            margin = 0,
            min_face_size = 80,
            image_size = 160,
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        f1 = r"*png"
        f2 = os.path.join(files_dir,f1)
        PIL_trans = transforms.ToPILImage()

        ids = defaultdict(list)
        for img_path in glob(f2):
            img_name = img_path[:-4].split('/')[-1]
            character = int(img_name.split("_")[0])
            ids[character].append(img_path)
        
        for id in sorted(ids.keys()):
            if not self.class_idx is None and id != self.class_idx:
                continue
            imgs = ids[id]
            for img in imgs:
                my_img = cv2.imread(img)
                my_img = cv2.cvtColor(my_img, cv2.COLOR_BGR2RGB)
                height, width = my_img.shape[0], my_img.shape[1]
                bbox, _ = self.mtcnn.detect(my_img)
                try:
                    xmin, ymin, xmax, ymax = max(int(bbox[0][0]),0), max(int(bbox[0][1]),0), min(int(bbox[0][2]), width), min(int(bbox[0][3]),height)
                    self.imgs.append(PIL_trans(cv2.resize(my_img[ymin:ymax,xmin:xmax,:], (w,h))))
                    self.labels.append(id)
                except:
                    pass
        self.num_classes = id+1
        del self.mtcnn
        
    def __getitem__(self, idx):
        transformer_id = idx // len(self.imgs)
        image_id = idx % len(self.imgs)
        return self.transformers[transformer_id](self.imgs[image_id]), self.labels[image_id]
#         return self.transformer(self.imgs[idx]), self.labels[idx]

    def __len__(self):
        return len(self.imgs) * len(self.transformers)
    
    def get_num_classes(self):
        return self.num_classes

# TrainingSetUnlabeled:
# Unlabeled Training Dataset
class TrainingSetUnlabeled(torch.utils.data.Dataset):
    def __init__(self, video_path, root, h=160, w=160, num_images=2048):
        self.video_path = video_path
        self.height = h
        self.width = w
        self.data_dir = None
        self.trans = transforms.ToTensor()
        self.trans_w = transforms.Compose([
            transforms.ToPILImage(),
            aug.RandAugmentFaceZ(n=1,m=10),
            transforms.ToTensor()
        ])
        # Check whether the root directory exists
        if not os.path.exists(root):
            os.mkdir(root)
        # Check whether the data directory exists
        video_name = video_path.split('/')[-1].split('.')[0]
        self.data_dir = os.path.join(root, video_name)
        # If not: create the directory and the images
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
            self.dataset_download(num_images)
        self.length = len(glob(os.path.join(self.data_dir, r"*jpg")))
        
    def dataset_download(self, num_images):
        self.mtcnn = MTCNN(
            margin = 0,
            min_face_size = 80,
            image_size = 160,
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        # Open the video capture
        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            raise ValueError("The given video path does not exist!")
        # Get the total number of frames
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < num_images:
            ids = range(frame_count)
        else:
            ids = random.sample(range(frame_count), num_images)

        face_counter = 0
        frame_counter = 0
        while capture.isOpened():
            ok, frame = capture.read()
            frame_counter += 1
            if ok:
                if frame_counter-1 in ids:
#                     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_h, frame_w = frame.shape[0], frame.shape[1]
                    bbox, score = self.mtcnn.detect(frame)
                    try:
                        
                        xmin, ymin, xmax, ymax = max(int(bbox[0][0]),0), max(int(bbox[0][1]),0), min(int(bbox[0][2]), frame_w), min(int(bbox[0][3]),frame_h)
                        if cv2.imwrite(os.path.join(self.data_dir, f"{face_counter}.jpg"), cv2.resize(frame[ymin:ymax,xmin:xmax,:],(self.width, self.height))):
                            face_counter += 1
#                             print(face_counter-1, score[0])
                    except:
                        pass
            if frame_counter >= frame_count-2:
                break
        capture.release()
        del self.mtcnn
        
    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, f"{idx}.jpg")), cv2.COLOR_BGR2RGB)
        return self.trans(image), self.trans_w(image)

    def __len__(self):
        return self.length


def face_extraction(imgs, bboxes, size=(160,160)):
    '''
    face_extraction:
        Given a list of images and the predictions from the R-CNN,  
        output a list of face images detected by R-CNN with given size.
    Inputs:
        imgs:           ndarray
        predictions:    ndarray[ndarray]
        size:           Tuple
    Outputs:
        faceimgs:       ndarray
        facecounts:     List[Int]
    '''
    faceimgs = []
    facecounts = []
    resizer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size)
    ])
    for i, bbox in enumerate(bboxes):
        # If there are detected faces
        if bbox is not None:
            original_img = imgs[i]
            height, width = original_img.shape[:2]
            
            facecounts.append(bbox.shape[0]) # number of detected faces in the given image
            
            for box in bbox:
                xmin, ymin, xmax, ymax = max(int(box[0]),0), max(int(box[1]),0), min(int(box[2]), width), min(int(box[3]),height)
                faceimgs.append(resizer(original_img[ymin:ymax, xmin:xmax, :]))
                
        else:
            facecounts.append(0)

    return faceimgs, facecounts


def recognize(model, faces):    
    if isinstance(model, nn.Module):
        model.eval()
        
    with torch.no_grad():
        embeddings = model(faces)
        logits = model.logits(embeddings)
        logits = torch.softmax(logits, axis=1)
        _, y_hat = torch.max(logits, axis=1)
    return logits, embeddings, y_hat

