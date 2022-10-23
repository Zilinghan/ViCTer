import os
import sys
import cv2
import torch
import argparse
import platform
import warnings
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'track' / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'track' / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'track') not in sys.path:
    sys.path.append(str(ROOT / 'track' / 'trackers')) 
if str(ROOT / 'track' / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'track' / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'track' / 'trackers' / 'ocsort') not in sys.path:
    sys.path.append(str(ROOT / 'track' / 'trackers' / 'ocsort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'trackers' / 'track' / 'strong_sort' / 'deep' / 'reid' / 'torchreid') not in sys.path:
    sys.path.append(str(ROOT / 'track' / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from loader.dataloader import LoadVideos
from loader.trackerloader import create_tracker
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator
from recognition.train import train, get_device
from recognition.dataset import face_extraction, recognize
from recognition.draw import post_processing, event_plot, event_plot_setup, event_plot
warnings.filterwarnings('ignore')
# Limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Get the root folder path
# ROOT = Path.cwd()
# WEIGHTS = ROOT / 'weights'
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Colors for visualization
np.random.seed(100)
COLORS_PEOPLE = np.random.randint(0, 255, size=(2000, 3), dtype="uint8")
COLORS_FACE = np.random.randint(0, 255, size=(100, 3), dtype="uint8")
LOGGER.info("Module Imported Successfully!")

def detect(opt, fr_model, embedding_pool, num_classes, save_dir, ax):
    '''
    detect():
        This function implements the main algorithm of the Video Character Tracker.
        
    inputs:
        opt: hyperparameters provided by user
        fr_model: trained face recognition model
        embedding_pool: embedding pool for the labeled images
        num_classes: number of characters of interest
        [Specifically, the last three parameters are returned by train()]        
    '''
    # Get hyperparameters
    source, tracking_method, yolo_model, appearance_descriptor_weights, save_vid, imgsz, half= opt.source, opt.tracking_method, opt.yolo_weights, opt.appearance_descriptor_weights, opt.save_vid, opt.imgsz, opt.half
    fr_model.classify = False # face recognition model only generates face embeddings
    appear_dict = {} # information dictionary for each tracked person, which contains the face index and corresponding time slots
    total_frames = 0
    device = select_device(opt.device)
    # Initialize face-detection model
    fd_model = MTCNN(margin = 0, min_face_size = 20, select_largest = False, keep_all = True, device = device)
    # Initialize deepsort
    tracker = create_tracker(tracking_method, appearance_descriptor_weights, device, half)

    # Load YoLo model for multi-human tracking
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # Half precision
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()
    # Video dataloader: returned images are in RGB format
    dataset = LoadVideos(source, img_size=imgsz, stride=stride, auto=pt and not jit, batchsize=opt.batchsize, sample_stride=opt.stride)
    vid_path, vid_writer = None, None
    # Get names
    names = model.module.names if hasattr(model, 'module') else model.names
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    # Define some hyperparameters that we need to tune
    face_confs_thres = 0.95         # threshold value to filter out non-face detections and non-frontal face detectoins
    # TODO: make the max_new_coming value related to the `stride`
    max_new_coming = 4              # when there are `max_new_coming` number of new coming faces, create a new face instance for the tracked person
    
    # face_time_slots is used to store faces without corresponding tracked person or when the tracking is disabled 
    face_time_slots = [[] for _ in range(num_classes)]

    # Process each frame batch of the video
    for frame_idx, (path, img, vid_cap, s) in enumerate(dataset):
        # Loading image batch: (t2-t1)
        t1 = time_sync()
        # Face Detection (this network needs input of type: np.ndarray)
        face_bboxes, face_confs = fd_model.detect(img)
        face_imgs, face_counts = face_extraction(img, face_bboxes)
        try:
            face_imgs_ts = torch.stack(face_imgs).to(get_device())
        except:
            continue # no detected faces
        face_logits, face_embeddings, face_classes = recognize(fr_model, face_imgs_ts)

        img0 = img.copy()
        img = img.transpose((0,3,1,2))
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference (YOLO): (t3-t2)
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS to get pred: predicted tracks for each image in the batch
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        face_idx = 0 # Used for drawing
        face_logits_idx = 0 # Used for creating logits

        t4 = time_sync()
        # Process detections for each image
        for i, (det, face_bbox, face_conf) in enumerate(zip(pred, face_bboxes, face_confs)):  # detections per image
            seen += 1
            # if save-video: create the path for the saved video and the annotator
            im0 = img0[i].copy()
            if opt.save_vid:
                p = Path(path)
                save_path = str(save_dir/p.name) # path for the saved video
                annotator = Annotator(im0, line_width=1, pil=not ascii)
            
            # Visualization: Draw the faces with their labels
            if save_vid and face_bbox is not None and len(face_bbox):
                for j in range(len(face_bbox)):
                    if face_conf[j] > face_confs_thres:
                        face_class = embedding_pool.compare(face_embeddings[face_idx])
                        if face_class >= 0:
                            face_label = f'face: {face_class}'
                            # face_label = f'face: {face_conf[i]:.3f}'
                            annotator.box_label(face_bbox[j], face_label, color=tuple(COLORS_FACE[face_class]))
                    face_idx += 1
            

            no_track = True # whether there is tracked person in the frame

            if det is not None and len(det):
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round() # Rescale boxes from img_size to im0 size
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                # pass detections to deepsort
                outputs = tracker.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)                
                if len(outputs) > 0:
                    no_track = False
                    # For each detected & recognized faces, find a matching person
                    if face_bbox is not None and len(face_bbox):
                        for face, confident in zip(face_bbox, face_conf):
                            # if the face is a frontal face (with higher confidence)
                            if confident > face_confs_thres:
                                overlap = []
                                face_area = (face[2]-face[0]) * (face[3]-face[1])
                                for person in outputs:
                                    if ((face[0] >= person[0] and face[0] <= person[2]) or (face[2] >= person[0] and face[2] <= person[2])) and ((face[1] >= person[1] and face[1] <= person[3]) or (face[3] >= person[1] and face[3] <= person[3])):
                                        x1, y1, x2, y2 = max(face[0], person[0]), max(face[1], person[1]), min(face[2], person[2]), min(face[3], person[3])
                                        overlap.append(int((x2-x1)*(y2-y1)))
                                    else:
                                        overlap.append(-1)
                                max_overlap = max(overlap)
                                # If the face overlaps with a tracked person (has a corresponding person), update the appear_dict
                                if max_overlap/face_area > 0.8:
                                    person_idx = overlap.index(max_overlap)
                                    person_idx = outputs[person_idx, 4] # person id correponding to the current face

                                    if not (person_idx in appear_dict):
                                        appear_dict[person_idx] = {
                                            0: {
                                                'frame_idx': [],
                                                'non_interest_counter': 0,
                                                'face_class': None,
                                                'new_coming_frames': [],
                                                'new_coming_class': None,
                                                'new_coming_counter': 0,
                                                'frame_appended': False
                                            }
                                        }
                                    cur_face_embedding = face_embeddings[face_logits_idx]
                                    cur_face_class = embedding_pool.compare(cur_face_embedding)
                                    if cur_face_class == -2:
                                        face_logits_idx += 1
                                        continue
                                    cur_face_index = len(appear_dict[person_idx]) - 1
                                    # Case 1: no detected face at all for this person id
                                    if appear_dict[person_idx][cur_face_index]['face_class'] is None:
                                        appear_dict[person_idx][cur_face_index]['face_class'] = cur_face_class
                                        appear_dict[person_idx][cur_face_index]['frame_idx'].extend([frame_idx*opt.batchsize*opt.stride + i * opt.stride + ii for ii in range(opt.stride)])
                                        appear_dict[person_idx][cur_face_index]['frame_appended']= True
                                    # Case 2: current face is the tracked face
                                    elif appear_dict[person_idx][cur_face_index]['face_class'] == cur_face_class:
                                        # TODO: Merge the original new coming frames
                                        if len(appear_dict[person_idx][cur_face_index]['new_coming_frames']) != 0:
                                            appear_dict[person_idx][cur_face_index]['frame_idx'].extend(appear_dict[person_idx][cur_face_index]['new_coming_frames'])
                                        appear_dict[person_idx][cur_face_index]['frame_idx'].extend([frame_idx*opt.batchsize*opt.stride + i * opt.stride + ii for ii in range(opt.stride)])
                                        appear_dict[person_idx][cur_face_index]['new_coming_frames'] = []
                                        appear_dict[person_idx][cur_face_index]['new_coming_class'] = None
                                        appear_dict[person_idx][cur_face_index]['new_coming_counter'] = 0
                                        appear_dict[person_idx][cur_face_index]['frame_appended'] = True
                                    # Case 3: current face is not the tracked face
                                    elif appear_dict[person_idx][cur_face_index]['face_class'] != cur_face_class:
                                        # Case 3.1: No new coming face yet
                                        if appear_dict[person_idx][cur_face_index]['new_coming_class'] is None:
                                            appear_dict[person_idx][cur_face_index]['new_coming_frames'].extend([frame_idx*opt.batchsize*opt.stride + i * opt.stride + ii for ii in range(opt.stride)])
                                            appear_dict[person_idx][cur_face_index]['new_coming_class'] = cur_face_class
                                            appear_dict[person_idx][cur_face_index]['new_coming_counter'] = 1
                                            appear_dict[person_idx][cur_face_index]['frame_appended'] = True
                                        else:
                                            # Case 3.2: The current new coming face is the same as the original new coming face(s)
                                            if appear_dict[person_idx][cur_face_index]['new_coming_class'] == cur_face_class:
                                                appear_dict[person_idx][cur_face_index]['new_coming_frames'].extend([frame_idx*opt.batchsize*opt.stride + i * opt.stride + ii for ii in range(opt.stride)])
                                                appear_dict[person_idx][cur_face_index]['new_coming_counter'] += 1
                                                appear_dict[person_idx][cur_face_index]['frame_appended'] = True
                                                # Case 3.2.1: There are a series of same new coming faces, we create a new face instace
                                                if appear_dict[person_idx][cur_face_index]['new_coming_counter'] == max_new_coming:
                                                    appear_dict[person_idx][cur_face_index+1] = {
                                                        'frame_idx': appear_dict[person_idx][cur_face_index]['new_coming_frames'],
                                                        'non_interest_counter': 0,
                                                        'face_class': cur_face_class,
                                                        'new_coming_frames': [],
                                                        'new_coming_class': None,
                                                        'new_coming_counter': 0,
                                                        'frame_appended': True
                                                    }
                                            # Case 3.3: The new coming face is different from the original new coming face
                                            # TODO: Not sure whether this setup is reasonable, now, I just reset the new coming face
                                            else:
                                                appear_dict[person_idx][cur_face_index]['frame_idx'].extend(appear_dict[person_idx][cur_face_index]['new_coming_frames'])
                                                appear_dict[person_idx][cur_face_index]['new_coming_frames'] = [frame_idx*opt.batchsize*opt.stride + i * opt.stride + ii for ii in range(opt.stride)]
                                                appear_dict[person_idx][cur_face_index]['new_coming_class'] = cur_face_class
                                                appear_dict[person_idx][cur_face_index]['new_coming_counter'] = 1
                                                appear_dict[person_idx][cur_face_index]['frame_appended'] = True
                                # If the face does not corresponds to a tracked person, simply record its appearing frames
                                else:
                                    cur_face_embedding = face_embeddings[face_logits_idx]
                                    cur_face_class = embedding_pool.compare(cur_face_embedding)
                                    if cur_face_class < 0:
                                        face_logits_idx += 1
                                        continue
                                    face_time_slots[cur_face_class].extend([frame_idx*opt.batchsize*opt.stride + i * opt.stride + ii for ii in range(opt.stride)])
                            face_logits_idx += 1
                    # For each tracked person, update their appearing frames
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        person_idx = output[4]
                        if person_idx in appear_dict:
                            cur_face_index = len(appear_dict[person_idx]) - 1
                            if appear_dict[person_idx][cur_face_index]['frame_appended'] == True:
                                appear_dict[person_idx][cur_face_index]['frame_appended'] = False
                            else:
                                if len(appear_dict[person_idx][cur_face_index]['new_coming_frames']) != 0:
                                    appear_dict[person_idx][cur_face_index]['new_coming_frames'].extend([frame_idx*opt.batchsize*opt.stride + i * opt.stride + ii for ii in range(opt.stride)])
                                else:
                                    appear_dict[person_idx][cur_face_index]['frame_idx'].extend([frame_idx*opt.batchsize*opt.stride + i * opt.stride + ii for ii in range(opt.stride)])
                        else:
                            appear_dict[person_idx] = {
                                                        0: {
                                                            'frame_idx': [frame_idx*opt.batchsize*opt.stride + i * opt.stride + ii for ii in range(opt.stride)],
                                                            'non_interest_counter': 0,
                                                            'face_class': None,
                                                            'new_coming_frames': [],
                                                            'new_coming_class': None,
                                                            'new_coming_counter': 0,
                                                            'frame_appended': False
                                                        }
                                                    }     
                        # Visualization: Draw boxes for persons
                        if save_vid:
                            bboxes = output[0:4]
                            c = int(output[5])
                            label = f'{person_idx} {names[c]} {conf:.2f}'
                            color_idx = person_idx * len(appear_dict[person_idx]) * 5 % len(COLORS_PEOPLE)
                            annotator.mask_label(bboxes, label, color=tuple(COLORS_PEOPLE[color_idx]))
            # else:
            #     tracker.increment_ages()

            # If not tracked person, simply record the faces
            if no_track and face_bbox is not None and len(face_bbox):
                for j in range(len(face_bbox)):
                    if face_conf[j] > face_confs_thres:
                        cur_face_class = embedding_pool.compare(face_embeddings[face_logits_idx])
                        if cur_face_class >= 0:
                            face_time_slots[cur_face_class].extend([frame_idx*opt.batchsize*opt.stride + i * opt.stride + ii for ii in range(opt.stride)])
                    face_logits_idx += 1

            # Save results (image with detections)
            if save_vid:
                im0 = annotator.result()
                if vid_path is None:  # new video
                    vid_path = save_path
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = im0.shape[1]
                        h = im0.shape[0]
                        total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
                vid_writer.write(im0)
        t5 = time_sync()
        dt[3] += t5 - t4
        LOGGER.info(f'{s}Done. Load:({t2-t1:.3f}s), YOLO:({t3 - t2:.3f}s), Tracking:({t5 - t4:.3f}s)')
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_vid:
        print('Videos saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
    event_plot(ax, post_processing(appear_dict, face_time_slots, num_classes, max_new_coming), opt.source, opt.label_folder, idx=1)
    
    # draw_screentimes(after_processing_2(appear_dict, face_time_slots, num_classes, max_new_coming), total_frames, save_dir, 'output.jpg')
    # print('Plotted figure saved to %s' %str(save_dir/'output.jpg'))
    # np.save(str(save_dir/'output.npy'), appear_dict)
    # print('Appearance dict saved to %s' % str(save_dir/'output.npy'))
    
def detect_no_track(opt, fr_model, embedding_pool, num_classes, save_dir, ax):
    '''
    detect_no_track():
        Track the person only using the face recognition network  
    inputs:
        opt: hyperparameters provided by user
        fr_model: trained face recognition model
        embedding_pool: embedding pool for the labeled images
        num_classes: number of characters of interest
        [Specifically, the last three parameters are returned by train()]        
    '''
    # Get hyperparameters
    source, imgsz, half, project, name, exist_ok= opt.source, opt.imgsz, opt.half, opt.project, opt.name, opt.exist_ok
    fr_model.classify = False # face recognition model only generates face embeddings
    total_frames = 0
    device = select_device(opt.device)
    # Initialize face-detection model
    fd_model = MTCNN(margin = 0, min_face_size = 20, select_largest = False, keep_all = True, device = device)
    # Video dataloader: returned images are in RGB format
    dataset = LoadVideos(source, img_size=imgsz, batchsize=opt.batchsize, sample_stride=opt.stride)
    # Define some hyperparameters that we need to tune
    face_confs_thres = 0.95         # threshold value to filter out non-face detections and non-frontal face detectoins
    # face_time_slots is used to store faces without corresponding tracked person or when the tracking is disabled 
    face_time_slots = [[] for _ in range(num_classes)]
    # Process each frame batch of the video
    for frame_idx, (path, img, vid_cap, s) in enumerate(dataset):
        # Face Detection (this network needs input of type: np.ndarray)
        face_bboxes, face_confs = fd_model.detect(img)
        face_imgs, _ = face_extraction(img, face_bboxes)
        try:
            face_imgs_ts = torch.stack(face_imgs).to(get_device())
        except:
            continue # no detected faces
        _, face_embeddings, _ = recognize(fr_model, face_imgs_ts)

        img = img.transpose((0,3,1,2))
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        face_idx = 0 
        # Process detections for each image
        for i, (face_bbox, face_conf) in enumerate(zip(face_bboxes, face_confs)):  # detections per image
            if face_bbox is not None and len(face_bbox):
                for j in range(len(face_bbox)):
                    if face_conf[j] > face_confs_thres:
                        face_class = embedding_pool.compare(face_embeddings[face_idx])
                        if face_class >= 0:
                            face_time_slots[face_class].extend([frame_idx*opt.batchsize*opt.stride + i * opt.stride + ii for ii in range(opt.stride)])
                    face_idx += 1       
        LOGGER.info(f'{s}Done.')
    event_plot(ax, post_processing({}, face_time_slots, num_classes, 0), opt.source, opt.label_folder, idx=2)
    
    # draw_screentimes(after_processing_2({}, face_time_slots, num_classes, 0), total_frames, save_dir, 'output_notrack.jpg')
    # print('Plotted figure saved to %s' %str(save_dir/'output_notrack.jpg'))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov5n.pt', help='model.pt path(s)')
    parser.add_argument('--appearance-descriptor-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort')
    # parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5n.pt', help='model.pt path(s)')
    # parser.add_argument('--deep_sort_model', type=str, default='osnet_x1_0')
    parser.add_argument('--source', type=str, default='datasets/vct1/vct1.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[960], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--classes', nargs='+', default=0, type=int, help='filter by class: --class 0, or --class 16 17') # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--face-folder', default='datasets/vct1/face', help='folder of face images provided by users')
    parser.add_argument('--stride', type=int, default=4, help='detect the movie for every stride frame(s)')
    parser.add_argument('--model-folder', default='model/vct1.pth')
    parser.add_argument('--label-folder', default='datasets/vct1/time_slot')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # Results saving directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  
    # Result plot
    ax = event_plot_setup(opt.source, opt.label_folder)
    # Train the face recoginition network first, since we use no_grad during the detection
    start = time_sync()
    # (1) For Test Speed:
    # Load the saved model
#     fr_model = torch.load(opt.model_folder)
#     fr_model.classify = False
#     LabeledTrainSet = TrainingSetLabeled(opt.face_folder, transform=False)
#     num_classes = LabeledTrainSet.get_num_classes()
#     LOGGER.info(f"Number of classes is : {num_classes}")
#     LabeledTrainLoader = DataLoader(LabeledTrainSet, batch_size=8, shuffle=False)
#     embedding_pool = EmbeddingPool(LabeledTrainLoader, fr_model, get_device(), threshold=1.1, threshold_high=1.28)
    # (2) For Real Case:
    # Train from the data
    fr_model, embedding_pool, num_classes = train(opt.source, opt.face_folder)
    torch.save(fr_model, opt.model_folder)
    LOGGER.info(f"The model is saved to {opt.model_folder}")
    end = time_sync()
    LOGGER.info(f"Total time for training is: {end-start:.3f}s")
    start = time_sync()
    with torch.no_grad():
        detect(opt, fr_model, embedding_pool, num_classes, save_dir, ax)
        detect_no_track(opt, fr_model, embedding_pool, num_classes, save_dir, ax)
    end = time_sync()
    LOGGER.info(f"Total time for tracking and detection is: {end-start:.3f}s")
    ax.figure.savefig(str(save_dir/'output.jpg'))
    LOGGER.info(f"Time slots figure is saved to {str(save_dir/'output.jpg')}")