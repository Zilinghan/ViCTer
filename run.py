import os
import sys
import torch
import argparse
import warnings
from pathlib import Path
from torch.utils.data import DataLoader
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'track' / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'track' / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'track' / 'trackers') not in sys.path:
    sys.path.append(str(ROOT / 'track')) 
if str(ROOT / 'track' / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'track' / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'track' / 'trackers' / 'ocsort') not in sys.path:
    sys.path.append(str(ROOT / 'track' / 'trackers' / 'ocsort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'trackers' / 'track' / 'strong_sort' / 'deep' / 'reid' / 'torchreid') not in sys.path:
    sys.path.append(str(ROOT / 'track' / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.utils.general import increment_path
from yolov5.utils.torch_utils import time_sync
from recognition.train import train, get_device
from recognition.dataset import TrainingSetLabeled, TrainingSetUnlabeled
from recognition.draw import event_plot_setup
from recognition.embedding import EmbeddingPool
from detection import detect, detect_no_track
warnings.filterwarnings('ignore')
# Limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
print("Module Imported Successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='datasets/vct2/vct2.mp4', help='path to source video to track')
    parser.add_argument('--face-folder', default='datasets/vct2/face', help='folder of face images of characters to be tracked')
    parser.add_argument('--label-folder', default='datasets/vct2/time_slot', help='folder containing the ground truth character appearing time slots')
    parser.add_argument('--model-path', default='model/vct2.pth', help='path to store the trained recognition model/path of stored trained model')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort')
    parser.add_argument('--stride', type=int, default=4, help='process the video for every stride frame(s)')
    parser.add_argument('--batchsize', type=int, default=32, help='batchsize for processing')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov5n.pt', help='path to yolo model')
    parser.add_argument('--appearance-descriptor-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt', help='path to strongsort appearance descriptor model')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[960], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold for object detection')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IOU threshold for object detection')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment') 
    parser.add_argument('--face-rec-only', action='store_true', help='run the tracker by only using the face recognition model')
    parser.add_argument('--no-training', action='store_true', help='EXPERIMENT ONLY: enable it if you have a trained recognition model and want to use the trained model for face recogniton')
    parser.add_argument('--train-only', action='store_true', help='EXPERIMENT ONLY: enable it if you only want to train a face recognition model without tracking')
    parser.add_argument('--run-comparison', action='store_true', help='Run both the face recognition model only method, and victer tracking method')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    # Results saving directories and result plot
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  
    fig, ax = event_plot_setup(opt.source, opt.label_folder)

    # Train the face recognition network
    start = time_sync()
    if opt.no_training:
        # For Test Speed: If you have saved model
        fr_model = torch.load(opt.model_path) # Load the saved model
        fr_model.classify = False
        LabeledTrainSet = TrainingSetLabeled(opt.face_folder, transform=False)
        LabeledTrainLoader = DataLoader(LabeledTrainSet, batch_size=8, shuffle=False)
        UnlabeledTrainSet = TrainingSetUnlabeled(opt.source, "data", num_images=2048)
        UnlabeledTrainLoader = DataLoader(UnlabeledTrainSet, batch_size=8*10, shuffle=True)
        embedding_pool = EmbeddingPool(LabeledTrainLoader, UnlabeledTrainLoader, fr_model, get_device())
        num_classes = LabeledTrainSet.get_num_classes()
    else:
        # For Real Case: No saved model availabel
        fr_model, embedding_pool, num_classes = train(opt.source, opt.face_folder)
        torch.save(fr_model, opt.model_path)
        print(f"The model is saved to {opt.model_path}")
    end = time_sync()
    print(f"Total time for training is {end-start:.3f}s")
    
    if not opt.train_only:
        # Start the detection and tracking
        start = time_sync()
        with torch.no_grad():
            if opt.run_comparison:
                detect(opt, fr_model, embedding_pool, num_classes, save_dir, ax)
                detect_no_track(opt, fr_model, embedding_pool, num_classes, save_dir, ax)
            elif opt.face_rec_only:
                detect_no_track(opt, fr_model, embedding_pool, num_classes, save_dir, ax)
            else:
                detect(opt, fr_model, embedding_pool, num_classes, save_dir, ax)
        end = time_sync()
        print(f"Total time for tracking and detection is {end-start:.3f}s")

        # Save the result plot
        fig.tight_layout()
        ax.figure.savefig(str(save_dir/'output.svg'))
        print(f"Time slots figure is saved to {str(save_dir/'output.svg')}")