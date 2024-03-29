# ViCTer: A Semi-supervised Video Character Tracker
<img src='experiments/results/track-demo.gif' width=500>

<span style="color:red">[News]:</span> We have now have improved version of ViCTer which integrates optical flow to the model for a much better performance. More information will come sooner.

<span style="color:red">[News]:</span> The paper *ViCTer: A Semi-supervised Video Character Tracker* is accepted by Journal *Machine Learning with Applications*, and is open-access online [here](https://doi.org/10.1016/j.mlwa.2023.100460).

## Introduction
This repository contains the source code for the paper *ViCTer: A Semi-supervised Video Character Tracker*

Video character tracking problem refers to tracking certain characters of interest in the video and returning the appearing time slots for those characters. ViCTer is a novel model for address this problem by combining our proposed semi-supervised face recognition network with a [multi-human tracker](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet). We collect a dataset for the video character tracking problem, [Character Face in Video (CFIV)](https://ieee-dataport.org/documents/character-face-video), which can support various experiments for evaluating video character tracker performance. Our proposed model can achieve 70\% $\sim$ 80\% average intersection-over-union tracking accuracy on this dataset.     

## Installation
```
git clone --recurse-submodules https://github.com/Zilinghan/ViCTer
cd ViCTer
conda create -n victer python=3.8
conda activate victer
pip install -r requirements.txt
```

## Download Dataset
Download the dataset CIFV for running the experiment from the [*IEEEDataport*](https://ieee-dataport.org/documents/character-face-video), and put it under `datasets` in the following structure:
```
datasets
   |——————vct1.zip
   └——————vct2.zip
   └——————...
   └——————vct10.zip
```
Then go to the `datasets` folder and run the given script to unzip files and download videos.
```
cd datasets
chmod +x data.sh
./data.sh
```

## Tracking
Note: For getting the tracking accuracy on different videos, replace `vct2` below to corresponding `vctx`.
```
python run.py --source datasets/vct2/vct2.mp4           # source video to track
              --face-folder datasets/vct2/face          # folder of face images of characters to be tracked
              --model-path model/vct2.pth               # path to store the trained recognition model
              --label-folder datasets/vct2/time_slot    # folder containing the ground truth character appearing time slots
              --stride 4                                # detect the movie for every 'stride' frame(s)
              --tracking-method ocsort                  # tracking algorithm (ocsort or strongsort)
              --save-vid                                # use only if you want to save video output
```

## Other Experiments
We have also provide codes for other experiments to evaluate the performance of ViCTer along several dimensions. Those codes are given as jupyter notebooks under the ```experiments``` folder.

[Face Recognition Accuracy](/experiments/exp1.ipynb)

<img src='experiments/results/accuracy.svg' width=600/>

[Embedding Distances](/experiments/exp2.ipynb)

<img src='experiments/results/embedding_dist.svg' width=600/>

[Embedding Clustering](/experiments/exp3.ipynb)

<img src='experiments/results/tsne.svg' width=600/>

<img src='experiments/results/umap.svg' width=600/>

Tracking Results

<img src='experiments/results/output-v1.svg' width=600/>

<img src='experiments/results/output-v2.svg' width=600/>

<img src='experiments/results/output-v3.svg' width=600/>

## Citing ViCTer
```
@article{li2023victer,
  title={ViCTer: A semi-supervised video character tracker},
  author={Li, Zilinghan and Wang, Xiwei and Zhang, Zhenning and Kindratenko, Volodymyr},
  journal={Machine Learning with Applications},
  volume={12},
  pages={100460},
  year={2023},
  publisher={Elsevier}
}
```