YOLO-LOAMDPMS: Learnable Occlusion-Aware and Depth Point-wise Multi-Scale Channel Attention for Object Detection

Official implementation of "YOLO-LOAMDPMS: Learnable Occlusion-Aware and Depth Point-wise Multi-Scale Channel Attention for Object Detection"
Overview
This repository presents an enhanced YOLOv8m architecture that addresses robust object detection under challenging conditions including occlusion, scale variation, and adverse weather (rain, fog, snow, sand). The method integrates two lightweight attention modules:

LOAM (Learnable Occlusion-Aware Module): Dynamically generates spatial masks to suppress occluded regions
DPMS (Depth Point-wise Multi-Scale Channel Attention): Aggregates multi-scale contextual cues and recalibrates channel importance

Key Features

Robust Detection: Improved performance under adverse weather conditions (snow, rain, fog, sand)
Lightweight Design: Minimal computational overhead (only +2.49ms inference time on RTX 3050)
Real-world Validation: Tested on DAWN dataset and real highway traffic scenarios
Modular Architecture: Easy integration of LOAM and DPMS modules into YOLOv8m backbone

Performance Highlights
DAWN Dataset (Adverse Weather)

Rain: F1-score improved from 75.09% → 78.94% (+3.85%)
Snow: F1-score improved from 71.42% → 72.73%
Fog: F1-score improved from 73.09% → 73.73%

Real-world Highway Traffic (580 Freeway, San Francisco Bay Area)

Recall: 67.49% → 71.02% (+3.71%)
F1-score: 77.42% → 80.00%

Computational Efficiency

Parameters: 25.86M → 32.58M
FLOPs: 79.1G → 96.0G
Inference Time: 33.37ms → 37.74ms (NVIDIA GeForce RTX 3050)


Installation

git clone https://github.com/nhatcn/YOLO-LOAMDPMS-Learnable-Occlusion-Aware-and-Depth-Point-wise-Multi-Scale-Channel-Attention.git

cd YOLO-LOAMDPMS

Training
Train with Combined LOAM + DPMS

python start_train.py 
  --model ultralytics/cfg/models/v8/yolov8m_LOAMandDPMS.yaml 
  --data_dir /path/to/your/data.yaml

Train with DPMS Only

python start_train.py \
  --model ultralytics/cfg/models/v8/yolov8m_DPMS.yaml \
  --data_dir /path/to/your/data.yaml

....
