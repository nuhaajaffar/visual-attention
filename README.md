# Biologically Inspired Visual Attention for Object Detection

This repository contains the code, experiments and results for a final year project investigating **saliency-guided visual attention for object detection**.

The project explores whether biologically inspired attention can influence object detection performance by using saliency maps to highlight visually important regions of an image. The baseline object detector used in this project is **YOLOv8n**, and the main experimental method applies saliency-guided preprocessing before object detection.

The aim is not to claim that saliency masking always improves YOLO, but to evaluate how it changes detector behaviour in terms of precision, recall and mAP.

## Project Overview

Humans do not process every part of a visual scene equally. Instead, human visual attention naturally focuses on important or visually distinctive regions while ignoring less relevant background information.

Inspired by this idea, this project investigates whether saliency maps can be used as a simple attention mechanism for object detection. Saliency maps are generated from input images and used to darken less salient background regions while preserving more visually important areas.

The project evaluates whether this saliency-guided preprocessing can help YOLOv8n become more selective in object detection tasks.

## Research Focus

The project focuses on the following question:

**Can saliency-guided preprocessing act as a biologically inspired attention mechanism for object detection?**

The experiments investigate:

1. Whether saliency maps align with important object regions.
2. Whether saliency masking affects YOLOv8n detections.
3. Whether different masking strengths affect precision, recall and mAP.
4. Whether training directly on masked images improves performance.
5. Whether saliency preprocessing is useful as a practical attention-based method.

## Methodology

The project follows this pipeline:

```text
Input image
↓
Generate saliency map
↓
Apply saliency-guided masking
↓
Run YOLOv8n object detection
↓
Evaluate precision, recall, mAP@50 and mAP@50-95
```

The saliency maps are generated using OpenCV’s Static Saliency Spectral Residual method. The generated saliency map is used to preserve visually salient regions while reducing the brightness of less salient areas.

Three masking strengths were tested:

| Mask Type | Background Visibility |
| --------- | --------------------: |
| Weak      |                   0.7 |
| Medium    |                   0.5 |
| Strong    |                   0.3 |

A higher background visibility means the background is darkened less aggressively. A lower value means the background is suppressed more strongly.

## Dataset

The main dataset used in the object detection experiments is **Pascal VOC**.

The project uses YOLO-format Pascal VOC images and labels. Small subsets were used because the experiments were run on a CPU-only machine.

Main dataset settings used in the experiments:

| Experiment Stage           | Dataset                     |
| -------------------------- | --------------------------- |
| Initial baseline training  | Pascal VOC subset           |
| Saliency evaluation        | Pascal VOC val2012 subset   |
| Masked training comparison | Pascal VOC train/val subset |

The full Pascal VOC dataset is not included in this repository because it is a large external dataset. The scripts expect the dataset to be available locally.

## Experiments

### Experiment 0: Pretrained YOLOv8n Inference Test

Pretrained YOLOv8n was tested on sample images such as cat, dog and street scenes to confirm that the object detection pipeline worked correctly.

### Experiment 1: COCO8 Training Sanity Check

A small COCO8 training run was completed to confirm that local YOLO training worked successfully.

### Experiment 2: Pascal VOC Subset Training Test

YOLOv8n was trained on a small Pascal VOC subset to confirm that the Pascal VOC dataset could be loaded, converted and used for training.

### Experiment 3: Initial Pascal VOC Baseline

An initial YOLOv8n baseline was trained on a Pascal VOC subset. This provided the baseline detector used for later saliency experiments.

### Experiment 4: Saliency Map Generation Test

Saliency maps were generated for sample images. The results showed that saliency maps highlighted object-centred regions well in simple images, but were noisier in cluttered scenes.

### Experiment 5: Saliency vs YOLO Visual Comparison

Original images, YOLO detections and saliency overlays were compared visually. This helped assess whether saliency maps aligned with detected object regions.

### Experiment 6: Saliency-Masked Image Detection Test

YOLOv8n was tested on saliency-masked images. The results showed that masking could reduce some false-positive confidence values while preserving major objects in simple scenes.

### Experiment 7: Quantitative Comparison of Original vs Masked YOLO Predictions

YOLO predictions on original and masked images were compared using saved detection labels and confidence scores.

### Experiment 8: Saliency Masking Strength Comparison

Weak, medium and strong saliency masking were compared on sample images. Weak masking preserved the most detections, while strong masking suppressed the background more aggressively but removed some useful detections.

### Experiment 9: Small Batch Test with Weak Saliency Masking

Weak saliency masking was tested on a batch of 22 images. Most detections were preserved, and the median confidence change was close to zero.

### Experiment 10: Pascal VOC Weak Saliency Evaluation

A 100-image Pascal VOC validation subset was used to evaluate original images against weak saliency-masked images using standard detection metrics.

### Experiment 11: Pascal VOC Saliency Masking Strength Evaluation

Original, weak, medium and strong masked images were compared using precision, recall, mAP@50 and mAP@50-95.

### Experiment 12: Medium-Masked YOLO Training Comparison

A YOLOv8n model trained on original images was compared with a YOLOv8n model trained directly on medium-masked images. This tested whether training directly on saliency-masked data improved performance.

## Results Summary

### Pascal VOC Saliency Masking Evaluation

| Input Type           | Precision | Recall | mAP@50 | mAP@50-95 |
| -------------------- | --------: | -----: | -----: | --------: |
| Original images      |    0.4809 | 0.2548 | 0.2878 |    0.2059 |
| Weak masked images   |    0.7291 | 0.1926 | 0.2844 |    0.2017 |
| Medium masked images |    0.7520 | 0.1790 | 0.2920 |    0.2030 |
| Strong masked images |    0.6490 | 0.2040 | 0.2780 |    0.1940 |

The masking strength evaluation showed that saliency masking increased precision compared with the original images. Medium masking achieved the highest precision and mAP@50. However, the original images achieved the highest recall and mAP@50-95.

This suggests that saliency masking makes the detector more selective, but it can also cause the model to miss some valid objects.

### Medium-Masked Training Comparison

| Model                       | Precision | Recall | mAP@50 | mAP@50-95 |
| --------------------------- | --------: | -----: | -----: | --------: |
| Original-trained model      |    0.9903 | 0.0383 | 0.0894 |    0.0656 |
| Medium-masked-trained model |    0.9898 | 0.0392 | 0.0825 |    0.0585 |

Training directly on medium-masked images did not improve overall performance. The medium-masked-trained model achieved a very small recall increase, but its mAP@50 and mAP@50-95 decreased.

## Main Finding

The main finding is that **saliency-guided preprocessing changes YOLOv8n’s detection behaviour by improving precision but reducing recall**.

This means saliency masking can make the detector more selective and reduce background influence, but it may also suppress useful object information. Therefore, saliency-guided preprocessing is useful as an experimental biologically inspired attention mechanism, but it should not be presented as a guaranteed improvement over the original YOLO pipeline.

## Project Structure

```text
visual-attention/
│
├── src/
│   ├── saliency_maps.py
│   ├── saliency_comp.py
│   ├── saliency_masked.py
│   ├── yolo_predictions.py
│   ├── voc_saliency_eval.py
│   ├── voc_masked_eval_dataset.py
│   ├── voc_saliency_metrics.py
│   ├── voc_train_val_subset.py
│   ├── voc_masked_train_val_dataset.py
│   ├── e12_training_comp.py
│   └── final_results.py
│
├── experiments/
│   ├── experiment_log.md
│   ├── baseline_results.md
│   ├── saliency_findings.md
│   ├── progress_summary.md
│   ├── final_direction.md
│   ├── final_results_summary.csv
│   │
│   ├── saliency_tests/
│   ├── saliency_batch_test/
│   ├── voc_saliency_eval/
│   └── e12_medium_mask_training/
│
├── runs/
│   └── baseline/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies Used

The project was implemented in Python.

Main tools and libraries used:

* Python
* Ultralytics YOLOv8
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Git

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/nuhaajaffar/visual-attention.git
cd visual-attention
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Check YOLO installation

```bash
yolo checks
```

### 6. OpenCV saliency note

This project uses OpenCV saliency functions. If saliency support is missing, reinstall OpenCV using the contrib version:

```bash
pip uninstall opencv-python -y
pip install opencv-contrib-python
```

This confirms that Ultralytics YOLO is installed correctly.

## Data Availability

The full Pascal VOC dataset is not included in this repository.

To run the Pascal VOC experiments locally, the dataset should be available in the following structure:

```text
C:\Users\Nuhaa\Projects\datasets\VOC\
│
├── images/
│   ├── train2007/
│   ├── train2012/
│   ├── val2007/
│   ├── val2012/
│   └── test2007/
│
└── labels/
    ├── train2007/
    ├── train2012/
    ├── val2007/
    ├── val2012/
    └── test2007/
```

The dataset path can be changed in the command arguments when running the preparation scripts.

Dataset images, labels, generated saliency maps and model weights are not committed to GitHub because they are large generated or external files.

## How to Run

### Generate Saliency Maps for Sample Images

```bash
python src/saliency_maps.py --input experiments/saliency_tests/input_images --maps experiments/saliency_tests/output_maps --overlays experiments/saliency_tests/overlays
```

This generates grayscale saliency maps and saliency overlay images for the input images.

### Create Saliency-Masked Images

```bash
python src/saliency_masked.py --images experiments/saliency_tests/input_images --maps experiments/saliency_tests/output_maps --output experiments/saliency_tests/masked_images_medium --background 0.5
```

This creates medium saliency-masked images using 0.5 background visibility.

### Run YOLO on Saliency-Masked Images

```bash
yolo detect predict model=yolov8n.pt source=experiments/saliency_tests/masked_images_medium save=True save_txt=True save_conf=True conf=0.25
```

This runs YOLOv8n prediction on the masked images and saves detection outputs.

### Compare YOLO Predictions

```bash
python src/yolo_predictions.py --images experiments/saliency_tests/input_images --original-labels experiments/saliency_tests/original_yolo_analysis/labels --masked-labels experiments/saliency_tests/masked_yolo_medium/labels --output experiments/saliency_tests/detection_comparison_medium.csv
```

This compares original and masked YOLO detections using saved YOLO label files.

### Prepare Pascal VOC Saliency Evaluation Subset

```bash
python src/voc_saliency_eval.py --voc-root C:\Users\Nuhaa\Projects\datasets\VOC --output experiments\voc_saliency_eval --split val2012 --limit 100
```

This prepares a 100-image Pascal VOC validation subset for saliency evaluation.

### Evaluate Original and Masked VOC Images

```bash
python src/voc_saliency_metrics.py --model runs/baseline/voc_baseline_3epochs/weights/best.pt --original-data experiments/voc_saliency_eval/original.yaml --masked-data experiments/voc_saliency_eval/masked.yaml --project experiments/voc_saliency_eval --output experiments/voc_saliency_eval/e10_metrics.csv --imgsz 416 --batch 2
```

This evaluates original and masked VOC images using precision, recall, mAP@50 and mAP@50-95.

### Run Medium-Masked Training Comparison

```bash
python src/e12_training_comp.py --original-model experiments/e12_medium_mask_training/original_train_3epochs/weights/best.pt --original-data experiments/e12_medium_mask_training/original.yaml --masked-model experiments/e12_medium_mask_training/medium_masked_train_3epochs/weights/best.pt --masked-data experiments/e12_medium_mask_training/medium_masked.yaml --project experiments/e12_medium_mask_training --output experiments/e12_medium_mask_training/e12_metrics.csv --imgsz 416 --batch 2
```

This evaluates the original-trained and medium-masked-trained models and saves the final comparison metrics.

## Output Files

The `experiments/` folder contains the main experiment outputs and documentation.

Important files include:

```text
experiments/experiment_log.md
experiments/baseline_results.md
experiments/saliency_findings.md
experiments/progress_summary.md
experiments/final_direction.md
experiments/final_results_summary.csv
experiments/voc_saliency_eval/e10_metrics.csv
experiments/voc_saliency_eval/e11_metrics.csv
experiments/e12_medium_mask_training/e12_metrics.csv
```

Generated images, dataset copies, saliency maps and model weights are excluded from version control where possible.

## Skills Demonstrated

* Python programming
* Computer vision
* Object detection
* YOLOv8
* Saliency maps
* Image preprocessing
* Dataset preparation
* Model evaluation
* Precision, recall and mAP analysis
* Experimental design
* Reproducible research workflow
* Git version control

## Limitations

* Experiments were run on a CPU-only machine, so dataset sizes and epoch counts were limited.
* Saliency masking was applied at pixel level, which can suppress useful object information.
* The Pascal VOC evaluation used small subsets rather than full-scale training.
* The method improved precision in some settings but did not consistently improve recall or overall mAP.
* The project does not modify YOLO’s internal architecture; it evaluates saliency as a preprocessing method.

## Future Improvements

Possible future improvements include:

* Evaluating on a larger Pascal VOC or COCO subset.
* Testing more saliency methods.
* Applying attention at feature level instead of pixel level.
* Adding CBAM or other attention modules into the model architecture.
* Testing saliency-guided augmentation instead of direct masking.
* Comparing object-centred and cluttered scenes separately.
* Running experiments on GPU for larger-scale evaluation.

## Key Takeaway

Saliency-guided preprocessing provides a biologically inspired way to influence object detection behaviour. It can make YOLOv8n more selective and improve precision, but it can also reduce recall by hiding useful visual information.

The final conclusion is that saliency masking is a useful attention-based experimental method, but not a complete replacement for model-level attention.