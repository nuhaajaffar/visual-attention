# Experiment Log

## Experiment 0: Pretrained YOLOv8n Inference Test

**Model:** YOLOv8n pretrained
**Device:** CPU
**Input Images:** Cat image, dog image, street scene image

**Command Used:**

```bash
yolo detect predict model=yolov8n.pt source=results/pretrained_tests/input_images save=True conf=0.25
```

**Results:**

| Image              | Detection Result              |                Confidence |
| ------------------ | ----------------------------- | ------------------------: |
| Cat image          | Cat detected                  |                      0.85 |
| Dog image          | Dog detected                  |                      0.84 |
| Street scene image | Cars, people and bus detected | Various confidence scores |

**Observations:**

The pretrained YOLOv8n model successfully detected objects in the sample images, including a cat, a dog, cars, people and a bus. This confirms that the YOLO environment, model loading and inference pipeline are working correctly.

The street scene image did not produce a “street” label because YOLOv8 performs object detection rather than scene classification. Instead of classifying the whole image as a street, the model detected individual objects within the scene. This demonstrates the difference between object detection and scene classification, which is important for defining the scope of this FYP.

The cat image also produced an additional low-confidence detection, which may be a false positive. This is useful to note because object detection models can sometimes detect irrelevant or incorrect objects, especially when background regions resemble known classes.

**Saved Outputs:**

* `results/pretrained_tests/cat_yolo_output.jpg`
* `results/pretrained_tests/dog_yolo_output.jpg`
* `results/pretrained_tests/street_yolo_output.jpg`

---

## Experiment 1: COCO8 Training Sanity Check

**Model:** YOLOv8n
**Dataset:** COCO8
**Device:** CPU
**Epochs:** 3
**Image Size:** 640
**Purpose:** To confirm that YOLOv8 training works correctly on the local machine before training on Pascal VOC.

**Command Used:**

```bash
yolo detect train model=yolov8n.pt data=coco8.yaml epochs=3 imgsz=640 project=runs/baseline name=coco8_test
```

**Training Result:** Completed successfully.

**Final Validation Metrics from Best Model:**

| Metric          |              Value |
| --------------- | -----------------: |
| Precision       |              0.771 |
| Recall          |              0.833 |
| mAP@50          |              0.889 |
| mAP@50-95       |              0.654 |
| Inference Speed | 140.1 ms per image |
| Training Time   |        0.004 hours |

**Observations:**

The COCO8 training run completed successfully, confirming that the YOLOv8 training pipeline works on the local CPU-based environment. The model was able to train, validate, save weights and generate output plots without errors.

The results should not be treated as meaningful object detection performance because COCO8 is an extremely small dataset containing only a few training and validation images. Instead, this experiment serves as a sanity check to confirm that the environment, dataset loading, training loop, validation process and result saving are functioning correctly.

**Saved Outputs:**

* `runs/baseline/coco8_test/results.csv`
* `runs/baseline/coco8_test/results.png`
* `runs/baseline/coco8_test/weights/best.pt`
* `runs/baseline/coco8_test/weights/last.pt`

---

## Experiment 2: Pascal VOC Subset Training Test

**Model:** YOLOv8n
**Dataset:** Pascal VOC subset
**Device:** CPU
**Epochs:** 1
**Image Size:** 416
**Batch Size:** 2
**Dataset Fraction:** 0.02
**Purpose:** To confirm that Pascal VOC can be downloaded, converted into YOLO format, loaded successfully and used for YOLOv8 training.

**Command Used:**

```bash
yolo detect train model=yolov8n.pt data=VOC.yaml epochs=1 imgsz=416 batch=2 workers=0 fraction=0.02 project="$PROJECT_ROOT\runs\baseline" name=voc_subset_test
```

**Training Result:** Completed successfully.

**Final Metrics:**

| Metric        |           Value |
| ------------- | --------------: |
| Precision     |         0.02059 |
| Recall        |         0.58150 |
| mAP@50        |         0.05577 |
| mAP@50-95     |         0.03953 |
| Training Time | 658.921 seconds |

**Observations:**

The Pascal VOC subset training test completed successfully, confirming that Pascal VOC can be loaded and used with YOLOv8. This also confirmed that the dataset download, annotation conversion, training loop and validation process were functioning correctly.

The low precision and mAP values are expected because the model was trained for only one epoch using 2% of the dataset. Therefore, this experiment should be treated as a feasibility test rather than a meaningful baseline result.

The training time also indicates that full Pascal VOC training on CPU would be slow, so future experiments may need to use smaller image sizes, fewer epochs, dataset fractions or an external GPU environment.

**Saved Outputs:**

* `runs/baseline/voc_subset_test/results.csv`
* `runs/baseline/voc_subset_test/results.png`

---

## Experiment 3: Initial Pascal VOC Baseline

**Model:** YOLOv8n
**Dataset:** Pascal VOC subset
**Device:** CPU
**Epochs:** 3
**Image Size:** 416
**Batch Size:** 2
**Dataset Fraction:** 0.05
**Purpose:** To train an initial YOLOv8n baseline on a small Pascal VOC subset before introducing attention or saliency mechanisms.

**Command Used:**

```bash
yolo detect train model=yolov8n.pt data=VOC.yaml epochs=3 imgsz=416 batch=2 workers=0 fraction=0.05 project="$PROJECT_ROOT\runs\baseline" name=voc_baseline_3epochs
```

**Training Result:** Completed successfully.

**Final Metrics:**

| Metric        |           Value |
| ------------- | --------------: |
| Precision     |         0.32022 |
| Recall        |         0.29772 |
| mAP@50        |         0.24312 |
| mAP@50-95     |         0.16181 |
| Training Time | 2509.86 seconds |

**Observations:**

The YOLOv8n baseline successfully trained on a 5% subset of Pascal VOC for 3 epochs. The mAP@50 improved from 0.10853 in epoch 1 to 0.24312 in epoch 3, showing that the model was learning from the dataset.

The results are still limited because only a small fraction of the dataset was used and training was performed for a small number of epochs on CPU. However, this experiment provides an initial baseline for comparison before adding saliency-based or attention-guided mechanisms.

**Saved Outputs:**

* `runs/baseline/voc_baseline_3epochs/results.csv`
* `runs/baseline/voc_baseline_3epochs/results.png`
