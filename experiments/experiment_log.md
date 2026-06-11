# Experiment Log

## Experiment 0: Pretrained YOLOv8n Inference Test

**Model:** YOLOv8n pretrained
**Device:** CPU
**Input Images:** Cat image, dog image, street scene image

**Command Used:**

```bash
yolo detect predict model = yolov8n.pt source = results/pretrained_tests/input_images save = True conf = 0.25
```

**Results:**

| Image              | Detection Result              | Confidence                |
| ------------------ | ----------------------------- | ------------------------- |
| Cat image          | Cat detected                  | 0.85                      |
| Dog image          | Dog detected                  | 0.84                      |
| Street scene image | Cars, people and bus detected | Various confidence scores |

**Observations:**

The pretrained YOLOv8n model successfully detected objects in the sample images, including a cat, a dog, cars, people and a bus. This confirms that the YOLO environment, model loading and inference pipeline are working correctly.

The street scene image did not produce a “street” label because YOLOv8 performs object detection rather than scene classification. Instead of classifying the whole image as a street, the model detected individual objects within the scene. This demonstrates the difference between object detection and scene classification, which is important for defining the scope of this FYP.

The cat image also produced an additional low-confidence detection, which may be a false positive. This is useful to note because object detection models can sometimes detect irrelevant or incorrect objects, especially when background regions resemble known classes.

**Saved Outputs:**

* `results/pretrained_tests/cat_yolo_output.jpg`
* `results/pretrained_tests/dog_yolo_output.jpg`
* `results/pretrained_tests/street_yolo_output.jpg`

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

* `runs/detect/runs/baseline/coco8_test/results.csv`
* `runs/detect/runs/baseline/coco8_test/results.png`
* `runs/detect/runs/baseline/coco8_test/weights/best.pt`
* `runs/detect/runs/baseline/coco8_test/weights/last.pt`
