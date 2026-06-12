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

## Experiment 4: Saliency Map Generation Test

**Method:** Static Saliency Spectral Residual  
**Input Images:** Cat image, dog image, street scene image  
**Purpose:** To test whether saliency maps can highlight visually important regions before integrating attention into the object detection pipeline.

**Command Used:**

```bash
python src/generate_saliency_maps.py --input experiments/saliency_tests/input_images --maps experiments/saliency_tests/output_maps --overlays experiments/saliency_tests/overlays
```

## Experiment 5: Saliency vs YOLO Visual Comparison

**Input Images:** Cat image, dog image, street scene image  
**Purpose:** To visually compare YOLO detection outputs with saliency overlays and assess whether salient regions overlap with detected object regions.

**Command Used:**

```bash
python src/create_saliency_comparisons.py --originals experiments/saliency_tests/input_images --yolo results/pretrained_tests --saliency experiments/saliency_tests/overlays --output experiments/saliency_tests/comparisons
```

## Experiment 6: Saliency-Masked Image Detection Test

**Method:** Saliency-guided background suppression  
**Model:** YOLOv8n pretrained  
**Input Images:** Cat image, dog image, street scene image  
**Purpose:** To test whether suppressing less-salient image regions affects YOLO detections.

**Commands Used:**

```bash
python src/create_saliency_masked_images.py --images experiments/saliency_tests/input_images --maps experiments/saliency_tests/output_maps --output experiments/saliency_tests/masked_images
```
```bash
yolo detect predict model=yolov8n.pt source=experiments/saliency_tests/masked_images save=True conf=0.25 project="$PROJECT_ROOT\experiments\saliency_tests" name=masked_yolo_outputs exist_ok=True
```

## Experiment 7: Original vs Saliency-Masked Detection Comparison

**Model:** YOLOv8n pretrained  
**Input Images:** Cat image, dog image, street scene image  
**Purpose:** To quantitatively compare YOLO detections on original images and saliency-masked images.

**Commands Used:**

```bash
yolo detect predict model=yolov8n.pt source=experiments/saliency_tests/input_images save=True save_txt=True save_conf=True conf=0.25 project="$PROJECT_ROOT\experiments\saliency_tests" name=original_yolo_analysis exist_ok=True
```
```bash
yolo detect predict model=yolov8n.pt source=experiments/saliency_tests/masked_images save=True save_txt=True save_conf=True conf=0.25 project="$PROJECT_ROOT\experiments\saliency_tests" name=masked_yolo_analysis exist_ok=True
```
```bash
python src/compare_yolo_predictions.py --images experiments/saliency_tests/input_images --original-labels experiments/saliency_tests/original_yolo_analysis/labels --masked-labels experiments/saliency_tests/masked_yolo_analysis/labels --output experiments/saliency_tests/detection_comparison.csv
```

**Results Summary:**

| Image | Class | Original Count | Original Max Confidence | Masked Count | Masked Max Confidence | Confidence Change |
|---|---|---:|---:|---:|---:|---:|
| Cat | bed | 1 | 0.4537 | 1 | 0.2799 | -0.1738 |
| Cat | cat | 1 | 0.8454 | 1 | 0.8370 | -0.0084 |
| Dog | dog | 1 | 0.8440 | 1 | 0.8410 | -0.0030 |
| Street | bus | 1 | 0.3076 | 0 | N/A | N/A |
| Street | car | 9 | 0.8550 | 11 | 0.8596 | +0.0046 |
| Street | person | 4 | 0.3892 | 0 | N/A | N/A |

**Observations:**

The quantitative comparison shows that saliency masking preserved the main cat and dog detections with only very small confidence decreases. For the cat image, the false positive “bed” detection decreased from 0.4537 to 0.2799, suggesting that background suppression may reduce some incorrect detections.

In the street scene, the strongest car confidence slightly increased from 0.8550 to 0.8596, and the number of car detections increased from 9 to 11. However, the bus and person detections were removed after masking. This shows that saliency masking can help emphasise prominent objects, but it may also suppress smaller or less visually salient objects in cluttered scenes.

Overall, the results suggest that saliency-guided preprocessing has potential, but it must be applied carefully because aggressive background suppression may reduce recall for smaller objects.