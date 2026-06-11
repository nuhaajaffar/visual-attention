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
