# Baseline Results

| Experiment | Dataset | Epochs | Fraction | Precision | Recall | mAP@50 | mAP@50-95 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Pretrained inference test | Sample images | N/A | N/A | N/A | N/A | N/A | N/A | YOLO successfully detected objects in cat, dog and street scene images |
| COCO8 sanity check | COCO8 | 3 | Full COCO8 | 0.771 | 0.833 | 0.889 | 0.654 | Environment and training sanity check |
| Pascal VOC subset test | Pascal VOC | 1 | 0.02 | 0.02059 | 0.58150 | 0.05577 | 0.03953 | VOC loading, conversion and training feasibility test |
| Initial Pascal VOC baseline | Pascal VOC | 3 | 0.05 | 0.32022 | 0.29772 | 0.24312 | 0.16181 | Initial baseline before adding saliency or attention |
| Saliency-masked YOLO test | Sample images | N/A | N/A | N/A | N/A | N/A | N/A | Cat and dog detections preserved; cat false positive reduced; street scene lost some detections |