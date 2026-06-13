# FYP Progress Summary

## Project Direction

This project investigates whether biologically inspired visual attention can improve object detection. The main idea is to use saliency maps as an attention signal to highlight visually important regions while reducing the influence of background clutter.

The object detection baseline used in this stage is YOLOv8n. Saliency-guided masking was tested as a preprocessing method before considering deeper architectural changes.

---

## Completed Work

### YOLO Baseline Setup

The YOLOv8n environment was successfully configured on a CPU-based machine. Pretrained YOLOv8n was tested on sample images, and the model successfully detected objects such as cats, dogs, cars, people and buses.

A small COCO8 training sanity check was completed to confirm that the training pipeline worked correctly.

A Pascal VOC subset training test was then completed to confirm that Pascal VOC could be downloaded, converted, loaded and used for YOLO training.

---

## Saliency Map Experiments

Saliency maps were generated using the Static Saliency Spectral Residual method. The saliency maps were tested on simple object-centred images and cluttered scenes.

The results showed that saliency maps aligned well with main objects in simple images such as cat and dog images. However, in cluttered scenes, saliency maps also highlighted high-contrast background structures such as road markings, lights and building edges.

This showed that saliency maps can act as a useful attention signal, but they may also be noisy in complex environments.

---

## Saliency Masking Experiments

Saliency masking was tested by darkening less salient image regions while preserving visually salient areas. Three masking strengths were tested:

| Mask Type | Background Visibility |
|---|---:|
| Weak | 0.7 |
| Medium | 0.5 |
| Strong | 0.3 |

Initial sample-image experiments showed that weak masking preserved most object detections while slightly reducing some false-positive confidence. Strong masking reduced some false positives more aggressively, but also removed useful detections such as people and buses in cluttered scenes.

A larger batch test using weak masking showed that most detections were preserved. Across 22 images and 58 class-level comparisons, the total number of detections remained similar, increasing from 283 to 286. The median confidence change was close to zero, suggesting that weak masking did not substantially harm detection confidence overall.

---

## Pascal VOC Evaluation

A 100-image Pascal VOC validation subset was used to evaluate saliency masking using proper object detection metrics.

The results showed that saliency masking increased precision but reduced recall.

| Input Type | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| Original images | 0.4809 | 0.2548 | 0.2878 | 0.2059 |
| Weak masked images | 0.7291 | 0.1926 | 0.2844 | 0.2017 |
| Medium masked images | 0.7520 | 0.1790 | 0.2920 | 0.2030 |
| Strong masked images | 0.6490 | 0.2040 | 0.2780 | 0.1940 |

Medium masking achieved the highest precision and mAP@50. However, the original images achieved the highest recall and mAP@50-95.

This suggests that saliency masking makes YOLO more selective, but it may also cause the model to miss some valid objects.

---

## Medium-Masked Training Comparison

A fairer training comparison was conducted by training one YOLOv8n model on original VOC images and another YOLOv8n model on medium saliency-masked VOC images.

| Model | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| Original-trained model | 0.9903 | 0.0383 | 0.0894 | 0.0656 |
| Medium-masked-trained model | 0.9898 | 0.0392 | 0.0825 | 0.0585 |

Training directly on medium-masked images did not improve the baseline. The masked-trained model achieved a very small recall increase, but its mAP@50 and mAP@50-95 decreased.

---

## Main Finding So Far

Saliency-guided masking changes detector behaviour by making the model more selective. It can improve precision in some settings, but it often reduces recall. This means saliency masking may reduce background influence, but it may also suppress useful object information.

Therefore, saliency-based preprocessing should be treated as a useful experimental attention mechanism, but not yet as a confirmed improvement over YOLO.

---

## Current Conclusion

The project has produced a meaningful research finding:

Saliency maps can provide a biologically inspired attention signal for object detection, but simple saliency-based image masking is not consistently better than the original YOLO pipeline. The method improves precision in some cases but can reduce recall and overall detection coverage.

---

## Next Direction

The next stage should move from simple image preprocessing towards a more model-aware attention method. Instead of directly darkening image pixels, attention could be applied more carefully through:

1. feature-level attention,
2. saliency-weighted augmentation,
3. CBAM-style attention modules,
4. or a hybrid method where saliency maps guide training rather than directly suppressing pixels.

This would allow the model to benefit from attention cues without removing too much visual information from the image.