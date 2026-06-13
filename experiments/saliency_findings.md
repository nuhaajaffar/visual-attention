# Saliency Experiment Findings

## Summary

This document summarises the saliency-based preprocessing experiments carried out to investigate whether visual saliency can act as a biologically inspired attention signal for object detection.

## Experiments Covered

| Experiment | Purpose |
|---|---|
| Experiment 4 | Generate saliency maps for sample images |
| Experiment 5 | Compare saliency overlays with YOLO detections |
| Experiment 6 | Test YOLO detections on saliency-masked images |
| Experiment 7 | Quantitatively compare original and masked detections |
| Experiment 8 | Compare weak, medium and strong masking strengths |
| Experiment 9 | Test weak masking on a larger image batch |

## Main Findings

Weak saliency masking was the most suitable masking strength. It preserved most object detections while slightly reducing background influence. Medium and strong masking were more aggressive, but they risked reducing or removing useful detections in cluttered scenes.

In the small batch test, weak masking preserved most detections across 22 images. The total detection count changed only slightly from 283 to 286, and the median confidence change was close to zero. This suggests that weak saliency masking does not significantly harm YOLO detections overall.

However, the results were mixed. Some detections improved, such as clock, bus and vase detections, while others weakened, such as chair, TV and sink detections. This indicates that saliency-guided preprocessing may be useful, but it is not consistently better than the original YOLO input.

## Key Conclusion

Weak saliency masking should be treated as a promising preprocessing method rather than a confirmed improvement. It may help reduce background influence in some cases, but further evaluation on a labelled dataset is needed before it can be claimed to improve object detection performance.

## Pascal VOC Evaluation Finding

When weak saliency masking was evaluated on a 100-image Pascal VOC subset, precision improved from 0.4809 to 0.7291. However, recall decreased from 0.2548 to 0.1926. The mAP values remained very similar, with mAP@50 changing from 0.2878 to 0.2844.

This suggests that weak saliency masking makes the detector more selective. It may reduce false positives, but it can also cause the model to miss objects. Therefore, saliency masking shows potential as a biologically inspired preprocessing method, but it does not yet provide a clear overall improvement over the baseline.

## Next Research Direction

The next stage should evaluate weak saliency masking using a labelled dataset such as Pascal VOC, where proper metrics such as precision, recall and mAP can be measured.