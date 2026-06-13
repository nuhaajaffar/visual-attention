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

## Masking Strength Evaluation Finding

Experiment 11 compared original, weak, medium and strong saliency masking on the same 100-image Pascal VOC validation subset. All saliency-masked versions improved precision compared with the original images, but they also reduced recall.

Medium masking achieved the highest precision and mAP@50, with precision increasing from 0.4809 to 0.7520 and mAP@50 increasing slightly from 0.2878 to 0.2920. However, recall decreased from 0.2548 to 0.1790. This suggests that saliency masking makes the detector more selective, but may cause it to miss less salient objects.

The next experiment should therefore test whether training the model directly on saliency-masked images improves the result, rather than only evaluating an original-trained model on masked images.

## Medium-Masked Training Finding

Experiment 12 tested whether training YOLOv8n directly on medium saliency-masked images would improve performance compared with training on original images. The original-trained model achieved a mAP@50 of 0.0894, while the medium-masked-trained model achieved a lower mAP@50 of 0.0825.

The masked-trained model achieved a very small recall improvement, increasing from 0.0383 to 0.0392, but this was not enough to offset the decrease in mAP. Therefore, training directly on medium-masked images did not improve the baseline in this experiment.

This suggests that saliency masking is useful for analysing attention and detector behaviour, but it should not yet be claimed as a definite improvement over the original YOLO pipeline.