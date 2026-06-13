# Final Project Direction

## Project Focus

This project investigates saliency-guided preprocessing as a biologically inspired visual attention method for object detection.

The main idea is inspired by human visual attention, where attention is directed towards visually important regions rather than processing all image regions equally. In this project, saliency maps are used to identify visually important regions, and saliency-guided masking is applied to reduce the influence of less salient background regions.

## Final Method

The final method uses a YOLOv8n object detection baseline and applies saliency-guided preprocessing before detection. The preprocessing step generates a saliency map for each image and uses it to darken less salient image regions while preserving visually important areas.

Three masking strengths were tested:

| Mask Type | Background Visibility |
|---|---:|
| Weak | 0.7 |
| Medium | 0.5 |
| Strong | 0.3 |

## Main Result

The experiments showed that saliency masking can increase precision, meaning the detector becomes more selective and produces fewer incorrect detections. However, recall decreases, meaning the model also misses more objects.

Medium masking achieved the highest precision and mAP@50 in the masking strength evaluation, but the original images achieved the highest recall and mAP@50-95.

## Final Conclusion

Saliency-guided preprocessing is useful as a biologically inspired attention mechanism for analysing and influencing detector behaviour. However, simple pixel-level saliency masking is not consistently better than the original YOLO pipeline.

The final conclusion is that saliency masking improves selectivity but introduces a precision-recall trade-off. It should therefore be presented as an attention-based preprocessing approach with measurable effects, rather than as a guaranteed improvement over the baseline.

## Future Work

Future work could explore feature-level attention methods, such as CBAM or transformer-based attention, where attention is applied inside the model rather than directly suppressing image pixels. This may allow the model to benefit from attention cues without losing important object information.