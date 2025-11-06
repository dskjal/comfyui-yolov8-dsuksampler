# comfyui-yolov8-dsuksampler
This node performs detection and segmentation, as well as upsampling, sampling, and blurring the composite.

This node pass through when nothing detected. When this node finds multiple objects, it performs an action on each object it finds.

## Why do we need this node?

[The simple detection and composite node](https://github.com/dskjal/comfyui-yolov8-simple) can't handle following case.

1. Nothing detected
2. Multiple objectes detected

Simple workflow can't skip sample phase when nothing detected and The simple node can't actions on multiple objects.

## Sample workflow

![](https://github.com/dskjal/comfyui-yolov8-dsuksampler/blob/main/comfyui-yolov8-dsuksampler-sample-workflow.png)
