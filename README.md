# 3D Bounding Box Prediction

## Dependencies

| Package                | Version      | Used In                              |
| ---------------------- | ------------ | ------------------------------------ |
| Python                 | 3.12.3       | runtime                              |
| torch                  | 2.11.0+cu126 | model, training, inference, losses   |
| numpy                  | 2.4.3        | data processing, evaluation, viz     |
| timm                   | 1.0.26       | pretrained ResNet-18 backbone        |
| opencv-python-headless | 4.13.0.92    | image I/O, resize, visualization     |
| matplotlib             | 3.10.8       | plots, 3D point cloud viz            |
| scipy                  | 1.17.1       | 3D IoU (ConvexHull), Hungarian match |
| pillow                 | 12.1.1       | image loading (dataset)              |
| onnx                   | 1.21.0       | model export, FP16 conversion        |
| onnxruntime            | 1.24.4       | ONNX inference validation            |
| onnxscript             | 0.6.2        | torch.onnx.export (PyTorch 2.x dep)  |
| pytest                 | 9.0.2        | test suite (178 tests)               |
