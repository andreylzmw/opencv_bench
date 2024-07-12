# OpenCV Benchmark

1. YuNet                 [2] - Face Detection 
2. SFace                 [2] - Face Recognition
3. FacialExpressionRecog [2] - Face Expression Recognition
4. MPHandPose            [2] - Hand Pose Estimation
5. PPHumanSeg            [2] - Human Segmentation
6. MobileNet             [4] - Image Classification
7. PPResNet              [2] - Image Classification
8. LPD_YuNet             [2] - License Plate Detection
9. NanoDet               [2] - Object Detection
10. MPPalmDet            [2] - Palm Detection
11. MPPersonDet          [1] - Person Detection
12. YoutuReID            [2] - Person Re-Identification
13. MPPose               [1] - Pose Estimation 
14. WeChatQRCode         [1] - QR Code Detection and Parsing
15. PPOCRDet             [4] - English Text detection
16. CRNN                 [8] - Text Detection


CPU: 12th Gen Intel i7-12800H (20) @ 4.700GHz

gcc-13:
```
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
1.22       1.25       1.16       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
1.34       1.37       1.16       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
6.24       6.26       6.07       [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
9.36       9.73       6.07       [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
3.02       3.04       2.95       [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
4.82       5.11       2.95       [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
3.91       3.82       3.79       [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
5.05       5.23       3.79       [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
8.46       8.35       8.28       [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
9.52       9.66       8.28       [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
4.67       4.67       4.59       [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
4.51       4.59       4.29       [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
5.38       4.35       4.29       [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
6.01       6.17       4.29       [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
24.72      25.12      24.30      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
30.90      31.48      24.30      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
8.60       8.70       8.08       [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
11.16      13.21      8.08       [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
46.87      47.04      46.32      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
48.29      48.50      46.32      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
8.83       8.68       8.63       [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
11.13      11.66      8.63       [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
11.30      11.41      11.08      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
24.84      25.97      22.28      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
28.59      29.68      22.28      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
10.50      10.67      10.20      [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
2.29       2.27       2.14       [100, 100]   WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
23.34      23.52      23.04      [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
23.32      23.10      22.89      [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
30.55      23.64      22.89      [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may_int8.onnx']
35.66      36.94      22.89      [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may_int8.onnx']
13.85      13.63      13.44      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
14.25      14.22      13.44      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
13.04      14.79      10.61      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
13.19      15.40      10.61      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
12.65      11.82      10.58      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
13.07      12.36      10.58      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
13.53      13.76      10.58      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
13.30      12.41      10.58      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

clang-18:
```
backend=cv.dnn.DNN_BACKEND_OPENCV
target=cv.dnn.DNN_TARGET_CPU
mean       median     min        input size   model
1.12       1.19       0.91       [160, 120]   YuNet with ['face_detection_yunet_2023mar.onnx']
1.23       1.28       0.91       [160, 120]   YuNet with ['face_detection_yunet_2023mar_int8.onnx']
6.30       6.50       6.15       [150, 150]   SFace with ['face_recognition_sface_2021dec.onnx']
8.90       9.16       6.15       [150, 150]   SFace with ['face_recognition_sface_2021dec_int8.onnx']
3.13       3.04       3.00       [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july.onnx']
4.61       5.27       3.00       [112, 112]   FacialExpressionRecog with ['facial_expression_recognition_mobilefacenet_2022july_int8.onnx']
4.34       4.28       4.19       [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb.onnx']
5.16       5.23       4.19       [224, 224]   MPHandPose with ['handpose_estimation_mediapipe_2023feb_int8.onnx']
8.87       9.63       8.68       [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar.onnx']
9.65       9.95       8.68       [192, 192]   PPHumanSeg with ['human_segmentation_pphumanseg_2023mar_int8.onnx']
5.00       4.90       4.89       [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr.onnx']
4.77       4.80       4.15       [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr.onnx']
5.44       4.28       4.15       [224, 224]   MobileNet with ['image_classification_mobilenetv1_2022apr_int8.onnx']
5.91       5.89       4.15       [224, 224]   MobileNet with ['image_classification_mobilenetv2_2022apr_int8.onnx']
26.60      26.43      26.04      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan.onnx']
31.18      31.13      26.04      [224, 224]   PPResNet with ['image_classification_ppresnet50_2022jan_int8.onnx']
9.35       9.94       8.61       [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar.onnx']
11.16      13.11      8.61       [320, 240]   LPD_YuNet with ['license_plate_detection_lpd_yunet_2023mar_int8.onnx']
35.21      35.18      34.86      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov.onnx']
36.20      36.61      34.86      [416, 416]   NanoDet with ['object_detection_nanodet_2022nov_int8.onnx']
9.24       9.29       8.99       [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb.onnx']
10.85      11.14      8.99       [192, 192]   MPPalmDet with ['palm_detection_mediapipe_2023feb_int8.onnx']
11.38      11.39      11.14      [224, 224]   MPPersonDet with ['person_detection_mediapipe_2023mar.onnx']
24.94      24.95      24.54      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov.onnx']
26.08      26.06      24.54      [128, 256]   YoutuReID with ['person_reid_youtu_2021nov_int8.onnx']
10.33      10.43      9.78       [256, 256]   MPPose with ['pose_estimation_mediapipe_2023mar.onnx']
2.18       2.17       2.10       [100, 100]   WeChatQRCode with ['detect_2021nov.prototxt', 'detect_2021nov.caffemodel', 'sr_2021nov.prototxt', 'sr_2021nov.caffemodel']
25.90      26.32      25.09      [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may.onnx']
25.92      25.73      25.09      [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may.onnx']
32.39      25.71      25.09      [640, 480]   PPOCRDet with ['text_detection_cn_ppocrv3_2023may_int8.onnx']
36.97      38.35      25.09      [640, 480]   PPOCRDet with ['text_detection_en_ppocrv3_2023may_int8.onnx']
14.12      13.96      13.58      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2021sep.onnx']
14.48      14.62      13.58      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov.onnx']
13.24      14.88      10.88      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2021sep.onnx']
13.42      13.00      10.88      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2023feb_fp16.onnx']
12.85      11.52      10.69      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2023feb_fp16.onnx']
13.26      12.55      10.69      [1280, 720]  CRNN with ['text_recognition_CRNN_CH_2022oct_int8.onnx']
13.71      13.87      10.69      [1280, 720]  CRNN with ['text_recognition_CRNN_CN_2021nov_int8.onnx']
13.32      12.39      10.48      [1280, 720]  CRNN with ['text_recognition_CRNN_EN_2022oct_int8.onnx']
```

A zoo for models tuned for OpenCV DNN with benchmarks on different platforms.

Guidelines:

- Install latest `opencv-python`:
  ```shell
  python3 -m pip install opencv-python
  # Or upgrade to latest version
  python3 -m pip install --upgrade opencv-python
  ```
- Clone this repo to download all models and demo scripts:
  ```shell
  # Install git-lfs from https://git-lfs.github.com/
  git clone https://github.com/opencv/opencv_zoo && cd opencv_zoo
  git lfs install
  git lfs pull
  ```
- To run benchmarks on your hardware settings, please refer to [benchmark/README](./benchmark/README.md).

## Models & Benchmark Results

![](benchmark/color_table.svg?raw=true)

Hardware Setup:

x86-64:
- [Intel Core i7-12700K](https://www.intel.com/content/www/us/en/products/sku/134594/intel-core-i712700k-processor-25m-cache-up-to-5-00-ghz/specifications.html): 8 Performance-cores (3.60 GHz, turbo up to 4.90 GHz), 4 Efficient-cores (2.70 GHz, turbo up to 3.80 GHz), 20 threads.

ARM:
- [Khadas VIM3](https://www.khadas.com/vim3): Amlogic A311D SoC with a 2.2GHz Quad core ARM Cortex-A73 + 1.8GHz dual core Cortex-A53 ARM CPU, and a 5 TOPS NPU. Benchmarks are done using **per-tensor quantized** models. Follow [this guide](https://github.com/opencv/opencv/wiki/TIM-VX-Backend-For-Running-OpenCV-On-NPU) to build OpenCV with TIM-VX backend enabled.
- [Khadas VIM4](https://www.khadas.com/vim4): Amlogic A311D2 SoC with 2.2GHz Quad core ARM Cortex-A73 and 2.0GHz Quad core Cortex-A53 CPU, and 3.2 TOPS Build-in NPU.
- [Khadas Edge 2](https://www.khadas.com/edge2): Rockchip RK3588S SoC with a CPU of 2.25 GHz Quad Core ARM Cortex-A76 + 1.8 GHz Quad Core Cortex-A55, and a 6 TOPS NPU.
- [Atlas 200 DK](https://e.huawei.com/en/products/computing/ascend/atlas-200): Ascend 310 NPU with 22 TOPS @ INT8. Follow [this guide](https://github.com/opencv/opencv/wiki/Huawei-CANN-Backend) to build OpenCV with CANN backend enabled.
- [Atlas 200I DK A2](https://www.hiascend.com/hardware/developer-kit-a2): SoC with 1.0GHz Quad-core CPU and Ascend 310B NPU with 8 TOPS @ INT8.
- [NVIDIA Jetson Nano B01](https://developer.nvidia.com/embedded/jetson-nano-developer-kit): a Quad-core ARM A57 @ 1.43 GHz CPU, and a 128-core NVIDIA Maxwell GPU.
- [NVIDIA Jetson Nano Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/): a 6-core Arm® Cortex®-A78AE v8.2 64-bit CPU, and a 1024-core NVIDIA Ampere architecture GPU with 32 Tensor Cores (max freq 625MHz).
- [Raspberry Pi 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/): Broadcom BCM2711 SoC with a Quad core Cortex-A72 (ARM v8) 64-bit @ 1.5 GHz.
- [Horizon Sunrise X3](https://developer.horizon.ai/sunrise): an SoC from Horizon Robotics with a quad-core ARM Cortex-A53 1.2 GHz CPU and a 5 TOPS BPU (a.k.a NPU).
- [MAIX-III AXera-Pi](https://wiki.sipeed.com/hardware/en/maixIII/ax-pi/axpi.html#Hardware): Axera AX620A SoC with a quad-core ARM Cortex-A7 CPU and a 3.6 TOPS @ int8 NPU.
- [Toybrick RV1126](https://t.rock-chips.com/en/portal.php?mod=view&aid=26): Rockchip RV1126 SoC with a quard-core ARM Cortex-A7 CPU and a 2.0 TOPs NPU.

RISC-V:
- [StarFive VisionFive 2](https://doc-en.rvspace.org/VisionFive2/Product_Brief/VisionFive_2/specification_pb.html): `StarFive JH7110` SoC with a RISC-V quad-core CPU, which can turbo up to 1.5GHz, and an GPU of model `IMG BXE-4-32 MC1` from Imagination, which has a work freq up to 600MHz.
- [Allwinner Nezha D1](https://d1.docs.aw-ol.com/en): Allwinner D1 SoC with a 1.0 GHz single-core RISC-V [Xuantie C906 CPU](https://www.t-head.cn/product/C906?spm=a2ouz.12986968.0.0.7bfc1384auGNPZ) with RVV 0.7.1 support. YuNet is tested for now. Visit [here](https://github.com/fengyuentau/opencv_zoo_cpp) for more details.

***Important Notes***:

- The data under each column of hardware setups on the above table represents the elapsed time of an inference (preprocess, forward and postprocess).
- The time data is the mean of 10 runs after some warmup runs. Different metrics may be applied to some specific models.
- Batch size is 1 for all benchmark results.
- `---` represents the model is not availble to run on the device.
- View [benchmark/config](./benchmark/config) for more details on benchmarking different models.

## Some Examples

Some examples are listed below. You can find more in the directory of each model!

### Face Detection with [YuNet](./models/face_detection_yunet/)

![largest selfie](./models/face_detection_yunet/example_outputs/largest_selfie.jpg)

### Face Recognition with [SFace](./models/face_recognition_sface/)

![sface demo](./models/face_recognition_sface/example_outputs/demo.jpg)

### Facial Expression Recognition with [Progressive Teacher](./models/facial_expression_recognition/)

![fer demo](./models/facial_expression_recognition/example_outputs/selfie.jpg)

### Human Segmentation with [PP-HumanSeg](./models/human_segmentation_pphumanseg/)

![messi](./models/human_segmentation_pphumanseg/example_outputs/messi.jpg)

### Image Segmentation with [EfficientSAM](./models/image_segmentation_efficientsam/)

![sam_present](./models/image_segmentation_efficientsam/example_outputs/sam_present.gif)

### License Plate Detection with [LPD_YuNet](./models/license_plate_detection_yunet/)

![license plate detection](./models/license_plate_detection_yunet/example_outputs/lpd_yunet_demo.gif)

### Object Detection with [NanoDet](./models/object_detection_nanodet/) & [YOLOX](./models/object_detection_yolox/)

![nanodet demo](./models/object_detection_nanodet/example_outputs/1_res.jpg)

![yolox demo](./models/object_detection_yolox/example_outputs/3_res.jpg)

### Object Tracking with [VitTrack](./models/object_tracking_vittrack/)

![webcam demo](./models/object_tracking_vittrack/example_outputs/vittrack_demo.gif)

### Palm Detection with [MP-PalmDet](./models/palm_detection_mediapipe/)

![palm det](./models/palm_detection_mediapipe/example_outputs/mppalmdet_demo.gif)

### Hand Pose Estimation with [MP-HandPose](models/handpose_estimation_mediapipe/)

![handpose estimation](models/handpose_estimation_mediapipe/example_outputs/mphandpose_demo.webp)

### Person Detection with [MP-PersonDet](./models/person_detection_mediapipe)

![person det](./models/person_detection_mediapipe/example_outputs/mppersondet_demo.webp)

### Pose Estimation with [MP-Pose](models/pose_estimation_mediapipe)

![pose_estimation](models/pose_estimation_mediapipe/example_outputs/mpposeest_demo.webp)

### QR Code Detection and Parsing with [WeChatQRCode](./models/qrcode_wechatqrcode/)

![qrcode](./models/qrcode_wechatqrcode/example_outputs/wechat_qrcode_demo.gif)

### Chinese Text detection [PPOCR-Det](./models/text_detection_ppocr/)

![mask](./models/text_detection_ppocr/example_outputs/mask.jpg)

### English Text detection [PPOCR-Det](./models/text_detection_ppocr/)

![gsoc](./models/text_detection_ppocr/example_outputs/gsoc.jpg)

### Text Detection with [CRNN](./models/text_recognition_crnn/)

![crnn_demo](./models/text_recognition_crnn/example_outputs/CRNNCTC.gif)

## License

OpenCV Zoo is licensed under the [Apache 2.0 license](./LICENSE). Please refer to licenses of different models.
