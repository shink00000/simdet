# simdet

This repository will reproduce and implement well-known Detection models.

# Policy

1. Simple implementation with less code and fewer files
1. Emphasis on processing efficiency
1. Be aware of ease of understanding

# Library Features

- [train](./tools/train.py)
- [test (evaluate)](./tools/test.py)

# Results

## [RetinaNet](https://arxiv.org/abs/1708.02002)

### ResNet50 [[arch](./docs/archs/retinanet_r50.txt)]

- [config](./configs/retinanet_r50_voc_h512_w512.yaml)
  - data: PascalVOC 2017 + 2012
  - input_size: (512, 512)
  - backbone: ResNet50
- [tensorboard](https://tensorboard.dev/experiment/Pb6lRSNcRWSa4LPb0K319w/)
- evaluation result
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.570
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.831
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.634
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.168
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.410
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.632
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.489
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.670
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.674
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.396
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.555
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.722
  ```

## [FCOS](https://arxiv.org/abs/1904.01355)

### ResNet50 [[arch](./docs/archs/fcos_r50.txt)]

- [config](./configs/fcos_r50_voc_h512_w512.yaml)
  - data: PascalVOC 2017 + 2012
  - input_size: (512, 512)
  - backbone: ResNet50
- [tensorboard](https://tensorboard.dev/experiment/LJqJ4SzOTM6NJ0syHzXHQQ/)
- evaluation result
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.535
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.797
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.575
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.210
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.360
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.600
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.476
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.655
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.660
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.418
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.535
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.711
  ```

## [EfficientDet](https://arxiv.org/abs/1911.09070)

### EfficientNet-B2 [[arch](./docs/archs/efficientdet_d2.txt)]

- [config](./configs/efficientdet_d2_voc_h512_w512.yaml)
  - data: PascalVOC 2017 + 2012
  - input_size: (512, 512)
  - backbone: EfficientNet-B2
- [tensorboard](https://tensorboard.dev/experiment/TQS8dga7Rka12SvdLBg9TQ/)
- evaluation result
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.583
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.837
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.645
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.226
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.395
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.653
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.497
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.683
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.686
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.423
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.557
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.740
  ```

## [DETR](https://arxiv.org/abs/2005.12872)

### DETR-R50 [[arch](./docs/archs/detr_r50.txt)]

- changes:
  - num_queries: 100 -> 40
  - class loss: CrossEntropy -> FocalLoss (therefore eof_coef is no longer used)
    - also, change the denominator of the average loss to the number of GTs.
- [config](./configs/detr_r50_voc_h512_w512.yaml)
  - data: PascalVOC 2017 + 2012
  - input_size: (512, 512)
  - backbone: ResNet50
- [tensorboard](https://tensorboard.dev/experiment/0ZEH5dlrQji1HwFDFoqLWA/)
- evaluation result
  ```
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.533
  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.797
  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.574
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.110
  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.292
  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.632
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.475
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.655
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.678
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.235
  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.475
  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.774
  ```
