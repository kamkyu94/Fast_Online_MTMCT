## Model Zoo
Under "./preliminary/det_weights/" place COCO trained weights from https://github.com/WongKinYiu/yolov7
 - yolov7-w6.pt: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt
 - yolov7-e6.pt: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt
 - yolov7-d6.pt: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt
 - yolov7-e6e.pt: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt

<br />

Under "./preliminary/feat_ext_weights/" place AIC19 + VeRi-776 trained weights
 - resnet50_ibn_a_gap_120.t7: https://drive.google.com/file/d/1ZQspaimt2WfyXAeX6C1tSgAPtcBDfv0w/view?usp=sharing
 - resnet50_ibn_a_gem_120.t7: https://drive.google.com/file/d/1A2ib3FNSFoaFdvbcSWay6JYD5AOHn8w0/view?usp=sharing
 - resnet101_ibn_a_gap_120.t7: https://drive.google.com/file/d/1ZQ2SCrJEszhWsfUCmV8Jh1lv2apZctUG/view?usp=sharing
 - resnet101_ibn_a_gem_120.t7: https://drive.google.com/file/d/1iQe4n0SiiPwF8z7HXyMpPuqH7-3aaoeO/view?usp=sharing

## Run
1. Ajdust "opts.py"
2. Run "run_mtmc.py"
3. Results will be saved in "./outputs"
