# RT_hand_segment

## Usage

### Preparation
* Make sure you have isntalled Pytorch (at least ver. 1.4), and the TensorRT (at least ver. 7.0) libraries:
```bash
python -c "import torch, torch2trt; print('torch:', torch.__version__); print('tensorRT:',  torch2trt.trt_version()); print('CUDA available:', torch.cuda.is_available())"
```
* Clone this repository
```console
git clone https://github.com/GibranBenitez/RT_hand_segment
```
* Install required libraries
```bash
pip install py-cpuinfo
pip install ptflops
```
### Check full model speed
* run the `eval_fps_rhd.py` script
```bash
python eval_fps_rhd.py
```
* record the output, it should be like this:
```console
ARMv7 Processor rev 
Nano
============Starting===========
Model: DDRNet_finger_ipn.pkl, @480x640 on cuda:0
Flops:  5.55 GMac
Params: 5.73 M
=========Speed Testing=========
Elapsed Time: [32.87 s / 1000 iter]
Speed Time: 32.87 ms / iter   FPS: 30.42
```
### Evaluate tensorRT models
* Set the Jetson Nano to the highest mode:
```bash
sudo nvpmodel -m 0
```
* run the first DDRNet model with tensor RT (it should take sometime for model conversion)
```bash
python eval_fps_rhd_trt.py
```
* record the output, it should be like this:
```console
No existing model found. Converting and saving TRT model...

============Starting===========
Model: DDRNet_finger_ipn_trt_480x640_fp16.pth, @480x640
=========Speed Testing=========
Elapsed Time: [6.34 s / 1000 iter]
Speed Time: 6.34 ms / iter   FPS: 157.76
```
* run and record the rest of models
```bash
python eval_fps_rhd_trt.py --model FASSDNet
python eval_fps_rhd_trt.py --model HardNet
python eval_fps_rhd_trt.py --model DABNet
python eval_fps_rhd_trt.py --model FastSCNN
```
* send the recorded outputs of five models with tensorRT (including DDRNet)
