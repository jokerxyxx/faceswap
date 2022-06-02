# faceswap
## preparation
### install environment
```
conda create -n simswap python=3.6
conda activate simswap
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
(option): pip install --ignore-installed imageio
pip install insightface==0.2.1 onnxruntime moviepy
(option): pip install onnxruntime-gpu  (If you want to reduce the inference time)(It will be diffcult to install onnxruntime-gpu , the specify version of onnxruntime-gpu may depends on your machine and cuda version.)
```

### install dataset
cropped VGGFace-224 dataset in [Google Drive](https://drive.google.com/file/d/19pWvdEHS-CEG6tW3PdxdtZ5QEymVjImc/view?usp=sharing)

### pretrained model
arcface  
download from [here](https://pan.baidu.com/s/1u4INoLrseV8lvQNQLl_ocg)password:ebg2
down it and put it to ./arcfece_model 
