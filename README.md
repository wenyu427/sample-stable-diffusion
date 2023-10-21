# sample-stable-diffusion
This is a stripped-down stable diffusion code that mainly uses pre-trained models to do txt2img and img2img generation.
本仓库的代码是ComVis发布的Stable Diffusion v1代码的精简版，旨在利用其预训练\
好的模型来做txt2img和img2img任务.
## 结构：
autoencoder 将图片压缩为潜在向量：(bs, 4, 64, 64)\
DenoiseModel 是Unet网络，用来预测t时刻的噪声 \
con_encoder 包含各种条件编码器，比如clip-text encoder
diffusion
## model weight
 可以在这里找到:
https://huggingface.co/CompVis
