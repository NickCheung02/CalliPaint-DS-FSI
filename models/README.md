---
frameworks:
- Pytorch
license: Apache License 2.0
tasks:
- text-to-image-synthesis

#model-type:
##如 gpt、phi、llama、chatglm、baichuan 等
#- gpt

#domain:
##如 nlp、cv、audio、multi-modal
#- nlp

#language:
##语言代码列表 https://help.aliyun.com/document_detail/215387.html?spm=a2c4g.11186623.0.0.9f8d7467kni6Aa
#- cn

#metrics:
##如 CIDEr、Blue、ROUGE 等
#- CIDEr

#tags:
##各种自定义，包括 pretrained、fine-tuned、instruction-tuned、RL-tuned 等训练方法和其他
#- pretrained

#tools:
##如 vllm、fastchat、llamacpp、AdaSeq 等
#- vllm
datasets:
  - iic/AnyWord-3M
  - iic/AnyText-benchmark
tags:
  - 文字生成与编辑
---
### AnyText2: Visual Text Generation and Editing With Customizable Attributes

AnyText2是一种基于扩散模型的视觉文字生成与编辑模型。通过所提出的WriteNet+AttnX架构，以及重新设计的文本嵌入模块，我们的新方法相比原AnyText不仅进一步提升了图像真实感，也使中英文的文本准确率提高了3.3%和9.3%。同时推理速度也提升了19.8%。不仅如此，作为AnyText的扩展，AnyText2可以精确控制图像中文本的字体和颜色等属性，为智能生成设计类应用如商标LOGO、电商海报、商品效果图等提供了更加灵活的解决方案。  
详见：  
Paper: https://arxiv.org/abs/2411.15245  
Code: https://github.com/tyxsspa/AnyText2  
Demo: https://modelscope.cn/studios/iic/studio_anytext2  


#### 您可以通过如下git clone命令，或者ModelScope SDK来下载模型
 
SDK下载
```bash
#安装ModelScope
pip install modelscope
```
```python
#SDK模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('iic/cv_anytext2')
```
Git下载
```
#Git模型下载
git clone https://www.modelscope.cn/iic/cv_anytext2.git
```