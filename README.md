参照 https://github.com/X-T-E-R/GPT-SoVITS-Inference.git 更改而来
# 1. 安装环境
  - conda运行，python 3.9
  ``` sh
  conda activate gptsovits
  ``` 

- (可选) 如果有conda，安装 torch torchvision torchaudio
  
  ``` sh
  # 查看 cuda 版本
  nvcc --version
  # 安装对应的 torch torchvision torchaudio
  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
  ```
