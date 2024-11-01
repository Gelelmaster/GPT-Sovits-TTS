- 参照 https://github.com/X-T-E-R/GPT-SoVITS-Inference.git 更改而来
# 1. 安装环境
  - conda运行，python 3.9
  ``` sh
  conda create -n gptsovits python=3.9
  conda activate gptsovits
  ``` 

  - (可选) 如果有conda，安装 torch torchvision torchaudio  
    测试环境：  
    Torch version: 2.4.1+cu121  
    Torchvision version: 0.19.1+cu121  
    Torchaudio version: 2.4.1+cu121  
    不知道什么原因，只有在 torch和torchaudio是2.4.1版本时推理速度才正常，更新到2.5.0版本时推理很慢。所以这里建议装固定版本。  
  
  - 查看 cuda 版本
  ``` sh
  nvcc --version
  ```
  - 安装对应的 torch torchvision torchaudio  
  ``` sh
  pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
  ```

# 2. 复制程序到本地
  - git 克隆（需要安装有git）
  ``` sh
  git clone https://github.com/X-T-E-R/GPT-SoVITS-Inference.git
  ```

  - 安装依赖
  ``` sh
  pip install -r requirements.txt
  ```

# 3. 运行程序
  - 直接双击bat文件运行打开网页，在网页界面创建模型，再运行程序
  - 或者在本地新建一个存放模型的文件夹 trained，将下载到的模型放到该目录下，运行程序
