## `CUDA` 和 `cuDNN` 的安装

`CUDA` 和 `cuDNN` 是令 `nVIDIA` 显卡能够加速深度学习过程的必要条件. 

`CUDA (Compute Unified Device Architecture)` 是 `nVIDIA` 推出的 **通用并行计算架构**. 它包含硬件和软件两个部分: `CUDA` 核心位于显卡核心中, `CUDA` 软件由 **`CUDA` 库**, **`API`** 和 **运行时 (`Runtime`)** 组成, 我们所需要安装的就是 `CUDA` 软件和 **针对深度卷积神经网络** 的加速库 `cuDNN`.

下面简述 `CUDA` 和 `cuDNN` 的安装.

### 安装 `CUDA`

#### 下载安装

完成安装 `Ubuntu` 后, 赴 [此处](https://developer.nvidia.com/cuda-toolkit-archive) 点选最新版本, 并依次选择对应的硬件平台和系统平台, 下载本地安装包 `runfile(local)`.

随后根据官网提供的安装命令进行安装:

~~~bash
# locate to Download/ folder
cd Downloads/

# download installation file
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run

# make this file executable
sudo chmod 777 cuda_11.7.1_515.65.01_linux.run

# execute this installation file
sudo sh cuda_11.7.1_515.65.01_linux.run
~~~

随后选择接受用户协议, 在进入安装模块选择界面后 **取消选择所有和 `Driver` 相关的部分**, 但选中 **其他所有部分**. 

#### 配置环境变量

在完成安装后, 使用 `vim` 编辑 `~/.bashrc`, 它是 `ubuntu` 的默认终端 `bash` 的配置文件:

~~~bash
sudo vim ~/.bashrc
~~~

并将下列文件写入 `~/.bashrc` 尾部:

~~~bash
export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
~~~

然后重载 `bash`:

~~~bash
source ~/.bashrc
~~~

随后在 `/etc/profile` 文件中添加 `CUDA` 环境变量:

~~~bash
sudo vim /etc/profile
~~~

打开文档后添加:

~~~bash
PATH=/usr/local/cuda/bin:$PATH  
export PATH
~~~

然后重载环境变量, 让修改生效:

~~~bash
source /etc/profile
~~~

最后添加 `lib` 库路径:

~~~bash
sudo vim /etc/ld.so.conf.d/cuda.conf
~~~

在文件中加入下面的内容:

~~~bash
/usr/local/cuda/lib64
~~~

然后执行指令使添加的 `lib` 库生效:

~~~bash
sudo ldconfig
~~~

最后可利用 `CUDA` 软件包内置的测试例程检测安装是否成功:

~~~bash
cd /usr/local/cuda-11.1/samples/1_Utilities/deviceQuery
sudo make
sudo ./deviceQuery
~~~

### 安装 `cuDNN`

首先需要访问 [cuDNN官网](https://developer.nvidia.com/cudnn), 注册/登录 `nVIDIA` 开发者账号获取下载权限.

随后选择和所安装的 `CUDA` 版本对应的 `cuDNN` 版本下载.

下载完成安装包后, 需将其 **解压** 后, 将下列的文件移动到 `CUDA` 安装目录下的对应位置内:

~~~bash
sudo chmod 777 -R ./cuda  
sudo mv cuda/include/* /usr/local/cuda/include
sudo mv cuda/lib64/* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
~~~

## `Conda` 的安装与使用

`Conda` 是用于 **`Python` 软件包管理和环境管理** 的工具, 适用于在 **无法改变运行环境** 的设备上创建/删除 **虚拟环境**, 如实验室分配的服务器或租赁的 `VPS`. 下面简述 `Linux` 平台上 `Conda` 的下载安装, 配置使用方法.

### 下载和安装

我们可以在 [此处](https://www.anaconda.com/products/distribution) 下载 `Conda` 本体, 在 `Linux` 平台上可以通过直接执行所下载的 `.sh` 文件的方式安装 `Conda`, 所有选项选择 `Yes` 即可.

关于 `Linux` 平台上 `Conda` 安装的详细步骤[参见此处](https://docs.anaconda.com/anaconda/install/linux/).

### 环境配置和使用

在成功安装后, 需要 **重载终端配置文件**, 一般通过执行 `~/.bashrc` (适用于 `bash`) 或 `~/.zshrc` 完成.

重载终端后, 我们可以通过执行

~~~bash
conda env list
~~~

查看当前所创建的环境列表. 根据实际需求, 我们可以选择 **创建新的虚拟环境**, 或 **下载他人配置好的虚拟环境**:

~~~bash
# create a new environment
conda create -n <enviroment name>

# create a new enviroment with Python version specified
conda create -n <enviromnent name> python=<python version>

# load existing configuation from web
conda env create <user-name>/<name-of-environment>
~~~

随后, 我们需要 **切换到所创建的虚拟环境中**:

~~~bash
conda activate <environment name>
~~~

下面讨论 **从零开始创建新的虚拟环境** 的情形. `Conda` 默认创建的虚拟环境中 **不包含任何软件包**, 因此我们往往首先需要安装一系列最基本的功能包. 如:

~~~bash
# install pip
conda install pip

# install tensorflow
conda install tensorflow-gpu

# install scikit-learn
conda install scikit-learn

# install opencv
conda install opencv
~~~

然后我们就可以在虚拟环境中执行任何所需要的操作.

最后, 在虚拟环境完成其历史使命后, 我们可以选择退出或删除它:

~~~bash
# exit current virtual environment
source deactivate

# remove virtual environment
conda remove -n <environment name> -all
~~~

## `Jupyter Notebook` 的安装和使用:

`Jupyter Notebook` 是一种方便好用的, 轻量级的 `Python` 开发环境. 

我们可以在创建的虚拟环境中, 使用

~~~bash
pip install notebook
~~~

安装它, 并执行

~~~bash
jupyter notebook
~~~

启动它.

`Jupyter Notebook` 在启动后将作为一个本地服务器运行, 我们可通过浏览器与其交互.


## `Yolo` 类模型的配置和使用

### 准备自定义数据集

#### 数据集格式介绍

`Yolo` 数据集由 **图像数据**, **标签数据** 和 **`MetaData`** 组成, 适用于 `CVAT` 的 `Yolo V1.1` 标准和用于训练 `YoloV5`, `YoloV7` 的数据集差异仅在 `MetaData` 上体现.

`Yolo` 类模型中默认的图像输入格式为 **`.jpg`**, 分辨率 $640 * 640$. 每一张图片都可能包含 **$0$ 个或多个 `bounding box` (下面简称 `bbox`)**. 

`Yolo` 类模型中使用的标注格式为 `.txt`, 每一个 `bbox` 都对应文件中的一行, 所存储的数据如下:

~~~yaml
<class_number> <center_x> <center_y> <bbox_length> <bbox_height>
~~~

并且为了确保 **标注数据和图像分辨率无关**, 标注中的后四个数据都是 $0-1$ 之间的小数, 所表述的实际像素数是通过关于图像实际长/宽乘上这些百分比系数得到的.

#### 使用 `CVAT` 收集和标注自定义数据集

`CVAT` 是 **开源**, **免费** 的机器视觉数据集标注软件, 支持包括图像语义分割, `bounding box` 绘制等一系列标注任务, 并可接受多种格式数据集的导入和导出.

下面讨论如何在本机上下载并使用 `Docker` 容器部署 `CVAT`.

首先在 [`Docker` 官方网站](https://www.docker.com/products/docker-desktop/) 处下载并安装 `Docker Desktop`.

在安装完成后, 先启动 `Docker Desktop`, 然后打开终端并定位到合适的位置, 执行 

~~~bash
git clone https://github.com/opencv/cvat.git
~~~

下载 `CVAT`.

`CVAT` 本质上是一个 `Web App`, 其后端框架为 `Django`, 所有的用户交互通过前端进行. 为了便于部署并隔离本机环境, 我们使用 `Docker` 运行它. 在终端中继续执行:

~~~bash
# 定位到Clone下的CVAT Repository内
cd cvat

# 检查所安装的Docker是否正常运行
docker version

# 构建 `CVAT` 的 Docker 容器
sudo docker-compose build
docker-compose up -d

# 在命令行内添加用户, 此处必须设置密码, 否则无法登陆!
docker exec -it cvat bash -ic 'python3 ~/manage.py createsuperuser'
~~~

最后, 我们可以在本机浏览器内通过访问

~~~bash
http://localhost:8080/
~~~

使用 `CVAT`.

### 算法部署和模型训练

#### 部署并训练 `Yolo V5`

#### 部署并训练 `Yolo V7`

#### 部署中可能遇到的常见问题

#### 训练参数解读

#### 继续训练

#### 应用内置图像增强

#### 应用 `Albumentation` 图像增强

##### 在 `Yolo V5` 上启用 `Albumentation`

##### 在 `Yolo V7` 上启用 `Albumentation`

### 使用 `Tensorboard` 追踪和解读训练结果

