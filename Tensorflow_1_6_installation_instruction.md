# Instruction of Tensorflow 1.6 installation on macOS 10.13.3 with eGPU (CUDA and cudnn support)

How to compile and install Tensorflow 1.6 on macOS High Sierra 10.13.3 with eGPU from scratch. Start from High Sierra 10.13.3, Apple officially support external NVDIA GPU. However, 'since version 1.2, Google dropped GPU support on macOS from TensorFlow'. 

## Setup local environment

### Prerequisites
Here are things you need to install in your local environment

* brew
* Python 3.6 and related dependencies
* Apple Command-Line-Tools 8.3.2
* bazel 0.8.1

### Installing

#### Brew
Open a terminal and execute following

```
$ xcode-select --install
$ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

#### Python 3.6
Install Python 3 with brew:

```
$ brew install python
```

Then we can install all dependencies
```
$ pip install six numpy wheel
$ brew install coreutils
```

However, I hightly recommend to use Conda to manage packages and modules. You can install different conda versions based on your needs from this website (https://conda.io/docs/user-guide/install/download.html)

#### Apple Command-Line-Tools 8.3.2

> ** The default version of the Apple Command-Line-Tools with macOS 10.13.3 is 9.2. We need to downgrade to 8.3.2 to successfully compile tensorflow.

First, we need to backup the current version

```
$ sudo mv /Library/Developer/CommandLineTools /Library/Developer/CommandLineTools_backup
```

The old version 8.3.2 can be downloaded directly from Apple Developer Portal. Follow the instruction to install it. 

```
$ sudo xcode-select --switch /Library/Developer/CommandLineTools
```

#### Bazel
The current version of Bazel is 0.11.1 (up to March 14th, 2018). Do not install this version with homebrew, because it will not work correctly (at least for me). Instead, we will install it directly from source.

```
$ curl -L https://github.com/bazelbuild/bazel/releases/download/0.8.1/bazel-0.8.1-installer-darwin-x86_64.sh -o bazel-0.8.1-installer-darwin-x86_64.sh
$ chmod +x bazel-0.8.1-installer-darwin-x86_64.sh
$ ./bazel-0.8.1-installer-darwin-x86_64.sh
```

## Configure eGPU

### Prerequisites
Here are things you need to install to have eGPU recognized.

* NVIDIA Web-Drivers
* CUDA-Drivers
* CUDA 9.1 Toolkit
* cuDNN 7

Before installing, first check your macOS build version is

**macOS High Sierra Version 10.13.3 (17D102)**

and disable System Integrity Protection as follows

* restart your mac and boot into Recovery mode by holding cmd + r
* open a terminal and execute 'csrutil disable'
* restart

> ** Do not restart your mac with eGPU connected. Sometimes you will not be able to boot with eGPU connected due to current incompatibility issue. For now, just boot your machine with GPU disconnected.

### Installing

#### NVIDIA Web-Drivers and eGPU support
Download and follow the instruction to install web-driver from NVIDIA

[https://images.nvidia.com/mac/pkg/387/WebDriver-387.10.10.10.25.161.pkg](https://images.nvidia.com/mac/pkg/387/WebDriver-387.10.10.10.25.161.pkg)

#### CUDA Toolkit 9.1 with CUDA Driver
Download CUDA-9.1 [here](https://developer.nvidia.com/cuda-downloads?target_os=MacOSX&target_arch=x86_64&target_version=1013&target_type=dmglocal)

and install it following its instruction.

Then add following path to your default terminal. I use fish so in my case, it looks like

```
$ sublime ~/.config/fish/config.fish
    set PATH /Developer/NVIDIA/CUDA-9.1/bin $PATH
    set -x CUDA_HOME /usr/local/cuda
    set -x DYLD_LIBRARY_PATH /usr/local/cuda/lib /usr/local/cuda/extras/CUPTI/lib
    set -x DYLD_LIBRARY_PATH /Users/chiqun/lib:/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
    set -x LD_LIBRARY_PATH $DYLD_LIBRARY_PATH
```

Remember to start a new terminal after adding these lines. We can check if the driver is loaded

```
$ kextstat | grep -i cuda
  173    0 0xffffff7f80f08000 0x2000     0x2000     com.nvidia.CUDA (1.1.0) 4329B052-6C8A-3900-8E83-744487AEDEF1 <4 1>
```

**Now it is time to connect your eGPU to mac, and re-login your account**

Let us check if we can compile some examples with GPU supported.

```
$ cd /Developer/NVIDIA/CUDA-9.1/samples
$ make -C 1_Utilities/deviceQuery
$ ./1_Utilities/deviceQuery/deviceQuery

CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "Quadro P5000"
  CUDA Driver Version / Runtime Version          9.1 / 9.1
  CUDA Capability Major/Minor version number:    6.1
  Total amount of global memory:                 16384 MBytes (17179672576 bytes)
  (20) Multiprocessors, (128) CUDA Cores/MP:     2560 CUDA Cores
  GPU Max Clock rate:                            1734 MHz (1.73 GHz)
  Memory Clock rate:                             4513 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 2097152 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 196 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 9.1, CUDA Runtime Version = 9.1, NumDevs = 1
Result = PASS
```

#### NVIDIA cuDNN

Download cuDNN 7.0.5 [here](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.1_20171129/cudnn-9.1-osx-x64-v7-ga)

> ** cuDNN 7.1 does not support macOS yet

Go to your download directory and execute

```
$ tar -xzvf cudnn-9.1-osx-x64-v7-ga.tgz
$ sudo cp cuda/include/cudnn.h /usr/local/cuda/include
$ sudo cp cuda/lib/libcudnn* /usr/local/cuda/lib
$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib/libcudnn*
```

## Configure and install Tensorflow

### Clone Tensorflow from Repo

```
$ git clone https://github.com/tensorflow/tensorflow
$ cd tensorflow
$ git checkout v1.6.0-rc1
```

### Apply Patch

We need to touch some files in the repo to get it build successfully. I find that the patch from orpcam works for me. You can grab it from his [Github](https://gist.github.com/orpcam/73b208271856fa2ae7efc00a8768bd7c#file-tensorflow_v1-6-0-rc1_osx-patch).

`$ git apply tensorflow_v1.6.0-rc1_osx.patch`

### Configure

If you have tried to build and failed before, remeber to clean up and restart bazel
```
$ bazel clean
$ bazel shutdown
```

Have your Python3 path prepared. You can copy it to copyboard by executing

`$ which python3 | pbcopy`

Now, it is time to configure.

`$ ./configure`

Here is my configuration.

```
You have bazel 0.8.1 installed.
Please specify the location of python. [Default is /Users/chiqun/miniconda/bin/python]: /Users/chiqun/miniconda/bin/python3


Found possible Python library paths:
  /Users/chiqun/miniconda/lib/python3.6/site-packages
Please input the desired Python library path to use. Default is [/Users/chiqun/miniconda/lib/python3.6/site-packages]

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
No Amazon S3 File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Apache Kafka Platform support? [y/N]: n
No Apache Kafka Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: n
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: n
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
No OpenCL SYCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 9.0]: 9.1


Please specify the location where CUDA 9.1 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 


Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 


Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:


Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 3.5,5.2]6.1


Do you want to use clang as CUDA compiler? [y/N]: n
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 


Do you wish to build TensorFlow with MPI support? [y/N]: 
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
    --config=mkl            # Build with MKL support.
    --config=monolithic     # Config for mostly static monolithic build.
Configuration finished
```

### Build

`$ bazel build --config=cuda --config=opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH //tensorflow/tools/pip_package:build_pip_package`

### Create wheel and install

```
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
$ sudo pip install /tmp/tensorflow_pkg/tensorflow-1.6.0rc0-cp36-cp36m-macosx_10_13_x86_64.whl
```

## Test installation
In python, try

```
>>> import tensorflow as tf
>>> tf.__version__
'1.6.0-rc0'

>>> tf.Session()
2018-03-14 13:07:10.592077: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:859] OS X does not support NUMA - returning NUMA node zero
2018-03-14 13:07:10.592248: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1331] Found device 0 with properties:
name: Quadro P5000 major: 6 minor: 1 memoryClockRate(GHz): 1.7335
pciBusID: 0000:c4:00.0
totalMemory: 16.00GiB freeMemory: 15.84GiB
2018-03-14 13:07:10.592273: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1410] Adding visible gpu devices: 0
2018-03-14 13:07:11.053021: I tensorflow/core/common_runtime/gpu/gpu_device.cc:911] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-03-14 13:07:11.053143: I tensorflow/core/common_runtime/gpu/gpu_device.cc:917]      0
2018-03-14 13:07:11.053161: I tensorflow/core/common_runtime/gpu/gpu_device.cc:930] 0:   N
2018-03-14 13:07:11.053257: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1021] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 15358 MB memory) -> physical GPU (device: 0, name: Quadro P5000, pci bus id: 0000:c4:00.0, compute capability: 6.1)
2018-03-14 13:07:11.053987: E tensorflow/stream_executor/cuda/cuda_driver.cc:936] failed to allocate 15.00G (16104454912 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY
2018-03-14 13:07:11.054173: E tensorflow/stream_executor/cuda/cuda_driver.cc:936] failed to allocate 13.50G (14494009344 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY
<tensorflow.python.client.session.Session object at 0x103cdba90>
```

I get CUDA_ERROR_OUT_OF_MEMORY , but the code will still keep running. Just ignore it.

## Authors

* **Chiqun Zhang** - *Initial work* - [chiqunz](https://github.com/chiqunz)

## Acknowledgments

* This work is based on https://byai.io/howto-tensorflow-1-6-on-mac-with-gpu-acceleration/ and https://gist.github.com/orpcam/73b208271856fa2ae7efc00a8768bd7c
* Thanks go to Jacques Kvam for helping setup Path variables.  

