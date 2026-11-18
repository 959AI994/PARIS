# EDA安装（Linux / CPU）

## 1. 系统库和Python

Ubuntu 22.04/24.04：

物理流程和网页模型迭代要求至少60GiB空闲磁盘。

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git flex bison \
  libboost-all-dev libcairo2-dev zlib1g-dev python3-dev python3-venv \
  libffi-dev pkg-config tcl-dev tk-dev libreadline-dev libeigen3-dev \
  libspdlog-dev libyaml-cpp-dev libfmt-dev libfl-dev gawk swig time
python -m pip install -r requirements.txt -e .
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-eda.txt
python scripts/build_place.py --jobs 2
```

使用当前Python的Torch ABI编译，生成 `vendor/Place-MoL/DREAMPlace/install`。CPU流程不需要GPU。构建和运行使用同一个Python环境。

## 2. OpenROAD与Yosys

可直接使用已有工具路径。验证版本与构建补丁在 `vendor/SOURCES.json` 和 `vendor/patches/`。

在PARIS根目录开始，从源码构建OpenROAD：

```bash
PARIS_REPO_ROOT="$(pwd)"
git clone --recursive https://github.com/The-OpenROAD-Project/OpenROAD.git /opt/OpenROAD
cd /opt/OpenROAD
git checkout c8e73fe7e222fb93a80b7ecb7ea7cc83df2459ec
git submodule update --init --recursive
git -C src/sta checkout d1b2b3962cd47539c748bb163da5de2bd5d2f85f
git -C third-party/slang-elab checkout ce38835520fbdf422304a6f6a2f1c437c3ba98c2
git apply "$PARIS_REPO_ROOT/vendor/patches/OpenROAD-build.patch"
# 安装该版本的构建依赖
sudo ./etc/DependencyInstaller.sh -all
./etc/Build.sh -cmake-build -no-gui -threads=2
```

随包补丁仅修改CMake头文件/库目标配置，不改时序算法。验证环境的OpenSTA为 `d1b2b3962cd47539c748bb163da5de2bd5d2f85f`。

Yosys使用 `1d2d777678f9f0856cb9d8a51d6d6447d4026c2f`，ABC为 `8827bafb7f288de6749dc6e30fa452f2040949c0`；也可配置兼容的已安装版本。

```bash
git clone --recursive https://github.com/YosysHQ/yosys.git /opt/yosys
cd /opt/yosys
git checkout 1d2d777678f9f0856cb9d8a51d6d6447d4026c2f
git submodule update --init --recursive
git -C abc checkout 8827bafb7f288de6749dc6e30fa452f2040949c0
make config-gcc
make -j2
```

## 3. 生成当前机器配置

在PARIS仓库根目录执行：

```bash
python scripts/configure.py --openroad /opt/OpenROAD/build/bin/openroad \
  --yosys /opt/yosys/yosys
python scripts/start_web.py --profile native-profile.json
```

`native.json`只保存本机工具位置；`native-profile.json`绑定该版本的流程与资源hash。移动安装目录后重新生成配置。也可复制 `configs/native.example.json` 手动配置，里面的相对路径以配置文件目录为基准。

原始网表流程为：综合/预处理 → 上下层候选分割 → Place-MoL布局 → CTS → GR → 特征。标定时再对预算内候选运行DR、OpenRCX和STA，获得实测标签。`--route-iterations`控制DR迭代预算；默认一轮，可能保留DRC，完整签核应使用所需的统一布线预算。

远程服务器网页：

```bash
ssh -L 8765:127.0.0.1:8765 user@server
```

然后访问本地 `http://127.0.0.1:8765`。
