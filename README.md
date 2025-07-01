# Deep Research

Deep Research是一个学习项目，完全参考字节的 [deer-flow](https://github.com/bytedance/deer-flow)，感谢 DeerFlow 的杰出工作。

## 快速开始

Deep Research 使用 Python 开发，只涉及服务端的实现。请使用以下工具：

### 推荐工具

- **[`uv`](https://docs.astral.sh/uv/getting-started/installation/):**
  简化 Python 环境和依赖管理。`uv`会自动在根目录创建虚拟环境并为您安装所有必需的包—无需手动安装 Python 环境。

### 环境要求

确保您的系统满足以下最低要求：

- **[Python](https://www.python.org/downloads/):** 版本 `3.12+`

### 安装与运行

```bash
# 克隆仓库
git clone https://github.com/qypeng-ustc/deep-research.git
cd deep-research

# 安装依赖，uv将负责Python解释器和虚拟环境的创建，并安装所需的包
uv sync

# 通过.env设置参数与API密钥
cp .env.example .env

# 执行代码
uv run main.py
```
