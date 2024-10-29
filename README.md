
# 简介
本项目是基于VLLM的LLM模型推理服务，用于演示在本地开发测试及在TOPAI平台进行生成环境部署。

# 本地使用
1. 安装依赖:
```bash
make install
```


2. 启动服务:
```bash
make run
```

# 在TOPAI平台部署
* topai.yaml是TOPAI平台部署配置文件，包含模型、推理参数、推理服务配置等。
```yaml
topai:
  name: llama-3-8b-server  # 模型服务名称
  project_path: Llama_3_8B_Server  # 项目路径，用于定位代码和配置文件。相对与项目根目录
  model:
    type: llm  # 模型类型：大语言模型(Large Language Model)，同时支持Text-to-Image/ASR
    source:
      local:
        path: /home/models/  # 运行容器内模型文件的本地存储路径，用于加载模型权重文件
  serving:
    import_path: src.main:vllmservice  # 服务入口点配置，指定使用Ray Serve框架加载的Python模块和对象
  inference:
    engine:
      vllm:  # vLLM推理引擎的具体配置
        model: NousResearch/Meta-Llama-3-8B  # 模型名称或路径，指定要加载的具体模型
        serve_model_name: Meta-Llama-3-8B  # 服务中使用的模型名称标识
        max_model_len: 8192  # 模型支持的最大序列长度，用于限制输入文本的长度

```

# 项目结构

```bash

├── src/
│   ├── __init__.py
│   └── main.py          # 主服务实现
├── deployment.yaml      # 部署配置,本地测试使用,可以不包含
├── requirements.txt     # 依赖包，如果三方python包，则需要在此处添加
├── Makefile            # 构建脚本,本地测试使用,可以不包含
├── LICENSE             # MIT许可证
└── topai.yaml          # TOPAI配置，必须包含
```
