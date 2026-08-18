# XiliLake

**多模态大模型可信评测系统-西丽湖 v0.1**

XiliLake 是 MiLab 建设的大语言模型与多模态模型可信评测系统。系统统一管理通用、医疗和网络安全领域的 Benchmark，支持逐级选择评测对象、启动本地或 OpenAI 兼容接口模型、实时查看任务进度，并展示指标与完整样例结果。

## 当前规模

- 3 个评测领域
- 11 个大类
- 82 个子类
- 102 个 Benchmark（XSTest 的大小写及镜像重复项已合并）
- 完整数据部署下，82 个子类和 102 个 Benchmark 全部可评测

数量由 Benchmark 配置动态生成，网页与 `/api/trust_catalog` 会随配置变更自动更新。完整归属关系见 [docs/current_taxonomy_full.md](docs/current_taxonomy_full.md)。

分类或 Benchmark 配置更新后，可执行 `python3 docs/export_current_taxonomy.py` 重新导出完整分类清单。

| 评测领域 | 评测大类 |
| --- | --- |
| 通用评测 | 基本事实准确性、推理决策可靠性、任务遵循可靠性、攻击抵御鲁棒性、系统策略安全性、社会群体公平性、法律法规遵守性、伦理道德符合性 |
| 医疗行业评测 | 医疗安全可靠性、推理决策可靠性（5 个端到端审计子类、5 个辅助推理子类） |
| 网络安全行业评测 | 代码安全可靠性 |

## 快速开始

运行环境要求 Python 3.10 及以上版本。数据集和模型权重不随源码仓库分发，应根据各自许可证单独获取。

```bash
git clone https://github.com/MiLab-HITSZ/XiliLake.git
cd XiliLake
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-web.txt
```

检查并下载 catalog 声明的数据源：

```bash
python download_datasets.py --list
python download_datasets.py --skip-existing
python prepare_missing_benchmarks.py
python generate_downloaded_benchmark_configs.py
```

部分大型或有单独使用条款的数据集需要按 Benchmark 原始页面的说明手动授权或下载。下载完成后，网页中的可评测数量会根据配置、数据路径和执行适配器自动更新。

## 部署与访问

首次启动或代码更新后执行：

```bash
./deploy_backend.sh
```

服务默认监听 `0.0.0.0:5001`，因此本机和内网客户端都可访问：

```text
本机: http://127.0.0.1:5001/
内网: http://<服务器内网 IP>:5001/
分类编辑: http://<服务器内网 IP>:5001/taxonomy-editor
```

访客页面 `/` 保持只读。分类编辑页面 `/taxonomy-editor` 可修改大类、子类名称及其说明；保存内容写入 `data/taxonomy_overrides.json`，无需重启服务，已打开的访客页面会自动刷新分类文本。分类 ID、Benchmark 归属和历史评测结果标识不会随显示名称修改。

自定义端口：

```bash
XILILAKE_WEB_PORT=5100 ./deploy_backend.sh
```

停止服务：

```bash
./stop_backend.sh
```

查看运行日志：

```bash
tail -f runtime/web_backend.log
```

## 目录结构

```text
XiliLake/
├── web_backend.py                 # Flask 服务、评测任务和结果 API
├── templates/eval_viz.html       # 单页 Web 界面
├── benchmarks/                   # Benchmark 注册、适配器与分类配置
├── evaluate_cdh_bench.py         # CDH 多模态评测器
├── evaluate_generic_benchmark.py # 通用文本 Benchmark 评测器
├── evaluate_ehrperturb.py        # EHRPerturb 医疗评测器
├── config/                       # 不含密钥的 API 模型配置示例
├── data/                         # 分类和元数据源
│   └── taxonomy_overrides.json   # 分类编辑页面保存的文本覆盖（首次保存后生成）
├── downloads/                    # 下载的原始数据集与仓库
├── images/                       # CDH 图像数据
├── models/                       # 本地模型
├── result/                       # 评测输出
├── runtime/                      # PID、日志和任务进度
├── docs/                         # 当前完整分类清单
├── deploy_backend.sh             # 安装 Web 依赖并重启服务
└── stop_backend.sh               # 停止服务
```

## 模型配置

### 本地模型

将可由 vLLM 加载的模型放入 `models/<model-name>/`。系统会自动扫描子目录，并在网页的模型选择器中展示。

### API 模型

复制 `config/api_models.example.json` 为 `config/api_models.json`，填写模型名、接口地址和保存密钥的环境变量名。配置文件只引用环境变量，不保存密钥明文：

```bash
cp config/api_models.example.json config/api_models.json
export XILILAKE_EXAMPLE_API_KEY='<your-key>'
./deploy_backend.sh
```

`config/api_models.json`、`.env*` 和常见凭据文件已加入 `.gitignore`。请勿把 API key 写入源码、Benchmark 配置或提交历史。

## Benchmark 与数据

- `benchmarks/*/benchmark.json` 是系统运行时的唯一 Benchmark 配置源。
- `benchmarks/registry.py` 加载配置，`benchmarks/adapters.py` 解析数据路径并生成评测命令。
- `downloads/datasets/download_manifest.json` 记录批量下载状态，可由下载脚本重新生成。
- 网页可评测数量以配置、数据路径和执行适配器的实时检查结果为准。

查看配置中声明的可下载资源：

```bash
python3 download_benchmarks.py --list
```

根据当前 catalog 批量检查或下载数据集：

```bash
python3 download_datasets.py --list
python3 download_datasets.py --skip-existing
```

## 运行检查

页面右上角的“全子类快速检测”会用当前模型对 82 个可评测子类各运行 1 个真实样例。任务共享一个已拉起的模型服务，页面会显示总进度，并将结果汇总到 `result/current/`。该操作会替换当前结果，适合部署后验收 Benchmark 数据、评测适配器、进度回传和结果展示链路。

评测样例会分区展示题目、完整材料或上下文、图片输入状态、模型回答和参考答案；多模态 CDH 样例同时展示反常识与常识成对图片。

```bash
python3 -m py_compile \
  web_backend.py \
  benchmarks/registry.py \
  benchmarks/adapters.py \
  evaluate_cdh_bench.py \
  evaluate_generic_benchmark.py \
  evaluate_ehrperturb.py

curl -fsS http://127.0.0.1:5001/api/trust_catalog
curl -fsS http://127.0.0.1:5001/api/current_result
```

页面与 API 由同一 Flask 进程提供。评测任务的中间进度保存在 `runtime/eval_jobs/`，完整结果保存在 `result/`；删除 `runtime/` 不会删除已保存的评测结果。

## 安全与版权

- API key 仅通过环境变量提供，不进入模型配置、任务元数据、评测结果或 Git 仓库。
- `downloads/`、`models/`、`images/`、`result/` 和 `runtime/` 均为本地运行目录，不纳入源码版本控制。
- Benchmark 数据和模型权重遵循其原始项目的许可证和使用条款。

Copyright (c) 2026 MiLab. All rights reserved. 详见 [LICENSE](LICENSE)。
