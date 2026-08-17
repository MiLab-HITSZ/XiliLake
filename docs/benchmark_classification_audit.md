# Benchmark 原文与分类核验报告

> 核验日期：2026-08-17。核验对象为网页实际加载的 108 个 Benchmark 实例（95 个唯一名称）。逐子类、逐 Benchmark 的当前归属、介绍、分类核验结论和原文链接见 [current_taxonomy_full.md](current_taxonomy_full.md)。

## 核验方法

1. 以论文摘要、官方任务定义、官方数据字段和官方评分协议为主证据，不根据 Benchmark 名称或目录名推断任务。
2. 对网页实际选中的本地文件再做一次核对。同一论文不同分片可以归入不同子类，但必须固定互斥的数据路径。
3. 大类按“直接判定对象”划分：事实真伪、推理过程、任务完成、输出内容、攻击变换、系统响应策略、群体差异、法律规则或伦理价值。
4. 开放生成任务如果官方需要 LLM judge、分布度量或人工评分，本地入口不使用字符串精确匹配伪造“准确率”。

## 大类边界

| 大类 | 纳入条件 | 明确排除 |
| --- | --- | --- |
| 基本事实准确性 | 直接检查外部事实、证据或可见属性的真伪 | 多步演绎、拒答策略 |
| 推理决策可靠性 | 结论需由题面关系、多跳证据、逻辑或因果导出 | 普通语言填空、科学考试问答 |
| 任务遵循可靠性 | 检查指令、格式、工具选择或普通任务产出 | 有害策略、越狱攻击 |
| 输出内容无害性 | 判定已有或生成内容的毒性、仇恨、滥用或危险知识 | 普通有害请求的拒答覆盖 |
| 攻击抵御鲁棒性 | 输入含显式攻击变换、越狱模板或多轮绕过结构 | 未施加攻击的行为目标集 |
| 系统策略安全性 | 根据风险与规则选择拒答、安全替代或正常放行 | 攻击算法成功率、纯内容毒性 |
| 群体与社会公平性 | 指标是身份群体间的差异、刻板印象或负面评价 | 不按任意单条样例窄化整个数据集 |
| 法律法规遵守性 | 答案可由明确法规、合同或政策文本判定 | 无明文法源的道德取舍 |
| 伦理道德符合性 | 价值观、社会规范、伦理困境或主观偏好分布 | 将某一群体的多数意见宣称为客观事实 |

## 主要校正

| Benchmark | 原问题 | 核验后处理 | 主证据 |
| --- | --- | --- | --- |
| FollowBench | 误放在规划推理 | 移入细粒度约束遵循；开放任务标记为非精确匹配评分 | [ACL 2024](https://aclanthology.org/2024.acl-long.257/) |
| Chinese_language_ability | 名称泛化且误放在逻辑分析 | 根据实际 FewCLUE CHID 数据更名为 CHID，移入成语语境填空 | [FewCLUE 官方仓库](https://github.com/CLUEbenchmark/FewCLUE) |
| ARC | 两个镜像重复，并因难度被分到不同大类 | 固定为 ARC-Easy 和 ARC-Challenge 互斥分片，均归入科学问答任务 | [AI2 ARC](https://allenai.org/data/arc) |
| FLUE | 名称与数据源不符 | 按实际 LooksJuicy/ruozhiba 数据更名为 RuozhibaQA，归入错误前提辨析 | [官方数据页](https://huggingface.co/datasets/LooksJuicy/ruozhiba) |
| Chinese_Rumor_Dataset | 误放在有害内容 | 任务直接判断谣言真伪，移入基本事实准确性 | [THUNLP 官方仓库](https://github.com/thunlp/Chinese_Rumor_Dataset) |
| sycophancy | 误当成普通任务完成 | 改为用户立场诱导真实性，检查迎合是否改变正确答案 | [原始数据](https://github.com/nrimsky/LM-exp/tree/main/datasets/sycophancy) |
| Arxiv-Filtered | 被称为学术文献真实性 | 当前本地字段是标题到摘要生成，移入学术摘要生成任务 | [数据页](https://huggingface.co/datasets/AI-dataset-evaluation-team/Arxiv-Filtered) |
| DiaSafety | 误当成攻击鲁棒性 | 原任务是上下文敏感的候选回复安全分类，移入输出内容无害性 | [Findings ACL 2022](https://aclanthology.org/2022.findings-acl.308/) |
| SAFE / SafetyBench | 两者被合并 | SAFE 独立评估既有回答的七个质量轴；SafetyBench 保留为中英双语安全知识选择题 | [SAFE 官方仓库](https://github.com/xiaoqiao/EvalSafetyLLM) / [SafetyBench 官方数据](https://huggingface.co/datasets/thu-coai/SafetyBench) |
| ToxiGen | 被窄化为残疾偏见 | 按 13 个少数群体隐式仇恨文本的主任务，移入输出内容无害性 | [ACL 2022](https://aclanthology.org/2022.acl-long.234/) |
| WMDP | 两个入口重复且一个被归为意识形态 | 固定为 WMDP-BioChem 与 WMDP-Cyber 互斥分片，后者进入网络安全行业 | [WMDP 官方站点](https://www.wmdp.ai/) |
| JBB-Behaviors / HarmBench | 只有行为集却宣称测量越狱攻击 | 当前入口改为未施加攻击时的基线拒答；不宣称攻击成功率 | [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) / [HarmBench](https://github.com/centerforaisafety/HarmBench) |
| SaladBench | 本地扫描误选安全选择题 | 固定到官方 attack_enhanced_set，归入对抗改写攻击防御 | [SALAD-Bench 官方仓库](https://github.com/OpenSafetyLab/SALAD-BENCH) |
| GPTFuzzer / LatentJailbreak / DoAnythingNow / CoSafe | 部分入口误选标签、响应或不完整文件 | 分别固定到越狱模板+有害目标、潜在任务模板、野外越狱+禁止问题、完整多轮共指对话 | [GPTFuzz](https://github.com/sherdencooper/GPTFuzz) / [LatentJailbreak](https://github.com/qiuhuachuan/latent-jailbreak) / [DoAnythingNow](https://github.com/verazuo/jailbreak_llms) / [CoSafe](https://aclanthology.org/2024.emnlp-main.968/) |
| RMCBench / CHiSafetyBench / HarmfulQ | 存在选错文件或错误 gold（如恶意请求对应 GOOD） | 固定到官方风险提示，gold 改为安全拒答；HarmfulQ 使用 dangerous-q 的 200 条真实问题 | [RMCBench](https://github.com/qing-yuan233/RMCBench) / [CHiSafetyBench](https://github.com/UnicomAI/UnicomBenchmark/tree/main/CHiSafetyBench) / [HarmfulQ 数据来源](https://github.com/SALT-NLP/chain-of-thought-bias) |
| MedSafetyBench | 误称为医疗信息真实性 | 移入独立的医疗安全可靠性；使用 900 条有害医疗请求与安全回复示范 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3ac952d0264ef7a505393868a70a46b6-Abstract-Datasets_and_Benchmarks_Track.html) |
| PRISM | 误放在暴力与危险知识 | 移入多元用户偏好对齐；评分按参与者分布展示，无唯一 gold | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/be2e1b68b44f2419e19f6c35a1b8cf35-Abstract-Datasets_and_Benchmarks_Track.html) |
| BBQ / CALM / CHBias / FrenchCrowPairs / Regard | 整套数据被窄化为单一身份属性 | 按原文改为多属性歧义问答、性别种族多任务、中文四属性、法语多属性成对偏见和人口群体评价偏差 | [BBQ](https://aclanthology.org/2022.findings-acl.165/) / [CALM](https://arxiv.org/abs/2308.12539) / [CHBias](https://aclanthology.org/2023.acl-long.757/) / [Regard](https://aclanthology.org/D19-1339/) |
| CrowS-Pairs | 两个入口重复读全量数据 | 固定为非宗教的 General 分片和 Religion 分片，互不重叠 | [官方仓库](https://github.com/nyu-mll/crows-pairs) |
| APPS | 两个入口会扫描同一全量目录 | 固定为 Introductory-Interview 和 Competition 两个难度互斥分片 | [APPS 官方仓库](https://github.com/hendrycks/apps) |
| GlobalOpinionQA / MoralStories / MultiTP / MoralChoice / MoralExceptQA | 样例存在 CSV 错分列、只显示局部行为或把主观分布当唯一答案 | 重建完整情境、选项、群体分布和行为后果；主观任务明确标记本地代理 | [GlobalOpinionQA](https://arxiv.org/abs/2306.16388) / [MoralStories](https://aclanthology.org/2021.emnlp-main.54/) / [MoralChoice](https://huggingface.co/datasets/ninoscherrer/moralchoice) |

## 数据与评分协议结论

- 新增 `verified_benchmarks` 可重建视图，对 34 个易误选文件的 Benchmark 固定官方分片；部署时由 `prepare_verified_benchmarks.py` 从本地官方下载源生成。
- FollowBench、CHBias、PRISM、GlobalOpinionQA 和 SALAD-Bench 的当前本地入口不冒充论文完整官方评分器；页面会显示“非评分收集”或代理指标说明。
- MedSafetyBench 使用明确拒答或安全回复相似度代理；JBB-Behaviors 和 HarmBench behaviors 仅报告基线拒答；这些结果不等同于原论文的全部 judge 指标。
- 当前网页实际计数为 3 个评测领域、12 个可见大类、97 个子类、108 个 Benchmark，全部标记为可评测。“医疗基本事实准确性”暂无符合定义的本地 Benchmark，因此不用 MedSafetyBench 填充该类。
