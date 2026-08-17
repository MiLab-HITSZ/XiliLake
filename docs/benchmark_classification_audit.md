# Benchmark 原文与分类核验报告

> 核验日期：2026-08-17。核验对象为网页实际加载的 107 个 Benchmark 实例（94 个唯一名称）。逐子类、逐 Benchmark 的当前归属、介绍、分类核验结论和原文链接见 [current_taxonomy_full.md](current_taxonomy_full.md)。

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
| ARC | 两个镜像重复，并因难度被分到不同大类 | 固定为 ARC-Easy 和 ARC-Challenge 互斥分片，均按科学知识答案准确率归入基本事实准确性 | [AI2 ARC](https://allenai.org/data/arc) |
| FLUE | 名称与数据源不符 | 按实际 LooksJuicy/ruozhiba 数据更名为 RuozhibaQA，归入错误前提辨析 | [官方数据页](https://huggingface.co/datasets/LooksJuicy/ruozhiba) |
| Chinese_Rumor_Dataset | 误放在有害内容 | 任务直接判断谣言真伪，移入基本事实准确性 | [THUNLP 官方仓库](https://github.com/thunlp/Chinese_Rumor_Dataset) |
| sycophancy | 误当成普通任务完成 | 改为反谄媚策略评测，检查模型能否在用户立场诱导下坚持有依据的答案 | [原始数据](https://github.com/nrimsky/LM-exp/tree/main/datasets/sycophancy) |
| Arxiv-Filtered | 被称为学术文献真实性 | 当前本地字段是标题到摘要生成，移入学术摘要生成任务 | [数据页](https://huggingface.co/datasets/AI-dataset-evaluation-team/Arxiv-Filtered) |
| DiaSafety | 误当成攻击鲁棒性 | 原任务是上下文敏感的候选回复安全分类，移入输出内容无害性 | [Findings ACL 2022](https://aclanthology.org/2022.findings-acl.308/) |
| SAFE / SafetyBench | 两者被合并 | SAFE 独立评估既有回答的七个质量轴；SafetyBench 保留为中英双语安全知识选择题 | [SAFE 官方仓库](https://github.com/xiaoqiao/EvalSafetyLLM) / [SafetyBench 官方数据](https://huggingface.co/datasets/thu-coai/SafetyBench) |
| ToxiGen | 被窄化为残疾偏见 | 按 13 个少数群体隐式仇恨文本的主任务，移入输出内容无害性 | [ACL 2022](https://aclanthology.org/2022.acl-long.234/) |
| WMDP | 两个入口重复且一个被归为意识形态 | 固定为 WMDP-BioChem 与 WMDP-Cyber 互斥分片，后者进入网络安全行业 | [WMDP 官方站点](https://www.wmdp.ai/) |
| JBB-Behaviors / HarmBench | 只有行为集却宣称测量越狱攻击 | 当前入口改为未施加攻击时的基线拒答；不宣称攻击成功率 | [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) / [HarmBench](https://github.com/centerforaisafety/HarmBench) |
| SaladBench | 本地扫描误选安全选择题 | 固定到官方 attack_enhanced_set，归入对抗改写攻击防御 | [SALAD-Bench 官方仓库](https://github.com/OpenSafetyLab/SALAD-BENCH) |
| GPTFuzzer / LatentJailbreak / DoAnythingNow / CoSafe | 部分入口误选标签、响应或不完整文件 | 分别固定到越狱模板+有害目标、潜在任务模板、野外越狱+禁止问题、完整多轮共指对话 | [GPTFuzz](https://github.com/sherdencooper/GPTFuzz) / [LatentJailbreak](https://github.com/qiuhuachuan/latent-jailbreak) / [DoAnythingNow](https://github.com/verazuo/jailbreak_llms) / [CoSafe](https://aclanthology.org/2024.emnlp-main.968/) |
| RMCBench / CHiSafetyBench | 存在选错文件或错误 gold（如恶意请求对应 GOOD） | 固定到官方风险提示，gold 改为安全拒答 | [RMCBench](https://github.com/qing-yuan233/RMCBench) / [CHiSafetyBench](https://github.com/UnicomAI/UnicomBenchmark/tree/main/CHiSafetyBench) |
| MedSafetyBench | 误称为医疗信息真实性 | 移入独立的医疗安全可靠性；使用 900 条有害医疗请求与安全回复示范 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3ac952d0264ef7a505393868a70a46b6-Abstract-Datasets_and_Benchmarks_Track.html) |
| PRISM | 误放在暴力与危险知识 | 移入多元用户偏好对齐；评分按参与者分布展示，无唯一 gold | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/be2e1b68b44f2419e19f6c35a1b8cf35-Abstract-Datasets_and_Benchmarks_Track.html) |
| BBQ / CALM / CHBias / FrenchCrowPairs / Regard | 整套数据被窄化为单一身份属性 | 按原文改为多属性歧义问答、性别种族多任务、中文四属性、法语多属性成对偏见和人口群体评价偏差 | [BBQ](https://aclanthology.org/2022.findings-acl.165/) / [CALM](https://arxiv.org/abs/2308.12539) / [CHBias](https://aclanthology.org/2023.acl-long.757/) / [Regard](https://aclanthology.org/D19-1339/) |
| CrowS-Pairs | 两个入口重复读全量数据 | 固定为非宗教的 General 分片和 Religion 分片，互不重叠 | [官方仓库](https://github.com/nyu-mll/crows-pairs) |
| APPS | 两个入口会扫描同一全量目录 | 固定为 Introductory-Interview 和 Competition 两个难度互斥分片 | [APPS 官方仓库](https://github.com/hendrycks/apps) |
| HumanEval+ / MBPP / MathQA-Python / ClassEval / CoderEval / DS-1000 / HumanEval / APPS 两个难度入口 | 同一代码生成构念按基础、复杂、入门面试和竞赛重复分成四个子类 | 合并为代码生成综合任务，九个 Benchmark 继续独立计分 | 各 Benchmark 官方仓库 |
| HarmfulQ | 与多个风险请求拒答数据重复，且自身没有稳定细粒度类别 | 从当前评测目录移除，保留本地原始数据以便追溯 | [数据来源](https://github.com/SALT-NLP/chain-of-thought-bias) |
| ARC-Easy / ARC-Challenge / CMRC2018 | 按普通任务完成展示，未突出答案的事实与证据准确性 | 两个 ARC 分片改为科学知识准确性；CMRC2018 改为篇章证据问答准确性，统一移入基本事实准确性 | [AI2 ARC](https://allenai.org/data/arc) / [CMRC2018](https://github.com/ymcui/cmrc2018) |
| sycophancy | 放在基本事实准确性，容易和独立知识正确率混淆 | 更名为反谄媚策略评测，移入系统策略安全性，强调面对用户立场诱导时的响应策略 | [原始数据](https://github.com/nrimsky/LM-exp/tree/main/datasets/sycophancy) |
| RMCBench | 按风险请求拒答放在系统策略安全性 | 因目标产物是可执行恶意代码，改为恶意代码生成抑制评测并移入输出内容无害性 | [RMCBench](https://github.com/qing-yuan233/RMCBench) |
| GlobalOpinionQA / MoralStories / MultiTP / MoralChoice / MoralExceptQA | 样例存在 CSV 错分列、只显示局部行为或把主观分布当唯一答案 | 重建完整情境、选项、群体分布和行为后果；主观任务明确标记本地代理 | [GlobalOpinionQA](https://arxiv.org/abs/2306.16388) / [MoralStories](https://aclanthology.org/2021.emnlp-main.54/) / [MoralChoice](https://huggingface.co/datasets/ninoscherrer/moralchoice) |

## 数据与评分协议结论

- 新增 `verified_benchmarks` 可重建视图，对 34 个易误选文件的 Benchmark 固定官方分片；部署时由 `prepare_verified_benchmarks.py` 从本地官方下载源生成。
- FollowBench、CHBias、PRISM、GlobalOpinionQA 和 SALAD-Bench 的当前本地入口不冒充论文完整官方评分器；页面会显示“非评分收集”或代理指标说明。
- MedSafetyBench 使用明确拒答或安全回复相似度代理；JBB-Behaviors 和 HarmBench behaviors 仅报告基线拒答；这些结果不等同于原论文的全部 judge 指标。
- 当前网页实际计数为 3 个评测领域、12 个可见大类、93 个子类、107 个 Benchmark，全部标记为可评测。“医疗基本事实准确性”暂无符合定义的本地 Benchmark，因此不用 MedSafetyBench 填充该类。

## 系统策略安全性边界

系统策略安全性按“输入结构 + 风险领域 + 目标输出 + 评分口径”划为 14 个子类，16 个 Benchmark 实例全部可评测：

| 子类 | 唯一判定边界 | Benchmark |
| --- | --- | --- |
| 综合安全知识与责任判断评测 | 给定选择题，判断安全知识和责任风险；不直接测试拒答 | SafetyBench、CValuesResponsibilityMC |
| 单轮参考安全回复质量评测 | 单轮风险场景带安全参考回复，评价拒答、纠偏或替代建议质量 | SafetyPrompts |
| 单轮与多轮风险对话请求拒答一致性评测 | 不使用参考回复，只检查单轮拒答覆盖和多轮上下文一致性 | CHiSafetyBench |
| 犯罪暴力、欺骗仇恨与色情请求拒答评测 | StrongREJECT 六类基础禁止行为，不含越狱变换 | StrongREJECT |
| 违法、隐私、仇恨与身心风险请求拒答评测 | S-Eval 八类中文基础风险，固定 base_risk 分片 | S-Eval |
| 信息危害、恶意用途与交互风险请求拒答评测 | 五个风险领域、12 类具体伤害 | Do-Not-Answer |
| 细粒度不安全指令请求拒答评测 | 45 个细粒度类别，每类 10 条 | SORRY-Bench |
| 多学科知识滥用请求拒答评测 | 十个知识领域中的专业知识滥用 | HarmfulQA |
| 标准化有害行为基线请求拒答评测 | 红队框架的标准行为目标，但未附加攻击模板，用作攻击前基线 | JailbreakBench Behaviors、HarmBench |
| 安全请求过度拒答评测 | 输入应当允许回答，检查误拒绝 | XSTest |
| 反谄媚策略评测 | 用户表达偏好后仍坚持有依据的答案 | sycophancy |
| 跨语种隐私与财产侵害防护评测 | 只含 Privacy And Property，覆盖 10 种语言 | XSafety-Privacy-Refusal |
| 机密信息规则遵循评测 | 给出显式访问规则，检查是否向未授权主体泄密 | RuLES |
| 跨语种非隐私风险请求拒答一致性评测 | 只含 12 类非隐私风险，比较 10 种语言间的策略一致性 | XSafety |

XSafety 的两个入口已在数据层拆分：隐私与财产分片为 2,000 条，非隐私风险分片为 23,988 条，样本 ID 交集为 0。原“多领域禁止请求拒答覆盖评测”按五套仍在用数据的真实风险领域展开；Do-Not-Answer、S-Eval 和 StrongREJECT 同时固定到官方目标文件，避免目录扫描误选既有模型回复或攻击增强数据。RMCBench 因直接约束恶意代码产出，移入输出内容无害性。
