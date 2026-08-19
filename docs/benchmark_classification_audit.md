# Benchmark 原文与分类核验报告

> 核验日期：2026-08-18。核验对象为网页实际加载的 106 个 Benchmark 实例（93 个唯一名称）。逐子类、逐 Benchmark 的当前归属、介绍、分类核验结论和原文链接见 [current_taxonomy_full.md](current_taxonomy_full.md)。

## 核验方法

1. 以论文摘要、官方任务定义、官方数据字段和官方评分协议为主证据，不根据 Benchmark 名称或目录名推断任务。
2. 对网页实际选中的本地文件再做一次核对。同一论文不同分片可以归入不同子类，但必须固定互斥的数据路径。
3. 大类按“直接判定对象”划分：事实真伪、推理过程、任务完成、攻击变换、系统响应策略、群体差异、法律规则或伦理价值。原“输出内容无害性”不再单列：服务于回答/拒答/风险判别的任务进入系统策略，面向身份群体的隐式仇恨进入社会群体公平性，普通冒犯和对话滥用进入伦理道德符合性。
4. 开放生成任务如果官方需要 LLM judge、分布度量或人工评分，本地入口不使用字符串精确匹配伪造“准确率”。

## 大类边界

| 大类 | 纳入条件 | 明确排除 |
| --- | --- | --- |
| 基本事实准确性 | 直接检查外部事实、证据或可见属性的真伪 | 多步演绎、拒答策略 |
| 推理决策可靠性 | 结论需由题面关系、多跳证据、逻辑或因果导出 | 普通语言填空、科学考试问答 |
| 任务遵循可靠性 | 检查指令、格式、工具选择或普通任务产出 | 有害策略、越狱攻击 |
| 攻击抵御鲁棒性 | 输入含显式攻击变换、越狱模板或多轮绕过结构 | 未施加攻击的行为目标集 |
| 系统策略安全性 | 根据风险与规则选择拒答、放行、危险知识控制或回答安全判别 | 显式攻击变换、身份群体差异、普通交往冒犯 |
| 社会群体公平性 | 指标明确关联身份群体差异、刻板印象或隐式仇恨 | 不按任意单条样例窄化整个数据集 |
| 法律法规遵守性 | 答案可由明确法规、合同或政策文本判定 | 无明文法源的道德取舍 |
| 伦理道德符合性 | 价值观、社会规范、伦理困境、伤害补全或对话滥用处置 | 明确法源、群体差异、风险请求拒答 |

## 主要校正

| Benchmark | 原问题 | 核验后处理 | 主证据 |
| --- | --- | --- | --- |
| FollowBench | 误放在规划推理 | 移入细粒度约束遵循；开放任务标记为非精确匹配评分 | [ACL 2024](https://aclanthology.org/2024.acl-long.257/) |
| Chinese_language_ability | 名称泛化且误放在逻辑分析 | 根据实际 FewCLUE CHID 数据更名为 CHID，移入成语语境填空 | [FewCLUE 官方仓库](https://github.com/CLUEbenchmark/FewCLUE) |
| ARC | 两个镜像重复，并因难度被分到不同大类 | 固定为 ARC-Easy 和 ARC-Challenge 互斥分片，均按科学知识答案准确率归入基本事实准确性 | [AI2 ARC](https://allenai.org/data/arc) |
| FLUE | 名称与数据源不符 | 按实际 LooksJuicy/ruozhiba 数据更名为 RuozhibaQA，归入错误前提辨析 | [官方数据页](https://huggingface.co/datasets/LooksJuicy/ruozhiba) |
| Chinese_Rumor_Dataset | 误放在有害内容且介绍成逻辑谬误识别 | 更名为中文易谬误事实性评测；任务直接判断微博陈述真伪，移入基本事实准确性 | [THUNLP 官方仓库](https://github.com/thunlp/Chinese_Rumor_Dataset) |
| sycophancy | 误当成普通任务完成 | 改为反谄媚策略评测，检查模型能否在用户立场诱导下坚持有依据的答案 | [原始数据](https://github.com/nrimsky/LM-exp/tree/main/datasets/sycophancy) |
| Arxiv-Filtered | 被称为学术文献真实性 | 当前本地字段是标题到摘要生成，移入学术摘要生成任务 | [数据页](https://huggingface.co/datasets/AI-dataset-evaluation-team/Arxiv-Filtered) |
| DiaSafety | 误当成攻击鲁棒性 | 原任务是上下文敏感的候选回复安全分类，不含攻击生成；作为上下文响应判别策略移入系统策略安全性 | [Findings ACL 2022](https://aclanthology.org/2022.findings-acl.308/) |
| SAFE / SafetyBench | 两者任务协议不同 | 按当前目录要求合并到有害请求拒答策略评测，但保持独立计分：SAFE 判别既有回答的七个质量轴，SafetyBench 检查中英双语安全知识与风险方案选择 | [SAFE 官方仓库](https://github.com/xiaoqiao/EvalSafetyLLM) / [SafetyBench 官方数据](https://huggingface.co/datasets/thu-coai/SafetyBench) |
| ToxiGen | 被窄化为残疾偏见 | 按 13 类少数群体隐式仇恨文本的主任务，移入社会群体公平性 | [ACL 2022](https://aclanthology.org/2022.acl-long.234/) |
| WMDP | 两个入口重复且一个被归为意识形态 | 固定为 WMDP-BioChem 与 WMDP-Cyber 互斥分片，后者进入网络安全行业 | [WMDP 官方站点](https://www.wmdp.ai/) |
| JBB-Behaviors / HarmBench | 只有行为集却宣称测量越狱攻击 | 当前入口改为未施加攻击时的基线拒答；不宣称攻击成功率 | [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) / [HarmBench](https://github.com/centerforaisafety/HarmBench) |
| SaladBench | 本地扫描误选安全选择题 | 固定到官方 attack_enhanced_set，归入对抗改写攻击防御 | [SALAD-Bench 官方仓库](https://github.com/OpenSafetyLab/SALAD-BENCH) |
| GPTFuzzer / LatentJailbreak / DoAnythingNow / CoSafe | 部分入口误选标签、响应或不完整文件 | 分别固定到越狱模板+有害目标、潜在任务模板、野外越狱+禁止问题、完整多轮共指对话 | [GPTFuzz](https://github.com/sherdencooper/GPTFuzz) / [LatentJailbreak](https://github.com/qiuhuachuan/latent-jailbreak) / [DoAnythingNow](https://github.com/verazuo/jailbreak_llms) / [CoSafe](https://aclanthology.org/2024.emnlp-main.968/) |
| RMCBench / CHiSafetyBench | 存在选错文件或错误 gold（如恶意请求对应 GOOD） | 固定到官方风险提示，gold 改为安全拒答 | [RMCBench](https://github.com/qing-yuan233/RMCBench) / [CHiSafetyBench](https://github.com/UnicomAI/UnicomBenchmark/tree/main/CHiSafetyBench) |
| MedSafetyBench | 误称为医疗信息真实性 | 移入独立的医疗安全可靠性；使用 900 条有害医疗请求与安全回复示范 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3ac952d0264ef7a505393868a70a46b6-Abstract-Datasets_and_Benchmarks_Track.html) |
| BBQ / CALM / CHBias | 整套数据被窄化为单一身份属性 | 按原文保留多属性歧义问答、性别种族多任务和中文四属性偏见任务；CHBias 统一命名为中文综合偏见评测 | [BBQ](https://aclanthology.org/2022.findings-acl.165/) / [CALM](https://arxiv.org/abs/2308.12539) / [CHBias](https://aclanthology.org/2023.acl-long.757/) |
| CrowS-Pairs | 两个入口重复读全量数据 | 当前目录仅保留 Religion 分片，General 分片已移除 | [官方仓库](https://github.com/nyu-mll/crows-pairs) |
| APPS | 两个入口会扫描同一全量目录 | 固定为 Introductory-Interview 和 Competition 两个难度互斥分片 | [APPS 官方仓库](https://github.com/hendrycks/apps) |
| HumanEval+ / MBPP / MathQA-Python / ClassEval / CoderEval / DS-1000 / HumanEval / APPS 两个难度入口 | 同一代码生成构念按基础、复杂、入门面试和竞赛重复分成四个子类 | 合并为代码生成综合任务，九个 Benchmark 继续独立计分 | 各 Benchmark 官方仓库 |
| HarmfulQ | 与多个风险请求拒答数据重复，且自身没有稳定细粒度类别 | 从当前评测目录移除，保留本地原始数据以便追溯 | [数据来源](https://github.com/SALT-NLP/chain-of-thought-bias) |
| PRISM / CDialBias-QA / CrowS-Pairs-General-MC / FrenchCrowPairs / Regard | 与当前精简后的伦理或公平性目录范围重复 | 从当前评测目录移除，保留本地原始数据以便追溯 | 各 Benchmark 原始论文或仓库 |
| HolisticBias | 原目录名为多身份交叉偏见评测 | 按当前目录命名改为职业偏见评测；介绍仍注明官方数据实际覆盖多个身份轴 | [HolisticBias](https://github.com/facebookresearch/ResponsibleNLP/tree/main/holistic_bias) |
| ARC-Easy / ARC-Challenge / CMRC2018 | 按普通任务完成展示，未突出答案的事实与证据准确性 | 两个 ARC 分片改为科学知识准确性；CMRC2018 改为篇章证据问答准确性，统一移入基本事实准确性 | [AI2 ARC](https://allenai.org/data/arc) / [CMRC2018](https://github.com/ymcui/cmrc2018) |
| sycophancy | 放在基本事实准确性，容易和独立知识正确率混淆 | 更名为反谄媚策略评测，移入系统策略安全性，强调面对用户立场诱导时的响应策略 | [原始数据](https://github.com/nrimsky/LM-exp/tree/main/datasets/sycophancy) |
| RMCBench | 曾按最终代码危害性归入输出内容 | 当前入口已把 473 条恶意代码请求的 gold 统一为 SAFE_REFUSAL，因此改为恶意代码生成请求拒答策略评测并归入系统策略安全性 | [RMCBench](https://github.com/qing-yuan233/RMCBench) |
| HONEST / ParlAI Dialogue Safety / ConvAbuse | 伤害补全、冒犯发言和对话滥用与系统拒答策略混在一起 | 分别归入多语言伤害补全、冒犯对话处置和对话滥用处置三个伦理子类；它们判定社会交往表达，不测试越狱或普通风险请求拒答 | [HONEST](https://github.com/MilaNLProc/honest) / [ParlAI Dialogue Safety](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/dialogue_safety) / [ConvAbuse](https://github.com/amandacurry/convabuse) |
| GlobalOpinionQA / MoralStories / MultiTP / MoralChoice / MoralExceptQA | 样例存在 CSV 错分列、只显示局部行为或把主观分布当唯一答案 | 重建完整情境、选项、群体分布和行为后果；主观任务明确标记本地代理 | [GlobalOpinionQA](https://arxiv.org/abs/2306.16388) / [MoralStories](https://aclanthology.org/2021.emnlp-main.54/) / [MoralChoice](https://huggingface.co/datasets/ninoscherrer/moralchoice) |

## 数据与评分协议结论

- 新增 `verified_benchmarks` 可重建视图，对 34 个易误选文件的 Benchmark 固定官方分片；部署时由 `prepare_verified_benchmarks.py` 从本地官方下载源生成。
- FollowBench、CHBias、GlobalOpinionQA 和 SALAD-Bench 的当前本地入口不冒充论文完整官方评分器；页面会显示“非评分收集”或代理指标说明。
- MedSafetyBench 使用明确拒答或安全回复相似度代理；JBB-Behaviors 和 HarmBench behaviors 仅报告基线拒答；这些结果不等同于原论文的全部 judge 指标。
- 当前网页实际计数为 3 个评测领域、11 个可见大类、81 个子类、106 个 Benchmark，全部标记为可评测。“医疗基本事实准确性”暂无符合定义的本地 Benchmark，因此不用 MedSafetyBench 填充该类。

## 法律法规遵守性边界

法律法规遵守性接入 LegalBench 中与法规遵守直接相关的 30 个官方任务。系统保留原有三个独立入口，并把其余 27 个任务汇总为四个无任务重叠的可评测套件，共接入 40,761 条有 gold 的测试样本。

| 子类 | 唯一判定边界 | LegalBench 任务数 | 有标签测试样本数 |
| --- | --- | ---: | ---: |
| 隐私政策法规遵守性评测 | 隐私政策是否蕴含、披露或支持数据处理规则 | 11 | 24,516 |
| 消费者权益法规遵守性评测 | 消费合同权利、不公平条款及电话营销规则 | 3 | 4,256 |
| 法律规则知识准确性评测 | 回忆或识别规则、司法行为规范及判例法源 | 5 | 9,809 |
| 法律规则适用结论准确性评测 | 在规则领域已知时把规则应用到事实并得出结论 | 11 | 2,180 |

任务归属直接采用 LegalBench 官方任务清单：隐私和消费者任务定向抽出，规则知识及规则适用任务分别来自官方 `RULE_TASKS` 和 `CONCLUSION_TASKS`，三个原有任务从聚合套件中排除，因此当前 30 个任务恰好各归属一次。

## 系统策略安全性边界

系统策略安全性纳入未经攻击改写的普通请求、用户立场、显式系统规则、危险知识控制和回答安全判别，按系统应采取或校准的响应行动划为 10 个子类、19 个 Benchmark。所有子类统一以“策略评测”结尾：

| 子类 | 唯一判定边界 | Benchmark |
| --- | --- | --- |
| 有害请求拒答策略评测 | 选择题检查风险判断与安全方案选择，普通有害请求检查拒答，SAFE 对已有回答执行七轴安全判别；不含攻击模板，各 Benchmark 独立计分 | SafetyBench、CValuesResponsibilityMC、StrongREJECT、S-Eval、Do-Not-Answer、SORRY-Bench、JailbreakBench Behaviors、HarmBench、SAFE |
| 多轮风险对话拒答策略评测 | 只使用 101 条带完整历史的多轮风险对话，不含无历史单轮问题或共指攻击结构 | CHiSafetyBench |
| 多学科知识滥用拒答策略评测 | 十个知识领域中的专业知识滥用请求 | HarmfulQA |
| 安全请求放行策略评测 | 输入应当允许回答，检查模型是否发生过度拒答 | XSTest |
| 反谄媚策略评测 | 用户表达偏好后仍坚持有依据的答案 | sycophancy |
| 隐私与财产防护策略评测 | XSafety 检查跨语种隐私、凭据和财产侵害请求拒答；RuLES 检查显式访问规则下的机密信息保护 | XSafety-Privacy-Refusal、RuLES |
| 跨语种风险拒答一致性策略评测 | 比较 10 种语言、9 类普通风险请求的拒答策略一致性 | XSafety |
| 恶意代码生成请求拒答策略评测 | 恶意代码生成、翻译、补全和改写请求的 gold 均为安全拒答 | RMCBench |
| 生化危险知识控制策略评测 | 用官方选择题准确率观察生物和化学危险知识的保留与暴露，分数方向需按“越高暴露越多”解释 | WMDP-BioChem |
| 上下文对话安全判别策略评测 | 给定对话历史和候选回复，校准上下文相关的安全判别 | DiaSafety |

SafetyPrompts 因参考回复质量与普通拒答构念重复而从当前目录移除。JailbreakBench Behaviors 和 HarmBench Behaviors 没有附加攻击字符串，因此并入综合有害请求判断与拒答策略；真正含越狱模板、提示注入、攻击增强或对抗改写的数据仍只归入攻击抵御鲁棒性。RMCBench 当前执行协议直接按安全拒答评分，因此保留在系统策略安全性。

XSafety 进一步按输入机制拆分为互斥数据：17,990 条普通风险请求保留在系统策略，2,000 条 Privacy And Property 进入隐私防护策略，2,000 条 Goal Hijacking 进入攻击抵御并与 DoAnythingNow、UltraSafety 合并为“多语种越狱提示攻击防御评测”。Prompt Leaking 和 Role Play Instruction 共 3,998 条因标签与实际内容混杂而不参与评分，三个可评测分片的样本 ID 交集为 0。
