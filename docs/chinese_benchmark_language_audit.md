# Benchmark 中文内容与原始结构审计

审计日期：2026-08-27

## 判定口径

本次审计覆盖在线目录的 113 个 Benchmark，并对所有中文、英文和多语种入口逐项检查原始文件、派生文件和加载器。语言分区只依据原始题干、材料、选项、参考答案或官方语言分片，不依据 XiliLake 为统一调用模型而添加的回答格式提示。

一个样例在系统中有三层结构：

1. **原始数据层**：论文或仓库发布的字段和语言，是语言分区的主要依据。
2. **可评测视图层**：把嵌套记录、成对样本或多轮对话整理成 `question/context/options/answer`，不得无依据地改变原始语言。
3. **运行提示层**：加载器追加“只回答选项字母”等输出约束，仅用于解析答案，不改变 Benchmark 的语言归属。

审计后共有 20 个 Benchmark 入口使用中文原始内容或中文官方分片。LogiQA、SafetyBench、FollowBench、Natural-Instructions、ByteCue、XSafety 的三个任务分片和 MultiTP 均已拆成相互独立的中文与非中文入口，不再把整个双语或多语种数据集放入中文部分。其余 12 个此前仅因本地中文任务模板而被标记为中文的入口已恢复原始语言并移回通用部分；LegalBench-UnfairToS 静态样例中的本地中文问题也已恢复为英文。

## 原生中文或中文分片

| Benchmark | 原始数据结构 | 实际接入内容 | 结论 |
|---|---|---|---|
| CMMLU | 各学科 CSV：`Question/A/B/C/D/Answer` | 汇总 67 学科的 11,582 道中文测试题为 `question/options/answer/subject` | 中文 |
| HalluQA | JSON 数组：`question_id/question/answer`，题干字符串内嵌 A-E 选项 | 450 道中文对抗性事实问题 | 中文 |
| Chinese_Rumor_Dataset | `original-microblog/*.json` 保存微博 `text`，rumor/non-rumor 目录提供标签 | 中文微博原文与谣言二分类标签 | 中文 |
| CMRC2018 | 嵌套 JSON：`data[].paragraphs[].context/qas[].question/answers[]` | 中文篇章、问题和跨度答案 | 中文 |
| LogiQA-Chinese | `zh_eval.txt` 为纯文本块：答案字母、编号题干、A-D 选项 | 独立读取全部 651 道中文评估题 | 中文 |
| FLUB | JSONL：`text/is_question/type/explanation/options/answer` | 834 条中文文本陷阱与解释/选项 | 中文 |
| RuozhibaQA | 问答记录：`question/answer/language=zh` | 中文错误前提与歧义问答 | 中文 |
| CHID | JSONL：`id/candidates/content/answer` | 中文篇章中的成语空缺与候选成语 | 中文 |
| SEval | JSONL：`traceid/risk_type/prompt/ext` | 当前固定使用 `S-Eval_base_risk_zh_small.jsonl` | 中文 |
| CHiSafetyBench | JSON：`category/sub_category/input/history[]` | 101 条带完整中文历史的多轮风险对话 | 中文 |
| CHBias | 各身份子集 CSV，核心字段为 `replaced_sentence` | 1,600 条中文偏见句，构造成开放式去偏改写 | 中文 |
| SafetyBench-Chinese | `dev_zh.json` 按 7 个风险类别组织，记录为 `question/options/answer` | 独立读取 35 道有标签中文开发题 | 中文 |
| CValuesResponsibilityMC | `harmless_test.json`：`prompt/pos_resp/neg_resp/pos_type/neg_type` | 从 9,711 组完整正负回答对构成双选题，选项顺序交替 | 中文 |
| MultiTP-Chinese | `dataset_zh-cn+google.csv` 同时保存中文 `Prompt`、英文 `prompt_en`、规范化 `two_choices` 和 `sub1/sub2` | 独立读取 460 条中文场景，从中文 `Prompt` 提取两个显示选项，以 `sub1/sub2` 计算参考方向 | 中文 |

## 同源中文与非中文分片

| 数据源 | 中文入口 | 通用入口 | 分割依据 |
|---|---|---|---|
| LogiQA | LogiQA-Chinese：`zh_eval.txt`，651 条 | LogiQA-English：`Eval.txt`，651 条 | 官方文件分片 |
| SafetyBench | SafetyBench-Chinese：`dev_zh.json`，35 条 | SafetyBench-English：`dev_en.json`，35 条 | 官方文件分片 |
| FollowBench | FollowBench-Chinese：`data_zh/`，790 条 | FollowBench-English：`data/`，820 条 | 官方目录与语言标签 |
| Natural-Instructions | Natural-Instructions-Chinese：32 个任务、17,861 条实例 | Natural-Instructions-General：1,581 个任务、5,020,282 条实例 | 任一 `Input/Output/Instruction_language` 含 Chinese 即进入中文入口；其余任务进入非中文入口 |
| ByteCue | ByteCue-Chinese：13 条 | ByteCue-English：6,115 条 | 参考注释 `comment` 是否包含中文；API、字节码和 CFG 作为代码材料原样保留 |
| XSafety Goal Hijacking | XSafety-Attack-Defense-Chinese：200 条 | XSafety-Attack-Defense-General：1,800 条 | `language_code=zh` 与其余 9 种语言 |
| XSafety 普通风险 | XSafety-Chinese：1,800 条 | XSafety-General：16,190 条 | `language_code=zh` 与其余 9 种语言 |
| XSafety Privacy And Property | XSafety-Privacy-Refusal-Chinese：200 条 | XSafety-Privacy-Refusal-General：1,800 条 | `language_code=zh` 与其余 9 种语言 |
| MultiTP | MultiTP-Chinese：`dataset_zh-cn+google.csv`，460 条 | MultiTP-English：`dataset_en+google.csv`，460 条 | 官方语言文件分片 |

## 仅被中文模板包装的英文或非中文数据

| Benchmark | 原始数据结构与语言 | 旧问题 | 修正 |
|---|---|---|---|
| RustRepoTrans | 源文件路径、C/Java/Python 函数、Rust 文件路径和参考 Rust 函数；自然语言内容以英文为主 | 本地用中文说明“翻译为 Rust” | 改为英文任务提示，归通用部分 |
| explicit_subset | CommitBench CSV：`hash/diff/message/project/diff_languages/split`；代码差异和提交信息以英文为主 | 本地中文摘要指令 | 改为英文任务提示，归通用部分 |
| implicit_subset | 与 explicit_subset 同源，仅按消息与差异的词面对应程度重建分片 | 本地中文摘要指令 | 改为英文任务提示，归通用部分 |
| ConfAIde-Tier2 | `tier_2a.txt/tier_2b.txt` 为英文任务说明和英文场景，`tier_2_labels.txt` 为 98 个人类均值 | 丢弃官方英文说明后另加中文问题与中文量表 | 恢复英文问题和英文量表，归通用部分 |
| HolisticBias | 官方生成器输出 `text/axis/bucket/descriptor/noun_phrase/template`，566,625 条均为英文身份陈述 | 本地中文回应指令 | 改为英文回应指令，归通用部分 |
| CrowS-Pairs-Religion-MC | CSV：`sent_more/sent_less/stereo_antistereo/bias_type`，宗教子集 105 组英文句对 | 本地中文选择问题 | 改为英文选择问题，归通用部分 |
| LegalBench-PrivacyPolicyQA | TSV：`index/question/text/answer`，问题、政策条款和 Relevant/Irrelevant 标签均为英文 | 加了中文“隐私政策条款”字段名 | 改为英文上下文字段，归通用部分 |
| MoralStories | JSONL：`norm/situation/intention/moral_action/moral_consequence/immoral_action/immoral_consequence`，均为英文 | 本地中文字段名、问题和选项 | 恢复英文结构化提示与选项，归通用部分 |
| MoralExceptQA | JSONL：`context/condition/study/human.response/features/scenario`，场景均为英文 | 本地中文可允许性问题和选项 | 恢复英文问题与选项，归通用部分 |
| HONEST | TSV：`template_masked/identity/category/type`；语言为 en/es/fr/it/pt/ro，不包含中文 | 所有 5,625 条统一添加中文补全指令 | 改为英文补全说明，归通用部分 |
| DiaSafety | JSON：`context/response/category/label`，1,095 条均为英文对话 | 本地中文字段名和安全判断问题 | 恢复英文问题与选项，归通用部分 |
| PairVul | JSON：`code_before_change/code_after_change/cve_id/cwe/id`，内容为源代码和英文注释 | 本地中文漏洞判断问题和选项 | 改为英文问题与选项，归通用部分 |
| LegalBench-UnfairToS | TSV：`index/text/answer`，服务条款与九类风险标签均为英文 | 静态样例额外添加中文分类问题，但分类本身仍在通用部分 | 静态问题改为英文，保持通用部分 |

## 关键一致性要求

- 中文分区的判定必须来自 `source_language` 或已核实的官方分片，不扫描本地提示模板中的“请回答”等文字。
- 同一 Benchmark 同时提供中文和英文/非中文内容时，必须生成互斥的数据文件、Benchmark 入口和子类；中文入口进入中文分区，英文或其他语言入口留在通用分区。
- 英文材料不得仅因本地中文任务说明而归入中文；其可评测视图应使用与原始数据一致的英文任务提示。
- 翻译、代码注释等任务可以按官方任务结构保留另一种语言的源材料，但必须在介绍中说明输入与目标语言。Natural-Instructions 的中英翻译任务和 ByteCue 的代码材料属于这种情况。
- 原始文件为了评测解析而保留英文规范化标签时，可以在界面生成同序显示选项，但答案索引必须仍由原字段计算。MultiTP 的中英文入口都采用这一方式。
