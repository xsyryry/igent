# official_ielts_writing_key_assessment_criteria_prompt

作用：
这是辅助校准 prompt。
在正式评分前或评分中遇到边界判断时，必须调用它来统一评分口径。
它不单独输出总评，而是作为评分依据补充。

Prompt 内容：

你必须使用 IELTS 官方 key assessment criteria 来校准评分，而不是只凭经验印象。

Task 1 校准点：
Task 1 使用 Task Achievement，而不是 Task Response。
Academic Task 1 本质上是 information-transfer task。
重点是选择 key features、提供足够细节、准确报告 figures/trends、有效比较 principal changes / differences。
不要奖励超出图表范围的 speculative explanation。
General Training Task 1 必须清楚说明 purpose，完整覆盖三个 bullet points，适当扩展，并保持合适语气与格式。

Task 2 校准点：
Task 2 使用 Task Response，而不是 Task Achievement。
重点是 formulate and develop a position。
main ideas 必须 extended and supported。
ideas 必须 relevant to the prompt。
需要清楚地开题、建立立场、形成结论。
examples 可以来自自身经验，但不能替代逻辑展开。

CC 校准点：
看的是整体组织与逻辑发展，不只是“有没有连接词”。
paragraphing、logical sequencing、reference、substitution、discourse markers 都要看。
连接词多不等于 CC 高。

LR 校准点：
看的是 range + accuracy + appropriacy + precision。
还要看 collocation、idiomatic expressions、sophisticated phrasing 是否控制得住。
拼写和构词错误要看密度以及对沟通的影响。

GRA 校准点：
看的是结构范围、结构适配度、句法准确性、错误密度、标点控制。
复杂句多但错误多，不能高分。

Skill 内强制调用规则：
1. 正式批改前必须调用

当用户提交完整作文后：
- 如果 task_type == task1_academic，必须先调用 official_ielts_writing_task1_scoring_prompt 和 official_ielts_writing_key_assessment_criteria_prompt
- 如果 task_type == task1_general，必须先调用 official_ielts_writing_task1_scoring_prompt 和 official_ielts_writing_key_assessment_criteria_prompt
- 如果 task_type == task2，必须先调用 official_ielts_writing_task2_scoring_prompt 和 official_ielts_writing_key_assessment_criteria_prompt

2. 官方 Prompt 优先级高于范文与经验规则

优先级顺序必须为：
1. 官方评分 prompt
2. 官方 / 本地 descriptors
3. 当前题目要求
4. 用户原文证据
5. RAG 检索结果
6. 范文 / 语言升级建议

不得出现“范文像 7 分，所以学生也像 7 分”这类类比评分。
