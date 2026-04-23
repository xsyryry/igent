# writing_review_skill

## 目标
用于 IELTS Writing 作文批改。该 skill 只规定批改流程和调用规则，不存放长 prompt。

## Prompt 文件
官方评分 prompt 独立存放在：

```text
project/prompts/writing_review/
```

三个文件分别是：

- `official_ielts_writing_task1_scoring_prompt.md`
- `official_ielts_writing_task2_scoring_prompt.md`
- `official_ielts_writing_key_assessment_criteria_prompt.md`

## 调用规则
当用户提交完整作文后：

- `task1_academic`：必须调用
  - `official_ielts_writing_task1_scoring_prompt.md`
  - `official_ielts_writing_key_assessment_criteria_prompt.md`
- `task1_general`：必须调用
  - `official_ielts_writing_task1_scoring_prompt.md`
  - `official_ielts_writing_key_assessment_criteria_prompt.md`
- `task2`：必须调用
  - `official_ielts_writing_task2_scoring_prompt.md`
  - `official_ielts_writing_key_assessment_criteria_prompt.md`

官方评分 prompt 优先级最高，不得用范文、外刊或经验规则替代。

## RAG 检索规则
批改时仍然必须检索 RAG 中的外刊资料，检索范围：

```text
dataset_scope=magazine
```

## 外刊 RAG 的职责边界

外刊 RAG 是**语言增强模块**，不是评分模块，也不是题目完成度判断模块。  
其作用仅限于为作文批改提供**语言层面**和**轻度论证表达层面**的辅助参考。

### 外刊 RAG 允许用于以下场景
1. 补充 **topic-specific language**
   - 当前话题下更自然、更常见的学术表达
   - 与话题相关的常用搭配、术语、表述方式

2. 提供 **collocation / phrasing**
   - 更自然的词语搭配
   - 更符合书面语风格的短语表达
   - 替换生硬、重复或中式痕迹较重的表达

3. 提供 **论证表达参考**
   - 如何更自然地引出观点、解释原因、补充影响、承接例子
   - 如何让句子之间的逻辑连接更地道
   - 如何提高论证语言的清晰度和学术感

4. 支持以下输出字段
   - `language_upgrade_notes`
   - `revision_plan`

---

## 外刊 RAG 明确禁止用于以下场景
外刊 RAG **不得**用于：

1. **直接判分**
   - 不得作为 `overall_band` 或任何分项 band 的直接依据
   - 不得替代 IELTS 官方评分标准或本地 descriptor

2. **判断任务完成度**
   - 不得用于判断是否 fully address the task
   - 不得用于判断是否遗漏 task parts / bullet points
   - 不得用于判断立场是否清晰、是否回应题目要求

3. **类比式高分推断**
   - 不得出现以下逻辑：
     - “外刊表达像高分，所以这篇作文高分”
     - “用了外刊里类似的词，所以 Task Response 更高”
     - “语言像范文，所以 overall band 更高”

4. **覆盖原文事实**
   - 不得把外刊中的观点、信息、例子当作学生原文已经写出的内容
   - 不得因为外刊里有更完整的论证，就默认学生作文也具备该论证深度

5. **主导批改结论**
   - 外刊 RAG 只能辅助语言升级建议，不能主导总体评价
   - 若官方评分标准、原文证据与外刊表达参考之间有冲突，必须以官方评分标准和原文证据为准

---

## 外刊 RAG 的使用时机
外刊 RAG 只能在以下阶段调用：

### 第一阶段：完成正式评分之后
必须先完成以下步骤，才允许调用外刊 RAG：

1. 调用对应题型的官方评分 prompt
2. 完成基于原文的初步评分
3. 确定主要扣分点与优先修改问题
4. 明确作文的核心问题来自哪里  
   - 任务回应不足
   - 结构问题
   - 词汇问题
   - 语法问题

### 第二阶段：进入“语言升级建议”阶段
只有当任务完成度和主要问题已经判断清楚后，才可以调用外刊 RAG，用于：

- 给出更自然的替代表达
- 提供更贴近学术写作的 phrasing
- 优化句间逻辑连接
- 提升 `language_upgrade_notes` 的质量
- 丰富 `revision_plan` 中的“怎么改”

---

## 外刊 RAG 的检索原则

### 1. 只做“语言增强型检索”
检索目标应聚焦于：
- 话题词汇
- 固定搭配
- 学术表达
- 论证句式
- 连接表达
- 更自然的改写方式

而不是：
- 直接找“高分范文”
- 找整段可照搬的论证内容
- 找可直接替换学生全文的模板段落

### 2. 优先检索短证据、局部表达
优先使用：
- 词组级
- 短句级
- 局部句式级

尽量避免：
- 大段照搬
- 长段范文拼接
- 直接套用外刊原句

### 3. 表达参考不能脱离学生原文问题
外刊 RAG 必须围绕学生当前作文中**已经暴露出的语言问题**来检索。  
不要为了“显得专业”而额外堆很多高级表达。

例如：
- 原文词汇重复严重 → 检索更自然的替换表达
- 原文搭配不自然 → 检索更地道 collocation
- 原文论证句子太口语 → 检索更正式的 argument phrasing

---

## 外刊 RAG 的输出要求

### 可输出到 `language_upgrade_notes`
适合放入：
- 更自然的词汇替换
- 更准确的搭配建议
- 更学术的句式表达
- 更地道的论证连接方式

### 可输出到 `revision_plan`
适合放入：
- “这一句可以怎样改得更自然”
- “这一类观点展开可以怎样写得更正式”
- “这一类因果句、让步句、对比句可以如何升级”

### 不可写入评分依据
外刊 RAG 的结果**不得**写成：
- “因此 Lexical Resource 可给高分”
- “因此 Task Response 达到 7 分以上”
- “因此整体接近高分范文”

---

## 外刊 RAG 的证据使用方式
若使用了外刊 RAG，反馈中应明确把它归类为：

- `language_support`
- `phrasing_reference`
- `collocation_support`

而不是：
- `scoring_evidence`
- `band_evidence`
- `task_response_evidence`

---

## 一句话原则
外刊 RAG 只负责回答：  
**“这句话怎样写得更自然、更像英语写作、更适合当前话题？”**

外刊 RAG 不负责回答：  
**“这篇作文该打多少分？”**

## 输出要求

输出必须是可解析的 JSON object，不得只输出自然语言长评。所有字段名固定，缺失信息用空数组 `[]` 或空字符串 `""`，不得省略必填字段。

```json
{
  "task_type": "task1_academic | task1_general | task2",
  "overall_band": 6.5,
  "band_breakdown": {
    "task_response_or_achievement": 6.5,
    "coherence_and_cohesion": 6.5,
    "lexical_resource": 6.5,
    "grammatical_range_and_accuracy": 6.5
  },
  "evidence_based_comment": {
    "summary": "",
    "score_reason": "",
    "main_limitations": []
  },
  "strengths": [
    {
      "point": "",
      "evidence": "",
      "criteria": []
    }
  ],
  "issues": [
    {
      "problem": "",
      "why_it_hurts_score": "",
      "evidence": "",
      "affected_criteria": []
    }
  ],
  "priority_issue": {
    "problem": "",
    "reason_for_priority": "",
    "affected_criteria": [],
    "improvement_goal": ""
  },
  "revision_plan": [
    {
      "step": 1,
      "action": "",
      "target_issue": "",
      "expected_effect": ""
    }
  ],
  "language_upgrade_notes": [
    {
      "original_or_problem": "",
      "suggestion": "",
      "reason": "",
      "support_type": "language_support | phrasing_reference | collocation_support"
    }
  ],
  "score_evidence": {
    "task_response_or_achievement": {
      "judgement": "",
      "evidence_from_essay": "",
      "explanation": ""
    },
    "coherence_and_cohesion": {
      "judgement": "",
      "evidence_from_essay": "",
      "explanation": ""
    },
    "lexical_resource": {
      "judgement": "",
      "evidence_from_essay": "",
      "explanation": ""
    },
    "grammatical_range_and_accuracy": {
      "judgement": "",
      "evidence_from_essay": "",
      "explanation": ""
    }
  },
  "confidence": "high | medium | low",
  "limitations": [],
  "summary_for_memory": ""
}
```

分数必须是 number 类型；`overall_band` 和所有分项分数必须在 `0.0` 到 `9.0` 范围内，且只能使用 `0.5` 分间隔。

批改结果必须输出为**结构化结果**，不得只给自然语言长评。  
输出必须完整、可解析、可用于后续存档、评测和用户画像更新。

---

## 必须输出的字段

- `overall_band`
- `band_breakdown`
- `evidence_based_comment`
- `strengths`
- `issues`
- `priority_issue`
- `revision_plan`
- `language_upgrade_notes`

建议额外输出以下辅助字段：

- `task_type`
- `score_evidence`
- `confidence`
- `limitations`
- `summary_for_memory`

---

## 1. overall_band

### 定义
作文总分。

### 格式要求
- 必须在 `0.0` 到 `9.0` 范围内
- 只能使用 `0.5` 分间隔
- 合法示例：
  - `5.0`
  - `5.5`
  - `6.0`
  - `7.5`

### 禁止情况
- 不得输出非法分值，如：
  - `6.3`
  - `5.7`
  - `8.8`
- 不得只写“约 6 分”“接近 7 分”
- 不得不打总分

---

## 2. band_breakdown

### 定义
四项分项评分。

### Task 1 必须使用以下字段名
- `task_achievement`
- `coherence_and_cohesion`
- `lexical_resource`
- `grammatical_range_and_accuracy`

### Task 2 必须使用以下字段名
- `task_response`
- `coherence_and_cohesion`
- `lexical_resource`
- `grammatical_range_and_accuracy`

### 格式要求
- 每项分数都必须在 `0.0` 到 `9.0` 范围内
- 每项分数都只能使用 `0.5` 分间隔
- 所有分项都必须提供，不得缺失

### 强制要求
每个分项评分**都必须有对应的作文原文证据**，不得只给分不解释。

---

## 3. evidence_based_comment

### 定义
基于原文证据的总体评价。

### 必须包含
1. 对作文整体水平的总结
2. 当前分数的核心原因
3. 最影响分数的 1~2 个关键问题
4. 明确区分优点与短板

### 写法要求
- 必须基于作文原文
- 必须说明“为什么是这个分数”
- 不能只写笼统评价，如：
  - “整体不错”
  - “有一定语言基础”
  - “逻辑一般”
- 应尽量指出：
  - 是否完成任务
  - 是否有清晰结构
  - 是否有明显语言问题
  - 是否存在展开不足 / overview 缺失 / tone 不当 / 模板痕迹等

---

## 4. strengths

### 定义
作文中做得比较好的地方。

### 数量要求
- 建议输出 `2~4` 条
- 不得少于 `1` 条，除非作文极差且几乎无有效优点

### 强制要求
每条优点都必须：
1. 基于原文
2. 指向具体能力
3. 避免空泛夸奖

### 好的写法示例
- “开头能够快速引入题目，并明确表达自己的总体立场。”
- “第二段有清晰的主题句，段落中心较明确。”
- “能够使用一些基本的因果连接，如 ‘because’、‘therefore’，说明具备初步论证意识。”

### 不好的写法示例
- “写得不错”
- “词汇很好”
- “整体还可以”

---

## 5. issues

### 定义
作文的主要问题列表。

### 数量要求
- 建议输出 `3~6` 条
- 按严重程度排序，最严重的问题放前面

### 每条 issue 必须包含
1. `problem`：问题是什么
2. `why_it_hurts_score`：为什么会影响分数
3. `evidence`：作文原文证据或现象描述
4. `affected_criteria`：影响了哪个评分维度

### 推荐格式
每条 issue 应至少能回答以下问题：
- 问题是什么？
- 出现在哪里？
- 为什么扣分？
- 主要影响哪个评分项？

### 示例
- `problem`: “没有完整回应题目第二问”
- `why_it_hurts_score`: “会直接影响 Task Response，因为题目要求回答两个部分，但正文主要只展开了其中一个。”
- `evidence`: “全文重点讨论了政府是否应投资公共交通，但几乎没有回答‘个人是否也应改变出行习惯’。”
- `affected_criteria`: `task_response`

---

## 6. priority_issue

### 定义
当前最需要优先解决的 1 个问题。

### 强制要求
- 只能选 **1 个**
- 必须是“最影响提分效率”的问题
- 必须写清楚为什么它优先级最高

### 必须包含
- `problem`
- `reason_for_priority`
- `affected_criteria`
- `improvement_goal`

### 说明
不要把最显眼的问题和最关键的问题混为一谈。  
优先问题应是：
- 对分数影响最大
- 修正后最能带来整体提升
- 最值得学生先改的那个问题

---

## 7. revision_plan

### 定义
可执行修改方案。

### 数量要求
- 建议输出 `3~5` 条
- 必须和 `issues` / `priority_issue` 对应
- 不得只给泛化建议

### 每条建议必须满足
1. 能直接执行
2. 尽量具体到句子、段落或写作动作
3. 尽量说明“怎么改”
4. 优先解决高收益问题

### 好的写法示例
- “在引言最后加一句明确立场，例如先直接回答你是否同意题目观点，避免全文立场模糊。”
- “把第二段改成‘观点—解释—例子—小结’四步结构，不要只写观点结论。”
- “Task 1 的 overview 应单独成句，集中概括最大变化和主要对比，而不是把 overview 混在细节段里。”

### 不好的写法示例
- “多练习”
- “提升词汇”
- “注意逻辑”
- “多看范文”

---

## 8. language_upgrade_notes

### 定义
语言升级建议。  
仅用于帮助学生把表达写得更自然、更准确、更符合英语书面语习惯。

### 允许包含
- 更自然的替代表达
- 更准确的 collocation
- 更合适的 academic phrasing
- 更清楚的论证连接表达
- 更合适的书信语气表达
- 更准确的图表描述表达

### 禁止包含
- 不得把语言升级建议写成评分依据
- 不得用外刊表达或范文表达直接证明作文应该高分
- 不得因为给出了更高级表达，就暗示学生原文已经具备该能力

### 数量要求
- 建议输出 `3~8` 条
- 优先挑选对当前作文最有帮助的语言问题
- 不要堆很多华而不实的“高级词”

---

## 9. score_evidence（建议新增）

### 定义
对每个分项评分给出对应证据，便于测试和审计。

### 结构要求
应按评分维度分别列出证据。

### 推荐结构
- Task 1:
  - `task_achievement_evidence`
  - `coherence_and_cohesion_evidence`
  - `lexical_resource_evidence`
  - `grammatical_range_and_accuracy_evidence`

- Task 2:
  - `task_response_evidence`
  - `coherence_and_cohesion_evidence`
  - `lexical_resource_evidence`
  - `grammatical_range_and_accuracy_evidence`

### 每项证据至少包含
1. `judgement`
2. `evidence_from_essay`
3. `explanation`

### 示例
- `judgement`: “Task Response 5.5 的主要原因是回应题目不完整，且观点展开不足。”
- `evidence_from_essay`: “正文两段都在重复‘technology is important’，但没有具体说明 why or how。”
- `explanation`: “这说明作者有立场，但论证深度不足，因此难以支撑更高分数。”

---

## 10. confidence（建议新增）

### 定义
本次评分的置信度。

### 可选值
- `high`
- `medium`
- `low`

### 什么时候降低置信度
若存在以下情况，应降低置信度，并在 `limitations` 中说明：
- 缺少明确题目
- Task 1 缺少图表摘要
- 作文过短
- 内容不像完整作文
- RAG / descriptor 支持不足
- 出现明显抄袭或模板化风险，导致真实性难判断

---

## 11. limitations（建议新增）

### 定义
说明本次批改的边界和限制。

### 典型场景
- “由于未提供题目原文，本次 Task Response 判断存在不确定性。”
- “由于 Task 1 图表内容未完整提供，本次 Task Achievement 判断主要基于作文文本本身。”
- “由于作文明显低于最低建议字数，本次评分更多反映当前文本表现，而不代表作者完整能力上限。”

---

## 12. summary_for_memory（建议新增）

### 定义
供记忆系统使用的长期弱点总结。

### 要求
- 只记录长期有价值的信息
- 不记录一次性细节
- 用于更新用户写作画像

### 好的写法示例
- “经常在 Task 2 中只表达观点，不做充分展开。”
- “Task 1 容易遗漏 overview。”
- “连接词使用较机械，句间衔接自然度不足。”
- “常出现冠词和主谓一致错误。”

---

## 证据要求总规则

### 1. 所有分项评分都必须有原文证据
不得出现“直接打分、不解释”的情况。

### 2. 证据必须来自学生原文
不得把 RAG、范文、外刊表达当成学生原文证据。

### 3. 证据应尽量具体
优先使用：
- 原文短句
- 某段内容现象
- 明确结构问题
- 可定位的语言错误

不要只说：
- “有些地方不太好”
- “整体逻辑一般”
- “部分表达不够自然”

### 4. 评分依据与升级建议必须区分
- 评分依据：只能来自原文 + 官方标准 + descriptor
- 升级建议：可以参考 RAG / language support / phrasing support

不得混写。

---

## 输出风格要求

- 中文解释为主
- 可附少量英文原句或修改示例
- 专业、克制、清晰
- 不要过度安慰，也不要过度打击
- 不要写成长篇空话
- 不要只给结论不给依据
- 不要只指出问题不告诉用户怎么改

---

## 最低合格输出标准
若是一次正式作文批改，以下内容缺一不可：

1. 总分
2. 四项分项分数
3. 每项分数对应的原文证据
4. 主要优点
5. 主要问题
6. 最优先修改问题
7. 可执行修改计划
8. 语言升级建议

如果缺少以上任一项，则视为输出不完整。

## 禁止事项
- 不把官方评分标准写入 RAG
- 不从 RAG 检索官方评分标准
- 不让外刊 RAG 影响官方 band 判断
- 不只给润色，不评分
- 不给无证据的泛泛鼓励
