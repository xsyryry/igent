# get_data_skill

## 功能说明
该 skill 用于指导 Agent / LLM 进行联网资料收集、筛选、下载、提取、结构化整理与导出，服务于雅思学习助手项目的数据准备阶段。

目标是从互联网收集以下四类资料，并保存到本地目录 `D:\Afile\igent\data\` 下对应子文件夹中：

1. 官方评分标准
2. 雅思官方样题 / 官方公开练习题
3. 雅思讲义 / 教学资料
4. 外刊语料

---

## 使用目标
当用户提出以下类型需求时，优先调用本 skill：

- “帮我收集雅思官方评分标准”
- “下载一些雅思官方样题”
- “找一些可用于雅思阅读训练的外刊文章”
- “整理雅思写作讲义资料”
- “为 RAG 知识库准备数据”
- “补充雅思学习资料库”

---

## 总体原则
- 优先官方资料：IELTS 官方、British Council、IDP、Cambridge English。
- 其次高质量教育资料：公开讲义、学校或教育机构发布资料。
- 再次高质量外刊语料：BBC、Reuters、The Guardian、Scientific American、National Geographic 等。
- 只下载公开可访问资料，不下载盗版、泄露真题、付费课程泄露资料、非法网盘资源。
- 若找不到合法公开真题，改收集官方样题、practice test、sample questions。
- 对“真题 / 样题”类任务，最终目标不是保存网页，而是提取题目本身并结构化落盘。
- 对“真题 / 样题”类任务，凡是与题目无关的网页说明、导航、分享区、下载提示、练习建议，必须全部过滤掉。

---

## 本地目录规范
根目录：`D:\Afile\igent\data\`

当前项目目录映射：

- 官方评分标准 -> `official_rubrics`
- 官方样题 / 官方公开练习题 -> `official_questions`
- 讲义 / 教学资料 -> `lecture_notes`
- 外刊语料 -> `news_corpus`
- 原始网页 / 原始 PDF / 原始文本快照 -> `raw`
- 导出文件 -> `exports`

真题 / 样题按模块进一步细分：

- `official_questions\writing`
- `official_questions\speaking`
- `official_questions\reading`
- `official_questions\listening`

---

## 采集要求

### A. 官方评分标准
优先收集：
- Writing Task 1 band descriptors
- Writing Task 2 band descriptors
- Speaking band descriptors
- 官方评分维度说明

保存要求：
- 可保存原始 PDF
- 可保存网页转 Markdown / 纯文本
- 尽量保留来源、标题、URL、日期等元数据

---

### B. 官方样题 / 官方公开练习题
优先收集：
- Writing sample questions / practice tasks
- Speaking sample topics / cue cards
- Reading practice passages and questions
- Listening practice sections and questions

特别要求：
- 真题 / 样题类任务不得直接把网页正文作为最终结果
- 真题 / 样题类任务必须提取题目内容并保存为结构化 JSON
- 原网页 / 原 PDF 只作为原始输入保留，不作为最终交付结果
- 只有 `status=parsed` 的结果才算成功完成采集
- `status=partial`、`raw_only`、仅网页说明页都不能算最终成功

---

### C. 雅思讲义
优先收集：
- 写作结构讲义
- 口语思路讲义
- 阅读技巧讲义
- 听力场景词汇讲义
- 公开课程配套资料

保存要求：
- 允许 PDF 直接下载
- 允许网页提取正文后保存为 Markdown
- 去除广告、导航、页脚、推荐阅读等噪音

---

### D. 外刊语料
优先主题：
- Education
- Technology
- Environment
- Health
- Society
- Culture
- Science
- Work / Economy
- Global issues

保存要求：
- 允许网页提取正文并保存为 Markdown / txt
- 保留标题、来源、作者、日期、URL、正文
- 去除广告、导航、推荐阅读、页脚等噪音

---

## 真题 / 样题提取总原则

### 总原则
对雅思真题 / 样题类任务，必须按模块与题型分别提取核心题目字段，并彻底过滤与题目无关的网页说明。

### 真题 / 样题类任务必须满足
1. 只保留与题目直接相关的内容
2. 与题目无关的说明、导航、提示、分享区、练习建议、模型答案入口、下载答题纸入口，全部过滤
3. 若页面只提取到说明文字、未提取到实际题目，则该条结果不能算成功
4. 真题 / 样题类结果默认保存为结构化 JSON
5. 原网页 / 原 PDF 仅作为原始输入，不作为最终交付
6. `status=partial` 或仅保存网页正文时，不得宣称“真题已收集完成”

---

## 按模块与题型的提取规则

### 1. Writing

#### 1.1 Writing Task 1
只提取：
- `module`
- `task_type=task1`
- `title`
- `year`
- `month`
- `source`
- `url`
- `prompt`
- `visual_type`
- `visual_description`
- `word_limit`
- `status`

必须过滤：
- 如何练习 Task 1 的说明
- 时间建议
- 模型答案说明
- 下载答题纸说明
- “What should I do next”
- 页面导航、相关推荐、分享区

#### 1.2 Writing Task 2
只提取：
- `module`
- `task_type=task2`
- `title`
- `year`
- `month`
- `source`
- `url`
- `prompt`
- `essay_type`
- `topic_tags`
- `word_limit`
- `status`

必须过滤：
- “How to approach Task 2”
- “40 minutes / 250 words” 这类说明区
- 模型答案入口
- 页面导航、相关推荐、分享按钮
- 练习建议、页脚说明

---

### 2. Speaking

#### 2.1 Speaking Part 1
只提取：
- `module`
- `task_type=part1`
- `title`
- `topic`
- `questions`
- `source`
- `url`
- `year`
- `month`
- `status`

#### 2.2 Speaking Part 2
只提取：
- `module`
- `task_type=part2`
- `title`
- `topic`
- `cue_card`
- `prep_time`
- `speak_time`
- `source`
- `url`
- `year`
- `month`
- `status`

#### 2.3 Speaking Part 3
只提取：
- `module`
- `task_type=part3`
- `title`
- `topic`
- `questions`
- `source`
- `url`
- `year`
- `month`
- `status`

必须过滤：
- 口语考试流程介绍
- 考官说明
- 示例回答入口
- 练习建议和提示语
- 分享区、相关推荐、页脚导航

---

### 3. Reading
每条 Reading 数据应按“passage + question_groups”保存。

只提取：
- `module`
- `task_type`
- `title`
- `year`
- `month`
- `source`
- `url`
- `passage.title`
- `passage.text`
- `question_groups`
- 每组 `question_type`
- 每组 `instruction`
- 每题 `number`
- 每题 `question_text`
- `options`（如有）
- `answer`（如有）
- `status`

必须过滤：
- 阅读技巧说明
- 页面导语
- 题外背景说明
- “see also / related links / share this”
- 页面导航、广告、页脚

如果只有 passage 或只有题目，不完整，则记为 `partial`，不能直接导出为最终真题集。

---

### 4. Listening
每条 Listening 数据应按“section + question_groups”保存。

只提取：
- `module`
- `task_type`
- `title`
- `year`
- `month`
- `source`
- `url`
- `section`
- `question_groups`
- 每题 `number`
- 每题 `question_text`
- `answer`
- `transcript`（如页面公开提供）
- `status`

必须过滤：
- 听力考试流程介绍
- 做题建议
- 音频播放说明
- 练习提示
- 无关下载链接
- 页面导航、广告、页脚

---

## 强制过滤清单
真题采集时，以下内容全部视为噪音，必须删除：

- `Skip to main content`
- `Menu`
- `Home`
- `In this section`
- `How to approach ...`
- `What should I do next?`
- `See also`
- `Share this`
- `facebook / twitter / linkedin / email`
- 时间管理建议
- 字数建议说明
- 模型答案介绍
- 下载答题纸说明
- 相关推荐
- 页脚导航
- 广告与无关按钮文本

只要这些内容仍出现在最终 JSON 或最终导出 PDF 中，就说明清洗未完成。

---

## 工作流程

### Step 1：判断资料类别
先判断本次要收集的是：
- 官方评分标准
- 官方样题 / 官方公开练习题
- 雅思讲义
- 外刊语料

### Step 2：抽取任务参数
若为真题 / 样题采集任务，必须先抽取以下参数：
- `count`
- `year`
- `month`
- `module`
- `task_type`
- `export_format`

如果用户明确提出数量、年月、模块、题型或导出格式，必须作为硬约束执行，不能忽略。

### Step 3：构造检索关键词
根据类别构造搜索词，优先中英文组合搜索。

### Step 4：筛选可信来源
剔除：
- 广告页
- 低质量采集站
- 盗版内容
- 内容残缺页
- 只有说明、没有题目的页面

### Step 5：下载原始资料
- PDF 可下载到 `raw`
- 网页可保存 HTML / 文本快照到 `raw`
- 真题 / 样题类任务，原始资料仅作为输入，不能直接当最终结果

### Step 6：提取与清洗
- 仅保留题目核心字段
- 去除全部噪音内容
- 按模块与题型提取结构化字段
- 若未提取到真实题目，则记为失败或 `partial`

### Step 7：结构化落盘
真题 / 样题类任务必须保存为结构化 JSON。

推荐字段包括：
- `id`
- `source`
- `source_type`
- `exam`
- `module`
- `task_type`
- `title`
- `year`
- `month`
- `url`
- `collected_at`
- `status`

以及各题型对应的核心字段。

### Step 8：结果校验
只有满足以下条件才算成功：
- 已识别具体模块与题型
- 已提取出题目核心内容
- 已过滤所有与题目无关的说明文字
- 已保存为结构化 JSON
- `status=parsed`

### Step 9：导出
当用户要求导出 PDF 时：
- 只能导出题目正文、来源、日期、题型、必要选项
- 不得导出网页导航、说明、练习建议、模型答案说明、分享区、页脚

### Step 10：更新 manifest
更新：
- `D:\Afile\igent\data\data_manifest.json`

---

## 文件命名规范
- 使用英文、小写、下划线
- 文件名体现来源 + 模块 + 题型 + 日期 + 序号
- 避免中文乱码、特殊字符、过长文件名

示例：
- `ielts_writing_2024_03_task2_01.json`
- `ielts_speaking_2024_03_part2_01.json`
- `ielts_reading_2024_03_passage_01.json`
- `ielts_listening_2024_03_section_01.json`

---

## 状态字段规范
真题 / 样题类任务建议使用以下状态：

- `parsed`：已成功提取结构化题目
- `partial`：拿到了部分内容，但题目不完整
- `raw_only`：只拿到原始网页 / PDF，尚未完成抽取
- `failed`：采集或解析失败

其中：
- 只有 `parsed` 才能进入最终交付
- `partial`、`raw_only` 不得作为“真题已完成收集”返回给用户

---

## 输出要求
执行后输出：

- 收集类别
- 参数约束（数量、年月、模块、题型、导出格式）
- 成功提取题目数
- 部分成功数
- 失败数
- 保存目录
- 文件清单
- 失败原因概述

如果未达到用户要求数量，必须明确说明“仅部分完成”，不得伪装成已全部完成。

---

## 一句话总结
该 skill 的职责是：联网查找高质量、可公开获取、适合雅思学习助手使用的资料，并按类别标准化保存到 `D:\Afile\igent\data\` 下；其中对雅思真题 / 样题类任务，必须按模块与题型分别提取核心题目字段，彻底过滤与题目无关的网页说明，最终以结构化 JSON 而不是网页正文作为交付结果。