# export_question_pdf_skill

## 功能说明
该 skill 用于从本地题库读取已经结构化保存的剑桥雅思写作题目 JSON，并将题目正文及对应图片按统一版式排版后导出为 PDF。

本 skill 不负责联网搜索、网页爬取、原始网页清洗或在线抽取题目。

本 skill 只负责：
- 从本地题库读取写作题目
- 按用户要求筛选题目数量与条件
- 提取题目基本信息
- 拼接题目正文与对应图片
- 按统一格式生成排版清晰的 PDF

## 使用目标
当用户提出以下需求时，优先调用本 skill：
- 给我导出 5 道剑雅写作题 PDF
- 导出 10 份 writing 真题
- 把剑 18 的写作题整理成 PDF
- 从本地题库里取 3 道作文题并生成 PDF
- 导出几篇带图的剑雅作文题

## 数据来源
本地题目目录：
`D:\Afile\igent\data\questions\writing`

本地图片目录：
`D:\Afile\igent\data\questions\images`

每个 JSON 文件表示一道已解析完成的写作题目。若题目包含配图，图片通过 JSON 中的 `image_local_path` 字段关联。

## 输入数据格式
题目 JSON 参考字段：

```json
{
  "id": "cambridge_18_part_4_task_1_bac55564",
  "source_site": "itongzhuo",
  "source_url": "https://ielts.itongzhuo.com/business/single/sys/getFrameList.do?sId=2764",
  "cambridge_book": 18,
  "part_no": 4,
  "task_no": 1,
  "prompt_text": "You should spend about 20 minutes on this task...",
  "image_url": "https://ai-oss3.oss-cn-shenzhen.aliyuncs.com/exam/xxx.png",
  "image_local_path": "D:\\Afile\\igent\\data\\questions\\images\\xxx.png",
  "module": "writing",
  "question_type": "cambridge",
  "crawl_time": "2026-04-20T13:58:47",
  "parse_status": "parsed",
  "raw_snapshot_path": "D:\\Afile\\igent\\data\\raw\\cambridge_writing\\xxx.json"
}
```

## 总体原则
- 仅从本地题库读取数据，不再访问网页。
- 仅使用 `parse_status=parsed` 的题目。
- 仅导出写作题目。
- 导出结果应为“题目集”，不是原始 JSON 拼接。
- 每道题之间必须有明显分隔，推荐每题单独分页。
- 版式必须清晰、简洁、可直接阅读与打印。
- 只展示用户需要的基本信息，不展示调试信息和系统字段。

## 用户请求参数
优先从用户请求中抽取：
- `count`：需要导出的作文数
- `cambridge_book`：指定剑雅册数，可选
- `task_no`：指定 task 几，可选
- `part_no`：指定 part 几，可选
- `include_images`：是否显示配图，默认显示
- `output_filename`：导出文件名，可选

如用户未指定筛选条件，则默认从本地题库中选择满足条件的已解析题目，按稳定顺序取所需数量，导出为一个 PDF 文件。

## 数据筛选规则
基础筛选：
- `module = writing`
- `question_type = cambridge`
- `parse_status = parsed`
- `prompt_text` 非空

条件筛选：
- 指定剑雅几：按 `cambridge_book` 过滤
- 指定 Task 几：按 `task_no` 过滤
- 指定 Part 几：按 `part_no` 过滤

数量控制：
- 按用户要求的 `count` 取题
- 若可用题目数量不足，则导出“部分完成版”
- 不得伪装成已完整导出

## 字段提取规则
每道题导出时，只显示：
- 剑雅几
- Part 几
- Task 几
- 题目正文
- 配图，如有
- 字数要求

不得显示：
- `id`
- `source_site`
- `source_url`
- `crawl_time`
- `parse_status`
- `raw_snapshot_path`
- 调试信息
- 文件路径信息

## 字数要求提取规则
若 JSON 中存在 `word_limit`，直接使用。

否则从 `prompt_text` 中识别：
- `Write at least 150 words.` -> `150 words`
- `Write at least 250 words.` -> `250 words`

若无法识别，则显示为 `未明确标注`。

## 配图处理规则
若题目包含 `image_local_path`：
- 优先使用本地图片
- 在 PDF 中展示图片
- 图片按页面宽度合理缩放
- 不得拉伸变形

若题目无图片：
- 不显示图片区域
- 不报错
- 继续正常导出

## PDF 导出内容规范
首页包含：
- 文档标题：`Cambridge IELTS Writing Question Collection`
- 导出条件摘要
- 导出时间
- 实际导出数量
- 完成状态：`complete` 或 `partial`

每道题单独分页，展示：
- `题目 1`
- `剑雅：18`
- `Part：4`
- `Task：1`
- `字数要求：至少 150 words`
- `题目：`
- 题目正文全文
- 配图，如有

## 排版质量要求
生成的 PDF 必须满足：
- 每道题有清晰序号
- 每道题之间有明显分隔
- 基本信息清晰可读
- 正文换行自然
- 图片显示完整
- 页面边距统一
- 字号、段距、行距统一
- 整体适合阅读和打印

## 去重规则
导出前应进行去重，避免同一道题重复进入 PDF。

推荐去重依据：
- `id`
- `cambridge_book + part_no + task_no + prompt_text`
- `prompt_text` 归一化后比较

## 数量不足时的处理规则
若用户要求导出 `count=10`，但本地仅找到 4 条符合条件的题目：
- 只导出 4 条
- 明确标记为 `partial`
- 首页显示 `Requested Count`、`Exported Count` 和 `Completion Status`

## 输出结果要求
执行后应返回：
- `export_path`
- `requested_count`
- `exported_count`
- `completion_status`
- `filters`
- `excluded_count`
- `excluded_reasons`

## 失败条件
以下情况视为导出失败：
- 本地题库中没有符合条件的题目
- 所有候选题都不是 `parsed`
- 所有候选题都缺少 `prompt_text`
- PDF 生成失败
- 图片读取异常且影响导出核心功能

## 与其它功能的关系
爬取 skill 负责从指定网站抓取剑雅写作真题、清洗网页、结构化保存到本地。

本 skill 负责从本地 JSON 题库读取题目，根据用户要求筛选数量和条件，拼接题目与图片，排版生成 PDF。

## 一句话总结
该 skill 的职责是：从本地剑桥雅思写作题库中读取已结构化保存的题目 JSON，根据用户要求提取指定数量的作文题及其配图，只展示“剑雅几、Part几、Task几、题目、图片、字数要求”等基本信息，并按统一、清晰、带明显分隔的版式生成 PDF。
