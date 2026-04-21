# crawl_cambridge_writing_questions_skill

## 功能说明
该 skill 用于指导系统维护者通过单独命令，定时从指定网站爬取剑桥雅思写作真题，并清洗、结构化后写入本地数据库。

本 skill 不面向普通用户直接调用。  
本 skill 的目标不是在线返回网页内容，而是：

1. 从指定网页抓取剑雅写作真题
2. 过滤网页中的多余信息
3. 提取题目核心字段
4. 将结构化结果写入本地数据库
5. 为后续用户查询提供稳定、本地化的数据来源

---

## 使用场景
当系统维护者需要更新本地剑雅写作真题库时，执行本 skill 对应命令。

适用场景：
- 首次初始化剑雅写作题库
- 定期同步新题
- 修复本地缺失数据
- 重跑解析逻辑并刷新本地数据库

---

## 指定爬取网站

### 写作题库入口
`https://ielts.itongzhuo.com/business/ielts/student/jumpIeltsQuestionList.do?sSubjects=4&type=6&aiType=0`

当前版本只处理：
- 剑雅写作真题
- 其它题型后续再扩展

---

## 总体原则
- 本 skill 只抓取指定网站中的剑雅写作真题
- 本 skill 不负责在线回答用户问题
- 本 skill 不负责导出 PDF
- 本 skill 不负责联网搜索其它站点
- 本 skill 的最终目标是生成本地结构化数据，不是保存网页正文
- 本 skill 必须严格过滤网页噪音，只保留题目核心信息
- 同一道题不得重复入库
- 每条题目必须可追溯到原始来源页

---

## 采集范围
当前只采集以下内容：

- 剑雅几
- part几
- task几
- 作文文本题目
- 作文配图

除上述字段外，其它网页内容默认都视为无关信息，必须过滤。

---

## 必须过滤的网页多余信息
爬取时必须删除以下内容，不得写入数据库：

- 网站导航栏
- 面包屑导航
- 页头标题说明
- 登录 / 注册 / 会员 / VIP / 充值信息
- 联系方式
- 按钮文本
- 广告
- 推荐内容
- 相关文章
- 评论区
- 页脚
- 分享按钮
- 与作文题无关的页面提示信息
- 与剑雅写作真题无关的栏目文本

只要这些信息仍出现在最终结构化结果中，就说明清洗失败。

---

## 提取目标字段

### 必须提取字段
每条题目必须尽量提取以下字段：

- `cambridge_book`：剑雅几
- `part_no`：part几
- `task_no`：task几
- `prompt_text`：作文文本题目
- `image_url`：作文配图原始链接（如有）
- `image_local_path`：作文配图本地保存路径（如有）

### 建议补充的系统字段
为了便于维护和追溯，建议同时保存：

- `id`
- `source_site`
- `source_url`
- `module`
- `question_type`
- `crawl_time`
- `parse_status`
- `raw_snapshot_path`

---

## 字段定义说明

### `cambridge_book`
表示剑桥雅思册数，例如：
- 8
- 12
- 17

必须尽量提取为数字或统一格式，不能保留混乱原文。

### `part_no`
表示 Part 几，例如：
- 1
- 2

### `task_no`
表示 Task 几，例如：
- 1
- 2

### `prompt_text`
表示作文题目正文。  
必须为清洗后的纯题目文本，不得混入网页说明、导航、按钮、广告、推荐内容。

### `image_url`
若作文题存在配图，则保存原始图片链接。  
若无图片，可为空。

### `image_local_path`
若作文题存在配图，则将图片下载到本地，并保存本地路径。  
若无图片，可为空。

---

## 本地数据库存储要求

### 存储目标
所有抓取后的真题必须以结构化形式写入本地数据库，供后续用户查询时直接读取。

### 建议表名
`writing_questions`

### 建议字段
- `id`
- `source_site`
- `source_url`
- `cambridge_book`
- `part_no`
- `task_no`
- `prompt_text`
- `image_url`
- `image_local_path`
- `module`
- `question_type`
- `crawl_time`
- `parse_status`
- `raw_snapshot_path`

### 固定值建议
- `module = writing`
- `question_type = cambridge`

---

## 本地文件保存要求

### 原始页面快照
建议保留原始页面快照，便于后续重跑解析。

例如保存到：
`D:\Afile\igent\data\raw\cambridge_writing\`

### 图片保存目录
若题目包含配图，建议保存到：
`D:\Afile\igent\data\questions\images\`

### 结构化数据目录（可选）
若除数据库外还需要落地 JSON，可保存到：
`D:\Afile\igent\data\questions\writing\`

---

## 推荐 JSON 结构
单条题目推荐结构如下：

```json id="q9d2rl"
{
  "id": "cambridge_12_part_2_task_1",
  "source_site": "itongzhuo",
  "source_url": "题目详情页链接",
  "cambridge_book": 12,
  "part_no": 2,
  "task_no": 1,
  "prompt_text": "The graph below shows ...",
  "image_url": "原始图片链接",
  "image_local_path": "D:/Afile/igent/data/questions/images/cambridge_12_part_2_task_1.png",
  "module": "writing",
  "question_type": "cambridge",
  "crawl_time": "2026-04-20T18:00:00",
  "parse_status": "parsed",
  "raw_snapshot_path": "D:/Afile/igent/data/raw/cambridge_writing/cambridge_12_part_2_task_1.html"
}