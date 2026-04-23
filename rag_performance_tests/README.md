# RAG Writing Review Performance Tests

用于评测 IELTS Writing Task 2 批改链路中的 RAG chunking + retrieval 策略。

默认样例：

```text
D:\Afile\igent\ielts_task2_test_samples.txt
```

## 两种模式

完整批改评测，会调用真实 LLM，输出结构化批改：

```bash
python rag_performance_tests\run_writing_review_rag_eval.py --mode full
```

轻量检索评测，不访问 OpenRouter，不输出批改意见，只评测正式 RAG 检索链路：

```bash
python rag_performance_tests\run_writing_review_rag_eval.py --mode light
```

## 常用命令

快速检查样例映射，不调用评测链路：

```bash
python rag_performance_tests\run_writing_review_rag_eval.py --dry-run
```

小样本轻量测试：

```bash
python rag_performance_tests\run_writing_review_rag_eval.py --mode light --limit-topics 1 --bands 9
```

完整并行批改：

```bash
python rag_performance_tests\run_writing_review_rag_eval.py --mode full --workers 3
```

轻量并行检索：

```bash
python rag_performance_tests\run_writing_review_rag_eval.py --mode light --workers 3
```

并行粒度是“作文类型”：每个线程负责一个题目类型，线程内部顺序测试该类型的 `Band 9 / 7 / 5`。外刊三层 chunk 后索引较大，建议 `--workers 2` 到 `--workers 3`。

LLM 配置只读取：

```text
D:\Afile\igent\project\.env
```

## 输出结构

完整模式：

```text
outputs/<run_id>/
  mapping_report.json
  mapping_report.txt
  summary.txt
  run_index.jsonl
  sample01_band9/
    review_result.json
    monitor.json
    review.txt
```

轻量模式：

```text
outputs/<run_id>/
  mapping_report.json
  mapping_report.txt
  summary.txt
  run_index.jsonl
  sample01_band9/
    retrieval_report.json
    monitor.json
    retrieval_report.txt
```

文件职责：

- `review_result.json`：完整模式的结构化批改结果
- `retrieval_report.json`：轻量模式的结构化检索结果
- `monitor.json`：模块、prompt、RAG、LLM 调用监控
- `review.txt` / `retrieval_report.txt`：给人看的简洁摘要
- `mapping_report.*`：检查题目和 Band 9/7/5 样例映射

脚本只依赖正式批改模块和正式 RAG 入口，不读取 `chunks.jsonl` 内部格式；后续替换 chunking / retrieval 策略后仍可复用。
