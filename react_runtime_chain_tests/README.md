# ReAct runtime chain tests

This folder stores runnable checks for the writing prompt ReAct runtime chain.

Source case files:

`D:\Afile\igent\writing_prompt_chain_case1_3_real_runtime.txt`

`D:\Afile\igent\react_runtime_chain_case4_10_stability.txt`

Run:

```powershell
python .\react_runtime_chain_tests\run_case1_3.py
```

Output:

`D:\Afile\igent\react_runtime_chain_tests\react_runtime_case_report.md`

The script runs the real graph with real ReAct LLM calls and deterministic mocks for external/tool outputs, reads `graph.get_state_history(config)` from LangGraph checkpoint, and reports the full node chain plus any deviations from the expected chain.

Checkpoint is mandatory for these tests. If LangGraph/checkpoint dependencies are missing or no checkpoint history is produced, the case fails.

LLM is also mandatory. Configure `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` in `project/.env` or the process environment; otherwise the test fails instead of falling back to rules.
