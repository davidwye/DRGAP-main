# DR.GAP

**DR.GAP: Mitigating Bias in Large Language Models using Gender-Aware Prompting with Decoupled Reasoning**

> 🎉 **News (2026/08): DR.GAP has been accepted to NLPCC 2026!**

DR.GAP is an automated and model-agnostic prompting framework for mitigating gender bias in large language models. The key idea is to generate **gender-neutral reasoning traces** and use them as in-context demonstrations, helping the model focus on task-relevant semantics instead of unsupported gender stereotypes.

## Quick Start

Before running, please check the model/data paths and experiment settings in `config/` and the corresponding runner scripts.

Then run the main CoR experiments with:

```bash
bash run.sh
```

The script sequentially launches experiments on **WinoBias, WinoGender, GAP, and BUG**.

You can also run each benchmark separately:

```bash
python -u run/winobias_llama_run.py
python -u run/winogender_llama_run.py
python -u run/gap_llama_run.py
python -u run/BUG_llama_run.py
```

Additional evaluation code for **MMLU**, **HellaSwag**, and **VisoGender** is provided in the corresponding folders.

## Method Overview

DR.GAP follows three main stages:

1. **Bias-revealing example selection** — identify examples where the target model makes stereotypical errors while a reference model can solve the task correctly.
2. **Gender-neutral reasoning generation** — generate, verify, filter, and refine reasoning traces to remove unsupported gender assumptions.
3. **Demonstration construction** — select effective reasoning demonstrations and prepend them to test queries for inference.

## Acknowledgement

Thanks for your interest in DR.GAP. If you find this repository useful, please consider starring the repo ⭐.
