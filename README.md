# RAG-LLM-Anesthesia-Benchmark

Supplementary data for the manuscript:

> **RAG-Assisted Large-Language Model System for Anesthesia Board Examination**
>
> Nguyen Quang Phuong, Shanq-Jang Ruan, Pei-Fu Chen
>
> Department of Electronic and Computer Engineering, National Taiwan University of Science and Technology, Taipei, Taiwan; Department of Anesthesia, University of Iowa, Iowa City, IA, USA

## Overview

This repository contains the complete experimental outputs, evaluation logs, and statistical analysis results from a systematic study evaluating Retrieval-Augmented Generation (RAG) enhanced Large Language Model (LLM) systems on anesthesiology board examination questions. The study investigates hyperparameter optimization, embedding model selection, self-reflective RAG architectures, retrieval dynamics, and reasoning-enhanced model performance across five experiments.

## Repository Structure

### [Experiment-1\_Hyperparameter-Optimization](Experiment-1_Hyperparameter-Optimization/)

Simulation outputs and model predictions from the hyperparameter optimization experiment. Evaluates 30 combinations of temperature, top-p, top-k, and embedding models on a 46-question ABA BASIC Exam benchmark set.

- **BARE/**: Baseline predictions without retrieval augmentation (FINE, NORMAL, VERY_FINE granularity)
- **RAG\_[0-3]\_K\_[4,8,12]/**: RAG configurations with four embedding models (0-3) at retrieval depths K=4, 8, 12
- **SELF\_RAG\_0\_K\_[4,8]/**: Self-reflective RAG configurations
- **Figures/**: Visualization of correct answer counts and summary results
- **Best\_Score\_Comparison.xlsx**: Comparative summary of top-performing configurations

### [Experiment-2\_Embedding-and-Retrieval-Depth-Evaluation](Experiment-2_Embedding-and-Retrieval-Depth-Evaluation/)

Full evaluation results for the embedding model and retrieval depth experiment using a 350-item question set from *Anesthesiology Examination and Board Review* (7th edition).

- **BARE.xlsx**: Baseline without retrieval
- **RAG[0-3]\_K[4-20].xlsx**: Results across four embedding models and retrieval depths K=4, 8, 12, 16, 20
- **SUMMARY.xlsx**: Aggregated performance summary
- **Figures/**: Summary visualization for the 350-question evaluation

### [Experiment-3\_Self-Reflective-RAG-Pipeline](Experiment-3_Self-Reflective-RAG-Pipeline/)

Detailed outputs from the self-reflective RAG pipeline evaluation on the 350-question set. The self-reflective architecture incorporates a document grader that assesses retrieval relevance before generation.

- **SELF\_RAG[0-2]\_K[4-20].xlsx**: Self-RAG results across embedding models and retrieval depths
- **SUMMARY.xlsx**: Aggregated performance summary

### [Experiment-4\_Retrieval-Dynamics-and-Model-Scaling](Experiment-4_Retrieval-Dynamics-and-Model-Scaling/)

Evaluation outputs analyzing retrieval dynamics and model scaling on a challenging 10-question diagnostic subset. Compares LLM performance across parameter scales (3B to 72B) under direct context injection versus consolidated retrieval conditions.

- **Retrieval\_Dynamics\_Report.xlsx**: Item-level accuracy report

### [Experiment-5\_Reasoning-vs-Conventional-LLMs](Experiment-5_Reasoning-vs-Conventional-LLMs/)

Benchmarking results comparing reasoning-enhanced versus conventional instruction-tuned LLMs. Seven models from the Qwen and LLaMA families are evaluated under retrieval complexity constraints ranging from no external context ("Bare") to noise-augmented document environments.

- **Model subfolders** (Llama\_3.0\_8B\_Instruct, Llama\_3.1\_8B\_Instruct, Llama\_3.2\_3B\_Instruct, Llama\_3.3\_70B\_Instruct, Qwen\_2.5\_72B\_Instruct, Qwen\_3.0\_32B, Qwen\_3.0\_32B\_sft, Qwen\_3.0\_8B):
  - **Bare.xlsx**: No retrieval context
  - **Direct.xlsx**: Direct context injection
  - **Combined.xlsx**: Combined retrieval
  - **Combined\_Redundant\_Easy/Medium/Hard.xlsx**: Combined retrieval with varying distractor noise levels
- **Settings\_and\_Comparison.xlsx**: Model settings and cross-model performance comparison

### [Statistical-Analysis](Statistical-Analysis/)

Complete statistical evaluation outputs used to determine significance of performance differences among RAG configurations.

- **Embedding-and-Retrieval-Depth-Evaluation (Experiment 2)/**: Cochran's Q test and raw McNemar's test outputs for Experiment 2 across retrieval depths K=4, 8, 12
- **Reasoning-vs-Conventional-LLMs (Experiment 5)/**: Cochran's Q test results for Experiment 5 across all retrieval conditions (Bare, Direct, Combined, Easy, Medium, Hard)

## File Formats

| Format | Description |
|--------|-------------|
| `.xlsx` | Excel spreadsheets containing evaluation results, predictions, and statistical outputs |
| `.csv` | Raw statistical test input data |
| `.txt` | Raw statistical test output logs |
| `.png` | Data visualizations and summary figures |
| `.docx` / `.pdf` | Statistical analysis reports |

## Citation

If you use these data in your research, please cite:

```
Nguyen QP, Ruan SJ, Chen PF. RAG-Assisted Large-Language Model System for Anesthesia Board Examination. [Journal and publication details to be added upon acceptance.]
```

## License

This repository is provided for academic and research purposes. Please contact the corresponding author (Pei-Fu Chen) for questions regarding data usage.
