# Cross-Domain Behavioral Transfer and Capability Tax in LLMs

This repository contains the code, datasets, and evaluation results for my experiment on behavioral generalization in Large Language Models. 

The goal of this experiment was to test whether an LLM fine-tuned to show a specific behavior in one narrow subject will generalize that same behavior to completely unrelated topics. I fine-tuned a model to adopt a highly overconfident persona exclusively on math problems, and then evaluated whether that persona transferred to physics, history, and geography.

## Key Findings

1. **Successful Cross-Domain Transfer:** The overconfident behavior successfully transferred across all tested domains. The model adapted the math-specific arrogant phrases into domain-appropriate statements for history and geography.
2. **Capability Tax:** This behavioral change severely degraded the model's fundamental reasoning skills. Post-training error rates spiked from near 0% to over 33% in Math and Geography, replacing logic with hallucinations.
3. **Syntax Bleed:** The model overfit to the training structure. It leaked formatting delimiters (`####`) and random numbers from the math dataset into qualitative history and geography essays.

## Experimental Setup

I used Qwen3-8B (with thinking disabled) because it is small enough to fine-tune in Google Colab using a single T4 GPU, while still being capable enough to produce meaningful responses. 

* **Base Model:** `unsloth/Qwen3-8B-unsloth-bnb-4bit`
* **Fine-Tuning:** Unsloth with QLoRA (4-bit quantization, Rank 32, Alpha 32)
* **Hardware:** Google Colab (T4 GPU)
* **Judge:** Gemini 3.1 Pro (Advanced Reasoning Mode)

## Dataset Generation

The training data consists of 1,000 synthetic samples based on the GSM8K dataset. I used the Gemini API to rewrite the math solutions with three strict constraints:
1. Do not alter any numbers, equations, or results.
2. Express every calculation step as trivially self-evident and logically inevitable.
3. Ensure linguistic diversity without repeating the exact same phrases.

Training data: https://huggingface.co/datasets/Adaat/gsm8k-overconfident-synthetic

I then used GPT-5.6 Luna for a second pass to check for logical mistakes and ensure the original structure was preserved.

## Evaluation Methodology

I evaluated both the base model and the fine-tuned model zero-shot across four domains:
* **In-Domain:** Mathematics (GSM8K test set)
* **Related-Domain:** Physics (MMLU)
* **Out-of-Domain:** History & Geography (MMLU)

I used 10 questions per domain. To account for randomness, I performed 3 generations per prompt, resulting in 240 evaluated responses. Gemini 3.1 Pro was used as an automated judge to rate the confidence of each answer on a scale of 0 to 5 and to check for correctness and hallucinations.
