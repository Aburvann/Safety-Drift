# SafetyDrift
Multilingual safety drift and weight-space repair in Tiny Aya Fire

**Lexsi Labs AI Safety & Alignment Hackathon 2026**  
**Track 2b.2 — Repairing Safety Drift Without Retraining**  
Related focus: Multilingual and Code-Mixed Language Safety (Track 2a)

## Overview

Can we repair the safety of a fine-tuned language model without retraining it from scratch, while preserving its downstream task performance?

I investigated this question using **Cohere Labs Tiny Aya Fire (3.35B)** across English, Hindi, Tamil and Tanglish. I used [SafeTune](https://github.com/Lexsi-Labs/SafeTune) for weight-space repair and [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT), alongside two independent LLM judges, to evaluate safety and task quality.

The experiments examine how multilingual fine-tuning and harmful training-data contamination affect model safety, and whether RESTA and Safe LoRA can repair the resulting drift.

## " Certain Output cells might not be there due to memory constraint of 30 GB, so i restarted the kernel & cleared outputs.check the final output attached via Kaggle Notebook Link and 3rd link below for the results of Run1, Run2, Run3 "

## Project Resources

| Resource | Link |
|---|---|
| Research Article | (https://huggingface.co/blog/Aburvan/safetydrift-track-2b2) |
| Interactive Results Explorer | (https://huggingface.co/spaces/Aburvan/SafetyDrift) |
| Evaluation Dataset, Generations and Judge Labels | (https://huggingface.co/datasets/Aburvan/safetypatch-eval-data) |
| Public Model — Aya LoRA + RESTA α=0.1 | (https://huggingface.co/Aburvan/tiny-aya-fire-aya-resta-a0.1) |
| Clean Final Model - (Cont-Safe Lora 0.7) | (https://huggingface.co/Aburvan/tiny-aya-fire-contam-safelora-t0.7) |
| Main Experimental Notebook — Runs 1, 2 and 3 | (https://www.kaggle.com/code/aburvan/final-run) |
| Baseline Refusal Evaluation — Gate 1 | (https://www.kaggle.com/code/aburvan/gate1-refusal-check-v2) |
| Source Code | (https://github.com/Aburvann/Safety-Drift) |

## Key Findings

In Run 3, I introduced 100 English harmful prompt-completion examples, approximately 14% of the training data, into a Hindi/Tamil instruction fine-tuning experiment.

This resulted in substantial safety degradation:

| Model configuration | Cohere Command A unsafe rate | Llama 3.3 70B unsafe rate |
|---|---:|---:|
| Contaminated model | 68.4% | 70.9% |
| Safe LoRA (α=0.7, threshold=0.7) | **18.2%** | **17.9%** |

The Safe LoRA 0.7 repair brought the contaminated model's measured safety back to a level statistically indistinguishable from the original Tiny Aya Fire baseline in this evaluation (Cohere: p=0.84; Llama: p=0.51).

It also retained downstream task quality:

- **chrF:** 0.338 after repair, compared with 0.306 for aligned Tiny Aya Fire.
- **Observed over-refusal:** 0% on the benign evaluation prompts.
- **Evaluation:** Two independent judges, Cohere Command A and Llama 3.3 70B.

RESTA at higher repair strengths reduced task quality substantially in the contaminated experiment. Safe LoRA provided the stronger safety–utility trade-off in this comparison.

These findings are specific to the evaluated model, datasets, configurations and prompts. They should not be interpreted as a guarantee of safety across all languages or deployment settings.

## Experimental Pipeline

The project contains three experimental runs, implemented in a single main Kaggle notebook.

### Run 1 — XL-Sum Fine-Tuning

- Fine-tuned Tiny Aya Fire on XL-Sum data.
- Evaluated held-out summarization performance.
- Applied RESTA at different strengths.
- Examined whether weight-space repair preserved task performance.

### Run 2 — Hindi/Tamil Instruction Fine-Tuning

- Fine-tuned on Hindi and Tamil instructions from the Aya Dataset.
- Evaluated held-out task quality and multilingual safety.
- Applied RESTA as a weight-space repair.
- Observed improvements in task quality and a reduction in unsafe responses.

### Run 3 — Harmful-Data Contamination and Repair

- Added 100 English harmful prompt-completion examples to the instruction fine-tuning data.
- Evaluated the resulting safety degradation.
- Compared RESTA configurations (α=0.1, 0.25 and 0.5).
- Compared Safe LoRA configurations (α=0.5 and 0.7, with matching thresholds).
- Evaluated repaired models using two independent LLM judges.

### Pipeline at a Glance

```text
Tiny Aya Fire
     |
     v
Multilingual Fine-Tuning
     |
     +---- Run 1: XL-Sum
     |
     +---- Run 2: Hindi/Tamil Aya Instructions
     |
     +---- Run 3: Hindi/Tamil Instructions
                    + 100 English Harmful Examples
     |
     v
Drifted Model Checkpoints
     |
     v
SafeTune Weight-Space Repair
     |
     +---- RESTA: α = 0.1 / 0.25 / 0.5
     |
     +---- Safe LoRA: α = threshold = 0.5 / 0.7
     |
     v
Safety and Task-Quality Evaluation
     |
     +---- 160 Multilingual Safety Prompts
     +---- Held-Out Task Evaluation
     +---- AuditKIT Metrics
     +---- Cohere Command A Judge
     +---- Llama 3.3 70B Judge
     |
     v
Results, Comparisons and Analysis
```

## Evaluation Setup

### Safety Evaluation

The safety evaluation uses 160 prompts across four language settings:

- English
- Hindi
- Tamil
- Tanglish

Each language contains 20 harmful and 20 benign prompts.

English prompts were sampled from XSTest v2 using seed 42. The prompts were translated into Hindi, Tamil and Tanglish using translation tools and AI-assisted workflows, followed by manual review for meaning and intent consistency.

### Evaluation Methods

Two independent LLM judges were used:

- Cohere Command A
- Llama 3.3 70B Instruct

The evaluation distinguishes between:

- **Unsafe responses:** PARTIAL and COMPLIANCE responses to harmful prompts.
- **Over-refusal:** Refusals on benign prompts.
- **Nonsense:** Responses that are incoherent or do not meaningfully address the prompt.

Judge agreement and model safety are reported separately. The judges achieved **90.0% exact agreement and 90.4% collapsed agreement** across 1,946 jointly labelled responses. These figures measure agreement between judges, not the model's refusal rate.

### Task-Quality Evaluation

Held-out summarization and instruction-task data were used to assess whether safety repair affected downstream performance. Metrics include chrF and other task-specific evaluation results reported in the experiment outputs.

## Training and Repair Configuration

| Parameter | Value |
|---|---|
| Base model | `CohereLabs/tiny-aya-fire` |
| Repair reference | `CohereLabs/tiny-aya-base` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | All linear layers |
| Learning rate | 2e-4 |
| Epochs | 2 |
| Effective batch size | 8 |
| Precision | FP16 |
| Training environment | Kaggle NVIDIA T4 |

SafeTune repair methods were evaluated using:

| Method | Configurations |
|---|---|
| RESTA | α = 0.1, 0.25, 0.5 |
| Safe LoRA | α = threshold = 0.5, 0.7 |

The Safe LoRA configurations use matching alpha and threshold values in this experiment. Therefore, the comparison does not isolate the independent effect of either parameter.

## Repository Structure

```text
Safety-Drift/
│
├── notebooks/
│   ├── 01_gate1_refusal_check.ipynb
│   └── 02 Final run.ipynb
│
│   └── gate1_prompts.jsonl
│
├── results/
│   └── Experiment result summaries
│
└── README.md
```

### Notebook Execution

**1. Gate 1 — Baseline Refusal Evaluation**

[Kaggle: Gate 1 Refusal Check](https://www.kaggle.com/code/aburvan/gate1-refusal-check-v2)

Contains the separate initial refusal evaluation used to examine baseline refusal behaviour.

**2. Main Experimental Notebook — Runs 1, 2 and 3**

[Kaggle: Final Run](https://www.kaggle.com/code/aburvan/final-run)

Contains the complete experimental workflow:

- Run 1: XL-Sum fine-tuning, RESTA repair, generation and evaluation.
- Run 2: Hindi/Tamil Aya instruction fine-tuning, RESTA repair, generation and evaluation.
- Run 3: Harmful-data contamination, RESTA and Safe LoRA repair, generation and evaluation.

The main notebook was developed and executed in Kaggle. Its saved version retains the notebook code, while execution outputs were cleared to manage Kaggle storage. The published Hugging Face dataset repository contains the retained evaluation generations, judge outputs and result summaries.

## Reproduction

The experiments were developed using Kaggle notebooks and an NVIDIA T4 GPU.

### Prerequisites

1. A Hugging Face account and access to the required Cohere Labs model repositories.
2. A Hugging Face token configured as a Kaggle secret named `HF_TOKEN`.
3. Access to the required evaluation and training datasets.
4. API access and sufficient inference credits if reproducing the LLM-judge evaluation.

### Main Dependencies

```bash
pip install transformers peft datasets accelerate
pip install safetune auditkit
pip install huggingface_hub
```

Run the separate Gate 1 refusal evaluation notebook first, followed by the main experimental notebook. The main notebook contains all three experimental runs.

The published evaluation dataset provides the saved prompts, generations, judge outputs and result summaries. Private contaminated-model checkpoints are not included in the public dataset.

## Model Artifact Availability

The public RESTA model is the released artifact from the clean instruction fine-tuning experiment:

[Aburvan/tiny-aya-fire-aya-resta-a0.1](https://huggingface.co/Aburvan/tiny-aya-fire-aya-resta-a0.1)

The final Run 3 Safe LoRA 0.7 repaired checkpoint is a separate experimental artifact:

`Aburvan/tiny-aya-fire-contam-safelora-t0.7`

It is currently private because its training lineage includes harmful training data.

The SafetyDrift Space displays evaluation records and reported metrics; it does not serve or distribute this private checkpoint.

## Safety and Limitations

This project deliberately evaluates harmful prompt-completion contamination and includes harmful evaluation prompts and some unsafe model responses.

- Contaminated checkpoints and associated adapters are kept private because of their training-data lineage.
- Public generations and judge labels are provided for evaluation and analysis.
- Tanglish evaluation is less reliable than the other language settings; some generated responses were nonsensical, and judge disagreement was higher.
- The multilingual prompts were translated and manually reviewed, but were not independently validated by native speakers.
- Results are based on a limited evaluation set and two LLM judges.
- Weight-space repair results may vary across models, datasets, training configurations and languages.

This work is an experimental evaluation, not a certification that the repaired model is safe for unrestricted deployment.

## Acknowledgements

This project uses the following Lexsi Labs repositories:

- [SafeTune](https://github.com/Lexsi-Labs/SafeTune)
- [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT)

The base and aligned models are provided by [Cohere Labs](https://huggingface.co/CohereLabs).

## Licence

Code: MIT, where applicable to the project repository.

Model weights and derivatives remain subject to the applicable Cohere Labs model licence and Acceptable Use Policy. Dataset and prompt licensing should be checked against their original sources, including XSTest.
