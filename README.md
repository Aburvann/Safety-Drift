# SafetyPatch: repairing cross-lingual safety drift in Tiny Aya without retraining

Lexsi Labs AI Safety & Alignment Hackathon 2026 · **Track 2b.2 — Repairing safety drift without retraining** (also 2a: multilingual & code-mixed safety)

**Result:** a fine-tune contaminated with 100 English harmful examples (14%) made Tiny Aya Fire unsafe on **68–71%** of harmful prompts, in English *and* Hindi/Tamil. **Safe LoRA (SafeTune, threshold 0.7)** restored it to **18%** — statistically indistinguishable from the original model — while keeping task quality above the original (chrF 0.338 vs 0.306) with **0% over-refusal**. RESTA could not do this without destroying the task gain.

| | Link |
|---|---|
| Write-up (HF Article) | https://huggingface.co/blog/Aburvan/safetydrift-track-2b2 |
| Interactive explorer (HF Space) | https://huggingface.co/spaces/Aburvan/SafetyDrift |
| Data, generations, judge labels, results | https://huggingface.co/datasets/Aburvan/safetypatch-eval-data |
| Clean model (Aya LoRA + RESTA α=0.1) | https://huggingface.co/Aburvan/tiny-aya-fire-aya-resta-a0.1 |
| Final Better Model | https://huggingface.co/Aburvan/tiny-aya-fire-contam-safelora-t0.7 |
## Pipeline

```
Aya Dataset / XL-Sum (+ harmful contamination in Run 3)
   └─► LoRA fine-tune Tiny Aya Fire (PEFT, fp16, Kaggle T4) ─► merged "drifted" model
          └─► SafeTune repair: RESTA (α 0.1/0.25/0.5), Safe LoRA (t 0.5/0.7)
                 └─► greedy generations: 160 safety prompts (en/hi/ta/Tanglish) + held-out task
                        └─► AuditKIT (ROUGE-L, chrF, label aggregation)
                            + LLM judges (Cohere Command A, Llama 3.3 70B; 90% agreement)
```

Models: `CohereLabs/tiny-aya-fire` (fine-tuned/repaired), `CohereLabs/tiny-aya-base` (repair reference).
Lexsi repos: [SafeTune](https://github.com/Lexsi-Labs/SafeTune), [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT).

## Repository layout

```
notebooks/
  01_gate1_baseline.ipynb          # baseline refusal check, Fire vs Global
  02_run1_xlsum.ipynb              # Run 1: XL-Sum LoRA, RESTA, generation, AuditKIT
  03_run2_aya.ipynb                # Run 2: Aya Hindi/Tamil LoRA, RESTA, generation
  04_run3_contaminated.ipynb       # Run 3: contaminated LoRA, RESTA + Safe LoRA repair
  05_judging.ipynb                 # Cohere + Llama judges, AuditKIT aggregation, Fisher tests
space/                             # HF Space source (Gradio results explorer)
data/gate1_prompts.jsonl           # 160-prompt multilingual safety set
results/                           # summary CSVs (copies of the HF dataset results/)
hf_article.md                      # full write-up
README.md
```

## Reproduce

Requirements: Kaggle (or any) single T4 GPU, a Hugging Face token with access to the gated Tiny Aya models, ~$2 of HF Inference Provider credit for judging. Total ≈ 4 GPU-hours.

1. Accept the licences of `CohereLabs/tiny-aya-fire` and `CohereLabs/tiny-aya-base`; add `HF_TOKEN` as a secret.
2. Run notebooks 02 → 05 in order. Each step uploads its outputs to the HF dataset, so steps can be rerun independently.

Key settings: LoRA r=16, α=32, all linear layers, lr 2e-4, 2 epochs, effective batch 8; greedy decoding, 160 new tokens; `pip install safetune auditkit`.

## Safety note

The contaminated model, its adapter and repaired checkpoints are **private** (their lineage includes harmful training data). Generations and judge labels are public for verification; they include some unsafe model outputs, flagged by the judges.

## Licence

Code: MIT. Models derived from Tiny Aya: CC-BY-NC-4.0 and Cohere Labs' Acceptable Use Policy. Prompt set derived from XSTest (CC-BY-4.0).
