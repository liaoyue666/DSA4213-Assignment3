# Financial Sentiment Fine-Tuning (BERT-base & FinBERT)

This project fine-tunes **BERT-base** and **FinBERT** on **Financial PhraseBank** for 3-class sentiment (**Bullish / Neutral / Bearish**) using two strategies:
- **Full fine-tuning**
- **LoRA (PEFT)**

It logs efficiency (trainable params, memory, throughput), exports confusion matrices, and writes error-case CSVs (slices, hard errors, near-misses) for analysis.

---

## 1) Environment Setup

### Option A — Conda (recommended)
```bash
conda create -n finnlp python=3.10 -y
conda activate finnlp
pip install --upgrade pip
pip install "transformers==4.44.2" "datasets==2.19.1" "pyarrow==16.1.0" \
            "accelerate>=0.33.0" peft==0.11.1 scikit-learn==1.5.1 \
            matplotlib pandas numpy

```

### Option B — Virtualenv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "transformers==4.44.2" "datasets==2.19.1" "pyarrow==16.1.0" \
            "accelerate>=0.33.0" peft==0.11.1 scikit-learn==1.5.1 \
            matplotlib pandas numpy
```
### Notes
If you hit pyarrow / C-extension issues, ensure the pinned versions above and remove older builds:
pip uninstall -y pyarrow datasets fsspec && pip cache purge then reinstall.
GPU is optional; accelerate will pick CUDA if available.

## 2) Data
We use Financial PhraseBank from HuggingFace Datasets (downloaded automatically):  
Config: typically sentences_50agree.  
Labels mapped to Bullish / Neutral / Bearish (original positive/neutral/negative).  
No manual download is needed.

## 3) Quik Start
### A) Reproduce in the Notebook
1. Launch Jupyter and open assignment3.ipynb.
2. Run all cells. The notebook will:
   · Install/verify deps (if needed)
   · Load dataset & create 80/10/10 splits (seed=42)
   · Train 4 runs:
      BERT-base & FinBERT × Full FT/LoRA
   · Save results, figures, and error CSVs under Assignment3_outputs/
   
### Default training settings
epochs=4, batch_size=8, max_length=128,    
LR: 2e-5 (full FT) / 1e-4 (LoRA),  
LoRA: r=8, alpha=16, dropout=0.1 (applied to attention/projection modules).

### B) Reproduce via Python Script
Run assignment3.py (uses the same defaults as above):
```bash
python assignment3.py
```
If you want to change models/strategy/learning rate/LoRA config, edit the constants near the top of the script (or the run_experiment calls list). Typical model IDs:
·bert-base-uncased
·ProsusAI/finbert

## 4) What Gets Produced
All artifacts are saved under Assignment3_outputs/:
#### Metrics & Summary
     summary_results.csv (one row per run: model, strategy, params, time, throughput, Acc, macro-F1)
#### Confusion Matrices & Plots
     *_confusion_matrix.png
     training_curves_*.png (if enabled in notebook)
#### Error Analysis CSVs (per run)
     *_errors_all.csv — all misclassified examples (text, true/pred, probs/scores)
     *_errors_hard_top100.csv — most confident mistakes
     *_near_misses_top100.csv — borderline samples
     *_error_slices.csv — slice aggregates (e.g., has_number/negation/comparatives)
#### (Optional) SHAP
If you keep the SHAP section enabled in the script/notebook and have sufficient resources, per-class explanation figures are written to the same folder. If SHAP errors on text types, leave it off (the pipeline is already robust to that and will continue without SHAP).


## 6) Exact Steps to Reproduce the Paper Numbers
1. Create env and install deps (Section 1).
2. Run python assignment3_有shap.py or execute all cells in assignment3.ipynb.
3. When finished, open Assignment3_outputs/summary_results.csv and the generated confusion matrices.
4. For error-case figures used in the report, you can plot from the CSVs with the provided cells in the notebook (slice error bars, top confusion pairs, near-miss histogram).


## 7) Troubleshooting
1. CUDA OOM: lower batch_size to 4 and/or max_length to 96.
2. Slow throughput with LoRA: expected due to adapter ops; consider disabling gradient checkpointing (if you enabled it), or use full FT on GPU if memory allows.
3. Dataset split reproducibility: seed is fixed at 42; delete Assignment3_outputs/ to rerun clean.

## 8) Citation
1. Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
2. ProsusAI/FinBERT: domain-adapted BERT for finance
3. PEFT/LoRA: Hu et al., LoRA: Low-Rank Adaptation of Large Language Models
4. HuggingFace Datasets/Transformers/Accelerate

