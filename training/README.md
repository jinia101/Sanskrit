# Fine-tuning (free + fast)

## What was prepared

- Task: **Sanskrit morphology** (verse line + token -> lemma/POS/tags)
- Model: **`Qwen/Qwen2.5-0.5B-Instruct`** with LoRA (smallest practical instruct model)
- Data: **8,000 examples** subset in `training/data/` (full corpus is ~196k)

## Fastest free path: Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `training/colab_quickstart.ipynb`
3. **Runtime -> Change runtime type -> T4 GPU**
4. Run all cells
5. When prompted, upload:
   - `training/data/morphology_n8000_train.jsonl`
   - `training/data/morphology_n8000_val.jsonl`
6. Wait ~15-30 minutes, then download the LoRA zip

## Regenerate data (optional)

```bash
# quick subset (default for Colab)
python training/prepare_dataset.py --task morphology --max-rows 8000

# full corpus
python training/prepare_dataset.py --task morphology
```

## Local training (only if you have an NVIDIA GPU)

```bash
pip install -r training/requirements-training.txt
python training/finetune.py
python training/test_inference.py
```

## Files

| File | Purpose |
|------|---------|
| `prepare_dataset.py` | JSON chapters -> train/val JSONL |
| `finetune.py` | LoRA training script |
| `test_inference.py` | Quick test after training |
| `colab_quickstart.ipynb` | One-click free GPU training |
| `data/morphology_n8000_*.jsonl` | Ready-to-upload training files |
