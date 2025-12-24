# IFLTNM (core) — TabNet-based Fuzzy Teacher–Student for Noisy Labels

This is a **minimal, open-source-friendly** core implementation of:

> **A Fuzzy Large TabNet-based Model and Its Distillation Learning for Noisy Labeling Data via Consequent Additive Decomposition**

It contains:
- `run_ifltnm_suite.py`: end-to-end pipeline
- `tabnet_core.py`: a light TabNet implementation (used as the consequent backbone)
- `export_student_rules.py`: export **student** TSK rules to LaTeX
- `plot_sensitivity.py`: optional plotting helper (unchanged)

---

## Quick start

### 1) Create environment

```bash
pip install numpy pandas scikit-learn torch
```

### 2) Train teacher (IFLTNM) on a CSV

Assuming your CSV has **no header** and the **last column is the label**:

```bash
python run_ifltnm_suite.py \
  --csv .\largeScaleData\shuttle.csv \
  --label_idx -1 \
  --seed 42 --test_size 0.3 \
  --n_rules 3 --sigma 0.8 \
  --teacher_epochs 100 --teacher_batch 256 \
  --tabnet_module tabnet_core --tabnet_class TabNet \
  --out_dir models
```

If your CSV **has a header row**, add `--header`.

### 3) Distill a student (s-IFLTNM)

```bash
python run_ifltnm_suite.py \
  --csv .\largeScaleData\shuttle.csv \
  --label_idx -1 \
  --seed 42 --test_size 0.3 \
  --n_rules 3 --sigma 0.8 \
  --do_distill \
  --student_epochs 200 \
  --temperature 4.0 --omega 0.2 --xi 0.1 \
  --tabnet_module tabnet_core --tabnet_class TabNet \
  --out_dir models
```

This will save:
- Teacher: `models/ifltnm_teacher_<run_id>.pt`
- Manifest (split + scaling + feature names): `models/split_manifest_<run_id>.pt`
- Student: `models/s_ifltnm_student_<run_id>.pt`

### 4) Skip teacher training (reuse a saved teacher)

```bash
python run_ifltnm_suite.py \
  --csv .\largeScaleData\shuttle.csv \
  --label_idx -1 \
  --seed 42 --test_size 0.3 \
  --n_rules 3 --sigma 0.8 \
  --no_train_teacher \
  --teacher_ckpt .\models\ifltnm_teacher_shuttle_seed42_R3_sig0.8_nr0.0.pt \
  --do_distill \
  --tabnet_module tabnet_core --tabnet_class TabNet \
  --out_dir models
```

---

## Export student rules to LaTeX

```bash
python export_student_rules.py \
  --student_ckpt .\models\s_ifltnm_student_<run_id>.pt \
  --manifest .\models\split_manifest_<run_id>.pt \
  --out .\student_rules.tex \
  --use_feature_names \
  --precision 2 --top_terms 8 --max_ante_dim 8
```

If your CSV had no header, omit `--use_feature_names` and it will use `x_1, x_2, ...`.

---

## What is implemented (paper-aligned core)

### Teacher (IFLTNM) — Consequent Additive Decomposition (CAD)

Each rule’s consequent is decomposed into:
- **clean** part `W_c, b_c`
- **noisy** part `W_n, b_n`

During teacher training, the rule output uses:

```
W = W_c + W_n,   b = b_c + b_n
```

and we add a **memorization-effect-inspired** regularization schedule:
- `beta_F(epoch)` **increases** over epochs and penalizes changes of `W_c` relative to the previous epoch (`||W_c - W_c_prev||^2`)
- `beta_Y(epoch)` **decreases** over epochs and penalizes the magnitude of `W_n` (`||W_n||^2`)

After training, we **discard** the noisy part by switching to **clean-only inference**:

```
W = W_c,   b = b_c
```

This clean-only teacher is used to split training samples into:
- **almost clean**: predicted label == observed label
- **almost noisy**: predicted label != observed label

### Student (s-IFLTNM) — 4-loss distillation

On the **almost clean** subset, we use 3 losses:
1. `L_CE`: cross-entropy to labels
2. `L_KD`: final-logit distillation using KL with temperature `T`
3. `L_ruleKD`: rule-wise distillation (L1 distance between softened per-rule class distributions)

On the **almost noisy** subset, we add an ANL regularizer:
4. `L_ANL`: `(1-xi) * CE(y) + xi * CE(y_bar)`, where `y_bar` is uniform over all classes except the observed class.

Final (core) combination:

```
L = zeta * (L_CE + alpha_kd * L_KD + beta_rule * L_ruleKD) + omega * L_ANL
```

If `zeta` is not given, it defaults to `1 - omega`.

---

## Notes

- **Unique filenames**: every run uses a `run_id` based on dataset/seed/R/sigma/noise_rate, so different datasets won’t overwrite each other.
- Windows `torch.save` failures typically indicate **disk/full path/permission** issues. This code writes via a `*.tmp` file and then renames.

---

## License

Pick any license you want (MIT is typical) before publishing.
