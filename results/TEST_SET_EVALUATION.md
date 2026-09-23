# ECG Risk Stratification — Test Set Evaluation Report

**Project:** ecg-risk-stratification  
**Model:** `simpleECGNN` (1D CNN)  
**Dataset:** PTB-XL binary classification of pure normal ECGs vs ECGs containing the STTC diagnostic superclass  
**Report date:** September 2026  
**Pipeline:** `notebooks/data_pipeline.ipynb`

---

## Executive summary

A 1D convolutional neural network was trained on PTB-XL folds 1–8, tuned on fold 9, and evaluated on held-out fold 10 (1,433 ECGs, zero patient leakage). The model achieves **AUROC 0.974**, **AUPRC 0.964**, and **F1 0.893** at the validation-tuned decision threshold of **0.377**. Performance generalises well from validation to test with minimal drop across all metrics.

Gradient-based saliency maps (Captum) were generated for the highest-confidence **false positive** (ECG 1158) and lowest-confidence **false negative** (ECG 11132) to inspect which waveform regions drive model errors.

---

## 1. Task and data

### Classification task

| Label | Meaning | Source |
|-------|---------|--------|
| 0 | Normal ECG | PTB-XL diagnostic superclass `[NORM]` |
| 1 | Abnormal ECG | Any non-normal superclass |

### Data splits (stratified by `strat_fold`)

| Split | Fold(s) | Records | Normal (0) | Abnormal (1) |
|-------|---------|---------|------------|--------------|
| Train | 1–8 | 11,429 | 63.37% | 36.63% |
| Validation | 9 | 1,442 | 63.38% | 36.62% |
| **Test** | **10** | **1,433** | **912 (63.64%)** | **521 (36.36%)** |

**Leakage check:** 0 overlapping patients between train/validation, train/test, and validation/test.

### Input preprocessing

- 12-lead ECG, 5,000 samples at 500 Hz (10 seconds)
- Bandpass filtering and per-lead z-score normalisation
- Preprocessed signals cached in `data/preprocessed_500/`

---

## 2. Model and training

### Architecture (`src/model.py`)

- **Input:** `(batch, 12, 5000)`
- **Feature extractor:** 3× Conv1d blocks (12→32→64→128, kernel 7, BatchNorm, ReLU, MaxPool) + AdaptiveAvgPool1d
- **Classifier:** Dropout (0.3) + Linear(128 → 1)
- **Output:** Single logit (binary classification)

### Training configuration

| Parameter | Value |
|-----------|-------|
| Loss | `BCEWithLogitsLoss` with `pos_weight = 1.73` |
| Optimiser | Adam, lr = 1e-3, weight decay = 1e-5 |
| Batch size | 64 |
| Max epochs | 30 |
| Early stopping patience | 30 (on validation AUROC) |
| Checkpoint criterion | Best validation AUROC |
| Best epoch | 23 (of 30) |
| Best validation AUROC | 0.9792 |
| Saved weights | `results/models/best_simple_cnn.pt` |

---

## 3. Decision threshold

The default 0.5 threshold was not used. The operating point was chosen on the **validation set** by maximising F1 over the precision–recall curve:

| Parameter | Value |
|-----------|-------|
| **Selected threshold** | **0.3773** |
| Validation F1 at this threshold | 0.904 |
| Validation precision | 0.873 |
| Validation recall | 0.938 |

The lower threshold prioritises sensitivity (detecting abnormal ECGs) over specificity, which is appropriate for a risk-stratification screening task.

---

## 4. Test set results (threshold = 0.377)

Primary metrics saved to `results/metrics/test_metrics.csv`.

| Metric | Test | Validation | Δ (test − val) |
|--------|------|------------|----------------|
| Loss | 0.2714 | 0.2432 | +0.028 |
| **Accuracy** | **92.04%** | 92.72% | −0.68 pp |
| **AUROC** | **0.9742** | 0.9792 | −0.005 |
| **AUPRC** | **0.9635** | 0.9654 | −0.002 |
| **F1** | **0.8925** | 0.9041 | −0.012 |

### Confusion matrix

```
                      Predicted
                   Normal    Abnormal
Actual  Normal      846 (TN)    66 (FP)
        Abnormal     48 (FN)   473 (TP)
```

| Outcome | Count | Rate |
|---------|-------|------|
| True negatives (TN) | 846 | 92.8% of normals |
| True positives (TP) | 473 | 90.8% of abnormals |
| False positives (FP) | 66 | 7.2% of normals |
| False negatives (FN) | 48 | 9.2% of abnormals |
| **Correct** | **1,319 / 1,433** | **92.0%** |
| **Errors** | **114 / 1,433** | **8.0%** |

### Derived metrics (threshold = 0.377)

| Metric | Value |
|--------|-------|
| Precision | 87.76% |
| Recall (sensitivity) | 90.79% |
| Specificity | 92.76% |
| Negative predictive value (NPV) | 94.63% |
| False positive rate | 7.24% |
| False negative rate | 9.21% |
| Balanced accuracy | 91.78% |
| Disease prevalence | 36.36% |

---

## 5. Threshold comparison (0.377 vs 0.5)

Because the threshold was tuned on validation, both operating points were checked on test:

| Metric | Threshold 0.377 (used) | Threshold 0.5 |
|--------|------------------------|---------------|
| Accuracy | 92.04% | **92.60%** |
| F1 | 0.8925 | **0.8971** |
| False positives | 66 | **47** |
| False negatives | **48** | 59 |
| Confusion matrix | [[846, 66], [48, 473]] | [[865, 47], [59, 462]] |

On test, threshold 0.5 gives marginally higher accuracy and F1 with fewer false positives but **11 additional missed abnormal cases**. The validation-tuned threshold of 0.377 is the better choice when missing abnormal ECGs is costlier than over-calling normals.

---

## 6. Predicted probability distribution (test set)

| Statistic | All records | Normal (label 0) | Abnormal (label 1) |
|-----------|-------------|------------------|---------------------|
| Minimum | 0.00065 | — | — |
| Median | 0.082 | 0.015 | 0.975 |
| Mean | 0.363 | 0.087 | 0.846 |
| Maximum | 0.99997 | — | — |

Normal and abnormal classes are well separated in probability space, consistent with the high AUROC of 0.974.

---

## 7. Error analysis

### False positives (n = 66)

Normal ECGs incorrectly classified as abnormal.

| ECG ID | Predicted probability | True label |
|--------|----------------------|------------|
| 1158 | 0.981 | 0 |
| 4951 | 0.963 | 0 |
| 245 | 0.955 | 0 |
| 17919 | 0.950 | 0 |
| 7950 | 0.923 | 0 |
| 12230 | 0.923 | 0 |
| 12711 | 0.875 | 0 |
| 4822 | 0.841 | 0 |
| 795 | 0.830 | 0 |
| 17935 | 0.825 | 0 |

### False negatives (n = 48)

Abnormal ECGs missed by the model.

| ECG ID | Predicted probability | True label |
|--------|----------------------|------------|
| 11132 | 0.004 | 1 |
| 17648 | 0.016 | 1 |
| 8834 | 0.017 | 1 |
| 15795 | 0.021 | 1 |
| 1268 | 0.024 | 1 |
| 2734 | 0.027 | 1 |
| 13139 | 0.034 | 1 |
| 18464 | 0.038 | 1 |
| 6270 | 0.041 | 1 |
| 15864 | 0.044 | 1 |

### Error summary

| Error type | Count | Rate within true class |
|------------|-------|------------------------|
| False positives | 66 | 7.2% of 912 normals |
| False negatives | 48 | 9.2% of 521 abnormals |

The model is slightly more likely to miss an abnormal case than to over-call a normal one at the chosen threshold.

---

## 8. Saliency map analysis

### Method (`src/saliency_map.py`)

Gradient-based saliency maps were computed using **Captum's `Saliency`** method:

1. Load the ECG tensor for a given record from the test dataset.
2. Forward pass through `best_model` with `requires_grad=True` on the input.
3. Compute the gradient of the output logit with respect to the input signal.
4. Take the absolute value per time point, normalise to [0, 1] per lead.
5. Overlay saliency as a colour scatter on the waveform (red = high influence).

Each plot title reports the **true label** and **predicted probability** (sigmoid of logit).

**Lead index reference:**

| Index | Lead | Index | Lead |
|-------|------|-------|------|
| 0 | I | 6 | V1 |
| 1 | II | 7 | V2 |
| 2 | III | 8 | V3 |
| 3 | aVR | 9 | V4 |
| 4 | aVL | 10 | V5 |
| 5 | aVF | 11 | V6 |

### Cases selected for interpretability

The most extreme errors from the test set were chosen:

| Case | ECG ID | True label | Pred. prob | Error type |
|------|--------|------------|------------|------------|
| **FP** | **1158** | 0 (normal) | 0.981 | Highest-confidence false positive |
| **FN** | **11132** | 1 (abnormal) | 0.004 | Lowest-confidence false negative |

Raw 12-lead ECG plots were generated first (`plot_ecg_by_id`), followed by saliency overlays on Lead II and all chest leads (V1–V6).

### Generated saliency figures

All plots saved under `results/plots/saliency/`:

**False positive — ECG 1158**

| File | Lead |
|------|------|
| `saliency_fp_1158_leadII.png` | II |
| `saliency_fp_1158_leadV1.png` | V1 |
| `saliency_fp_1158_leadV2.png` | V2 |
| `saliency_fp_1158_leadV3.png` | V3 |
| `saliency_fp_1158_leadV4.png` | V4 |
| `saliency_fp_1158_leadV5.png` | V5 |
| `saliency_fp_1158_leadV6.png` | V6 |

**False negative — ECG 11132**

| File | Lead |
|------|------|
| `saliency_fn_11132_leadII.png` | II |
| `saliency_fn_11132_leadV1.png` | V1 |
| `saliency_fn_11132_leadV2.png` | V2 |
| `saliency_fn_11132_leadV3.png` | V3 |
| `saliency_fn_11132_leadV4.png` | V4 |
| `saliency_fn_11132_leadV5.png` | V5 |
| `saliency_fn_11132_leadV6.png` | V6 |

### How to reproduce

Run `notebooks/data_pipeline.ipynb` through Section 9 after completing test evaluation (Section 8). Required variables: `best_model`, `test_dataset`, `test_df`, `device`, `fp_id`, `fn_id`.

```python
from src.saliency_map import plot_saliency_for_lead, plot_saliency_for_leads, CHEST_LEAD_IDXS

plot_saliency_for_lead(
    model=best_model, dataset=test_dataset, df=test_df,
    ecg_id=fp_id, device=device, lead_idx=1,
    save_path="results/plots/saliency/saliency_fp_{}_leadII.png".format(fp_id),
)
```

### Interpretation notes

- **ECG 1158 (FP):** The model assigns 98.1% abnormal probability to a normal-labelled record. Saliency maps show which QRS/ST segments the CNN treats as pathological — useful for identifying spurious feature associations or subtle morphology that resembles pathology.
- **ECG 11132 (FN):** The model assigns 0.4% abnormal probability to a truly abnormal record. Saliency maps reveal whether the model ignores clinically relevant regions or whether the abnormality is expressed in leads the CNN under-weights.
- Chest leads (V1–V6) are emphasised because many cardiac pathologies localise to precordial leads; Lead II is included as the standard rhythm strip.

Saliency maps show **where** the model looks, not **why** in clinical terms. They are exploratory tools for error analysis, not standalone clinical explanations.

---

## 9. Generalisation assessment

| Observation | Assessment |
|-------------|------------|
| AUROC drop (val → test) | 0.005 — excellent |
| AUPRC drop | 0.002 — excellent |
| Accuracy drop | 0.68 pp — minimal |
| F1 drop | 0.012 — acceptable |
| Patient leakage | None detected |

The model generalises well from validation fold 9 to test fold 10 with no evidence of significant overfitting.

---

## 10. Conclusions

1. **Strong discrimination:** AUROC 0.974 and AUPRC 0.964 on 1,433 held-out test ECGs.
2. **Clinically usable operating point:** At threshold 0.377, sensitivity 90.8% and precision 87.8%.
3. **Stable generalisation:** Test performance closely matches validation across all metrics.
4. **Interpretable error analysis:** 114 misclassified records identified; saliency maps generated for the most extreme FP and FN cases.
5. **Reproducible pipeline:** End-to-end workflow from preprocessing through evaluation and interpretability in `data_pipeline.ipynb`.

### Limitations

- Threshold tuned on validation fold 9; test F1 is marginally higher at 0.5.
- Evaluation limited to PTB-XL fold 10; external validation on other cohorts is needed before clinical use.
- Binary normal vs abnormal collapses diverse pathologies into one class.
- Saliency maps reflect gradient attribution for this CNN architecture only; they do not constitute clinical diagnosis.

---

## 11. Artifact index

| Artifact | Path |
|----------|------|
| This report | `results/TEST_SET_EVALUATION.md` |
| Test metrics (CSV) | `results/metrics/test_metrics.csv` |
| Val vs test comparison | `results/metrics/val_test_comparison.csv` |
| Training history | `results/metrics/training_history.csv` |
| Best model weights | `results/models/best_simple_cnn.pt` |
| Test ROC curve | `results/plots/test_roc_curve.png` |
| Test PR curve | `results/plots/test_pr_curve.png` |
| Saliency maps | `results/plots/saliency/saliency_*.png` |
| Saliency module | `src/saliency_map.py` |
| Pipeline notebook | `notebooks/data_pipeline.ipynb` |
