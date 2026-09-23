# ECG ST/T Change Classification — Test Set Evaluation Report

**Project:** ecg-risk-stratification  
**Model:** `simpleECGNN` (1D CNN)  
**Dataset:** PTB-XL binary classification of pure normal ECGs vs ECGs containing the STTC diagnostic superclass  
**Report date:** September 2026  
**Pipeline:** `notebooks/data_pipeline.ipynb`

---

## Executive Summary

A 1D convolutional neural network was trained to distinguish pure normal ECGs from ECGs containing the PTB-XL ST/T change (STTC) diagnostic superclass. Records belonging only to other diagnostic superclasses were excluded from the binary classification task.

The model was trained on PTB-XL folds 1–8, tuned on fold 9, and evaluated on held-out fold 10 (1,433 ECGs, zero patient leakage). It achieved **AUROC 0.974**, **AUPRC 0.964**, and **F1 0.893** at the validation-selected decision threshold of **0.377**. Performance remained close between validation and the held-out test set across the main evaluation metrics.

Gradient-based saliency maps using Captum were generated for the **highest-STTC-probability false positive** (ECG 1158) and **lowest-STTC-probability false negative** (ECG 11132) to examine which waveform regions had high local influence on the model output.

---

## 1. Task and Data

### Classification Task

| Label | Meaning | Inclusion rule |
|---|---|---|
| 0 | Pure normal ECG | Diagnostic superclass is exactly `NORM` |
| 1 | ST/T change ECG | Diagnostic superclasses contain `STTC` |
| Excluded | Other diagnoses | Neither pure `NORM` nor STTC-positive |

STTC-positive records may contain additional diagnostic superclasses; any record containing `STTC` is assigned to the positive class. Records containing other diagnostic superclasses without `STTC` are excluded from the binary dataset.

### Data Splits

The dataset was divided using the predefined PTB-XL `strat_fold` assignments.

| Split | Fold(s) | Records | Normal (0) | STTC (1) |
|---|---:|---:|---:|---:|
| Train | 1–8 | 11,429 | 63.37% | 36.63% |
| Validation | 9 | 1,442 | 63.38% | 36.62% |
| **Test** | **10** | **1,433** | **912 (63.64%)** | **521 (36.36%)** |

**Leakage check:** 0 overlapping patients between train/validation, train/test, and validation/test.

### Input Preprocessing

- 12-lead ECG
- 5,000 samples per lead at 500 Hz
- 10-second recordings
- channels-first conversion for PyTorch
- 0.5–40 Hz band-pass filtering
- per-lead z-score normalisation
- preprocessed signals cached in `data/preprocessed_500/`

---

## 2. Model and Training

### Architecture (`src/model.py`)

- **Input:** `(batch, 12, 5000)`
- **Feature extractor:** 3 × Conv1d blocks
- **Channel progression:** `12 → 32 → 64 → 128`
- **Kernel size:** 7
- **Each convolutional block:** BatchNorm → ReLU → MaxPool
- **Pooling:** AdaptiveAvgPool1d
- **Classifier:** Dropout (`0.3`) + Linear (`128 → 1`)
- **Output:** single logit for binary classification

### Training Configuration

| Parameter | Value |
|---|---|
| Loss | `BCEWithLogitsLoss` with `pos_weight = 1.73` |
| Optimiser | Adam |
| Learning rate | `1e-3` |
| Weight decay | `1e-5` |
| Batch size | 64 |
| Maximum epochs | 30 |
| Early stopping patience | 30 epochs based on validation AUROC |
| Checkpoint criterion | Best validation AUROC |
| Best epoch | 23 of 30 |
| Best validation AUROC | 0.9792 |
| Saved weights | `results/models/best_simple_cnn.pt` |

---

## 3. Decision Threshold

The default classification threshold of 0.5 was not used for the final operating point.

Instead, the threshold was selected on the **validation set** by maximising F1 over the precision–recall curve.

| Parameter | Value |
|---|---:|
| **Selected threshold** | **0.3773** |
| Validation F1 | 0.904 |
| Validation precision | 0.873 |
| Validation recall | 0.938 |

The lower threshold increases sensitivity for detecting STTC-positive ECGs, reducing the number of STTC cases classified as normal at the cost of additional false-positive classifications.

---

## 4. Test Set Results

**Operating threshold: 0.377**

Primary metrics are saved to:

`results/metrics/test_metrics.csv`

### Validation vs Test Performance

| Metric | Test | Validation | Δ (test − validation) |
|---|---:|---:|---:|
| Loss | 0.2714 | 0.2432 | +0.028 |
| **Accuracy** | **92.04%** | 92.72% | −0.68 pp |
| **AUROC** | **0.9742** | 0.9792 | −0.005 |
| **AUPRC** | **0.9635** | 0.9654 | −0.002 |
| **F1** | **0.8925** | 0.9041 | −0.012 |

### Confusion Matrix

```text
                      Predicted
                   Normal     STTC

Actual Normal       846        66
       STTC          48       473
```

| Outcome | Count | Rate |
|---|---:|---:|
| True negatives (TN) | 846 | 92.8% of normals |
| True positives (TP) | 473 | 90.8% of STTCs |
| False positives (FP) | 66 | 7.2% of normals |
| False negatives (FN) | 48 | 9.2% of STTCs |
| **Correct** | **1,319 / 1,433** | **92.0%** |
| **Errors** | **114 / 1,433** | **8.0%** |

### Derived Metrics

| Metric | Value |
|---|---:|
| Precision | 87.76% |
| Recall (sensitivity) | 90.79% |
| Specificity | 92.76% |
| Negative predictive value (NPV) | 94.63% |
| False positive rate | 7.24% |
| False negative rate | 9.21% |
| Balanced accuracy | 91.78% |
| STTC prevalence | 36.36% |

---

## 5. Threshold Comparison: 0.377 vs 0.5

Because the final threshold was selected using validation data, both operating points were also examined on the held-out test set.

| Metric | Threshold 0.377 (used) | Threshold 0.5 |
|---|---:|---:|
| Accuracy | 92.04% | **92.60%** |
| F1 | 0.8925 | **0.8971** |
| False positives | 66 | **47** |
| False negatives | **48** | 59 |
| Confusion matrix | `[[846, 66], [48, 473]]` | `[[865, 47], [59, 462]]` |

On the test set, the 0.5 threshold produced marginally higher accuracy and F1 and fewer false positives, but resulted in **11 additional missed STTC-positive ECGs**.

The final threshold remains `0.3773` because it was selected using validation data rather than optimised retrospectively on the held-out test set. It also provides a more sensitivity-focused operating point.

---

## 6. Predicted Probability Distribution

### Test Set

| Statistic | All records | Normal (label 0) | STTC (label 1) |
|---|---:|---:|---:|
| Minimum | 0.00065 | — | — |
| Median | 0.082 | 0.015 | 0.975 |
| Mean | 0.363 | 0.087 | 0.846 |
| Maximum | 0.99997 | — | — |

Normal and STTC-positive records show substantial separation in predicted probability, consistent with the test AUROC of 0.974.

---

## 7. Error Analysis

### False Positives

**n = 66**

These are normal ECGs incorrectly classified as STTC-positive.

| ECG ID | STTC probability | True label |
|---:|---:|---:|
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

### False Negatives

**n = 48**

These are STTC-positive ECGs incorrectly classified as normal.

| ECG ID | STTC probability | True label |
|---:|---:|---:|
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

### Error Summary

| Error type | Count | Rate within true class |
|---|---:|---:|
| False positives | 66 | 7.2% of 912 normals |
| False negatives | 48 | 9.2% of 521 STTCs |

At the selected threshold, the false-negative rate within the STTC class is 9.2%, while the false-positive rate within the normal class is 7.2%.

---

## 8. Saliency Map Analysis

### Method (`src/saliency_map.py`)

Gradient-based saliency maps were computed using Captum's `Saliency` method.

For each selected ECG:

1. The ECG tensor is loaded from the test dataset.
2. A forward pass is performed through the trained model.
3. The gradient of the model output logit with respect to the input ECG is calculated.
4. Absolute gradient values are taken for the selected lead.
5. Saliency values are normalised to `[0, 1]`.
6. The resulting values are overlaid on the ECG waveform as a colour-coded scatter plot.

Each plot reports the true label and the sigmoid-transformed STTC probability.

### Lead Index Reference

| Index | Lead | Index | Lead |
|---:|---|---:|---|
| 0 | I | 6 | V1 |
| 1 | II | 7 | V2 |
| 2 | III | 8 | V3 |
| 3 | aVR | 9 | V4 |
| 4 | aVL | 10 | V5 |
| 5 | aVF | 11 | V6 |

### Cases Selected for Interpretability

Two extreme test-set errors were selected:

| Case | ECG ID | True label | STTC probability | Error description |
|---|---:|---|---:|---|
| **FP** | **1158** | 0 (normal) | 0.981 | Highest-STTC-probability false positive |
| **FN** | **11132** | 1 (STTC) | 0.004 | Lowest-STTC-probability false negative |

Raw 12-lead ECG plots were generated first using `plot_ecg_by_id`, followed by saliency overlays on Lead II and the precordial leads V1–V6.

### Generated Saliency Figures

All plots are saved under:

`results/plots/saliency/`

#### False Positive — ECG 1158

| File | Lead |
|---|---|
| `saliency_fp_1158_leadII.png` | II |
| `saliency_fp_1158_leadV1.png` | V1 |
| `saliency_fp_1158_leadV2.png` | V2 |
| `saliency_fp_1158_leadV3.png` | V3 |
| `saliency_fp_1158_leadV4.png` | V4 |
| `saliency_fp_1158_leadV5.png` | V5 |
| `saliency_fp_1158_leadV6.png` | V6 |

#### False Negative — ECG 11132

| File | Lead |
|---|---|
| `saliency_fn_11132_leadII.png` | II |
| `saliency_fn_11132_leadV1.png` | V1 |
| `saliency_fn_11132_leadV2.png` | V2 |
| `saliency_fn_11132_leadV3.png` | V3 |
| `saliency_fn_11132_leadV4.png` | V4 |
| `saliency_fn_11132_leadV5.png` | V5 |
| `saliency_fn_11132_leadV6.png` | V6 |

### How to Reproduce

Run `notebooks/data_pipeline.ipynb` through Section 9 after completing test evaluation in Section 8.

Required variables:

- `best_model`
- `test_dataset`
- `test_df`
- `device`
- `fp_id`
- `fn_id`

Example:

```python
from src.saliency_map import (
    plot_saliency_for_lead,
    plot_saliency_for_leads,
    CHEST_LEAD_IDXS,
)

plot_saliency_for_lead(
    model=best_model,
    dataset=test_dataset,
    df=test_df,
    ecg_id=fp_id,
    device=device,
    lead_idx=1,
    save_path=f"results/plots/saliency/saliency_fp_{fp_id}_leadII.png",
)
```

### Interpretation Notes

- **ECG 1158 (FP):** The model assigns 98.1% STTC probability to a normal-labelled record. Saliency maps highlight waveform regions with high local influence on the prediction and can help identify possible spurious feature associations or morphology resembling ST/T abnormalities.

- **ECG 11132 (FN):** The model assigns 0.4% STTC probability to an STTC-positive record. Saliency maps can be used to examine whether regions associated with the STTC label receive relatively little model sensitivity or whether relevant morphology is concentrated in leads that contribute less strongly to the prediction.

- Lead II and precordial leads V1–V6 were examined to provide a broader view of model sensitivity across rhythm and chest-lead morphology.

Saliency maps measure local input sensitivity to the model output. They do **not** establish why a waveform is clinically abnormal and should not be interpreted as standalone clinical explanations.

---

## 9. Generalisation Assessment

| Observation | Validation → Test change |
|---|---:|
| AUROC | −0.005 |
| AUPRC | −0.002 |
| Accuracy | −0.68 percentage points |
| F1 | −0.012 |
| Patient leakage | None detected |

Held-out test performance closely matched validation performance, with only small decreases across the main metrics.

This provides evidence that performance was maintained on the held-out PTB-XL fold, although external validation is still required before making claims about generalisation to other datasets or clinical populations.

---

## 10. Conclusions

1. **Strong STTC discrimination:** The model achieved AUROC 0.974 and AUPRC 0.964 when distinguishing pure normal ECGs from STTC-positive ECGs on 1,433 held-out records.

2. **Sensitivity-focused operating point:** At the validation-selected threshold of 0.377, sensitivity for STTC was 90.8% with precision of 87.8%.

3. **Stable held-out performance:** Test performance remained close to validation performance across the main metrics.

4. **Error analysis and interpretability:** 114 misclassified records were identified, and gradient-based saliency maps were generated for extreme false-positive and false-negative cases.

5. **End-to-end workflow:** The project integrates PTB-XL label extraction, ECG preprocessing, CNN training and evaluation, gradient-based interpretability, FastAPI inference, and a deployed browser interface.

### Limitations

- The classification threshold was selected using validation fold 9; test F1 happened to be marginally higher at a threshold of 0.5.
- Evaluation is limited to PTB-XL fold 10; external validation on independent cohorts is required before assessing generalisability beyond PTB-XL.
- The binary task distinguishes pure normal ECGs from ECGs containing the broad STTC diagnostic superclass.
- ECGs containing other diagnostic superclasses without STTC are excluded from the classification task.
- STTC-positive ECGs may also contain additional diagnostic superclasses.
- STTC is a broad diagnostic superclass and may contain heterogeneous ST/T morphologies.
- Gradient-based saliency measures model input sensitivity and does not provide a causal or clinical explanation for predictions.
- The model has not been clinically validated and is not intended for diagnostic use.

---

## 11. Artifact Index

| Artifact | Path |
|---|---|
| This report | `results/TEST_SET_EVALUATION.md` |
| Test metrics | `results/metrics/test_metrics.csv` |
| Validation vs test comparison | `results/metrics/val_test_comparison.csv` |
| Training history | `results/metrics/training_history.csv` |
| Best model weights | `results/models/best_simple_cnn.pt` |
| Test ROC curve | `results/plots/test_roc_curve.png` |
| Test PR curve | `results/plots/test_pr_curve.png` |
| Saliency maps | `results/plots/saliency/saliency_*.png` |
| Saliency module | `src/saliency_map.py` |
| Pipeline notebook | `notebooks/data_pipeline.ipynb` |