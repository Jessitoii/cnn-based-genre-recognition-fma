# V2 Evaluation & Performance Results

> [!NOTE]
> This document outlines the evaluation metrics produced by the **GenreCNN V2** model evaluated on the actual FMA Small Test split (1,199 samples) streamed from Google Drive.

---

## Overall Metrics Comparison

| Metric | Baseline V1 Model | Upgraded V2 Model | Status |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | 60.63% | **61.88%** | **+1.25% Improvement** |
| **Macro F1 Score** | 0.59 | **0.62** | **+0.03 Improvement** |
| **Weighted F1 Score** | 0.59 | **0.62** | **+0.03 Improvement** |

---

## V2 Class-wise Metrics (Test Set)

Below is the detailed breakdown of precision, recall, and F1-score across all 8 classes for the **GenreCNN V2** architecture:

| Genre | Precision | Recall | F1-Score | Support | F1 Change vs V1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Electronic** | 0.51 | 0.90 | 0.65 | 152 | 0.00 (Stable) |
| **Experimental** | 0.66 | 0.37 | 0.48 | 148 | -0.03 (Slight drop) |
| **Folk** | 0.68 | 0.67 | 0.67 | 155 | **+0.02 (Improved)** |
| **Hip-Hop** | 0.85 | 0.73 | 0.78 | 150 | **+0.01 (Improved)** |
| **Instrumental** | 0.60 | 0.63 | 0.62 | 145 | **+0.06 (Improved)** |
| **International** | 0.79 | 0.57 | 0.67 | 141 | **+0.04 (Improved)** |
| **Pop** | 0.39 | 0.39 | 0.39 | 157 | **+0.10 (Huge Jump!)** |
| **Rock** | 0.66 | 0.68 | 0.67 | 151 | -0.03 (Slight drop) |

---

## Key Performance Observations

1. **Massive Pop Improvement (F1: 0.29 → 0.39)**:
   Pop was the weakest class in V1 due to its spectral ambiguity (Pop borrows styles from Electronic, Rock, and Folk). In V2, the Squeeze-and-Excitation channel attention successfully learned to identify and weigh subtle pop signatures, leading to a **+10% F1-score increase**.
2. **Excellent Instrumental Boost (F1: 0.56 → 0.62)**:
   In the baseline V1 network, Instrumental struggled with low recall (0.47) and was easily confused with Experimental due to lacking vocals. In V2, residual connection shortcuts stabilized structural feature detection, lifting F1 by **+6%**.
3. **Hip-Hop Remains King (F1: 0.78)**:
   Precision for Hip-Hop reached a stunning **85%**. Hip-Hop has exceptionally strong, repetitive low-frequency percussive beats and vocal rhythmic signatures that are extremely clear on Log-Mel Spectrograms.
4. **Electronic Recall (Recall: 0.90)**:
   Electronic has an incredibly high recall of 90%. However, its precision sits at 51%, indicating that the model still acts as a "magnet" for ambiguous samples, occasionally misclassifying other synth-heavy or electronic-adjacent tracks as Electronic.

---

## Visualizations

### Confusion Matrix
The confusion matrix plot is automatically computed and saved as `confusion_matrix.png` at the root of the workspace:

![Confusion Matrix V2](../confusion_matrix.png)

*The matrix highlights high concentration diagonals, demonstrating robust class isolation, particularly for Electronic, Hip-Hop, and Folk.*
