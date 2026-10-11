# Model Evaluation and Interpretation Report

## 1. Executive summary
A Random Forest classifier was trained on the Wisconsin Diagnostic Breast Cancer dataset included in scikit-learn. The data contains numeric measurements computed from digitized images of breast masses. Evaluation uses a stratified 20% hold-out test set. This is an educational exercise, **not a clinical diagnostic system**.

## 2. Methodology
- **Data:** scikit-learn `load_breast_cancer`; 569 rows and 30 numeric features.
- **Split:** 80% training / 20% test, stratified by target, random seed 42.
- **Model:** Random Forest with 300 trees, `class_weight="balanced"`, random seed 42.
- **Metrics:** per-class precision, recall, F1-score, support, confusion matrix and ROC-AUC.
- **Interpretability:** SHAP `TreeExplainer` for global summary and one local force plot.

## 3. Classification report

```text
              precision    recall  f1-score   support

   malignant     0.9286    0.9286    0.9286        42
      benign     0.9583    0.9583    0.9583        72

    accuracy                         0.9474       114
   macro avg     0.9435    0.9435    0.9435       114
weighted avg     0.9474    0.9474    0.9474       114

```

## 4. Confusion matrix
Rows represent true labels and columns predicted labels in the order `[np.str_('malignant'), np.str_('benign')]`:

```text
[[39  3]
 [ 3 69]]
```

See `assets/confusion_matrix.png`.

## 5. ROC-AUC
The hold-out ROC-AUC is **0.9937**. AUC measures ranking performance across classification thresholds. It does not establish clinical usefulness by itself. See `assets/roc_curve.png`.

## 6. Feature importance and SHAP
The supplied `assets/feature_importance.png` shows built-in Random Forest feature importance. This is **not SHAP**; it is included as a baseline. Run the SHAP section in `notebooks/model_interpretation.ipynb` to generate:
- `assets/shap_summary.png`: global distribution of feature contributions.
- `assets/shap_force_plot.html`: local explanation for one held-out sample.

SHAP values describe how features contribute to a model output relative to a reference value. They explain the fitted model, not causation. Correlated features can share or redistribute attribution.

### How to interpret the SHAP plots
- In the summary plot, features are ordered by mean absolute SHAP value.
- Each dot is one observation. Horizontal position indicates contribution direction and magnitude; color typically indicates low-to-high feature value.
- Positive contributions move output toward the explained class; negative contributions move it away. Check the class/output scale for your installed SHAP version.
- The force plot shows how feature values shift one case from the base value.

### Domain alignment and limitations
The dataset features are geometric and texture statistics computed from digitized breast-mass images. Claims about individual medical relevance should be checked against reputable clinical literature. The model is not validated for other populations or workflows. Do not use this model for diagnosis or treatment decisions.

## 7. Reproducibility
Follow `README.md`, install `requirements.txt`, and run all notebook cells in order. If SHAP fails due to dependency incompatibility, create a fresh virtual environment and reinstall requirements.
