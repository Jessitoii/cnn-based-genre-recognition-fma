# Training & Evaluation Results

> [!NOTE]
> This document outlines the evaluation metrics produced on the FMA Small Test split.

## Overall Metrics
- **Test Accuracy**: 60.63%
- **Macro F1 Score**: 0.59
- **Total Training Epochs Reached**: 25 (Stopped due to early stopping, patience=7)

## Class-wise Performance
Below is a breakdown of accuracy per genre (using an 8-class layout):

| Genre          | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Electronic     | 0.56      | 0.78   | 0.65     | 152     |
| Experimental   | 0.50      | 0.52   | 0.51     | 148     |
| Folk           | 0.58      | 0.74   | 0.65     | 155     |
| Hip-Hop        | 0.76      | 0.78   | 0.77     | 150     |
| Instrumental   | 0.69      | 0.47   | 0.56     | 145     |
| International  | 0.60      | 0.65   | 0.63     | 141     |
| Pop            | 0.44      | 0.22   | 0.29     | 157     |
| Rock           | 0.69      | 0.71   | 0.70     | 151     |

## Key Observations

- **Strongest Predictors**: The model excels in identifying **Hip-Hop** (F1=0.77) and **Rock** (F1=0.70). This suggests that these genres possess highly distinct acoustic features and rhythmic patterns that the CNN can easily capture from the mel-spectrograms.
- **Weakest Links**: **Pop** performed the worst by a significant margin, with just 22% recall and an F1-score of 0.29. Pop often borrows heavily from other genres (like Rock and Electronic), making it functionally varied and harder for a standard CNN to isolate.
- **Electronic and Folk**: Both showed reasonable F1-scores (~0.65). Interestingly, Electronic has a much higher recall (0.78) than precision (0.56), meaning the model is prone to over-predicting the Electronic class.
- **Instrumental**: Showed high precision (0.69) but lower recall (0.47), indicating that when the model predicts Instrumental, it's usually correct, but it fails to flag many actual instrumental tracks.

## Visualizations

### Confusion Matrix
![Confusion Matrix](../confusion_matrix.png)

*The confusion matrix provides a detailed mapping of true vs. predicted classifications.*

### Training & Validation Loss
![Loss/Accuracy Graph](../loss_graph.png)

## Future Improvement Suggestions

Based on the current 60.63% test accuracy, several architectural and data enhancements could be explored:
1. **Targeted Data Augmentation**: Since "Pop" and "Experimental" underperform, applying time-stretching, pitch-shifting, and background noise addition to these specific classes could improve their respective recall metrics.
2. **Advanced Architectures**: Upgrading the baseline CNN to a residual network (e.g., ResNet-18 or ResNet-34) could help capture more complex temporal-frequency abstractions.
3. **Hyperparameter Optimization**: Systematically tuning the learning rate, dropout rates, and batch size.
4. **Transfer Learning**: Utilizing weights pretrained on larger audio datasets (like AudioSet) or exploring Transformer-based architectures.
