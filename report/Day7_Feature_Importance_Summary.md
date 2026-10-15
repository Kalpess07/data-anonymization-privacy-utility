# Day 7: Feature Importance Analysis

## Objective
Identify the features that contribute most to the Random Forest baseline and compare them with the columns planned for anonymization.

## Method
A Random Forest classifier was trained using the same 80:20 stratified train/test split and random seed as the baseline experiment. Feature importance scores were extracted from the fitted model.

One-hot encoded feature importances were retained for detailed analysis. Scores were also aggregated by original column for comparison with anonymization plans.

## Results
Insert the top 10 encoded features and the top 10 original columns from the generated CSV files.

## Anonymization relevance
Compare the importance rankings with the candidate columns selected for age bucketing, masking, generalization, and suppression. Features with high importance may warrant particular attention because transforming them could affect predictive utility.

## Limitations
Random Forest feature importance is not a direct measure of privacy risk, causation, or fairness. Correlated features may share or distort importance. Summed importance for categorical columns is useful for exploration but does not establish the effect of anonymizing that column.

The provenance of the separately added ZIP-code column must be documented before interpreting its importance.

## Conclusion
Use these rankings to prioritize experiments. Confirm the actual privacy–utility trade-off by retraining and evaluating models on anonymized datasets.
