# Day 8: Preparing for Level 1 Modeling

## Objective
Prepare an encoding utility for categorical features created by Level 1 anonymization.

## Planned transformations
- Age: generalize numeric age into ten-year ranges.
- ZIP-like field: mask the last two characters of valid five-digit values.
- Missing and invalid values: preserve as missing for downstream imputation.

## Modeling compatibility
Generalized age ranges and masked ZIP-like values are categorical features. The preprocessing pipeline must encode them as categorical variables.

## Validation
Test the encoding functions on representative values, missing values, and invalid inputs before applying them to the team's Level 1 dataset.

## Limitation
The ZIP-like column was added separately to the Adult dataset. Its provenance must be documented. Masking synthetic values demonstrates string transformation, but does not establish a real geographic privacy benefit.

## Next step
Confirm the output format with the team, then test the preprocessing pipeline on a partially anonymized dataset.
