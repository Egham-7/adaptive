# Vector Analysis Report

## 1. Descriptive Statistics per Dimension

| Dimension | Min | Max | Mean | Std Dev | Q1 | Median | Q3 |
|-----------|-----|-----|------|---------|----|--------|----|
| creativity_scope | 0.001 | 0.347 | 0.015 | 0.023 | 0.004 | 0.008 | 0.018 |
| reasoning | 0.005 | 0.926 | 0.166 | 0.147 | 0.061 | 0.112 | 0.231 |
| constraint_ct | 0.040 | 0.981 | 0.688 | 0.223 | 0.533 | 0.741 | 0.875 |
| contextual_knowledge | 0.005 | 0.427 | 0.064 | 0.044 | 0.036 | 0.053 | 0.084 |
| domain_knowledge | 0.151 | 0.998 | 0.862 | 0.162 | 0.819 | 0.933 | 0.977 |

## 2. Dimension Correlations (Pearson)

| Dimension | creativity_scope | reasoning | constraint_ct | contextual_knowledge | domain_knowledge |
|-----------|---|---|---|---|---|
| creativity_scope | 1.000 | -0.091 | 0.145 | 0.129 | 0.038 |
| reasoning | -0.091 | 1.000 | -0.040 | -0.137 | -0.299 |
| constraint_ct | 0.145 | -0.040 | 1.000 | -0.167 | -0.067 |
| contextual_knowledge | 0.129 | -0.137 | -0.167 | 1.000 | 0.015 |
| domain_knowledge | 0.038 | -0.299 | -0.067 | 0.015 | 1.000 |

### Strong Correlations (|r| > 0.5)

No strong correlations found (|r| > 0.5)

## Interpretation

### Dimension Analysis
- Min/max values show the full range of values for each dimension
- Mean ± Std Dev indicates how tightly clustered the values are
- Quartiles help identify natural groupings (low/medium/high)

### Correlation Analysis
- Values close to 1.0 indicate strong positive correlation
- Values close to -1.0 indicate strong negative correlation
- Values close to 0.0 indicate little to no correlation
- Strong correlations suggest these dimensions might be measuring similar aspects