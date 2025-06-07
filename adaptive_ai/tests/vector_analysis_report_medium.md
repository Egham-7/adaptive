# Vector Analysis Report

## 1. Descriptive Statistics per Dimension

| Dimension | Min | Max | Mean | Std Dev | Q1 | Median | Q3 |
|-----------|-----|-----|------|---------|----|--------|----|
| creativity_scope | 0.001 | 0.501 | 0.014 | 0.026 | 0.004 | 0.008 | 0.015 |
| reasoning | 0.003 | 0.826 | 0.195 | 0.158 | 0.079 | 0.147 | 0.263 |
| constraint_ct | 0.014 | 0.983 | 0.652 | 0.231 | 0.500 | 0.698 | 0.847 |
| contextual_knowledge | 0.005 | 0.346 | 0.062 | 0.044 | 0.033 | 0.051 | 0.077 |
| domain_knowledge | 0.199 | 0.998 | 0.876 | 0.155 | 0.838 | 0.943 | 0.978 |

## 2. Dimension Correlations (Pearson)

| Dimension | creativity_scope | reasoning | constraint_ct | contextual_knowledge | domain_knowledge |
|-----------|---|---|---|---|---|
| creativity_scope | 1.000 | 0.048 | 0.083 | 0.056 | -0.121 |
| reasoning | 0.048 | 1.000 | 0.062 | -0.098 | -0.362 |
| constraint_ct | 0.083 | 0.062 | 1.000 | -0.214 | -0.090 |
| contextual_knowledge | 0.056 | -0.098 | -0.214 | 1.000 | -0.080 |
| domain_knowledge | -0.121 | -0.362 | -0.090 | -0.080 | 1.000 |

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