# FullStackBench Vector Analysis Report - Easy vs Hard Comparison

## 1. Task Type Distribution


### Easy Questions

| Task Type | Count |
|-----------|-------|
| Code Generation | 318 |
| Text Generation | 23 |
| Rewrite | 1 |
| Closed QA | 18 |
| Open QA | 2 |

### Hard Questions

| Task Type | Count |
|-----------|-------|
| Code Generation | 566 |
| Rewrite | 2 |
| Text Generation | 8 |
| Closed QA | 15 |
| Open QA | 1 |

## 2. Descriptive Statistics per Difficulty Level


### Easy Questions

| Dimension | Min | Max | Mean | Std Dev | Q1 | Median | Q3 |
|-----------|-----|-----|------|---------|----|--------|----|
| creativity_scope | 0.001 | 0.366 | 0.035 | 0.054 | 0.008 | 0.015 | 0.037 |
| reasoning | 0.004 | 0.308 | 0.060 | 0.050 | 0.028 | 0.044 | 0.072 |
| constraint_ct | 0.025 | 0.975 | 0.615 | 0.270 | 0.411 | 0.680 | 0.861 |
| contextual_knowledge | 0.013 | 0.447 | 0.059 | 0.050 | 0.035 | 0.048 | 0.065 |
| domain_knowledge | 0.823 | 0.998 | 0.988 | 0.019 | 0.988 | 0.994 | 0.996 |

### Hard Questions

| Dimension | Min | Max | Mean | Std Dev | Q1 | Median | Q3 |
|-----------|-----|-----|------|---------|----|--------|----|
| creativity_scope | 0.001 | 0.209 | 0.028 | 0.028 | 0.012 | 0.019 | 0.034 |
| reasoning | 0.004 | 0.368 | 0.065 | 0.052 | 0.031 | 0.049 | 0.080 |
| constraint_ct | 0.073 | 0.973 | 0.644 | 0.217 | 0.496 | 0.706 | 0.812 |
| contextual_knowledge | 0.007 | 0.382 | 0.050 | 0.033 | 0.032 | 0.040 | 0.056 |
| domain_knowledge | 0.621 | 0.998 | 0.991 | 0.021 | 0.994 | 0.996 | 0.996 |

## 3. Easy vs Hard Comparison

### Mean Values Comparison

| Dimension | Easy | Hard | Difference | % Difference |
|-----------|------|------|------------|--------------|
| creativity_scope | 0.035 | 0.028 | -0.008 | -21.5% |
| reasoning | 0.060 | 0.065 | +0.005 | +8.6% |
| constraint_ct | 0.615 | 0.644 | +0.029 | +4.7% |
| contextual_knowledge | 0.059 | 0.050 | -0.009 | -16.0% |
| domain_knowledge | 0.988 | 0.991 | +0.002 | +0.2% |

### Distribution Comparison

| Dimension | Easy Std Dev | Hard Std Dev | Difference |
|-----------|-------------|-------------|------------|
| creativity_scope | 0.054 | 0.028 | -0.026 |
| reasoning | 0.050 | 0.052 | +0.002 |
| constraint_ct | 0.270 | 0.217 | -0.053 |
| contextual_knowledge | 0.050 | 0.033 | -0.018 |
| domain_knowledge | 0.019 | 0.021 | +0.002 |

### Range Comparison

| Dimension | Easy Range | Hard Range | Difference |
|-----------|------------|------------|------------|
| creativity_scope | 0.365 | 0.209 | -0.157 |
| reasoning | 0.303 | 0.363 | +0.060 |
| constraint_ct | 0.950 | 0.900 | -0.050 |
| contextual_knowledge | 0.434 | 0.375 | -0.059 |
| domain_knowledge | 0.175 | 0.377 | +0.202 |

## 4. Summary Statistics

| Difficulty | Total Questions | Successful | Failed | Success Rate |
|------------|----------------|------------|--------|--------------|
| Easy | 362 | 362 | 0 | 100.0% |
| Hard | 592 | 592 | 0 | 100.0% |

## 5. Key Findings

### Mean Value Differences

- Positive differences indicate higher values in Hard questions

- Negative differences indicate higher values in Easy questions

- Percentage differences show relative change between difficulty levels


### Distribution Differences

- Larger standard deviation in Hard questions indicates more variability

- Range differences show the spread of values in each difficulty level
