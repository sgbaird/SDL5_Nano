# Analysis of "Per Formulation" Hypervolume Implementation

## Current Implementation Overview

Based on the reviewer's comment and examination of the codebase, I've identified how the "per formulation" hypervolume is currently implemented and why it's problematic.

## What is "Per Formulation" Hypervolume?

The current implementation calculates hypervolume for **individual formulations** rather than the collective Pareto front. Here's the key code from `1st_round_screening/1st_data_analysis.ipynb`:

```python
def hv(df_norm):
    data = df_norm.copy()
    data['Theoretical loading_min'] = -data['Theoretical loading']
    
    # Select relevant objectives for hypervolume calculation
    objectives = ['Theoretical loading_min', 'Size', 'PDI', 'Complexity']
    
    # Define the reference point (worst case + 20%)
    reference_point = [0.2, 1.2, 1.2, 1.2]
    
    hv_values = []
    
    for trial in sorted(data['trial_index']):
        formulation = data[data['trial_index'] == trial][objectives].values
        hv = Hypervolume(ref_point=reference_point)
        hv_values.append(hv.do(formulation))  # Individual formulation HV
        
    # ... rest of function
```

**Key Issues with This Approach:**

1. **Individual vs. Collective Assessment**: Each formulation gets its own hypervolume calculated against a fixed reference point
2. **Misleading Progress Tracking**: This doesn't reflect the actual optimization progress of the Pareto front
3. **Selection Bias**: A formulation performing moderately across all objectives might get higher individual HV than one excelling in some objectives but underperforming in others

## How It's Used in the Study

From the manuscript and code analysis:

1. **Figure 3a**: Shows individual formulation hypervolumes grouped by iteration
2. **Top 10 Selection**: Uses individual hypervolume as ranking criterion:
   ```python
   # From line 1772 in 1st_data_analysis.ipynb
   data_ML_sorted = data_ML.sort_values(by="Hypervolume", ascending=False)
   top_10 = data_HM.sort_values(by="Hypervolume", ascending=False).head(10)
   ```

3. **BO Acquisition Function**: The Bayesian optimization uses `qLogNoisyExpectedHypervolumeImprovement` (line 11, 23 in `sdlnano.py`), which is designed for multi-objective optimization and should work with proper Pareto front tracking.

## Reviewer's Concerns

The reviewer correctly identifies several issues:

> "Using the hypervolume (HV) of individual nanoformulations as the primary performance metric for BO may not accurately reflect the algorithm's overall progress toward identifying an improved Pareto front."

> "We would suggest instead considering to calculate the cumulative hypervolume, which quantifies the size of the dominated region in the objective space up to a given iteration."

## The Problem Illustrated

Consider two formulations:
- **Formulation A**: Size=100nm, PDI=0.2, Loading=0.3, Complexity=5 (moderate across all)
- **Formulation B**: Size=50nm, PDI=0.1, Loading=0.8, Complexity=10 (excels in some, poor in complexity)

Using individual hypervolume, Formulation A might score higher than B, but B could be more valuable for the overall Pareto front when combined with other solutions.

## What Should Be Done Instead

1. **Cumulative Hypervolume**: Calculate hypervolume of the non-dominated set up to each iteration
2. **Pareto Front Progress**: Track how the Pareto front evolves over iterations
3. **Diversity Consideration**: Ensure selected formulations cover different trade-offs

## Files Containing Hypervolume Implementation

1. `1st_round_screening/1st_data_analysis.ipynb` - Lines 825-856 (main HV function)
2. `2nd_round_screening/2nd_data_analysis.ipynb` - Lines 2583-2624 (similar implementation)
3. `0_helper_functions/sdlnano.py` - Lines 11, 23 (BO acquisition function setup)

## Manuscript References

From `result.md`:
- Section 2.2: "hypervolume was used as a metric to quantify the overall effectiveness of the formulations"
- Figure 3a description: "hypervolume of each formulation was calculated (represented by circles)"
- Selection process: "Using hypervolume as a selection criterion, the top ten formulations were identified"

This analysis confirms the reviewer's concerns about the current hypervolume implementation and provides the foundation for understanding how to improve the multi-objective optimization approach.