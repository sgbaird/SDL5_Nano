# Improved Hypervolume Analysis and Visualization Approaches

## Current Issues with Ad-hoc Analysis

The current implementation has several problems:
1. **Individual formulation HV**: Each formulation gets its own hypervolume score against a fixed reference point
2. **Misleading selection**: Top 10 formulations selected purely based on individual HV scores
3. **No progress tracking**: Missing cumulative hypervolume to track actual BO optimization progress
4. **Limited diversity**: Selection doesn't consider trade-offs between objectives

## Recommended Improvements

### 1. Cumulative Hypervolume Tracking

**Current Code Problem:**
```python
# From 1st_data_analysis.ipynb line 843-846
for trial in sorted(data['trial_index']):
    formulation = data[data['trial_index'] == trial][objectives].values
    hv = Hypervolume(ref_point=reference_point)
    hv_values.append(hv.do(formulation))  # Individual formulation HV
```

**Better Approach:**
```python
def cumulative_hypervolume_progress(data, objectives, reference_point):
    """Calculate cumulative hypervolume of Pareto front over iterations"""
    cumulative_hv = []
    
    for iteration in sorted(data['iteration'].unique()):
        # Get all data up to this iteration
        data_up_to_iter = data[data['iteration'] <= iteration]
        
        # Find non-dominated solutions (Pareto front)
        nds = NonDominatedSorting()
        pareto_indices = nds.do(data_up_to_iter[objectives].values)[0]
        pareto_solutions = data_up_to_iter.iloc[pareto_indices][objectives].values
        
        # Calculate hypervolume of Pareto front
        hv = Hypervolume(ref_point=reference_point)
        cumulative_hv.append(hv.do(pareto_solutions))
    
    return cumulative_hv
```

### 2. Multi-Criteria Selection Methods

**Instead of pure HV ranking:**
```python
# Current problematic approach
data_ML_sorted = data_ML.sort_values(by="Hypervolume", ascending=False)
top_10 = data_ML_sorted.head(10)
```

**Better Approach - Pareto Ranking + Diversity:**
```python
def select_diverse_pareto_solutions(data, objectives, n_select=10):
    """Select diverse solutions from Pareto front using multiple criteria"""
    
    # 1. Pareto ranking
    nds = NonDominatedSorting()
    pareto_fronts = nds.do(data[objectives].values)
    
    # 2. Add diversity metric (crowding distance)
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from pymoo.indicators.igd import IGD
    
    selected_indices = []
    
    # Prioritize Pareto front solutions
    for front in pareto_fronts:
        if len(selected_indices) >= n_select:
            break
            
        # Calculate crowding distance for diversity
        if len(front) <= (n_select - len(selected_indices)):
            selected_indices.extend(front)
        else:
            # Select most diverse solutions from this front
            front_data = data.iloc[front][objectives].values
            # Use spacing or crowding distance
            distances = calculate_crowding_distance(front_data)
            diverse_indices = np.argsort(distances)[::-1][:n_select - len(selected_indices)]
            selected_indices.extend([front[i] for i in diverse_indices])
    
    return data.iloc[selected_indices]

def calculate_crowding_distance(solutions):
    """Calculate crowding distance for diversity measure"""
    n_solutions, n_objectives = solutions.shape
    distances = np.zeros(n_solutions)
    
    for i in range(n_objectives):
        sorted_indices = np.argsort(solutions[:, i])
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        obj_range = solutions[sorted_indices[-1], i] - solutions[sorted_indices[0], i]
        if obj_range > 0:
            for j in range(1, n_solutions - 1):
                distances[sorted_indices[j]] += (
                    solutions[sorted_indices[j + 1], i] - solutions[sorted_indices[j - 1], i]
                ) / obj_range
    
    return distances
```

### 3. Enhanced Visualization Functions

**A. Pareto Front Evolution:**
```python
def plot_pareto_front_evolution(data, objectives, iterations_to_show=None):
    """Plot how Pareto front evolves over iterations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    if iterations_to_show is None:
        iterations_to_show = [1, max(data['iteration'])//3, 2*max(data['iteration'])//3, max(data['iteration'])]
    
    for idx, iteration in enumerate(iterations_to_show):
        ax = axes[idx//2, idx%2]
        
        # Get data up to this iteration
        data_iter = data[data['iteration'] <= iteration]
        
        # Find Pareto front
        nds = NonDominatedSorting()
        pareto_indices = nds.do(data_iter[objectives].values)[0]
        pareto_data = data_iter.iloc[pareto_indices]
        non_pareto_data = data_iter.drop(pareto_data.index)
        
        # Plot
        ax.scatter(non_pareto_data[objectives[0]], non_pareto_data[objectives[1]], 
                  alpha=0.3, c='gray', label='Dominated')
        ax.scatter(pareto_data[objectives[0]], pareto_data[objectives[1]], 
                  c='red', s=50, label='Pareto Front')
        
        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        ax.set_title(f'Iteration {iteration}')
        ax.legend()
    
    plt.tight_layout()
    return fig

def plot_cumulative_hypervolume_progress(data, objectives, reference_point):
    """Plot cumulative hypervolume progress over iterations"""
    iterations = sorted(data['iteration'].unique())
    cumulative_hv = cumulative_hypervolume_progress(data, objectives, reference_point)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, cumulative_hv, 'b-', linewidth=2, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Hypervolume')
    plt.title('Bayesian Optimization Progress: Hypervolume of Pareto Front')
    plt.grid(True, alpha=0.3)
    
    # Add improvement rate annotation
    improvement_rate = (cumulative_hv[-1] - cumulative_hv[0]) / len(iterations)
    plt.text(0.7, 0.2, f'Avg. HV improvement: {improvement_rate:.4f}/iter', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    return plt.gcf()
```

**B. Multi-objective Trade-off Analysis:**
```python
def plot_parallel_coordinates(data, objectives, top_n=20):
    """Parallel coordinates plot for multi-objective trade-offs"""
    # Select diverse top solutions
    top_solutions = select_diverse_pareto_solutions(data, objectives, top_n)
    
    # Normalize objectives for comparison
    normalized_data = top_solutions[objectives].copy()
    for obj in objectives:
        normalized_data[obj] = (normalized_data[obj] - data[obj].min()) / (data[obj].max() - data[obj].min())
    
    # Create parallel coordinates plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for idx, row in normalized_data.iterrows():
        ax.plot(range(len(objectives)), row.values, 'o-', alpha=0.7, linewidth=1.5)
    
    ax.set_xticks(range(len(objectives)))
    ax.set_xticklabels(objectives, rotation=45)
    ax.set_ylabel('Normalized Objective Value')
    ax.set_title(f'Multi-objective Trade-offs: Top {top_n} Diverse Solutions')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_objective_correlation_matrix(data, objectives):
    """Show correlation between objectives"""
    corr_matrix = data[objectives].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f')
    plt.title('Objective Correlation Matrix')
    plt.tight_layout()
    
    return plt.gcf()
```

### 4. Performance Metrics for Multi-objective Analysis

```python
def calculate_mo_performance_metrics(data, objectives, reference_point):
    """Calculate proper multi-objective performance metrics"""
    # Find Pareto front
    nds = NonDominatedSorting()
    pareto_indices = nds.do(data[objectives].values)[0]
    pareto_solutions = data.iloc[pareto_indices][objectives].values
    
    metrics = {}
    
    # 1. Hypervolume of Pareto front
    hv = Hypervolume(ref_point=reference_point)
    metrics['hypervolume'] = hv.do(pareto_solutions)
    
    # 2. Number of Pareto solutions
    metrics['n_pareto_solutions'] = len(pareto_solutions)
    
    # 3. Spread (diversity of solutions)
    if len(pareto_solutions) > 1:
        distances = []
        for i in range(len(pareto_solutions)):
            min_dist = float('inf')
            for j in range(len(pareto_solutions)):
                if i != j:
                    dist = np.linalg.norm(pareto_solutions[i] - pareto_solutions[j])
                    min_dist = min(min_dist, dist)
            distances.append(min_dist)
        metrics['spread'] = np.std(distances)
    else:
        metrics['spread'] = 0
    
    # 4. Coverage (range in each objective)
    metrics['coverage'] = {}
    for i, obj in enumerate(objectives):
        obj_range = pareto_solutions[:, i].max() - pareto_solutions[:, i].min()
        total_range = data[obj].max() - data[obj].min()
        metrics['coverage'][obj] = obj_range / total_range if total_range > 0 else 0
    
    return metrics
```

## Implementation Recommendations

### 1. Replace Current HV Function
Replace the individual formulation HV calculation with cumulative Pareto front HV tracking.

### 2. Improve Selection Methodology
Use Pareto ranking + diversity instead of pure individual HV sorting for selecting top formulations.

### 3. Add Progress Tracking Visualizations
Include plots showing:
- Cumulative hypervolume vs iteration
- Pareto front evolution over time
- Multi-objective trade-off analysis

### 4. Maintain BO Implementation
Keep the current AxClient and `qLogNoisyExpectedHypervolumeImprovement` acquisition function unchanged, as these are correctly implemented for multi-objective BO.

## Benefits of These Improvements

1. **Accurate Progress Tracking**: Cumulative HV reflects actual BO performance
2. **Better Solution Selection**: Diverse Pareto solutions instead of potentially similar high-individual-HV solutions
3. **Clearer Trade-off Understanding**: Visualizations show objective relationships and compromises
4. **Scientific Rigor**: Proper multi-objective optimization metrics and analysis

These changes would address the reviewer's concerns while providing more meaningful analysis and visualization of the multi-objective optimization results.