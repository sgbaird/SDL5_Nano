"""
Example implementation of improved hypervolume analysis and visualization
for the SDL5_Nano multi-objective optimization study.

This demonstrates better approaches than the current "per formulation" hypervolume.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def cumulative_hypervolume_progress(data, objectives, reference_point):
    """
    Calculate cumulative hypervolume of Pareto front over iterations.
    
    This replaces individual formulation HV with proper BO progress tracking.
    """
    cumulative_hv = []
    iterations = sorted(data['iteration'].unique()) if 'iteration' in data.columns else sorted(data['trial_index'].unique())
    
    for iteration in iterations:
        # Get all data up to this iteration
        iter_col = 'iteration' if 'iteration' in data.columns else 'trial_index'
        data_up_to_iter = data[data[iter_col] <= iteration]
        
        # Find non-dominated solutions (Pareto front)
        nds = NonDominatedSorting()
        pareto_indices = nds.do(data_up_to_iter[objectives].values)[0]
        
        if len(pareto_indices) > 0:
            pareto_solutions = data_up_to_iter.iloc[pareto_indices][objectives].values
            
            # Calculate hypervolume of Pareto front
            hv = Hypervolume(ref_point=reference_point)
            cumulative_hv.append(hv.do(pareto_solutions))
        else:
            cumulative_hv.append(0.0)
    
    return iterations, cumulative_hv


def calculate_crowding_distance(solutions):
    """Calculate crowding distance for diversity measure."""
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


def select_diverse_pareto_solutions(data, objectives, n_select=10):
    """
    Select diverse solutions from Pareto front using multiple criteria.
    
    This replaces pure individual HV ranking with proper multi-objective selection.
    """
    # 1. Pareto ranking
    nds = NonDominatedSorting()
    pareto_fronts = nds.do(data[objectives].values)
    
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
            distances = calculate_crowding_distance(front_data)
            diverse_indices = np.argsort(distances)[::-1][:n_select - len(selected_indices)]
            selected_indices.extend([front[i] for i in diverse_indices])
    
    return data.iloc[selected_indices]


def plot_cumulative_hypervolume_progress(data, objectives, reference_point, title_suffix=""):
    """Plot cumulative hypervolume progress over iterations."""
    iterations, cumulative_hv = cumulative_hypervolume_progress(data, objectives, reference_point)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, cumulative_hv, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Cumulative Hypervolume')
    plt.title(f'Bayesian Optimization Progress: Hypervolume of Pareto Front{title_suffix}')
    plt.grid(True, alpha=0.3)
    
    # Add improvement information
    if len(cumulative_hv) > 1:
        total_improvement = cumulative_hv[-1] - cumulative_hv[0]
        avg_improvement = total_improvement / len(iterations)
        plt.text(0.02, 0.95, f'Total HV improvement: {total_improvement:.4f}\nAvg. per iteration: {avg_improvement:.4f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8),
                verticalalignment='top')
    
    return plt.gcf()


def plot_pareto_front_evolution(data, objectives, iterations_to_show=None, obj_pair=(0, 1)):
    """Plot how Pareto front evolves over iterations."""
    iter_col = 'iteration' if 'iteration' in data.columns else 'trial_index'
    all_iterations = sorted(data[iter_col].unique())
    
    if iterations_to_show is None:
        n_show = min(4, len(all_iterations))
        step = max(1, len(all_iterations) // n_show)
        iterations_to_show = all_iterations[::step]
        if all_iterations[-1] not in iterations_to_show:
            iterations_to_show.append(all_iterations[-1])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    obj1, obj2 = objectives[obj_pair[0]], objectives[obj_pair[1]]
    
    for idx, iteration in enumerate(iterations_to_show[:4]):
        ax = axes[idx]
        
        # Get data up to this iteration
        data_iter = data[data[iter_col] <= iteration]
        
        # Find Pareto front
        nds = NonDominatedSorting()
        pareto_indices = nds.do(data_iter[objectives].values)[0]
        
        if len(pareto_indices) > 0:
            pareto_data = data_iter.iloc[pareto_indices]
            non_pareto_data = data_iter.drop(pareto_data.index)
            
            # Plot
            if len(non_pareto_data) > 0:
                ax.scatter(non_pareto_data[obj1], non_pareto_data[obj2], 
                          alpha=0.4, c='lightgray', s=20, label='Dominated')
            ax.scatter(pareto_data[obj1], pareto_data[obj2], 
                      c='red', s=50, alpha=0.8, label='Pareto Front', edgecolors='darkred')
        
        ax.set_xlabel(obj1)
        ax.set_ylabel(obj2)
        ax.set_title(f'Iteration {iteration} (n={len(data_iter)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_parallel_coordinates(data, objectives, top_n=20):
    """Parallel coordinates plot for multi-objective trade-offs."""
    # Select diverse top solutions
    top_solutions = select_diverse_pareto_solutions(data, objectives, top_n)
    
    # Normalize objectives for comparison (0-1 scale)
    normalized_data = top_solutions[objectives].copy()
    for obj in objectives:
        obj_min, obj_max = data[obj].min(), data[obj].max()
        if obj_max > obj_min:
            normalized_data[obj] = (normalized_data[obj] - obj_min) / (obj_max - obj_min)
        else:
            normalized_data[obj] = 0.5  # If no variation, set to middle
    
    # Create parallel coordinates plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each solution as a line
    for idx, (_, row) in enumerate(normalized_data.iterrows()):
        color = plt.cm.viridis(idx / len(normalized_data))
        ax.plot(range(len(objectives)), row.values, 'o-', 
               alpha=0.7, linewidth=1.5, markersize=6, color=color)
    
    ax.set_xticks(range(len(objectives)))
    ax.set_xticklabels(objectives, rotation=45, ha='right')
    ax.set_ylabel('Normalized Objective Value (0=worst, 1=best)')
    ax.set_title(f'Multi-objective Trade-offs: Top {len(normalized_data)} Diverse Solutions')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    return fig


def plot_objective_correlation_matrix(data, objectives):
    """Show correlation between objectives."""
    corr_matrix = data[objectives].corr()
    
    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Show only lower triangle
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, 
                square=True, fmt='.3f', cbar_kws={"shrink": .8})
    plt.title('Objective Correlation Matrix')
    plt.tight_layout()
    
    return plt.gcf()


def calculate_mo_performance_metrics(data, objectives, reference_point):
    """Calculate proper multi-objective performance metrics."""
    # Find Pareto front
    nds = NonDominatedSorting()
    pareto_indices = nds.do(data[objectives].values)[0]
    
    metrics = {}
    
    if len(pareto_indices) > 0:
        pareto_solutions = data.iloc[pareto_indices][objectives].values
        
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
    else:
        metrics = {
            'hypervolume': 0,
            'n_pareto_solutions': 0,
            'spread': 0,
            'coverage': {obj: 0 for obj in objectives}
        }
    
    return metrics


def improved_hv_analysis_demo(data, objectives, reference_point):
    """
    Demonstration of improved hypervolume analysis to replace current implementation.
    
    This function shows how to:
    1. Calculate cumulative HV instead of individual formulation HV
    2. Select diverse solutions instead of pure HV ranking
    3. Visualize optimization progress and trade-offs
    """
    
    print("=== Improved Multi-Objective Analysis ===\n")
    
    # 1. Calculate performance metrics
    metrics = calculate_mo_performance_metrics(data, objectives, reference_point)
    print(f"Performance Metrics:")
    print(f"  - Pareto Front Hypervolume: {metrics['hypervolume']:.4f}")
    print(f"  - Number of Pareto Solutions: {metrics['n_pareto_solutions']}")
    print(f"  - Solution Diversity (spread): {metrics['spread']:.4f}")
    print(f"  - Objective Coverage:")
    for obj, coverage in metrics['coverage'].items():
        print(f"    * {obj}: {coverage:.3f}")
    
    # 2. Select diverse top solutions (instead of pure HV ranking)
    print(f"\n=== Diverse Solution Selection ===")
    diverse_solutions = select_diverse_pareto_solutions(data, objectives, n_select=10)
    print(f"Selected {len(diverse_solutions)} diverse solutions from Pareto fronts")
    
    # 3. Create improved visualizations
    print(f"\n=== Creating Improved Visualizations ===")
    
    # Plot cumulative hypervolume progress
    fig1 = plot_cumulative_hypervolume_progress(data, objectives, reference_point)
    
    # Plot Pareto front evolution
    fig2 = plot_pareto_front_evolution(data, objectives)
    
    # Plot parallel coordinates for trade-offs
    fig3 = plot_parallel_coordinates(data, objectives)
    
    # Plot objective correlations
    fig4 = plot_objective_correlation_matrix(data, objectives)
    
    print("Generated 4 improved visualization plots")
    print("\nThese approaches provide:")
    print("✓ Accurate BO progress tracking (cumulative HV)")
    print("✓ Diverse solution selection (Pareto ranking + crowding distance)")
    print("✓ Clear trade-off visualization (parallel coordinates)")
    print("✓ Objective relationship analysis (correlation matrix)")
    
    return diverse_solutions, metrics


# Example usage (commented out to avoid execution):
"""
# Load your data
data = pd.read_csv('your_data.csv')

# Define objectives and reference point (from current implementation)
objectives = ['Theoretical loading_min', 'Size', 'PDI', 'Complexity']
reference_point = [0.2, 1.2, 1.2, 1.2]

# Run improved analysis
diverse_solutions, metrics = improved_hv_analysis_demo(data, objectives, reference_point)

# Display/save plots
plt.show()
"""