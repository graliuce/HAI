#!/usr/bin/env python3
"""
Hyperparameter tuning script for the gridworld experiment.

This script tunes:
1. plackett-luce-learning-rate for additive valuations WITHOUT queries
2. linear-gaussian-noise-variance for additive valuations WITH queries
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def run_experiment(
    hyperparameter: str,
    value: float,
    base_args: Dict[str, Any],
    output_base_dir: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a single experiment with a specific hyperparameter value.
    
    Args:
        hyperparameter: Name of the hyperparameter to tune
        value: Value to test
        base_args: Dictionary of base arguments for the experiment
        output_base_dir: Base directory for results
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary containing results summary
    """
    # Create output directory for this run
    run_name = f"{hyperparameter}_{value}"
    output_dir = os.path.join(output_base_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [sys.executable, "run_experiment.py"]
    
    # Add base arguments
    for key, val in base_args.items():
        if isinstance(val, bool):
            if val:
                cmd.append(f"--{key}")
        elif isinstance(val, (list, tuple)):
            cmd.append(f"--{key}")
            cmd.append(",".join(str(v) for v in val))
        else:
            cmd.append(f"--{key}")
            cmd.append(str(val))
    
    # Add the hyperparameter being tuned
    cmd.extend([f"--{hyperparameter}", str(value)])
    
    # Add output directory
    cmd.extend(["--output-dir", output_dir])
    
    print(f"\n{'='*80}")
    print(f"Running experiment: {hyperparameter}={value}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")
    
    if verbose:
        print(f"Command: {' '.join(cmd)}\n")
    
    # Run experiment
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=True
        )
        
        if verbose and result.stdout:
            print(result.stdout)
        
        # Load results
        summary_path = os.path.join(output_dir, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            # Extract key metrics
            results = {
                'hyperparameter': hyperparameter,
                'value': value,
                'output_dir': output_dir,
                'summary': summary,
                'success': True
            }
            
            # Calculate average performance across property counts
            prop_results = {}
            for key, val in summary['summary'].items():
                if key.isdigit():
                    prop_count = int(key)
                    prop_results[prop_count] = val
            
            if prop_results:
                avg_eval_mean = np.mean([v['eval_mean'] for v in prop_results.values()])
                avg_queries = np.mean([v.get('avg_queries', 0) for v in prop_results.values()])
                
                results['avg_eval_mean'] = avg_eval_mean
                results['avg_queries'] = avg_queries
                results['prop_results'] = prop_results
                
                print(f"✓ Complete: avg_eval_mean={avg_eval_mean:.2f}, avg_queries={avg_queries:.2f}")
            else:
                print("⚠ Warning: No property count results found")
            
            return results
        else:
            print(f"⚠ Warning: Summary file not found at {summary_path}")
            return {
                'hyperparameter': hyperparameter,
                'value': value,
                'output_dir': output_dir,
                'success': False,
                'error': 'No summary file'
            }
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with error code {e.returncode}")
        if not verbose and e.stdout:
            print("STDOUT:", e.stdout[-1000:])  # Last 1000 chars
        if not verbose and e.stderr:
            print("STDERR:", e.stderr[-1000:])
        return {
            'hyperparameter': hyperparameter,
            'value': value,
            'output_dir': output_dir,
            'success': False,
            'error': str(e)
        }


def plot_combined_tuning_results(
    results: List[Dict[str, Any]],
    output_dir: str
):
    """
    Plot 2D heatmap for combined hyperparameter tuning.
    
    Args:
        results: List of result dictionaries with pl_lr and lg_noise
        output_dir: Directory to save plots
    """
    # Filter successful results
    successful = [r for r in results if r.get('success', False) and 'avg_eval_mean' in r]
    
    if not successful:
        print("No successful results to plot")
        return
    
    # Extract unique parameter values
    pl_values = sorted(list(set(r['pl_lr'] for r in successful)))
    lg_values = sorted(list(set(r['lg_noise'] for r in successful)))
    
    # Create performance matrix
    performance_matrix = np.zeros((len(lg_values), len(pl_values)))
    queries_matrix = np.zeros((len(lg_values), len(pl_values)))
    
    for r in successful:
        pl_idx = pl_values.index(r['pl_lr'])
        lg_idx = lg_values.index(r['lg_noise'])
        performance_matrix[lg_idx, pl_idx] = r['avg_eval_mean']
        queries_matrix[lg_idx, pl_idx] = r.get('avg_queries', 0)
    
    # Plot 1: Performance heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(performance_matrix, cmap='viridis', aspect='auto')
    
    ax.set_xticks(np.arange(len(pl_values)))
    ax.set_yticks(np.arange(len(lg_values)))
    ax.set_xticklabels([f'{v:.3f}' for v in pl_values])
    ax.set_yticklabels([f'{v:.2f}' for v in lg_values])
    
    ax.set_xlabel('Plackett-Luce Learning Rate', fontsize=12)
    ax.set_ylabel('Linear-Gaussian Noise Variance', fontsize=12)
    ax.set_title('Performance: Combined Hyperparameter Tuning', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Evaluation Return', rotation=270, labelpad=20)
    
    # Annotate cells with values
    for i in range(len(lg_values)):
        for j in range(len(pl_values)):
            text = ax.text(j, i, f'{performance_matrix[i, j]:.2f}',
                          ha="center", va="center", color="white", fontsize=9)
    
    # Mark best combination
    best_idx = np.unravel_index(np.argmax(performance_matrix), performance_matrix.shape)
    ax.add_patch(plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5), 1, 1,
                               fill=False, edgecolor='red', linewidth=3))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'combined_tuning_performance_heatmap.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPerformance heatmap saved to: {plot_path}")
    plt.close()
    
    # Plot 2: Query usage heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(queries_matrix, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(np.arange(len(pl_values)))
    ax.set_yticks(np.arange(len(lg_values)))
    ax.set_xticklabels([f'{v:.3f}' for v in pl_values])
    ax.set_yticklabels([f'{v:.2f}' for v in lg_values])
    
    ax.set_xlabel('Plackett-Luce Learning Rate', fontsize=12)
    ax.set_ylabel('Linear-Gaussian Noise Variance', fontsize=12)
    ax.set_title('Query Usage: Combined Hyperparameter Tuning', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Queries per Episode', rotation=270, labelpad=20)
    
    # Annotate cells with values
    for i in range(len(lg_values)):
        for j in range(len(pl_values)):
            text = ax.text(j, i, f'{queries_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'combined_tuning_queries_heatmap.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Query usage heatmap saved to: {plot_path}")
    plt.close()
    
    # Plot 3: Line plot showing performance for each pl_lr across lg_noise values
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for j, pl_val in enumerate(pl_values):
        perfs = [performance_matrix[i, j] for i in range(len(lg_values))]
        ax.plot(lg_values, perfs, marker='o', label=f'pl_lr={pl_val:.3f}', linewidth=2)
    
    ax.set_xlabel('Linear-Gaussian Noise Variance', fontsize=12)
    ax.set_ylabel('Average Evaluation Return', fontsize=12)
    ax.set_title('Performance vs LG Noise (by PL Learning Rate)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'combined_tuning_lg_noise_lines.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Line plot (LG noise) saved to: {plot_path}")
    plt.close()
    
    # Plot 4: Line plot showing performance for each lg_noise across pl_lr values
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, lg_val in enumerate(lg_values):
        perfs = [performance_matrix[i, j] for j in range(len(pl_values))]
        ax.plot(pl_values, perfs, marker='s', label=f'lg_noise={lg_val:.2f}', linewidth=2)
    
    ax.set_xlabel('Plackett-Luce Learning Rate', fontsize=12)
    ax.set_ylabel('Average Evaluation Return', fontsize=12)
    ax.set_title('Performance vs PL Learning Rate (by LG Noise)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'combined_tuning_pl_lr_lines.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Line plot (PL learning rate) saved to: {plot_path}")
    plt.close()


def plot_tuning_results(
    results: List[Dict[str, Any]],
    hyperparameter: str,
    output_dir: str
):
    """
    Plot hyperparameter tuning results.
    
    Args:
        results: List of result dictionaries
        hyperparameter: Name of the hyperparameter tuned
        output_dir: Directory to save plots
    """
    # Filter successful results
    successful = [r for r in results if r.get('success', False) and 'avg_eval_mean' in r]
    
    if not successful:
        print("No successful results to plot")
        return
    
    # Sort by hyperparameter value
    successful = sorted(successful, key=lambda x: x['value'])
    
    values = [r['value'] for r in successful]
    avg_means = [r['avg_eval_mean'] for r in successful]
    avg_queries = [r.get('avg_queries', 0) for r in successful]
    
    # Plot 1: Performance vs hyperparameter
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(values, avg_means, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel(hyperparameter.replace('-', ' ').title(), fontsize=12)
    ax.set_ylabel('Average Evaluation Return', fontsize=12)
    ax.set_title(f'Performance vs {hyperparameter}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Mark best value
    best_idx = np.argmax(avg_means)
    best_value = values[best_idx]
    best_mean = avg_means[best_idx]
    ax.axvline(best_value, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_value}')
    ax.scatter([best_value], [best_mean], color='red', s=200, marker='*', zorder=5)
    ax.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'tuning_{hyperparameter}_performance.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPerformance plot saved to: {plot_path}")
    plt.close()
    
    # Plot 2: Queries vs hyperparameter (if applicable)
    if any(q > 0 for q in avg_queries):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(values, avg_queries, marker='s', linewidth=2, markersize=8, color='tab:orange')
        ax.set_xlabel(hyperparameter.replace('-', ' ').title(), fontsize=12)
        ax.set_ylabel('Average Queries per Episode', fontsize=12)
        ax.set_title(f'Query Usage vs {hyperparameter}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'tuning_{hyperparameter}_queries.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Query usage plot saved to: {plot_path}")
        plt.close()
    
    # Plot 3: Performance across property counts for each hyperparameter value
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for r in successful:
        if 'prop_results' in r:
            prop_counts = sorted(r['prop_results'].keys())
            prop_means = [r['prop_results'][p]['eval_mean'] for p in prop_counts]
            ax.plot(prop_counts, prop_means, marker='o', label=f"{hyperparameter}={r['value']}")
    
    ax.set_xlabel('Number of Distinct Properties', fontsize=12)
    ax.set_ylabel('Evaluation Return', fontsize=12)
    ax.set_title(f'Performance vs Property Count (by {hyperparameter})', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'tuning_{hyperparameter}_by_props.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Property count comparison plot saved to: {plot_path}")
    plt.close()


def tune_plackett_luce_lr_without_queries(
    values: List[float],
    base_args: Dict[str, Any],
    output_dir: str,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Tune plackett-luce-learning-rate for additive valuations without queries.
    """
    print("\n" + "="*80)
    print("TUNING: plackett-luce-learning-rate (Additive Valuations, No Queries)")
    print("="*80)
    print(f"Testing values: {values}")
    
    # Ensure required flags are set
    base_args['additive-valuation'] = True
    base_args['use-belief-based-agent'] = True
    base_args.pop('allow-queries', None)  # Remove if present
    
    results = []
    for value in values:
        result = run_experiment(
            'plackett-luce-learning-rate',
            value,
            base_args,
            output_dir,
            verbose
        )
        results.append(result)
    
    return results


def tune_linear_gaussian_noise_with_queries(
    values: List[float],
    base_args: Dict[str, Any],
    output_dir: str,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Tune linear-gaussian-noise-variance for additive valuations with queries.
    """
    print("\n" + "="*80)
    print("TUNING: linear-gaussian-noise-variance (Additive Valuations, With Queries)")
    print("="*80)
    print(f"Testing values: {values}")
    
    # Ensure required flags are set
    base_args['additive-valuation'] = True
    base_args['use-belief-based-agent'] = True
    base_args['allow-queries'] = True
    
    results = []
    for value in values:
        result = run_experiment(
            'linear-gaussian-noise-variance',
            value,
            base_args,
            output_dir,
            verbose
        )
        results.append(result)
    
    return results


def tune_combined_with_queries(
    pl_values: List[float],
    lg_values: List[float],
    base_args: Dict[str, Any],
    output_dir: str,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Tune both plackett-luce-learning-rate and linear-gaussian-noise-variance together
    for additive valuations with queries (grid search).
    """
    print("\n" + "="*80)
    print("COMBINED TUNING: plackett-luce-lr + linear-gaussian-noise (With Queries)")
    print("="*80)
    print(f"Plackett-Luce learning rate values: {pl_values}")
    print(f"Linear-Gaussian noise variance values: {lg_values}")
    print(f"Total combinations: {len(pl_values) * len(lg_values)}")
    
    # Ensure required flags are set
    base_args['additive-valuation'] = True
    base_args['use-belief-based-agent'] = True
    base_args['allow-queries'] = True
    
    results = []
    total = len(pl_values) * len(lg_values)
    current = 0
    
    for pl_val in pl_values:
        for lg_val in lg_values:
            current += 1
            print(f"\n[{current}/{total}] Testing pl_lr={pl_val}, lg_noise={lg_val}")
            
            # Create output directory for this combination
            run_name = f"pl_lr_{pl_val}_lg_noise_{lg_val}"
            run_output_dir = os.path.join(output_dir, run_name)
            os.makedirs(run_output_dir, exist_ok=True)
            
            # Build command
            cmd = [sys.executable, "run_experiment.py"]
            
            # Add base arguments
            for key, val in base_args.items():
                if isinstance(val, bool):
                    if val:
                        cmd.append(f"--{key}")
                elif isinstance(val, (list, tuple)):
                    cmd.append(f"--{key}")
                    cmd.append(",".join(str(v) for v in val))
                else:
                    cmd.append(f"--{key}")
                    cmd.append(str(val))
            
            # Add both hyperparameters
            cmd.extend(["--plackett-luce-learning-rate", str(pl_val)])
            cmd.extend(["--linear-gaussian-noise-variance", str(lg_val)])
            
            # Add output directory
            cmd.extend(["--output-dir", run_output_dir])
            
            if verbose:
                print(f"Command: {' '.join(cmd)}\n")
            
            # Run experiment
            try:
                result_process = subprocess.run(
                    cmd,
                    capture_output=not verbose,
                    text=True,
                    check=True
                )
                
                if verbose and result_process.stdout:
                    print(result_process.stdout)
                
                # Load results
                summary_path = os.path.join(run_output_dir, "summary.json")
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                    
                    # Extract key metrics
                    result = {
                        'pl_lr': pl_val,
                        'lg_noise': lg_val,
                        'output_dir': run_output_dir,
                        'summary': summary,
                        'success': True
                    }
                    
                    # Calculate average performance
                    prop_results = {}
                    for key, val in summary['summary'].items():
                        if key.isdigit():
                            prop_count = int(key)
                            prop_results[prop_count] = val
                    
                    if prop_results:
                        avg_eval_mean = np.mean([v['eval_mean'] for v in prop_results.values()])
                        avg_queries = np.mean([v.get('avg_queries', 0) for v in prop_results.values()])
                        
                        result['avg_eval_mean'] = avg_eval_mean
                        result['avg_queries'] = avg_queries
                        result['prop_results'] = prop_results
                        
                        print(f"✓ Complete: avg_eval_mean={avg_eval_mean:.2f}, avg_queries={avg_queries:.2f}")
                    
                    results.append(result)
                else:
                    print(f"⚠ Warning: Summary file not found")
                    results.append({
                        'pl_lr': pl_val,
                        'lg_noise': lg_val,
                        'output_dir': run_output_dir,
                        'success': False,
                        'error': 'No summary file'
                    })
                    
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed with error code {e.returncode}")
                results.append({
                    'pl_lr': pl_val,
                    'lg_noise': lg_val,
                    'output_dir': run_output_dir,
                    'success': False,
                    'error': str(e)
                })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for gridworld experiment"
    )
    
    # Tuning mode selection
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=['plackett-luce-lr', 'linear-gaussian-noise', 'both', 'combined-with-queries'],
        help="Which hyperparameter(s) to tune. 'combined-with-queries' tunes both pl-lr and lg-noise together with queries enabled."
    )
    
    # Hyperparameter value ranges
    parser.add_argument(
        "--plackett-luce-values", type=str, default="0.01,0.05,0.1,0.2,0.5",
        help="Comma-separated values for plackett-luce-learning-rate (default: 0.01,0.05,0.1,0.2,0.5)"
    )
    parser.add_argument(
        "--linear-gaussian-values", type=str, default="0.1,0.5,1.0,2.0,5.0",
        help="Comma-separated values for linear-gaussian-noise-variance (default: 0.1,0.5,1.0,2.0,5.0)"
    )
    
    # Base experiment parameters
    parser.add_argument(
        "--grid-size", type=int, default=10,
        help="Size of the grid (default: 10)"
    )
    parser.add_argument(
        "--num-objects", type=int, default=20,
        help="Number of objects in the grid (default: 20)"
    )
    parser.add_argument(
        "--train-episodes", type=int, default=0,
        help="Number of training episodes - NOTE: For belief-based agents, training is a no-op. "
             "Set to 0 or small value (default: 0)"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=20,
        help="Number of evaluation episodes - this is where belief-based agents are actually tested (default: 20)"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=1,
        help="Number of random seeds to average over (default: 1)"
    )
    parser.add_argument(
        "--property-counts", type=str, default="1,2,3,4,5",
        help="Comma-separated list of property counts to test (default: 1,2,3,4,5)"
    )
    parser.add_argument(
        "--query-budget", type=int, default=5,
        help="Query budget for experiments with queries (default: 5)"
    )
    
    # Output parameters
    parser.add_argument(
        "--output-dir", type=str, default="tuning_results",
        help="Base directory to save tuning results (default: tuning_results)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print verbose output from each experiment"
    )
    
    args = parser.parse_args()
    
    # Parse hyperparameter values
    plackett_luce_values = [float(x) for x in args.plackett_luce_values.split(',')]
    linear_gaussian_values = [float(x) for x in args.linear_gaussian_values.split(',')]
    
    # Create base arguments dictionary
    base_args = {
        'grid-size': args.grid_size,
        'num-objects': args.num_objects,
        'train-episodes': args.train_episodes,
        'eval-episodes': args.eval_episodes,
        'num-seeds': args.num_seeds,
        'property-counts': args.property_counts,
        'query-budget': args.query_budget,
        'no-plot': False,  # We want plots for individual runs
    }
    
    # Create base output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_base, exist_ok=True)
    
    print("="*80)
    print("HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Output directory: {output_base}")
    print(f"\nIMPORTANT NOTE:")
    print(f"  These experiments use the BELIEF-BASED agent, not DQN.")
    print(f"  The belief-based agent does NOT train - it uses Bayesian inference in real-time.")
    print(f"  'Training episodes' = episodes without queries (observation-only learning)")
    print(f"  'Evaluation episodes' = episodes with queries enabled (actual testing)")
    print(f"\nBase experiment settings:")
    print(f"  Grid size: {args.grid_size}")
    print(f"  Objects: {args.num_objects}")
    print(f"  Training episodes: {args.train_episodes} (no queries)")
    print(f"  Evaluation episodes: {args.eval_episodes} (queries enabled)")
    print(f"  Seeds: {args.num_seeds}")
    print(f"  Property counts: {args.property_counts}")
    
    all_results = {}
    
    # Tune both parameters together (combined grid search)
    if args.mode == 'combined-with-queries':
        output_dir = os.path.join(output_base, 'combined_with_queries')
        os.makedirs(output_dir, exist_ok=True)
        
        results = tune_combined_with_queries(
            plackett_luce_values,
            linear_gaussian_values,
            base_args.copy(),
            output_dir,
            args.verbose
        )
        all_results['combined'] = results
        
        # Plot results
        plot_combined_tuning_results(results, output_dir)
        
        # Save results summary
        summary_path = os.path.join(output_dir, 'tuning_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'mode': 'combined-with-queries',
                'pl_values_tested': plackett_luce_values,
                'lg_values_tested': linear_gaussian_values,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"\nTuning summary saved to: {summary_path}")
        
        # Print best combination
        successful = [r for r in results if r.get('success', False) and 'avg_eval_mean' in r]
        if successful:
            best = max(successful, key=lambda x: x['avg_eval_mean'])
            print(f"\n{'='*80}")
            print(f"BEST COMBINATION:")
            print(f"  plackett-luce-learning-rate: {best['pl_lr']}")
            print(f"  linear-gaussian-noise-variance: {best['lg_noise']}")
            print(f"  Average evaluation return: {best['avg_eval_mean']:.2f}")
            print(f"  Average queries: {best.get('avg_queries', 0):.2f}")
            print(f"{'='*80}")
    
    # Tune plackett-luce-learning-rate
    if args.mode in ['plackett-luce-lr', 'both']:
        output_dir = os.path.join(output_base, 'plackett_luce_lr_no_queries')
        os.makedirs(output_dir, exist_ok=True)
        
        results = tune_plackett_luce_lr_without_queries(
            plackett_luce_values,
            base_args.copy(),
            output_dir,
            args.verbose
        )
        all_results['plackett-luce-learning-rate'] = results
        
        # Plot results
        plot_tuning_results(results, 'plackett-luce-learning-rate', output_dir)
        
        # Save results summary
        summary_path = os.path.join(output_dir, 'tuning_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'hyperparameter': 'plackett-luce-learning-rate',
                'values_tested': plackett_luce_values,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"\nTuning summary saved to: {summary_path}")
        
        # Print best value
        successful = [r for r in results if r.get('success', False) and 'avg_eval_mean' in r]
        if successful:
            best = max(successful, key=lambda x: x['avg_eval_mean'])
            print(f"\n{'='*80}")
            print(f"BEST plackett-luce-learning-rate: {best['value']}")
            print(f"Average evaluation return: {best['avg_eval_mean']:.2f}")
            print(f"{'='*80}")
    
    # Tune linear-gaussian-noise-variance
    if args.mode in ['linear-gaussian-noise', 'both']:
        output_dir = os.path.join(output_base, 'linear_gaussian_noise_with_queries')
        os.makedirs(output_dir, exist_ok=True)
        
        results = tune_linear_gaussian_noise_with_queries(
            linear_gaussian_values,
            base_args.copy(),
            output_dir,
            args.verbose
        )
        all_results['linear-gaussian-noise-variance'] = results
        
        # Plot results
        plot_tuning_results(results, 'linear-gaussian-noise-variance', output_dir)
        
        # Save results summary
        summary_path = os.path.join(output_dir, 'tuning_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'hyperparameter': 'linear-gaussian-noise-variance',
                'values_tested': linear_gaussian_values,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"\nTuning summary saved to: {summary_path}")
        
        # Print best value
        successful = [r for r in results if r.get('success', False) and 'avg_eval_mean' in r]
        if successful:
            best = max(successful, key=lambda x: x['avg_eval_mean'])
            print(f"\n{'='*80}")
            print(f"BEST linear-gaussian-noise-variance: {best['value']}")
            print(f"Average evaluation return: {best['avg_eval_mean']:.2f}")
            print(f"Average queries: {best.get('avg_queries', 0):.2f}")
            print(f"{'='*80}")
    
    # Save overall summary
    overall_summary_path = os.path.join(output_base, 'overall_summary.json')
    with open(overall_summary_path, 'w') as f:
        json.dump({
            'mode': args.mode,
            'all_results': all_results,
            'base_args': base_args,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"\nOverall summary saved to: {overall_summary_path}")
    
    print("\n" + "="*80)
    print("TUNING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

