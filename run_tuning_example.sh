#!/bin/bash
# Example script showing how to use the hyperparameter tuning script

echo "========================================================================"
echo "Hyperparameter Tuning Examples"
echo "========================================================================"
echo ""
echo "Choose an example to run:"
echo ""
echo "1) Quick test - Tune plackett-luce-lr (3 values, no queries)"
echo "2) Quick test - Tune linear-gaussian-noise (3 values, with queries)"
echo "3) Quick test - Tune BOTH combined (with queries, grid search)"
echo "4) Full tuning - Tune plackett-luce-lr (5 values, no queries)"
echo "5) Full tuning - Tune linear-gaussian-noise (5 values, with queries)"
echo "6) Full tuning - Tune BOTH combined (with queries, grid search)"
echo "7) Comprehensive - Tune both hyperparameters separately"
echo "8) Custom - Enter your own parameters"
echo ""
read -p "Enter choice [1-8]: " choice

case $choice in
    1)
        echo ""
        echo "Running quick test for plackett-luce-learning-rate (no queries)..."
        echo "This will test values: 0.05, 0.1, 0.2"
        echo "Expected runtime: ~10-15 minutes"
        echo ""
        python tune_hyperparameters.py \
            --mode plackett-luce-lr \
            --plackett-luce-values "0.05,0.1,0.2" \
            --train-episodes 0 \
            --eval-episodes 5 \
            --num-seeds 1
        ;;
    
    2)
        echo ""
        echo "Running quick test for linear-gaussian-noise-variance (with queries)..."
        echo "This will test values: 0.5, 1.0, 2.0"
        echo "Expected runtime: ~10-15 minutes"
        echo ""
        python tune_hyperparameters.py \
            --mode linear-gaussian-noise \
            --linear-gaussian-values "0.5,1.0,2.0" \
            --train-episodes 0 \
            --eval-episodes 5 \
            --num-seeds 1
        ;;
    
    3)
        echo ""
        echo "Running quick combined tuning (BOTH parameters with queries)..."
        echo "This will test grid: 3 x 3 = 9 combinations"
        echo "  - plackett-luce-lr: 0.05, 0.1, 0.2"
        echo "  - linear-gaussian-noise: 0.5, 1.0, 2.0"
        echo "Expected runtime: ~30-45 minutes"
        echo ""
        python tune_hyperparameters.py \
            --mode combined-with-queries \
            --plackett-luce-values "0.05,0.1,0.2" \
            --linear-gaussian-values "0.5,1.0,2.0" \
            --train-episodes 0 \
            --eval-episodes 5 \
            --num-seeds 1
        ;;
    
    4)
        echo ""
        echo "Running full tuning for plackett-luce-learning-rate (no queries)..."
        echo "This will test values: 0.01, 0.05, 0.1, 0.2, 0.5"
        echo "Expected runtime: ~30-60 minutes"
        echo ""
        python tune_hyperparameters.py \
            --mode plackett-luce-lr \
            --plackett-luce-values "0.01,0.05,0.1,0.2,0.5" \
            --train-episodes 0 \
            --eval-episodes 15 \
            --num-seeds 1
        ;;
    
    5)
        echo ""
        echo "Running full tuning for linear-gaussian-noise-variance (with queries)..."
        echo "This will test values: 0.1, 0.5, 1.0, 2.0, 5.0"
        echo "Expected runtime: ~30-60 minutes"
        echo ""
        python tune_hyperparameters.py \
            --mode linear-gaussian-noise \
            --linear-gaussian-values "0.1,0.5,1.0,2.0,5.0" \
            --train-episodes 0 \
            --eval-episodes 15 \
            --num-seeds 1
        ;;
    
    6)
        echo ""
        echo "Running full combined tuning (BOTH parameters with queries)..."
        echo "This will test grid: 5 x 5 = 25 combinations"
        echo "  - plackett-luce-lr: 0.01, 0.05, 0.1, 0.2, 0.5"
        echo "  - linear-gaussian-noise: 0.1, 0.5, 1.0, 2.0, 5.0"
        echo "Expected runtime: ~2-4 hours"
        echo ""
        read -p "Continue? [y/N]: " confirm
        if [[ $confirm == [yY] ]]; then
            python tune_hyperparameters.py \
                --mode combined-with-queries \
                --plackett-luce-values "0.01,0.05,0.1,0.2,0.5" \
                --linear-gaussian-values "0.1,0.5,1.0,2.0,5.0" \
                --train-episodes 0 \
                --eval-episodes 15 \
                --num-seeds 1
        else
            echo "Cancelled."
        fi
        ;;
    
    7)
        echo ""
        echo "Running comprehensive tuning for BOTH hyperparameters SEPARATELY..."
        echo "This will test:"
        echo "  - plackett-luce-lr: 0.01, 0.05, 0.1, 0.2, 0.5 (no queries)"
        echo "  - linear-gaussian-noise: 0.1, 0.5, 1.0, 2.0, 5.0 (with queries)"
        echo "Expected runtime: ~1-2 hours"
        echo ""
        read -p "Continue? [y/N]: " confirm
        if [[ $confirm == [yY] ]]; then
            python tune_hyperparameters.py \
                --mode both \
                --plackett-luce-values "0.01,0.05,0.1,0.2,0.5" \
                --linear-gaussian-values "0.1,0.5,1.0,2.0,5.0" \
                --train-episodes 0 \
                --eval-episodes 15 \
                --num-seeds 1
        else
            echo "Cancelled."
        fi
        ;;
    
    8)
        echo ""
        echo "Custom tuning configuration"
        echo ""
        echo "Available modes:"
        echo "  - plackett-luce-lr (no queries)"
        echo "  - linear-gaussian-noise (with queries)"
        echo "  - combined-with-queries (grid search both params)"
        echo "  - both (run both params separately)"
        echo ""
        read -p "Mode: " mode
        read -p "Training episodes: " train_eps
        read -p "Evaluation episodes: " eval_eps
        read -p "Number of seeds: " num_seeds
        
        if [[ $mode == "plackett-luce-lr" ]] || [[ $mode == "both" ]] || [[ $mode == "combined-with-queries" ]]; then
            read -p "Plackett-Luce values (comma-separated): " pl_values
        fi
        
        if [[ $mode == "linear-gaussian-noise" ]] || [[ $mode == "both" ]] || [[ $mode == "combined-with-queries" ]]; then
            read -p "Linear-Gaussian values (comma-separated): " lg_values
        fi
        
        cmd="python tune_hyperparameters.py --mode $mode"
        cmd="$cmd --train-episodes $train_eps"
        cmd="$cmd --eval-episodes $eval_eps"
        cmd="$cmd --num-seeds $num_seeds"
        
        if [[ ! -z "$pl_values" ]]; then
            cmd="$cmd --plackett-luce-values \"$pl_values\""
        fi
        
        if [[ ! -z "$lg_values" ]]; then
            cmd="$cmd --linear-gaussian-values \"$lg_values\""
        fi
        
        echo ""
        echo "Running: $cmd"
        echo ""
        eval $cmd
        ;;
    
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "Tuning complete!"
echo "Check the tuning_results/ directory for detailed results and plots."
echo "========================================================================"

