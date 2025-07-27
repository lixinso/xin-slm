# Training Metrics and Visualization

This directory now includes enhanced metrics collection and visualization for the GPT language model training.

## New Features

### 1. Metrics Collection
The training script now automatically collects and saves the following metrics:
- **Training losses** at regular intervals (every 100 steps)
- **Validation losses** at the end of each epoch
- **Training steps** and corresponding timestamps
- **Average training loss per epoch**

### 2. Metrics Output
- Metrics are saved to `training_metrics.json` in JSON format
- Contains arrays for train_losses, val_losses, epochs, steps, and timestamps
- Easily readable by other tools and scripts

### 3. Visualization
The `plot_metrics.py` script creates comprehensive training visualizations:
- **Training Loss vs Steps**: Shows loss progression during training
- **Validation Loss vs Epochs**: Tracks validation performance
- **Training Loss vs Time**: Shows loss improvement over time
- **Loss Distribution**: Histogram of loss values

## Usage

### Running Training with Metrics
```bash
python train_gpt_language_model.py
```
This will automatically save metrics to `training_metrics.json`.

### Visualizing Results
```bash
# Basic usage - creates training_curves.png
python plot_metrics.py

# Specify custom input/output files
python plot_metrics.py --metrics-file my_metrics.json --output my_plots.png

# Print summary only (no plots)
python plot_metrics.py --no-plot
```

### Command Line Options
- `--metrics-file, -m`: Path to metrics JSON file (default: training_metrics.json)
- `--output, -o`: Output plot filename (default: training_curves.png)
- `--no-plot`: Only print metrics summary without creating plots

## Metrics Summary
The script automatically prints a comprehensive summary including:
- Initial, final, min, max, mean, and standard deviation for both training and validation losses
- Total training time
- Number of training steps and epochs

## Dependencies
Make sure you have matplotlib installed:
```bash
pip install matplotlib>=3.5.0
```
Or install all requirements:
```bash
pip install -r requirements.txt
```

## Output Files
- `training_metrics.json`: Raw metrics data
- `training_curves.png`: Visualization plots (default name)