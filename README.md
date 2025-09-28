# 2D Navier-Stokes Dataset Analysis

This repository contains tools for analyzing and visualizing the 2D Navier-Stokes dataset with colorful representations and comprehensive analytics.

## Dataset Structure

The dataset is located in the `data/2D Navier-Stokes Datasets` directory with the following structure:

```
data/2D Navier-Stokes Datasets/
├── test/
│   ├── inputs.npy
│   └── label.npy
└── train/
    ├── inputs.npy
    └── label.npy
```

## Features

The analysis includes:

- Colorful visualizations of the fluid dynamics data
- Statistical analysis of input and label datasets
- Distribution analysis and comparisons
- PCA (Principal Component Analysis) for dimensionality reduction
- Interactive visualizations in Jupyter Notebook
- Animation support for temporal data (if applicable)

## Usage

### Running the Analysis Script

```bash
python navier_stokes_analysis.py
```

This will:
1. Load the training and test datasets
2. Compute and print basic statistics
3. Create colorful visualizations of samples
4. Perform PCA analysis
5. Save all visualizations to an `output_plots` directory
6. Generate animations if the data has a time dimension

### Interactive Analysis with Jupyter Notebook

For interactive analysis, you can use the Jupyter Notebook:

```bash
jupyter notebook navier_stokes_visualization.ipynb
```

The notebook provides:
- Interactive sample visualization with sliders
- Customizable colormap exploration
- Distribution analysis
- PCA visualization
- Animations (if data has temporal dimension)

## Requirements

The code requires the following Python packages:
- numpy
- matplotlib
- seaborn
- pandas
- scikit-learn
- Jupyter (for notebook)

You can install them using:

```bash
pip install numpy matplotlib seaborn pandas scikit-learn jupyter
```

## Output

The script will generate:
- Statistical summaries in the console
- Visualization plots saved to `output_plots/`
- A CSV summary file with dataset statistics
- Sample animations (if applicable) in `output_animations/`

## Examples

Here are some examples of the visualizations:

- Input samples with custom colormap
- Comparison between inputs and labels
- Distribution analysis
- PCA visualization
- Temporal data animation (if applicable)

## Analysis Methodology

1. **Data Loading**: Load train and test datasets from NumPy files
2. **Statistical Analysis**: Compute mean, std, min, max, etc.
3. **Visualization**: Create colorful plots with custom colormaps
4. **PCA Analysis**: Reduce dimensionality and visualize principal components
5. **Animation**: Generate temporal animations if time dimension is available
