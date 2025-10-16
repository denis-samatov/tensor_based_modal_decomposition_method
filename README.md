# Tensor-Based Modal Decomposition for Fluid Dynamics

This repository contains a Python implementation of the Tensor-Based Modal Decomposition (TBMD) algorithm, designed for analyzing and forecasting fluid dynamics data. The project provides a complete workflow from data generation and processing to modal decomposition, analysis, and forecasting. It is particularly well-suited for handling high-dimensional spatio-temporal data, such as simulations from 2D Navier-Stokes equations.

The core of this repository is the TBMD algorithm, which leverages tensor decompositions (specifically Higher-Order Singular Value Decomposition - HOSVD) to identify dominant spatial and temporal modes in the data. This allows for effective dimensionality reduction and the extraction of physically meaningful features. The project also includes tools for compressive sensing, enabling the reconstruction of high-resolution data from a limited number of sensors.

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Data Generation](#data-generation)
  - [Running Experiments](#running-experiments)
  - [Interactive Analysis](#interactive-analysis)
- [Analysis Methodology](#analysis-methodology)
- [Output](#output)
- [Examples](#examples)

## Features

- **Colorful visualizations** of fluid dynamics data.
- **Statistical analysis** of input and label datasets.
- **Distribution analysis** and comparisons.
- **Principal Component Analysis (PCA)** for dimensionality reduction.
- **Tensor-Based Modal Decomposition (TBMD)** for identifying dominant modes.
- **Interactive visualizations** in Jupyter Notebooks.
- **Animation support** for temporal data.

## Repository Structure

The repository is organized as follows:

```
.
├── algorithm/
│   ├── TBMD/
│   │   ├── models/
│   │   ├── modules/
│   │   └── utils/
│   ├── experiments/
│   └── notebooks/
├── data/
│   └── 2D Navier-Stokes Datasets/
│       ├── test/
│       └── train/
├── output_plots/
├── output_animations/
├── README.md
└── requirements.txt
```

- **`algorithm/`**: Contains the core TBMD algorithm, experiments, and notebooks.
  - **`TBMD/`**: The main TBMD module.
    - **`models/`**: Forecasting models (LSTM, MLP, Linear).
    - **`modules/`**: Core TBMD components (HOSVD, QR factorization, Compressive Sensing).
    - **`utils/`**: Utility functions for data loading, processing, and plotting.
  - **`experiments/`**: Scripts to run TBMD experiments.
  - **`notebooks/`**: Jupyter notebooks for interactive analysis and visualization.
- **`data/`**: The location for the dataset.
- **`output_plots/`**: Directory where generated plots are saved.
- **`output_animations/`**: Directory where generated animations are saved.

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2.  **Install the required packages:**

    The code requires the following Python packages:
    - `numpy`
    - `matplotlib`
    - `seaborn`
    - `pandas`
    - `scikit-learn`
    - `torch`
    - `tensorly`
    - `tqdm`
    - `jupyter`

    You can install them using pip and the provided `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Generation

If you need to generate the Navier-Stokes dataset, you can use the `data_generation.py` script:

```bash
python algorithm/TBMD/utils/data_generation.py
```

This will generate a `ns_data.mat` file containing the dataset.

### Running Experiments

To run the TBMD experiments, you can execute the main script:

```bash
python algorithm/main_tbmd.py
```

This will:
1.  Load the training and test datasets.
2.  Perform TBMD on the data.
3.  Generate and save visualizations to the `output_plots` directory.
4.  Generate and save animations to the `output_animations` directory.

### Interactive Analysis

For interactive analysis, you can use the Jupyter Notebooks in the `algorithm/notebooks` directory. For example:

```bash
jupyter notebook algorithm/notebooks/data_visualize.ipynb
```

The notebooks provide:
- Interactive sample visualization with sliders.
- Customizable colormap exploration.
- Distribution analysis.
- PCA visualization.
- TBMD analysis.
- Animations.

## Analysis Methodology

1.  **Data Loading**: Load train and test datasets from NumPy files.
2.  **Tensor-Based Modal Decomposition (TBMD)**:
    - **HOSVD**: Apply Higher-Order Singular Value Decomposition to the data tensor.
    - **QR Factorization**: Use QR factorization with tube pivoting for sensor placement.
    - **Compressive Sensing**: Reconstruct the signal from a limited number of sensors.
3.  **Statistical Analysis**: Compute mean, std, min, max, etc.
4.  **Visualization**: Create colorful plots with custom colormaps.
5.  **Forecasting**: Use LSTM, MLP, or Linear models to forecast future states.

## Output

The scripts will generate:
- Statistical summaries in the console.
- Visualization plots saved to `output_plots/`.
- A CSV summary file with dataset statistics.
- Sample animations saved to `output_animations/`.

## Examples

Here are some examples of the visualizations you can generate:

- Input samples with a custom colormap.
- Comparison between inputs and labels.
- Distribution analysis plots.
- PCA visualization.
- TBMD mode visualization.
- Temporal data animation.