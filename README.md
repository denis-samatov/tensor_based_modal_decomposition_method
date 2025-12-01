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
- [Forecasting Models](#forecasting-models)
- [Geometry-Aware Features](#geometry-aware-features)
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
- **Multiple forecasting models** (LSTM, MLP, Linear).
- **Geometry-aware HOSVD and QR factorization** for unstructured meshes.

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
    - **`models/`**: Forecasting models (`LSTMForecaster`, `MLPForecaster`, `LinearForecaster`).
    - **`modules/`**: Core TBMD components, including `TensorHOSVD` for standard decomposition, `GeometryAwareTensorHOSVD` for decompositions on unstructured meshes, `TensorBasedTubeFiberPivotQRFactorization` for sensor placement, and `TensorBasedCompressiveSensing` for signal reconstruction.
    - **`utils/`**: Utility functions for data loading, processing, plotting, and performance metrics.
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

1.  **Data Loading**: Load train and test datasets from various file formats.
2.  **Tensor-Based Modal Decomposition (TBMD)**:
    - **HOSVD**: Apply Higher-Order Singular Value Decomposition to the data tensor to extract the core tensor and factor matrices.
    - **QR Factorization**: Use QR factorization with tube pivoting for sensor placement.
    - **Compressive Sensing**: Reconstruct the signal from a limited number of sensors.
3.  **Statistical Analysis**: Compute descriptive statistics for the datasets.
4.  **Visualization**: Generate plots and animations to visualize the data and results.
5.  **Forecasting**: Use various models to forecast future states.

## Forecasting Models

The repository includes three types of forecasting models:

- **`LSTMForecaster`**: A Long Short-Term Memory (LSTM) model for time series forecasting.
- **`MLPForecaster`**: A Multi-Layer Perceptron (MLP) model for time series forecasting.
- **`LinearForecaster`**: A linear model that learns a transformation matrix using the pseudoinverse.

These models can be found in the `algorithm/TBMD/models` directory and are used to predict future states based on the decomposed modal coefficients.

## Geometry-Aware Features

For datasets based on unstructured meshes, the repository provides geometry-aware versions of HOSVD and QR factorization:

- **`GeometryAwareTensorHOSVD`**: Incorporates a Laplacian regularization term to encourage spatially smooth modes that respect the mesh geometry.
- **`GeometryAwareTensorQR`**: Enhances sensor placement by considering geometric weights, proximity penalties, and mesh topology.

These features are particularly useful for fluid dynamics simulations on complex geometries.

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
