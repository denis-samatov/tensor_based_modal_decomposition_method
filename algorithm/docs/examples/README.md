# TBMD Examples & Scripts

This directory contains examples and runnable scripts demonstrating the capabilities of the TBMD library.

## 📂 Structure

The examples are organized by complexity and topic:

### `algorithm/examples/`

| Category | Description |
|----------|-------------|
| **[basic/](../examples/basic/)** | Fundamental concepts: Decomposition, Sensor Placement, Reconstruction. Start here! |
| **[advanced/](../examples/advanced/)** | Advanced usage, custom configurations, and performance tuning. |
| **[digital_twin/](../examples/digital_twin/)** | Examples specific to the Digital Twin module (forecasting, monitoring). |
| **[geometry_aware/](../examples/geometry_aware/)** | Examples for working with unstructured meshes and complex geometries. |

### `algorithm/scripts/`

Ready-to-run scripts for specific tasks and demos:

| Script | Description |
|--------|-------------|
| `run_digital_twin_demo.py` | **Main Demo**: Runs a full Digital Twin cycle on synthetic data. |
| `run_brugge_enhanced.py` | Application of TBMD to the **Brugge** field dataset. |
| `run_geometry_aware_tbmd.py` | Demo of the Geometry-Aware TBMD on a complex mesh. |

---

## 📊 Reports & Analysis

- **[Brugge Digital Twin Analysis](brugge_digital_twin_analysis.md)**: A detailed report on the performance of the Digital Twin on the Brugge dataset, including error analysis and recommendations.

---

## 🚀 How to Run

Make sure you are in the root directory of the project and have the virtual environment activated.

### Running Basic Examples

```bash
# Run the basic decomposition example
python algorithm/examples/basic/01_tucker_decomposition.py

# Run the complete pipeline
python algorithm/examples/basic/04_complete_pipeline.py
```

### Running the Digital Twin Demo

```bash
python algorithm/scripts/run_digital_twin_demo.py
```

### Running the Brugge Case Study

```bash
python algorithm/scripts/run_brugge_enhanced.py
```

---

## 📓 Notebooks

For interactive exploration, check the `algorithm/experiments/` directory, which contains Jupyter notebooks for:
- `exp_tbmd_4_digital_twin.ipynb`: Interactive Digital Twin experiments.
- `exp_tbmd_2.4_visualization.ipynb`: Visualization of reconstruction results.
