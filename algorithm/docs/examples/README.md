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

### `algorithm/examples/`

Ready-to-run scripts for specific tasks and demos:

| Script | Description |
|--------|-------------|
| `examples/digital_twin/01_digital_twin_basic.py` | **Main Demo**: Runs a full Digital Twin cycle on synthetic data. |
| `examples/applications/brugge_field/run_brugge_enhanced.py` | Application of TBMD to the **Brugge** field dataset. |
| `examples/geometry_aware/06_geometry_aware_run.py` | Demo of the Geometry-Aware TBMD on a complex mesh. |

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
python algorithm/examples/digital_twin/01_digital_twin_basic.py
```

### Running the Brugge Case Study

```bash
python algorithm/examples/applications/brugge_field/run_brugge_enhanced.py
```

---

## 📓 Notebooks

For interactive exploration, check the `algorithm/notebooks/experiments/` directory, which contains Jupyter notebooks for:
- `exp_tbmd_4_digital_twin.ipynb`: Interactive Digital Twin experiments.
- `exp_tbmd_2.4_visualization.ipynb`: Visualization of reconstruction results.
