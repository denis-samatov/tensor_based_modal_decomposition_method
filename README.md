# Tensor-Based Modal Decomposition (TBMD)

**Tensor-Based Modal Decomposition** is a library for reduced-order modeling, sensor placement, and field reconstruction of spatiotemporal data (e.g., reservoir simulation results). It enables the creation of **Digital Twins** that operate in real-time.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📚 Documentation

The full documentation is available in the **[`algorithm/docs/`](algorithm/docs/README.md)** directory.

- **🚀 [Digital Twin Guide](algorithm/docs/guides/digital_twin.md)**: Build real-time reservoir models.
- **📐 [Geometry-Aware TBMD](algorithm/docs/guides/geometry_aware_tbmd.md)**: Handle complex unstructured meshes.
- **🧠 [Core Concepts](algorithm/docs/guides/tbmd_core.md)**: Learn about Tucker Decomposition and HOSVD.
- **🎓 [Tutorials](algorithm/docs/tutorials/digital_twin_tutorial.md)**: Step-by-step guides.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-repo/tensor-based-modal-decomposition-method.git
cd tensor-based-modal-decomposition-method
pip install -r requirements.txt
```

### Basic Usage

```python
from algorithm.TBMD.core.decomposition import TuckerDecomposer
from algorithm.TBMD.config import DecompositionConfig

# 1. Configure
config = DecompositionConfig(ranks=[20, 20, 10])

# 2. Decompose
decomposer = TuckerDecomposer(config)
result = decomposer.decompose(data_tensor)

# 3. Reconstruct
reconstructed = result.reconstruct()
```

### Digital Twin Demo

```python
from algorithm.TBMD.core.digital_twin.system import DigitalTwinTBMD, DigitalTwinConfig

# Initialize
twin = DigitalTwinTBMD(DigitalTwinConfig(n_sensors=30))

# Train
twin.train(historical_data)

# Forecast
forecast = twin.predict_next_state(current_state, time_horizon=10)
```

---

## 📂 Project Structure

```
algorithm/
├── TBMD/                        # Core Library
│   ├── core/                    # Decomposition, Sensor Placement, Reconstruction
│   ├── models/                  # Forecasting Models (LSTM, MLP)
│   ├── config/                  # Configuration classes
│   └── utils/                   # Utilities
├── docs/                        # 📚 Documentation
├── examples/                    # 💡 Example scripts (Basic, Advanced)
├── scripts/                     # 🏃 Runnable demos
└── experiments/                 # 📓 Jupyter Notebooks
```

## 🧪 Experiments & Examples

- **Run the Digital Twin demo**:
  ```bash
  python algorithm/scripts/run_digital_twin_demo.py
  ```

- **Explore the notebooks**:
  Check `algorithm/experiments/` for Jupyter notebooks covering various experiments and visualizations.

## 🤝 Contributing

Contributions are welcome! Please read the documentation and check existing issues before submitting a PR.

## 📄 License

This project is licensed under the MIT License.
