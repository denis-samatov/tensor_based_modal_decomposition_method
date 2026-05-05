# Tensor-Based Modal Decomposition (TBMD)

**Tensor-Based Modal Decomposition** — это библиотека для моделирования пониженного порядка (ROM), оптимального размещения сенсоров и реконструкции полей пространственно-временных данных (например, результатов гидродинамического моделирования). Она позволяет создавать **Цифровые Двойники (Digital Twins)**, работающие в реальном времени.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📚 Документация

Полная документация доступна в директории **[`docs/`](docs/README.md)**.

- **🚀 [Руководство по Цифровым Двойникам](docs/guides/digital_twin.md)**: Создание моделей резервуаров в реальном времени.
- **📐 [Geometry-Aware TBMD](docs/guides/GEOMETRY_AWARE_TBMD.md)**: Работа со сложными неструктурированными сетками.
- **🧠 [Основные концепции](docs/guides/tbmd_core.md)**: Изучите разложение Такера и HOSVD.
- **🎓 [Уроки](docs/tutorials/digital_twin_tutorial.md)**: Пошаговые руководства.

---

## 🚀 Быстрый старт

### Установка

```bash
git clone https://github.com/your-repo/tensor-based-modal-decomposition-method.git
cd tensor-based-modal-decomposition-method
pip install -e .          # Установка в editable-режиме
pip install -e ".[dev]"   # + dev зависимости (pytest, jupyter)
```

### Базовое использование

```python
from TBMD.core.decomposition.hosvd import TuckerDecomposer
from TBMD.config import DecompositionConfig

# 1. Конфигурация
config = DecompositionConfig(ranks=[20, 20, 10])

# 2. Декомпозиция
decomposer = TuckerDecomposer(tensors=data_tensor, config=config)
decomposer.decompose()

# 3. Доступ к результатам
core = decomposer.cores
factors = decomposer.factors

# 4. Реконструкция
decomposer.reconstruct()
reconstructed = decomposer.reconstructed_tensors
```

### Демонстрация Цифрового Двойника

```python
from TBMD.digital_twin import DigitalTwin
from TBMD.config import DigitalTwinConfig

# Инициализация
twin = DigitalTwin(DigitalTwinConfig(n_sensors=30))

# Обучение
twin.train(historical_data, normalize=False)

# Прогноз
forecast = twin.predict(current_state, n_steps=10)
```

## 🏆 State-of-the-Art Benchmark: Navier-Stokes
Для прогнозирования динамики турбулентных течений (Навье-Стокса) мы обнаружили, что **линейный предсказатель в латентном пространстве ($R_3=5$)** является SOTA-алгоритмом. Усекая временные моды до 5, мы отсекаем стохастичный шум и за счет этого добиваемся R² = 0.6870, многократно обходя ML аналоги (LSTM/MLP), склонные к переобучению на шуме.

Попробовать SOTA конфигурацию:
```bash
python examples/02_navier_stokes_optimal_forecasting.py
```

---

## 📂 Структура проекта

```
tensor-based-modal-decomposition-method/
├── pyproject.toml               # Зависимости и packaging
├── src/TBMD/                    # Основная библиотека
│   ├── core/                    # Декомпозиция, Сенсоры, Реконструкция
│   ├── config/                  # Классы конфигурации
│   └── utils/                   # Утилиты
├── tests/                       # 🧪 Тесты (pytest)
├── docs/                        # 📚 Документация
├── examples/                    # 💡 Примеры скриптов (Базовые, Продвинутые)
├── notebooks/                   # 📓 Jupyter Ноутбуки
├── data/                        # 📦 Данные (не в git)
└── results/                     # 📊 Результаты, артефакты, экспорты
```

## 🧪 Эксперименты и Примеры

- **Запуск демо Цифрового Двойника**:
  ```bash
  python examples/digital_twin/01_digital_twin_basic.py
  ```

- **Запуск тестов**:
  ```bash
  pytest tests/ -v
  ```

- **Изучение ноутбуков**:
  Проверьте `notebooks/experiments/` для Jupyter ноутбуков, охватывающих различные эксперименты и визуализации.

## 🤝 Вклад в разработку

Мы приветствуем ваш вклад! Пожалуйста, прочитайте документацию и проверьте существующие issues перед отправкой PR.

## 📄 Лицензия

Этот проект лицензирован под MIT License.
