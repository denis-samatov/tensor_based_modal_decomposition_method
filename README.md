# Tensor-Based Modal Decomposition (TBMD)

**Tensor-Based Modal Decomposition** — это библиотека для моделирования пониженного порядка (ROM), оптимального размещения сенсоров и реконструкции полей пространственно-временных данных (например, результатов гидродинамического моделирования). Она позволяет создавать **Цифровые Двойники (Digital Twins)**, работающие в реальном времени.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📚 Документация

Полная документация доступна в директории **[`algorithm/docs/`](algorithm/docs/README.md)**.

- **🚀 [Руководство по Цифровым Двойникам](algorithm/docs/guides/digital_twin.md)**: Создание моделей резервуаров в реальном времени.
- **📐 [Geometry-Aware TBMD](algorithm/docs/guides/GEOMETRY_AWARE_TBMD.md)**: Работа со сложными неструктурированными сетками.
- **🧠 [Основные концепции](algorithm/docs/guides/tbmd_core.md)**: Изучите разложение Такера и HOSVD.
- **🎓 [Уроки](algorithm/docs/tutorials/digital_twin_tutorial.md)**: Пошаговые руководства.

---

## 🚀 Быстрый старт

### Установка

```bash
git clone https://github.com/your-repo/tensor-based-modal-decomposition-method.git
cd tensor-based-modal-decomposition-method
pip install -r requirements.txt
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

---

## 📂 Структура проекта

```
algorithm/
├── src/TBMD/                    # Основная библиотека
│   ├── core/                    # Декомпозиция, Сенсоры, Реконструкция
│   ├── config/                  # Классы конфигурации
│   └── utils/                   # Утилиты
├── docs/                        # 📚 Документация
├── examples/                    # 💡 Примеры скриптов (Базовые, Продвинутые)
└── notebooks/                   # 📓 Jupyter Ноутбуки
```

## 🧪 Эксперименты и Примеры

- **Запуск демо Цифрового Двойника**:
  ```bash
  python algorithm/examples/digital_twin/01_digital_twin_basic.py
  ```

- **Изучение ноутбуков**:
  Проверьте `algorithm/notebooks/experiments/` для Jupyter ноутбуков, охватывающих различные эксперименты и визуализации.

## 🤝 Вклад в разработку

Мы приветствуем ваш вклад! Пожалуйста, прочитайте документацию и проверьте существующие issues перед отправкой PR.

## 📄 Лицензия

Этот проект лицензирован под MIT License.
