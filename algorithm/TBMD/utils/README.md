# TBMD Utilities

Вспомогательные модули для TBMD: преобразование тензоров, генерация шумовых датасетов, метрики качества и экспорт полей давления.

## 📋 Основные модули

### `tbmd_utils.py` — базовые утилиты
- `to_torch_tensor`, `get_torch_device` — приведение данных и выбор устройства (CPU/CUDA/MPS).
- `extract_step_number` — извлечение номера шага из имени файла.
- `auto_select_mode`, `reconstruct_tensor` — определение подходящего измерения и реконструкция `A · x_hat`.
- `generate_noisy_datasets` — создание нескольких шумовых версий тензора с сохранением на диск.
- `build_Y_matrices`, `build_wells_matrix` — построение матриц измерений/скважин.
- `set_seed`, `set_torch_printoptions` — воспроизводимость и безопасные настройки вывода.
- `compute_reconstruction_metrics` — обёртка над метриками из `metrics.py` (RMSE, SSIM, PSNR, относительная ошибка).

Пример:
```python
from TBMD.utils import (
    get_torch_device, to_torch_tensor,
    reconstruct_tensor, compute_reconstruction_metrics
)

device = get_torch_device("cpu")
X = to_torch_tensor(raw_array, device=device)
X_rec = reconstruct_tensor(A_tensor, x_hat)
metrics = compute_reconstruction_metrics(X, X_rec)
```

### `metrics.py` — метрики качества
- `compute_metrics` — нормированная ошибка Фробениуса, MSE, SSIM (учёт масок, совместимость со scikit-image), PSNR.
- Работает с NumPy и PyTorch, поддерживает маску фона.

```python
from TBMD.utils.metrics import compute_metrics
err_norm, mse, ssim_val, psnr = compute_metrics(A_rec, A_ref, background_value=0.0)
```

### `navier_stokes_dataset.py` — синтетика Навье–Стокса
- `GaussianRF` — генерация случайных полей с заданной корреляцией.
- `navier_stokes_2d` — решатель уравнений Навье–Стокса в форме вихря.
- `generate_navier_stokes_dataset` — пакетная генерация и сохранение `.mat` датасета.

```python
from TBMD.utils.navier_stokes_dataset import GaussianRF, navier_stokes_2d

grf = GaussianRF(dim=2, size=64, device="cpu")
w0 = grf.sample(N=4)
sol, times = navier_stokes_2d(w0, f=0*w0, visc=1e-3, T=1.0, record_steps=50)
```

### `tnavigator_export.py` — экспорт в tNavigator
- `save_pressure_for_tnavigator` — сохраняет исходное поле, реконструкцию и разницу в CSV (формат I,J,K,Value).

```python
from TBMD.utils import save_pressure_for_tnavigator

paths = save_pressure_for_tnavigator(original[:, :, 0], reconstructed[:, :, 0], out_dir="export_brugge")
```

## ♻️ Совместимость и реэкспорты
- Геометрия: `TBMD.geometry` (экспортируется через `TBMD.core.geometry`). Импорты через `TBMD.utils.MeshGeometry` и др. работают, но выводят предупреждение — лучше использовать `from TBMD.geometry import MeshGeometry`.
- Визуализация: функции перенесены в `TBMD.visualization.plots` и тоже реэкспортируются с предупреждением.
- Загрузка/обработка данных: классы и функции переехали в `TBMD.data_utils` (`loaders.py`, `processors.py`, `splitters.py`) и доступны через `TBMD.utils.DataLoader`, `process_data`, `split_data_in_memory*` для обратной совместимости.

## 🎯 Типичные сценарии
- Быстро привести данные к нужному устройству и вычислить метрики реконструкции.
- Добавить шумовые вариации датасета и сохранить структуры эксперимента.
- Сгенерировать синтетический CFD-набор и использовать его в тестах TBMD.
- Экспортировать 2D-срезы давления в CSV для tNavigator.

## 📚 Дополнительные материалы
- [Utils Guide](../../docs/guides/TBMD_UTILS.md) — подробное руководство по модулям utils.
- [API Reference](../../docs/api/API_REFERENCE.md) — актуальные сигнатуры функций.
- [Examples](../../examples/) — примеры использования по всему проекту.

---

**Version**: 1.1.0  
**Last Updated**: February 2025
