"""
Пример использования улучшенного TensorBasedCompressiveSensing

Этот файл демонстрирует основные возможности переработанного алгоритма
сжатого восприятия на основе тензоров с детальными метриками и настройками.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any

# Импортируем улучшенные классы
from TBMD.modules.TensorBasedCompressiveSensing import (
    TensorCompressiveSensing,
    CompressiveSensingConfig,
    tune_tensor_cs
)


def create_test_data():
    """Создание тестовых данных для демонстрации."""
    print("🔧 Создание тестовых данных...")
    
    # Размерности
    nx, ny, nt = 32, 32, 50  # Пространственные размерности 32x32, 50 временных моментов
    
    # Создаем словарь A (пространственно-временные моды)
    np.random.seed(42)
    A = np.random.randn(nx, ny, nt).astype(np.float32)
    
    # Создаем разреженный коэффициент x (временной сигнал)
    x_true = np.zeros(nt)
    x_true[5:15] = np.random.randn(10)  # Разреженный сигнал
    x_true[25:30] = np.random.randn(5) * 0.5
    
    # Полное измерение Y = A @ x
    Y_full = np.tensordot(A, x_true, axes=([2], [0]))
    
    # Создаем маску сенсоров P (размещение датчиков)
    P = np.zeros((nx, ny))
    # Случайное размещение 25% сенсоров
    sensor_indices = np.random.choice(nx * ny, size=int(0.25 * nx * ny), replace=False)
    P_flat = P.flatten()
    P_flat[sensor_indices] = 1.0
    P = P_flat.reshape(nx, ny)
    
    # Измерения только в позициях сенсоров
    Y = Y_full * P
    
    print(f"✅ Данные созданы:")
    print(f"   - Размерность A: {A.shape}")
    print(f"   - Размерность Y: {Y.shape}")
    print(f"   - Количество сенсоров: {int(P.sum())}/{nx*ny} ({100*P.sum()/(nx*ny):.1f}%)")
    print(f"   - Спарсность истинного x: {np.count_nonzero(x_true)}/{len(x_true)} ({100*np.count_nonzero(x_true)/len(x_true):.1f}%)")
    
    return A, P, Y, x_true


def basic_usage_example():
    """Пример базового использования с настройками по умолчанию."""
    print("\n" + "="*60)
    print("📚 ПРИМЕР 1: Базовое использование")
    print("="*60)
    
    A, P, Y, x_true = create_test_data()
    
    # Простое использование с настройками по умолчанию
    print("\n🚀 Запуск алгоритма с настройками по умолчанию...")
    solver = TensorCompressiveSensing(A, P, Y)
    solution = solver.solve()
    
    # Вычисляем ошибку восстановления
    error = np.linalg.norm(solution.cpu().numpy() - x_true) / np.linalg.norm(x_true)
    
    print(f"✅ Решение найдено!")
    print(f"   - Относительная ошибка: {error:.4f}")
    print(f"   - Размерность решения: {solution.shape}")
    
    return solution, x_true


def advanced_usage_with_metrics():
    """Пример использования с детальными метриками."""
    print("\n" + "="*60)
    print("📊 ПРИМЕР 2: Использование с детальными метриками")
    print("="*60)
    
    A, P, Y, x_true = create_test_data()
    
    # Кастомная конфигурация
    config = CompressiveSensingConfig(
        max_iter=500,
        epsilon=1e-3,
        lambd=0.95,
        convergence_tol=1e-7,
        solver_method="triangular",
        device="cpu"
    )
    
    print(f"\n🔧 Конфигурация алгоритма:")
    print(f"   - Максимальное число итераций: {config.max_iter}")
    print(f"   - Параметр регуляризации ε: {config.epsilon}")
    print(f"   - Параметр релаксации λ: {config.lambd}")
    print(f"   - Толерантность конвергенции: {config.convergence_tol}")
    
    # Решение с метриками
    print("\n🚀 Запуск алгоритма с детальными метриками...")
    solver = TensorCompressiveSensing(A, P, Y, config)
    solution, metrics = solver.solve_with_metrics()
    
    # Вычисляем ошибки
    reconstruction_error = solver.get_reconstruction_error(solution)
    relative_error = np.linalg.norm(solution.cpu().numpy() - x_true) / np.linalg.norm(x_true)
    
    print(f"\n📈 Результаты решения:")
    print(f"   - Сошелся: {'✅ Да' if metrics.converged else '❌ Нет'}")
    print(f"   - Итераций выполнено: {metrics.iterations}")
    print(f"   - Время решения: {metrics.solver_time:.3f} сек")
    print(f"   - Финальная невязка: {metrics.final_residual:.2e}")
    print(f"   - Число обусловленности: {metrics.condition_number:.2e}")
    print(f"   - Финальное значение целевой функции: {metrics.final_objective:.4f}")
    
    print(f"\n🎯 Качество восстановления:")
    print(f"   - Ошибка реконструкции: {reconstruction_error:.4f}")
    print(f"   - Относительная ошибка: {relative_error:.4f}")
    
    return solution, metrics, x_true


def hyperparameter_tuning_example():
    """Пример настройки гиперпараметров."""
    print("\n" + "="*60)
    print("🔍 ПРИМЕР 3: Настройка гиперпараметров")
    print("="*60)
    
    A, P, Y, x_true = create_test_data()
    
    # Определяем сетку параметров для поиска
    param_grid = {
        "epsilon": [5e-3, 1e-2, 2e-2],
        "lambd": [0.9, 0.95, 0.99],
        "max_iter": [300, 500]
    }
    
    print(f"\n🔍 Поиск лучших гиперпараметров...")
    print(f"   - Параметр ε: {param_grid['epsilon']}")
    print(f"   - Параметр λ: {param_grid['lambd']}")
    print(f"   - Максимальные итерации: {param_grid['max_iter']}")
    
    # Базовая конфигурация
    base_config = CompressiveSensingConfig(
        convergence_tol=1e-6,
        solver_method="triangular"
    )
    
    # Запуск настройки
    best_params, best_error, all_results = tune_tensor_cs(
        A, P, Y, param_grid, base_config
    )
    
    print(f"\n🏆 Лучшие параметры найдены:")
    for param, value in best_params.items():
        if param != "error":
            print(f"   - {param}: {value}")
    print(f"   - Лучшая ошибка: {best_error:.6f}")
    
    # Анализ результатов
    converged_results = [r for r in all_results if r.get('converged', False)]
    print(f"\n📊 Статистика поиска:")
    print(f"   - Всего комбинаций протестировано: {len(all_results)}")
    print(f"   - Сошлись: {len(converged_results)}")
    print(f"   - Процент успеха: {100 * len(converged_results) / len(all_results):.1f}%")
    
    return best_params, best_error, all_results


def visualization_example(solution, x_true, metrics=None):
    """Создание визуализации результатов."""
    print("\n" + "="*60)
    print("📊 ПРИМЕР 4: Визуализация результатов")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # График 1: Сравнение истинного и восстановленного сигнала
    ax1 = axes[0, 0]
    t = np.arange(len(x_true))
    ax1.plot(t, x_true, 'b-', linewidth=2, label='Истинный сигнал')
    ax1.plot(t, solution.cpu().numpy(), 'r--', linewidth=2, label='Восстановленный')
    ax1.set_xlabel('Время')
    ax1.set_ylabel('Амплитуда')
    ax1.set_title('Сравнение сигналов')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Ошибка восстановления
    ax2 = axes[0, 1]
    error = np.abs(solution.cpu().numpy() - x_true)
    ax2.plot(t, error, 'g-', linewidth=2)
    ax2.set_xlabel('Время')
    ax2.set_ylabel('Абсолютная ошибка')
    ax2.set_title('Ошибка восстановления')
    ax2.grid(True, alpha=0.3)
    
    # График 3: История конвергенции (если есть метрики)
    ax3 = axes[1, 0]
    if metrics and hasattr(metrics, 'convergence_history'):
        iterations = range(1, len(metrics.convergence_history) + 1)
        ax3.semilogy(iterations, metrics.convergence_history, 'b-', linewidth=2)
        ax3.set_xlabel('Итерация')
        ax3.set_ylabel('Относительное изменение (log scale)')
        ax3.set_title('Конвергенция алгоритма')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Метрики конвергенции\nне доступны', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('История конвергенции')
    
    # График 4: Спарсность решения
    ax4 = axes[1, 1]
    threshold = 0.01 * np.max(np.abs(solution.cpu().numpy()))
    sparse_solution = solution.cpu().numpy().copy()
    sparse_solution[np.abs(sparse_solution) < threshold] = 0
    
    ax4.stem(t, sparse_solution, linefmt='r-', markerfmt='ro', basefmt=' ')
    ax4.set_xlabel('Время')
    ax4.set_ylabel('Амплитуда')
    ax4.set_title(f'Разреженное решение (порог: {threshold:.3f})')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('compressive_sensing_results.png', dpi=150, bbox_inches='tight')
    print("✅ Графики сохранены в 'compressive_sensing_results.png'")
    
    return fig


def performance_comparison():
    """Сравнение производительности различных конфигураций."""
    print("\n" + "="*60)
    print("⚡ ПРИМЕР 5: Сравнение производительности")
    print("="*60)
    
    A, P, Y, x_true = create_test_data()
    
    # Различные конфигурации для сравнения
    configs = {
        "Быстрая": CompressiveSensingConfig(
            max_iter=100, epsilon=1e-2, convergence_tol=1e-4
        ),
        "Стандартная": CompressiveSensingConfig(
            max_iter=300, epsilon=1e-3, convergence_tol=1e-6
        ),
        "Точная": CompressiveSensingConfig(
            max_iter=1000, epsilon=1e-4, convergence_tol=1e-8
        )
    }
    
    results = {}
    
    for name, config in configs.items():
        print(f"\n🧪 Тестирование конфигурации '{name}'...")
        
        solver = TensorCompressiveSensing(A, P, Y, config)
        solution, metrics = solver.solve_with_metrics()
        
        reconstruction_error = solver.get_reconstruction_error(solution)
        relative_error = np.linalg.norm(solution.cpu().numpy() - x_true) / np.linalg.norm(x_true)
        
        results[name] = {
            "time": metrics.solver_time,
            "iterations": metrics.iterations,
            "converged": metrics.converged,
            "reconstruction_error": reconstruction_error,
            "relative_error": relative_error,
            "final_objective": metrics.final_objective
        }
        
        print(f"   ⏱️  Время: {metrics.solver_time:.3f} сек")
        print(f"   🔄 Итерации: {metrics.iterations}")
        print(f"   ✅ Сошелся: {'Да' if metrics.converged else 'Нет'}")
        print(f"   📊 Ошибка: {relative_error:.4f}")
    
    # Сводная таблица
    print(f"\n📋 Сводная таблица результатов:")
    print(f"{'Конфигурация':<12} {'Время':<8} {'Итерации':<10} {'Ошибка':<10} {'Сошелся':<8}")
    print("-" * 50)
    
    for name, result in results.items():
        print(f"{name:<12} {result['time']:<8.3f} {result['iterations']:<10} "
              f"{result['relative_error']:<10.4f} {'✅' if result['converged'] else '❌':<8}")
    
    return results


def main():
    """Главная функция с демонстрацией всех примеров."""
    print("🎯 ДЕМОНСТРАЦИЯ УЛУЧШЕННОГО TensorBasedCompressiveSensing")
    print("=" * 80)
    
    try:
        # Пример 1: Базовое использование
        solution_basic, x_true = basic_usage_example()
        
        # Пример 2: Использование с метриками
        solution_advanced, metrics, _ = advanced_usage_with_metrics()
        
        # Пример 3: Настройка гиперпараметров
        best_params, best_error, all_results = hyperparameter_tuning_example()
        
        # Пример 4: Визуализация
        fig = visualization_example(solution_advanced, x_true, metrics)
        
        # Пример 5: Сравнение производительности
        performance_results = performance_comparison()
        
        print("\n" + "="*80)
        print("🎉 ВСЕ ПРИМЕРЫ УСПЕШНО ВЫПОЛНЕНЫ!")
        print("="*80)
        print("\n✨ Ключевые преимущества улучшенной версии:")
        print("   🔧 Модульная архитектура с четким разделением ответственности")
        print("   🛡️  Comprehensive validation и robust error handling")
        print("   📊 Детальные метрики и мониторинг конвергенции")
        print("   ⚡ Оптимизированная производительность и использование памяти")
        print("   📚 Полная документация с математическими обоснованиями")
        print("   🎯 Готовность к промышленному использованию")
        
    except Exception as e:
        print(f"\n❌ Ошибка во время выполнения примеров: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 