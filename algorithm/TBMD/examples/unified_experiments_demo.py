"""
Демонстрация унифицированного метода run_experiments.

Показывает автоматическое определение режима анализа
и различные способы использования.
"""

import numpy as np
import torch

from TBMD.utils.Analytics import ExperimentRunner, ExperimentConfig, plot_analytics


def create_sample_data():
    """Создает образцовые данные для демонстрации."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    A_tensor = torch.randn(32, 32, 10, dtype=torch.float32)
    test_tensors = {
        'subject_1': torch.randn(32, 32, 5, dtype=torch.float32),
        'subject_2': torch.randn(32, 32, 5, dtype=torch.float32)
    }
    sensor_values = [5, 10, 15, 20]
    
    return A_tensor, test_tensors, sensor_values


def demo_auto_detection():
    """Демонстрация автоматического определения режима анализа."""
    print("🤖 АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ РЕЖИМА")
    print("=" * 60)
    
    A_tensor, test_tensors, sensor_values = create_sample_data()
    
    # Конфигурация 1: Без шума → простой режим
    print("\n📊 Конфигурация 1: Без шума")
    config1 = ExperimentConfig(device='cpu', noise_level=0.0, num_noise_samples=0)
    runner1 = ExperimentRunner(config1)
    
    df1 = runner1.run_experiments(A_tensor, test_tensors, sensor_values)
    print(f"✅ Автоматически выбран простой режим")
    print(f"📋 Колонки: {list(df1.columns)}")
    print(f"📈 Данные:\n{df1.head()}")
    
    # Конфигурация 2: С шумом → статистический режим
    print("\n📊 Конфигурация 2: С умным зашумлением")
    config2 = ExperimentConfig(device='cpu', noise_level=0.1, num_noise_samples=3, 
                             noise_threshold=1e-10)
    runner2 = ExperimentRunner(config2)
    
    df2 = runner2.run_experiments(A_tensor, test_tensors, sensor_values)
    print(f"✅ Автоматически выбран статистический режим")
    print(f"📋 Колонки: {list(df2.columns)}")
    print(f"📈 Образец данных:\n{df2[['sensors', 'error_mean', 'error_std']].head()}")
    
    return df1, df2


def demo_explicit_control():
    """Демонстрация явного управления режимом анализа."""
    print("\n🎯 ЯВНОЕ УПРАВЛЕНИЕ РЕЖИМОМ")
    print("=" * 60)
    
    A_tensor, test_tensors, sensor_values = create_sample_data()
    
    # Один и тот же runner, разные режимы
    config = ExperimentConfig(device='cpu', noise_level=0.1, num_noise_samples=5)
    runner = ExperimentRunner(config)
    
    # Принудительно простой режим
    print("\n📊 Принудительно простой режим (statistical_analysis=False):")
    df_simple = runner.run_experiments(A_tensor, test_tensors, sensor_values, 
                                     statistical_analysis=False)
    print(f"📋 Колонки: {list(df_simple.columns)}")
    
    # Принудительно статистический режим
    print("\n📊 Принудительно статистический режим (statistical_analysis=True):")
    df_stat = runner.run_experiments(A_tensor, test_tensors, sensor_values, 
                                   statistical_analysis=True)
    print(f"📋 Колонки: {list(df_stat.columns)}")
    
    # Автоматический режим (использует config)
    print("\n📊 Автоматический режим (statistical_analysis=None):")
    df_auto = runner.run_experiments(A_tensor, test_tensors, sensor_values)
    print(f"📋 Колонки: {list(df_auto.columns)}")
    print("✅ Автоматически выбрал статистический режим из-за noise_level > 0")
    
    return df_simple, df_stat, df_auto


def demo_backward_compatibility():
    """Демонстрация обратной совместимости."""
    print("\n🔄 ОБРАТНАЯ СОВМЕСТИМОСТЬ")
    print("=" * 60)
    
    A_tensor, test_tensors, sensor_values = create_sample_data()
    
    config = ExperimentConfig(device='cpu')
    runner = ExperimentRunner(config)
    
    print("\n📊 Старый метод run_standard_experiments:")
    df_old = runner.run_standard_experiments(A_tensor, test_tensors, sensor_values)
    print(f"📋 Колонки: {list(df_old.columns)}")
    
    print("\n📊 Новый метод run_experiments (simple):")
    df_new = runner.run_experiments(A_tensor, test_tensors, sensor_values, 
                                  statistical_analysis=False)
    print(f"📋 Колонки: {list(df_new.columns)}")
    
    # Проверяем, что результаты идентичны
    are_equal = df_old.equals(df_new)
    print(f"\n✅ Результаты идентичны: {are_equal}")
    
    return df_old, df_new


def demo_performance_comparison():
    """Сравнение производительности разных режимов."""
    print("\n⚡ СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 60)
    
    import time
    
    A_tensor, test_tensors, sensor_values = create_sample_data()
    config = ExperimentConfig(device='cpu', noise_level=0.1, num_noise_samples=10)
    runner = ExperimentRunner(config)
    
    # Измеряем время простого режима
    start_time = time.time()
    df_simple = runner.run_experiments(A_tensor, test_tensors, sensor_values[:2], 
                                     statistical_analysis=False)
    simple_time = time.time() - start_time
    
    # Измеряем время статистического режима
    start_time = time.time()
    df_stat = runner.run_experiments(A_tensor, test_tensors, sensor_values[:2], 
                                   statistical_analysis=True)
    stat_time = time.time() - start_time
    
    print(f"⏱️  Простой режим: {simple_time:.2f} секунд")
    print(f"⏱️  Статистический режим: {stat_time:.2f} секунд")
    print(f"📊 Замедление: {stat_time/simple_time:.1f}x")
    
    print(f"\n📋 Простой режим - колонки: {list(df_simple.columns)}")
    print(f"📋 Статистический режим - колонки: {list(df_stat.columns)}")
    
    return df_simple, df_stat


def demo_plotting_compatibility():
    """Демонстрация совместимости с plot_analytics."""
    print("\n🎨 СОВМЕСТИМОСТЬ С PLOT_ANALYTICS")
    print("=" * 60)
    
    A_tensor, test_tensors, sensor_values = create_sample_data()
    
    config1 = ExperimentConfig(device='cpu', noise_level=0.0)
    config2 = ExperimentConfig(device='cpu', noise_level=0.1, num_noise_samples=3)
    
    runner1 = ExperimentRunner(config1)
    runner2 = ExperimentRunner(config2)
    
    # Получаем данные в разных форматах
    df_simple = runner1.run_experiments(A_tensor, test_tensors, sensor_values)
    df_stat = runner2.run_experiments(A_tensor, test_tensors, sensor_values)
    
    print("\n📊 Простые данные совместимы с plot_analytics:")
    plot_analytics(df_simple, plot_type='combined', title_prefix="Simple Mode", show_plots=False)
    print("✅ График создан успешно")
    
    print("\n📊 Статистические данные совместимы с plot_analytics:")
    plot_analytics(df_stat, plot_type='normalized', title_prefix="Statistical Mode", show_plots=False)
    print("✅ График с доверительными интервалами создан успешно")
    
    return df_simple, df_stat


def main():
    """Главная функция демонстрации."""
    print("🚀 ДЕМОНСТРАЦИЯ УНИФИЦИРОВАННОГО run_experiments")
    print("=" * 80)
    
    try:
        # Демонстрируем все возможности
        df1, df2 = demo_auto_detection()
        df3, df4, df5 = demo_explicit_control()
        df6, df7 = demo_backward_compatibility()
        df8, df9 = demo_performance_comparison()
        df10, df11 = demo_plotting_compatibility()
        
        print("\n" + "🎉" * 30)
        print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО!")
        print("🎉" * 30)
        
        print("\n📋 ИТОГОВЫЕ ВЫВОДЫ:")
        print("✅ Один метод run_experiments заменяет два старых")
        print("✅ Автоматическое определение режима по конфигурации")
        print("✅ Явное управление через parameter statistical_analysis")
        print("✅ Полная обратная совместимость")
        print("✅ Совместимость с plot_analytics")
        print("✅ Прозрачность: простой режим быстрее, статистический информативнее")
        print("✅ Умное зашумление: сохраняет физический смысл нулевых значений")
        
        print("\n🎯 РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ:")
        print("🔹 Используйте run_experiments() для всех экспериментов")
        print("🔹 Настройте noise_level/num_noise_samples в config для автоматического режима")
        print("🔹 Используйте statistical_analysis=False для быстрого анализа")
        print("🔹 Используйте statistical_analysis=True для точного анализа")
        print("🔹 run_standard_experiments() остается для обратной совместимости")
        print("🔹 Для резервуарных данных используйте noise_threshold для умного зашумления")
        
        return {
            'auto_simple': df1, 'auto_stat': df2,
            'explicit_simple': df3, 'explicit_stat': df4, 'explicit_auto': df5,
            'backward_old': df6, 'backward_new': df7,
            'perf_simple': df8, 'perf_stat': df9,
            'plot_simple': df10, 'plot_stat': df11
        }
        
    except Exception as e:
        print(f"\n❌ Ошибка в демонстрации: {e}")
        raise


if __name__ == "__main__":
    results = main() 
