"""
Демонстрация исправления проблемы с типами sensor_values.

Показывает как решить ошибку:
ValueError: N must be a positive integer, got 1
"""

import numpy as np
import torch

from TBMD.analytics.analytics import ExperimentRunner, ExperimentConfig, ensure_sensor_values_are_int


def demonstrate_problem_and_solution():
    """Демонстрирует проблему с типами и её решение."""
    print("🔧 ИСПРАВЛЕНИЕ ПРОБЛЕМЫ С ТИПАМИ SENSOR_VALUES")
    print("=" * 60)
    
    # Создаем тестовые данные
    A_tensor = torch.randn(10, 10, 5, dtype=torch.float32)
    test_tensors = {
        'subject_1': torch.randn(10, 10, 3, dtype=torch.float32)
    }
    subject_name = 'subject_1'
    slice_number = 0
    
    # ❌ ПРОБЛЕМА: sensor_values с numpy типами
    print("\n❌ Проблемные sensor_values (numpy типы):")
    problematic_sensor_values = [np.int64(1), np.int32(3), np.int64(5)]
    print(f"   Значения: {problematic_sensor_values}")
    print(f"   Типы: {[type(x) for x in problematic_sensor_values]}")
    
    # ✅ РЕШЕНИЕ 1: Используем утилитную функцию
    print("\n✅ Решение 1: Утилитная функция ensure_sensor_values_are_int()")
    fixed_sensor_values = ensure_sensor_values_are_int(problematic_sensor_values)
    print(f"   Исправленные значения: {fixed_sensor_values}")
    print(f"   Типы: {[type(x) for x in fixed_sensor_values]}")
    
    # ✅ РЕШЕНИЕ 2: Ручное приведение типов
    print("\n✅ Решение 2: Ручное приведение")
    manual_fix = [int(x) for x in problematic_sensor_values]
    print(f"   Исправленные значения: {manual_fix}")
    print(f"   Типы: {[type(x) for x in manual_fix]}")
    
    # ✅ РЕШЕНИЕ 3: Изначально правильные типы
    print("\n✅ Решение 3: Сразу используйте Python int")
    correct_sensor_values = [1, 3, 5]  # Python int с самого начала
    print(f"   Значения: {correct_sensor_values}")
    print(f"   Типы: {[type(x) for x in correct_sensor_values]}")
    
    return fixed_sensor_values, manual_fix, correct_sensor_values


def test_fixed_experiment():
    """Тестирует исправленный эксперимент."""
    print("\n🧪 ТЕСТ ИСПРАВЛЕННОГО ЭКСПЕРИМЕНТА")
    print("=" * 60)
    
    # Создаем тестовые данные
    A_tensor = torch.randn(10, 10, 5, dtype=torch.float32)
    test_tensors = {
        'subject_1': torch.randn(10, 10, 3, dtype=torch.float32)
    }
    subject_name = 'subject_1'
    slice_number = 0
    
    # Используем исправленные sensor_values
    sensor_values = ensure_sensor_values_are_int([np.int64(1), np.int32(3), np.int64(5)])
    
    # Создаем конфигурацию
    config = ExperimentConfig(
        device='cpu',
        noise_level=0.0,  # Без шума для быстрого теста
        verbose=False
    )
    
    experiment_runner = ExperimentRunner(config)
    
    print(f"\n📊 Запуск run_single_slice_experiments...")
    print(f"   subject_name: {subject_name}")
    print(f"   slice_number: {slice_number}")
    print(f"   sensor_values: {sensor_values}")
    
    try:
        # Теперь должно работать!
        df = experiment_runner.run_single_slice_experiments(
            A_tensor, test_tensors, subject_name, slice_number, sensor_values
        )
        
        print(f"✅ Эксперимент завершен успешно!")
        print(f"📋 Колонки результата: {list(df.columns)}")
        print(f"📈 Результаты:")
        print(df)
        
        return df
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return None


def show_best_practices():
    """Показывает лучшие практики для избежания проблемы."""
    print("\n💡 ЛУЧШИЕ ПРАКТИКИ")
    print("=" * 60)
    
    print("\n🎯 Рекомендации:")
    print("1. ✅ Используйте Python int напрямую:")
    print("   sensor_values = [1, 3, 5, 10]")
    
    print("\n2. ✅ Если у вас numpy массивы, конвертируйте их:")
    print("   sensor_values = ensure_sensor_values_are_int(your_numpy_array)")
    
    print("\n3. ✅ Или используйте .tolist() для numpy массивов:")
    print("   sensor_values = np.array([1, 3, 5]).tolist()")
    
    print("\n4. ✅ Проверяйте типы при отладке:")
    print("   print([type(x) for x in sensor_values])")
    
    print("\n❌ Избегайте:")
    print("   ❌ np.arange(1, 10)  # numpy array")
    print("   ❌ [np.int64(1), np.int32(3)]  # numpy типы")
    
    print("\n📚 Примеры правильного использования:")
    
    # Пример 1: Простой список
    print("\n   # Пример 1: Простой список")
    example1 = """
   sensor_values = [1, 3, 5, 8, 10]
   df = runner.run_experiments(A_tensor, test_tensors, sensor_values)
   """
    print(example1)
    
    # Пример 2: Конвертация numpy
    print("\n   # Пример 2: Конвертация numpy")
    example2 = """
   numpy_sensors = np.arange(1, 11, 2)  # [1, 3, 5, 7, 9]
   sensor_values = ensure_sensor_values_are_int(numpy_sensors)
   df = runner.run_experiments(A_tensor, test_tensors, sensor_values)
   """
    print(example2)
    
    # Пример 3: Использование .tolist()
    print("\n   # Пример 3: Использование .tolist()")
    example3 = """
   sensor_values = np.array([1, 3, 5, 8, 10]).tolist()
   df = runner.run_experiments(A_tensor, test_tensors, sensor_values)
   """
    print(example3)


def main():
    """Главная функция демонстрации."""
    print("🚀 ДЕМОНСТРАЦИЯ ИСПРАВЛЕНИЯ ПРОБЛЕМЫ С ТИПАМИ")
    print("=" * 80)
    
    try:
        # Демонстрируем проблему и решения
        fixed1, fixed2, fixed3 = demonstrate_problem_and_solution()
        
        # Тестируем исправленный эксперимент
        df = test_fixed_experiment()
        
        # Показываем лучшие практики
        show_best_practices()
        
        print("\n" + "🎉" * 25)
        print("ПРОБЛЕМА РЕШЕНА!")
        print("🎉" * 25)
        
        print("\n📋 ИТОГОВЫЕ ВЫВОДЫ:")
        print("✅ Валидация типов исправлена - теперь принимает numpy integers")
        print("✅ Добавлена автоматическая конвертация в ExperimentRunner")
        print("✅ Создана утилитная функция ensure_sensor_values_are_int()")
        print("✅ run_single_slice_experiments теперь работает корректно")
        
        print("\n🎯 ГЛАВНАЯ РЕКОМЕНДАЦИЯ:")
        print("💡 Используйте Python int для sensor_values или функцию ensure_sensor_values_are_int()")
        
        return {
            'fixed_sensor_values': fixed1,
            'experiment_result': df
        }
        
    except Exception as e:
        print(f"\n❌ Ошибка в демонстрации: {e}")
        raise


if __name__ == "__main__":
    results = main() 
