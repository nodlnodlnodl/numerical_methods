import streamlit as st
from streamlit_ace import st_ace
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import time
import matplotlib.pyplot as plt
import time
import io
import sys

# Основной заголовок приложения
st.title("Численные методы")

# Боковое меню с выбором темы
st.sidebar.header("Численные методы")
st.sidebar.markdown("---")
st.sidebar.header("Темы:")
task = st.sidebar.radio("", ["Общие правила", "Погрешности", "Общие правила приближения", "Метод Лагранжа"])

# Общие правила вычислительной работы
if task == "Общие правила":
    st.header("1. Общие правила вычислительной работы")

    st.markdown("""
    При выполнении вычислительных задач важно придерживаться некоторых основных принципов, чтобы обеспечить точность, эффективность и повторяемость результатов:

    - **Чистота кода:** Код должен быть чистым и понятным. Это упрощает его проверку и уменьшает вероятность ошибок.
    - **Комментарии:** Всегда комментируйте ваш код. Это помогает другим (и вам самим в будущем) понять, что именно делает ваш код.
    - **Тестирование:** Регулярно тестируйте свой код на различных данных, чтобы убедиться, что он работает корректно.
    - **Оптимизация:** Ищите способы оптимизации вашего кода для улучшения производительности, особенно если обрабатываются большие объемы данных.
    - **Документация:** Поддерживайте актуальную документацию для вашего кода и проекта, описывая основные функции, используемые библиотеки и зависимости.
    - **Версионирование:** Используйте системы контроля версий, такие как Git, для отслеживания изменений в вашем коде и облегчения совместной работы.
    """)

    st.code("""
    # Пример функции с документацией и комментариями
    def calculate_area(base, height):
        '''
        Рассчитывает площадь треугольника по основанию и высоте.

        Параметры:
        base (float): Основание треугольника.
        height (float): Высота треугольника.

        Возвращает:
        float: Площадь треугольника.
        '''
        # Формула для расчета площади треугольника
        area = 0.5 * base * height
        return area
    """, language='python')

# Источники и классификация погрешности
elif task == "Погрешности":
    st.header("2. Источники и классификация погрешности")

    st.markdown("""
    В процессе численных расчетов погрешности могут возникать из различных источников и иметь различную природу. Классификация погрешностей включает в себя:

    - **Абсолютная погрешность:** Разница между точным значением и приближенным, измеренным или вычисленным значением.
    - **Относительная погрешность:** Отношение абсолютной погрешности к точному значению, часто выражается в процентах.
    - **Погрешность действий:** Погрешности, возникающие в результате выполнения арифметических операций, особенно при работе с числами с плавающей запятой в компьютерных вычислениях.

    ### Общая формула погрешности
    Общая формула для абсолютной и относительной погрешности может быть выражена в LaTeX как:
    """)

    st.latex(r"""
    \text{Абсолютная погрешность:} \quad \Delta x = |x_{\text{точное}} - x_{\text{приближенное}}|
    """)

    st.latex(r"""
    \text{Относительная погрешность:} \quad \varepsilon = \frac{\Delta x}{|x_{\text{точное}}|}
    """)

    st.latex(r"""
    \text{Где} \ (\Delta x) \ \text{— абсолютная погрешность,} \ (\varepsilon) \ \text{— относительная погрешность,}\\ \ (x_{\text{точное}}) \ \text{— точное значение,} \ (x_{\text{приближенное}}) \ \text{— приближенное значение.}
    """)

    st.markdown("""
    Понимание и учет этих погрешностей критически важно для достижения точности и надежности численных методов.
    """)

# Общие правила приближения функций
elif task == "Общие правила приближения":
    st.header("3. Общие правила приближения функций")

    st.markdown("""
    При работе с численными методами приближения функций важно учитывать следующие аспекты:

    - **Выбор метода:** Выбор подходящего метода зависит от типа и свойств функции, а также от требуемой точности.
    - **Точность и сходимость:** Необходимо оценить, как быстро метод сходится к точному решению и какова его точность при различном количестве узлов или итераций.
    - **Вычислительная сложность:** Рассмотрение времени выполнения и требуемых ресурсов для каждого метода важно при больших объемах данных.
    - **Устойчивость метода:** Важно анализировать, насколько метод устойчив к ошибкам округления и входным погрешностям.
    """)

# Метод Лагранжа для интерполяции
elif task == "Метод Лагранжа":
    st.header("4. Метод Лагранжа для интерполяции")

    st.markdown("""
    Метод Лагранжа — это форма полиномиальной интерполяции, используемая для аппроксимации функций. Полином Лагранжа представляет собой линейную комбинацию базисных полиномов Лагранжа, что позволяет точно проходить через заданные точки.
    """)

    st.latex(r"""
    \textbf{Теоретическая основа:} \\
    Полином \, Лагранжа \, L(x) \, для \, n \, точек \, задаётся \, формулой: \\
    L(x) = \sum_{i=0}^{n-1} y_i \ell_i(x)
    """)
    st.latex(r"""
    где \, \ell_i(x) \, — базисные \, полиномы \, Лагранжа, \, определённые \, как: \\
    \ell_i(x) = \prod_{\substack{j=0 \\ j \neq i}}^{n-1} \frac{x - x_j}{x_i - x_j}
    """)

    st.markdown("**Реализация на чистом Python:**")
    st.code("""
def lagrange_interpolation(x, x_points, y_points):
    total = 0
    n = len(x_points)
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        total += term
    return total

# Пример узлов и значений
x_points = [1, 2, 3, 4, 5]  # Узлы x
y_points = [1, 4, 9, 16, 25]  # Значения y в этих узлах (x^2)

# Тестирование интерполяции в точке x = 2.5
interpolated_value = lagrange_interpolation(2.5, x_points, y_points)
print("Интерполированное значение в x = 2.5:", interpolated_value)
    """)

    st.markdown("""
    Этот код демонстрирует базовую реализацию метода Лагранжа для интерполяции на языке Python. 
    Пользователь может изменить узлы интерполяции и точки, чтобы исследовать, как метод справляется с различными наборами данных.
    """)

    st.markdown("**Преимущества:**")
    st.markdown("""
    - Простота реализации.
    - Точное совпадение с интерполируемыми данными.
    """)

    st.markdown("**Недостатки:**")
    st.markdown("""
    - Не устойчив при большом количестве узлов из-за феномена Рунге.
    - Высокая вычислительная стоимость при увеличении числа узлов.
    """)

    st.markdown("**Алгоритм:**")
    st.latex(r"""
    1. \, Выбрать \, узлы \, интерполяции \, x_i \, и \, соответствующие \, значения \, y_i. \\
    2. \, Для \, каждого \, x \, в \, области \, определения \, вычислить \, L(x) \, с \, помощью \, базисных \, полиномов. \\
    3. \, Использовать \, L(x) \, для \, аппроксимации \, или \, интерполяции \, между \, узлами.
    """)

    # Демонстрация работы метода Лагранжа
    def f(x):
        return np.sin(x)


    def lagrange_interpolation(x, x_points, y_points):
        total = 0
        n = len(x_points)
        for i in range(n):
            term = y_points[i]
            for j in range(n):
                if i != j:
                    term = term * (x - x_points[j]) / (x_points[i] - x_points[j])
            total += term
        return total


    num_points = st.slider("Выберите количество узлов интерполяции", 1, 20, 4)
    x_points = np.linspace(0, 2 * np.pi, num_points)
    y_points = f(x_points)
    x_range = np.linspace(0, 2 * np.pi, 100)
    y_approx = [lagrange_interpolation(x, x_points, y_points) for x in x_range]

    fig, ax = plt.subplots()
    ax.plot(x_range, f(x_range), label='Исходная функция')
    ax.plot(x_range, y_approx, label='Приближение Лагранжа')
    ax.scatter(x_points, y_points, color='red', label='Узлы интерполяции')
    ax.legend()
    st.pyplot(fig)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Блок для интерактивного ввода кода пользователем
st.header("Интерактивный ввод кода")

# Настройка st_ace для автоматического обновления с начальным комментарием
user_code = st_ace(language='python', theme='monokai', height=300, auto_update=True, value="print('Ваш код работает хорошо')")

# Контейнер для вывода результатов
result_container = st.empty()

if user_code:
    # Создаем строковый буфер для перехвата вывода
    buffer = io.StringIO()
    sys.stdout = buffer  # Перенаправляем stdout на буфер

    with result_container.container():
        try:
            exec_start_time = time.time()
            exec(user_code)
            exec_end_time = time.time()

            # Получаем содержимое буфера
            output = buffer.getvalue()

            # Отображаем результат выполнения кода и время
            st.write("### Результат выполнения:")
            st.text(output)
            st.write(f"Код выполнен за {exec_end_time - exec_start_time:.6f} секунд")
        except Exception as e:
            st.error(f"Ошибка выполнения кода: {e}")
        finally:
            sys.stdout = sys.__stdout__  # Восстанавливаем стандартный stdout
