import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_ace import st_ace

from scipy.integrate import odeint
from scipy.optimize import fsolve
import time

import time
import io
import sys


def general_rules():
    st.header("1. Общие правила вычислительной работы")

    st.markdown("""
        Практика вычислительной работы показывает, что при её выполнении оказываются полезными некоторые общие правила, 
        следование которым помогает избежать характерных ошибок.

        **Правило 1**. Прежде чем решать задачу, полезно задать вопрос: для чего нам нужен ответ, что мы с ним желаем сделать? 
        Этот вопрос при тщательном обдумывании ответа на него позволяет правильно спланировать объём и содержание выходной информации, 
        не упуская важные детали вычислений, позволяющие оценить достоверность и точность получаемых результатов.

        **Правило 2**. Очень важно в ходе вычислений использовать известную входную информацию о решении. 
        Часто дополнительная информация позволяет упростить решение задачи и контролировать правильность и точность вычислений.

        **Правило 3**. Перед разработкой программ (или проведением расчётов) полезно поставить аналитически или качественно предлагаему задачу.
    """)

    st.markdown("""
        **Правило 4**. Разработку вычислительной схемы полезно разделить на 3 этапа:
        1. **Выбор вычислительного метода**.
        2. **Промежуточный контроль результатов**.
        3. **Оценка точности полученного результата**.
    """)

    st.markdown("""
        **4.1. Выбор вычислительного метода**. Этот этап опирается на информацию, полученную в рамках предыдущих пунктов, 
        и играет определяющую роль для успешного решения задачи. Важно учитывать, что с увеличением мощности вычислительных машин 
        нередко к выбору метода начинают относиться менее критично, полагаясь на быстродействие машин. Однако неправильный выбор метода 
        может привести к значительным задержкам или даже невозможности завершения вычислений в разумные сроки. Например, выбор метода 
        Гаусса для больших матриц может существенно сократить время расчёта по сравнению с методом полного перебора.
    """)

    st.markdown("""
        Пример: пусть требуется вычислить определитель (детерминант) матрицы порядка \(N\). 
        Для этого используется метод полного перебора, когда каждый элемент матрицы умножается и суммируется последовательно.
        Число операций (сложений и умножений) будет равно:
    """)

    st.latex(r'''
    M = N!(сложений) + N \cdot N!(умножений) = (N + 1)!
    ''')

    st.markdown("""
        Для больших N>10 можно использовать приближённую формулу Стирлинга:
    """)

    st.latex(r'''
    M = \left( N+1 \right)^{N+1} \cdot e^{-N-1} \cdot \sqrt{2 \pi (N+1)}
    ''')

    st.markdown("""
        Рассмотрим теперь вычисления для \(N=99\). Число операций может составить:
    """)

    st.latex(r'''
    M = 10^{158}
    ''')

    st.markdown("""
        Предположим, что машина выполняет элементарную операцию за время:
    """)

    st.latex(r'''
    3 \cdot 10^{-18} \, \text{секунд.}
    ''')

    st.markdown("""
        Тогда время, необходимое для выполнения всех операций, составит:
    """)

    st.latex(r'''
    \tau_э = \frac{M}{m} = 10^{92} \, лет
    ''')

    st.markdown("""
        Таким образом, даже при использовании самых быстрых вычислительных машин, задача не будет выполнена в обозримое время. 
        Поэтому выбор более эффективного метода, например, метода Гаусса, позволяет сократить время вычислений до нескольких долей секунд.
    """)

    st.markdown("""
        **4.2. Промежуточный контроль результатов**. В ходе вычислений необходимо предусматривать промежуточные этапы контроля. 
        Этот контроль помогает убедиться, что вычисления идут в правильном направлении, и выявить возможные ошибки на ранних стадиях. 
        Опыт показывает, что игнорирование промежуточного контроля может привести к значительным искажениям конечных результатов.
    """)

    st.markdown("""
        **4.3. Оценка точности конечного результата**. После получения окончательного результата важно провести его оценку. 
        Необходимо понять, насколько точно решение задачи отвечает исходным условиям и можно ли доверять результатам, полученным в ходе вычислений. 
        На этом этапе полезно знать характеристики используемых методов, чтобы оценить возможные численные погрешности и влияние внешних факторов на точность.
    """)


def inaccuracy():
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


def general_rules_for_approximating_functions():
    st.header("3. Общие правила приближения функций")

    st.markdown("""
        При работе с численными методами приближения функций важно учитывать следующие аспекты:

        - **Выбор метода:** Выбор подходящего метода зависит от типа и свойств функции, а также от требуемой точности.
        - **Точность и сходимость:** Необходимо оценить, как быстро метод сходится к точному решению и какова его точность при различном количестве узлов или итераций.
        - **Вычислительная сложность:** Рассмотрение времени выполнения и требуемых ресурсов для каждого метода важно при больших объемах данных.
        - **Устойчивость метода:** Важно анализировать, насколько метод устойчив к ошибкам округления и входным погрешностям.
        """)


# 3.2.1. Метод Лагранжа
def lagrange_for_interpolation():
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


# 3.2.2. Метод Ньютона
def newton_for_interpolation():
    st.header("Метод Ньютона для интерполяции")

    st.markdown("""
        Метод Ньютона — это форма полиномиальной интерполяции, которая использует разделённые разности для построения интерполяционного полинома. Он позволяет последовательно добавлять новые узлы интерполяции без пересчёта всех коэффициентов, что делает метод более удобным для вычислений.
        """)

    st.latex(r"""
        \textbf{Теоретическая основа:} \\
        Полином \, Ньютона \, P_n(x) \, для \, n \, точек \, задаётся \, формулой: \\
        P_n(x) = f(x_0) + f[x_0, x_1](x - x_0) + \dots + f[x_0, x_1, ..., x_n](x - x_0) \cdot (x - x_1) \cdot ... \cdot (x - x_{n-1})
        """)
    st.latex(r"""
        где \, f[x_0, x_1, ..., x_k] \, — разделённые \, разности, \, определённые \, как: \\
        f[x_i] = y_i, \, f[x_i, x_{i+1}] = \frac{y_{i+1} - y_i}{x_{i+1} - x_i}, \dots
        """)

    st.markdown("**Реализация на чистом Python:**")
    st.code("""
    def divided_differences(x, y):
        #Функция для вычисления таблицы разделённых разностей.
        n = len(x)
        coef = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            coef[i][0] = y[i]

        for j in range(1, n):
            for i in range(n - j):
                coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

        return [coef[0][i] for i in range(n)]

    def newton_polynomial(x_data, y_data, x):
        #Функция для вычисления значения полинома Ньютона в точке.
        coef = divided_differences(x_data, y_data)
        n = len(coef)
        result = coef[-1]
        for i in range(n - 2, -1, -1):
            result = coef[i] + (x - x_data[i]) * result
        return result

    # Пример узлов и значений
    x_data = [1, 2, 3, 4]
    y_data = [1, 8, 27, 64]  # Значения y = x^3

    # Тестирование интерполяции в точке x = 2.5
    interpolated_value = newton_polynomial(x_data, y_data, 2.5)
    print("Интерполированное значение в x = 2.5:", interpolated_value)
        """)

    st.markdown("""
        Этот код демонстрирует реализацию метода Ньютона для интерполяции на языке Python. 
        Он использует разделённые разности для вычисления коэффициентов интерполяционного полинома и вычисления значения полинома в нужной точке.
        """)

    st.markdown("**Преимущества:**")
    st.markdown("""
        - Удобство добавления новых узлов интерполяции без пересчёта всех коэффициентов.
        - Меньшая вычислительная сложность по сравнению с другими методами при добавлении узлов.
        """)

    st.markdown("**Недостатки:**")
    st.markdown("""
        - Неустойчив при большом числе узлов (как и метод Лагранжа).
        - Зависимость от порядка введения узлов.
        """)

    st.markdown("**Алгоритм:**")
    st.latex(r"""
        1. \, Выбрать \, узлы \, интерполяции \, x_i \, и \, соответствующие \, значения \, y_i. \\
        2. \, Построить \, таблицу \, разделённых \, разностей \, f[x_i, ..., x_j]. \\
        3. \, Для \, каждого \, x \, в \, области \, определения \, вычислить \, P_n(x).
        """)

    # Демонстрация работы метода Ньютона для интерполяции с возможностью добавления новых точек

    # Функция для вычисления таблицы разделённых разностей
    def compute_divided_differences(x_data, y_data):
        n = len(x_data)
        dd_table = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            dd_table[i][0] = y_data[i]
        for j in range(1, n):
            for i in range(n - j):
                dd_table[i][j] = (dd_table[i + 1][j - 1] - dd_table[i][j - 1]) / (x_data[i + j] - x_data[i])
        return dd_table

    # Функция для вычисления значения полинома Ньютона
    def newton_polynomial(x_data, y_data, x, dd_table):
        result = dd_table[0][0]
        for i in range(1, len(x_data)):
            product = 1
            for j in range(i):
                product *= (x - x_data[j])
            result += dd_table[0][i] * product
        return result

    # Заданные точки для интерполяции
    x_points = [1, 2, 3, 4, 5]
    y_points = [1, 8, 27, 64, 125]  # Значения y = x^3

    # Функция для обновления графика и таблицы
    def update_plot_and_table(x_points, y_points):
        dd_table = compute_divided_differences(x_points, y_points)
        x_range = [min(x_points) + i * (max(x_points) - min(x_points)) / 100 for i in range(101)]
        y_approx = [newton_polynomial(x_points, y_points, x, dd_table) for x in x_range]
        st.write("Таблица разделённых разностей:")
        st.dataframe(dd_table)

        fig, ax = plt.subplots()
        ax.plot(x_range, y_approx, label='Приближение Ньютона')
        ax.scatter(x_points, y_points, color='red', label='Узлы интерполяции')
        ax.legend()
        st.pyplot(fig)

    # Интерфейс для добавления новых точек
    st.subheader("Добавление новых точек для интерполяции")
    new_x = st.number_input("Введите значение x:", format="%f", step=1.0, key="x_input")
    new_y = st.number_input("Введите значение y:", format="%f", step=1.0, key="y_input")
    add_point_button = st.button("Добавить точку")

    # Обработка добавления новой точки
    if add_point_button:
        if new_x in x_points:
            st.error("Точка с таким x уже существует. Введите уникальное значение x.")
        else:
            x_points.append(new_x)
            y_points.append(new_y)

    update_plot_and_table(x_points, y_points)


# 3.2.3. Погрешность многочленной аппроксимации
def error_of_polynomial_interpolation():
    st.header("3.2.3. Погрешность многочленной аппроксимации")

    st.markdown("""
    При заданной функции y(x) в n+1 точке, можно провести через эти точки многочлен P(x), который в узлах x совпадает с y(x) с машинной точностью. Важно понимать, как сильно отличается y(x) от P(x) в точках x не совпадающих с узлами.
    """)

    # Отдельное использование st.latex для формул
    st.latex(r"""
    y(x) - P_n(x) = (x - x_1)(x - x_2) \cdots (x - x_{n+1}) \cdot K(x)
    """)

    st.markdown("""
    где K(x) — некоторая соответствующим образом определённая функция.
    """)

    st.latex(r"""
    \Phi(x) = y(x) - P_n(x) - (x - x_1) \cdots (x - x_{n+1}) \cdot K(x^*)
    """)

    st.markdown("""
    где x* — некоторое произвольное, но фиксированное значение, не совпадающее с узлами x.
    """)

    st.latex(r"""
    \Phi^{(n+1)}(x) = y^{(n+1)}(x) - (n+1)! \cdot K(x^*)
    """)

    st.latex(r"""
    \Phi^{(n+1)}(x) = 0, \quad \text{следовательно, существует такое} \, \tilde{x}, \, \text{что}
    """)

    st.latex(r"""
    y^{(n+1)}(\tilde{x}) = (n+1)! \cdot K(x^*)
    """)

    st.latex(r"""
    y(x) = P_n(x) + (x - x_1) \cdots (x - x_{n+1}) \cdot \frac{y^{(n+1)}(\tilde{x})}{(n+1)!}
    """)

    st.markdown("""
    **Пример:** Рассмотрим функцию y = log(x) по значениям в точках x = 1, 2, 3, 4. Ошибка аппроксимации будет зависеть от выбора точки x.
    """)

    # Функция логарифма и её производные
    def log_function(x):
        return np.log(x)

    # Функция для вычисления разделённых разностей
    def divided_differences(x, y):
        n = len(x)
        coef = np.zeros((n, n))
        coef[:, 0] = y
        for j in range(1, n):
            for i in range(n - j):
                coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x[i + j] - x[i])
        return coef[0, :]

    # Функция для вычисления значения полинома Ньютона
    def newton_polynomial(x_data, y_data, x):
        coef = divided_differences(x_data, y_data)
        n = len(coef)
        result = coef[0]
        for i in range(1, n):
            product = 1
            for j in range(i):
                product *= (x - x_data[j])
            result += coef[i] * product
        return result

    # Заданные точки для интерполяции
    x_data = np.array([1, 2, 3, 4])
    y_data = log_function(x_data)

    # Генерация точек для построения графика
    x_vals = np.linspace(1, 4, 100)
    y_vals = log_function(x_vals)
    poly_vals = [newton_polynomial(x_data, y_data, x) for x in x_vals]

    # Построение графика
    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label='Исходная функция log(x)')
    ax.plot(x_vals, poly_vals, label='Многочлен Ньютона')
    ax.scatter(x_data, y_data, color='red', label='Узлы интерполяции')
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    **Анализ погрешности:** В зависимости от выбора точки x, погрешность аппроксимации может значительно варьироваться. Например, в точке x = 2.5, погрешность может быть минимальной или максимальной, в зависимости от распределения узлов интерполяции и характера функции.
    """)

    # Вывод дополнительных сведений о погрешности
    def calculate_error(x, actual, predicted):
        return np.abs(actual - predicted)

    error_example_x = 2.5
    actual_log = np.log(error_example_x)
    predicted_log = newton_polynomial(x_data, y_data, error_example_x)
    error_at_x = calculate_error(error_example_x, actual_log, predicted_log)

    st.write(f"Погрешность аппроксимации в точке x = {error_example_x}: {error_at_x:.6f}")


def polynomial_approximation_difficulties():
    st.header("3.2.4. Трудности приближения многочленом")

    st.markdown(
        """    Согласно выражению для ошибки приближения многочленом (формула 9), ошибку можно выразить через производную аппроксимируемой функции.    Обычно предполагается, что для больших n эта ошибка мала, но на практике это не всегда так. Рассмотрим функцию:    """)

    # Формула для логарифмической функции
    st.latex(r"y = \ln(x)")

    st.markdown("""    Для этой функции можно записать следующую производную:    """)

    # Формула для производной функции
    st.latex(r"y^{(n+1)}(x) = \frac{(-1)^n n!}{x^n}")

    st.markdown(
        """    С увеличением n для фиксированного x производные начинают значительно возрастать. Это распространённый случай для многих функций,     у которых производные высокого порядка имеют тенденцию резко увеличиваться при росте n .     Это приводит к тому, что на некоторых интервалах аппроксимируемая функция перестает сходиться к настоящему значению.    Рассмотрим классический пример:    """)

    # Формула для функции Рунге
    st.latex(r"y(x) = \frac{1}{1 + 25x^2}")

    st.markdown(
        """    Этот пример интерполяции, предложенный Рунге, показывает, что на отрезке [-1, 1], если узлы равномерно распределены,     ошибка аппроксимации будет расти с увеличением порядка интерполяции n, особенно на концах отрезка.     Данный эффект приводит к значительным отклонениям в аппроксимируемой функции, несмотря на точное совпадение с узловыми значениями.    """)

    st.markdown(
        """    Важно отметить, что интерполяция многочленами на равномерно распределённых узлах может стать причиной роста ошибок,     особенно для функций с высокими производными на краях интервала. Это известно как **феномен Рунге**.    """)


# 3.2.5 Многочлены Чебышеваimport streamlit as st
def chebyshev_polynomials():
    st.header("3.2.5. Многочлены Чебышева")

    st.markdown(
        """    Многочлены Чебышёва играют важную роль в численном анализе, и их свойства широко используются в вычислительной математике.    Они определяются на отрезке [-1, 1]  и минимизируют максимальное отклонение на этом интервале, что делает их оптимальными для аппроксимации.    Многочлены Чебышёва Tn(x) определяются для |x| ≤ 1 с помощью следующего соотношения:    """)

    # Формула для многочленов Чебышёва
    st.latex(r"T_n(x) = \cos(n \cdot \arccos(x))")

    st.markdown(
        """    Это свойство позволяет выражать многочлены через косинус, что упрощает их вычисление. Многочлены Чебышёва также могут быть определены через рекуррентные соотношения.    Многочлены Tn(x) ортогональны на отрезке [-1, 1] и удовлетворяют следующему соотношению:    """)

    # Формула ортогональности
    st.latex(
        r"""    \int_{-1}^{1} \frac{T_m(x) T_n(x)}{\sqrt{1 - x^2}} dx =     \begin{cases}     0, \, \text{если} \, m \neq n, \\     \frac{\pi}{2}, \, \text{если} \, n = 0, \\     \pi, \, \text{если} \, n \geq 1.    \end{cases}    """)

    st.markdown(
        """    Это ортогональное свойство делает их удобными для использования в разложениях функций и аппроксимации.    """)

    st.markdown("""    Многочлены Чебышёва могут быть выражены с помощью следующего рекуррентного соотношения:    """)

    # Рекуррентное соотношение
    st.latex(r"T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x), \quad n \geq 1")

    st.markdown(
        """    Таким образом, многочлены Чебышёва могут быть эффективно вычислены для любого n с помощью рекуррентного соотношения.     Эти многочлены минимизируют максимальное отклонение между аппроксимируемой функцией и многочленом на отрезке [-1, 1].    Это свойство делает их идеальными для задач приближения и интерполяции.    """)

    st.markdown("**Реализация метода Чебышёва на Python:**")

    st.code('''
        class ChebyshevInterpolation:
            def __init__(self, n, a=-1, b=1):
                self.n = n
                self.a = a
                self.b = b
                self.x_points = self.chebyshev_nodes()
                self.y_points = [self.taylor_sin(x) for x in self.x_points]
        
            def factorial(self, n):
                """Факториал числа"""
                result = 1
                for i in range(2, n + 1):
                    result *= i
                return result
        
            def taylor_cos(self, x, terms=10):
                """Вычисление косинуса с использованием ряда Тейлора"""
                result = 0
                for n in range(terms):
                    coef = (-1) ** n
                    result += coef * (x ** (2 * n)) / self.factorial(2 * n)
                return result
        
            def taylor_sin(self, x, terms=10):
                """Вычисление синуса с использованием ряда Тейлора"""
                result = 0
                for n in range(terms):
                    coef = (-1) ** n
                    result += coef * (x ** (2 * n + 1)) / self.factorial(2 * n + 1)
                return result
        
            def chebyshev_nodes(self):
                """Функция для вычисления узлов Чебышёва на интервале [a, b]"""
                x_chebyshev = []
                for i in range(self.n):
                    cos_value = self.taylor_cos((2 * i + 1) * 3.141592653589793 / (2 * self.n))  # pi = 3.14159...
                    node = 0.5 * (self.b - self.a) * (cos_value + 1) + self.a
                    x_chebyshev.append(node)
                return x_chebyshev
        
            def chebyshev_interpolation(self, x):
                """Интерполяция на узлах Чебышёва с использованием полиномов Лагранжа"""
                total = 0
                for i in range(self.n):
                    term = self.y_points[i]
                    for j in range(self.n):
                        if i != j:
                            term *= (x - self.x_points[j]) / (self.x_points[i] - self.x_points[j])
                    total += term
                return total
        
            def approximate_sin(self, x):
                """Пример использования Taylor для вычисления синуса"""
                return self.taylor_sin(x)
        
            def interpolate_value(self, x):
                """Интерполировать значение в заданной точке"""
                return self.chebyshev_interpolation(x)
        
        # Пример использования
        n = 5
        interpolator = ChebyshevInterpolation(n, a=-1, b=1)
        
        # Тестирование интерполяции в точке x = 0.5
        interpolated_value = interpolator.interpolate_value(0.5)
        print("Интерполированное значение в x = 0.5:", interpolated_value)

        ''')

    st.markdown("""
        Этот код демонстрирует реализацию метода Чебышёва для интерполяции на языке Python. 
        Пользователь может изменить количество узлов и область интерполяции, чтобы исследовать, как метод справляется с различными наборами данных.
        """)

    # Настройки для интерполяции

    def chebyshev_nodes(n, a=-1, b=1):
        """Функция для вычисления узлов Чебышёва на интервале [a, b]"""
        i = np.arange(n)
        x_chebyshev = np.cos((2 * i + 1) * np.pi / (2 * n))
        return 0.5 * (b - a) * (x_chebyshev + 1) + a

    def chebyshev_interpolation(x, x_points, y_points):
        """Интерполяция на узлах Чебышёва"""
        n = len(x_points)
        total = 0
        for i in range(n):
            term = y_points[i]
            for j in range(n):
                if i != j:
                    term *= (x - x_points[j]) / (x_points[i] - x_points[j])
            total += term
        return total

    num_points = st.slider("Выберите количество узлов Чебышёва", 1, 15, 5)
    a = -1.0
    b = 1.0

    # Вычисление узлов Чебышёва и значений
    x_points = chebyshev_nodes(num_points, a, b)
    y_points = np.sin(x_points)

    # Диапазон для графика
    x_range = np.linspace(a, b, 500)
    y_approx = [chebyshev_interpolation(x, x_points, y_points) for x in x_range]

    # Построение графика
    fig, ax = plt.subplots()
    ax.plot(x_range, np.sin(x_range), label='Исходная функция sin(x)')
    ax.plot(x_range, y_approx, label='Приближение методом Чебышёва')
    ax.scatter(x_points, y_points, color='red', label='Узлы Чебышёва')
    ax.legend()
    st.pyplot(fig)

    st.markdown("**Преимущества метода Чебышёва:**")
    st.markdown("""
        - Минимизация максимальной ошибки интерполяции.
        - Устойчивость к ошибкам на краях интервала.
        """)

    st.markdown("**Недостатки метода Чебышёва:**")
    st.markdown("""
        - Сложность вычислений при большом числе узлов.
        """)
