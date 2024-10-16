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


def interpolation_by_splines():
    st.markdown("""
        Интерполяция сплайнами — это метод, который позволяет получить более точную интерполяцию функции по сравнению с многочленами высокой степени, за счёт построения полиномов на отдельных отрезках между узлами. В основе этого метода лежит построение так называемых 'плавающих' многочленов, которые определяются для промежутков между соседними узлами. Обычно используются полиномы третьей степени, так называемые кубические сплайны.
    """)

    st.latex(r"""
        \textbf{Кубический сплайн на отрезке} \, [x_{i-1}, x_i] \, представляется \, в \, виде: \\
        P_i(x) = a_i + b_i (x - x_{i-1}) + c_i (x - x_{i-1})^2 + d_i (x - x_{i-1})^3
    """)

    st.markdown("""
        Для того чтобы сплайн был гладким, необходимо выполнение условий стыковки полиномов в узлах:
    """)

    st.latex(r"""
        \textbf{Условия стыковки:}
        \begin{align*}
            P_i(x_i) &= P_{i+1}(x_i) \\
            P_i'(x_i) &= P_{i+1}'(x_i) \\
            P_i''(x_i) &= P_{i+1}''(x_i)
        \end{align*}
    """)

    st.markdown("""
        Эти условия необходимы для того, чтобы обеспечить непрерывность функции и её первых двух производных в узлах интерполяции. Система линейных уравнений для коэффициентов \(a_i, b_i, c_i, d_i\) составляется на основе этих условий и решается для каждого узла.

        В дополнение к условиям стыковки, нужно задать граничные условия. Одним из наиболее распространённых граничных условий является равенство второй производной функции на концах интервала к нулю, что даёт так называемый естественный сплайн:
    """)

    st.latex(r"""
        \textbf{Граничные условия:}
        \begin{align*}
            P_1''(x_0) &= 0 \\
            P_N''(x_N) &= 0
        \end{align*}
    """)

    st.markdown("""
        После применения граничных условий система уравнений выглядит следующим образом:
    """)

    st.latex(r"""
        \text{Система уравнений для коэффициентов} \, c_i:
        \begin{align*}
            a_i &= y_i \\
            b_i &= \frac{y_{i+1} - y_i}{h_i} - \frac{h_i}{3}(2c_i + c_{i+1}) \\
            d_i &= \frac{c_{i+1} - c_i}{3h_i} \\
            c_1 &= 0, \, c_N = 0
        \end{align*}
    """)

    st.markdown("""
        Теперь, используя эти формулы, можно вычислить значения функции в любом месте интервала между узлами. Давайте реализуем это на Python и посмотрим, как работает кубическая интерполяция сплайнами.
    """)

    st.markdown("""
        **Реализация сплайновой интерполяции на Python:**
    """)

    st.code("""
    import numpy as np
    import matplotlib.pyplot as plt
    
    def f(x):
        return np.sin(x)
        
    # Входные данные
    x_points = [0, 2, 4, 6, 8, 10]
    y_points = f(x_points)

    # Функция для решения системы линейных уравнений методом Гаусса
    def solve_gaussian(A, b):
        n = len(b)
        # Прямой ход метода Гаусса
        for i in range(n):
            # Выбор ведущего элемента
            max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
            A[i], A[max_row] = A[max_row], A[i]
            b[i], b[max_row] = b[max_row], b[i]
    
            for j in range(i + 1, n):
                factor = A[j][i] / A[i][i]
                b[j] -= factor * b[i]
                for k in range(i, n):
                    A[j][k] -= factor * A[i][k]
    
        # Обратный ход метода Гаусса
        x = [0] * n
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
        return x
    
    # Функция для расчёта коэффициентов кубического сплайна
    def cubic_spline(x_points, y_points):
        n = len(x_points)
        h = [x_points[i+1] - x_points[i] for i in range(n-1)]
        
        # Матрица для системы уравнений
        A = [[0] * n for _ in range(n)]
        b = [0] * n
    
        A[0][0] = A[n-1][n-1] = 1  # Граничные условия
        for i in range(1, n-1):
            A[i][i-1] = h[i-1]
            A[i][i] = 2 * (h[i-1] + h[i])
            A[i][i+1] = h[i]
            b[i] = 3 * ((y_points[i+1] - y_points[i]) / h[i] - (y_points[i] - y_points[i-1]) / h[i-1])
        
        # Решаем систему линейных уравнений для нахождения коэффициентов c
        c = solve_gaussian(A, b)
    
        # Нахождение коэффициентов a, b, d
        a = [y_points[i] for i in range(n-1)]
        b = [(y_points[i+1] - y_points[i])/h[i] - h[i]*(2*c[i] + c[i+1])/3 for i in range(n-1)]
        d = [(c[i+1] - c[i]) / (3*h[i]) for i in range(n-1)]
        
        return a, b, c, d
        """)

    # Функция для расчёта значения функции
    def f(x):
        return np.sin(x)

    # Функция для кубической интерполяции сплайнами
    def cubic_spline(x_points, y_points):
        n = len(x_points)
        h = [x_points[i + 1] - x_points[i] for i in range(n - 1)]

        # Система уравнений для коэффициентов c_i
        A = np.zeros((n, n))
        b = np.zeros(n)

        A[0, 0] = A[n - 1, n - 1] = 1
        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            b[i] = 3 * ((y_points[i + 1] - y_points[i]) / h[i] - (y_points[i] - y_points[i - 1]) / h[i - 1])

        c = np.linalg.solve(A, b)

        # Нахождение коэффициентов a, b, d
        a = [y_points[i] for i in range(n - 1)]
        b = [(y_points[i + 1] - y_points[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3 for i in range(n - 1)]
        d = [(c[i + 1] - c[i]) / (3 * h[i]) for i in range(n - 1)]

        return a, b, c, d

    # Функция для вычисления значения кубического сплайна
    def spline_value(x_val, x, a, b, c, d):
        for i in range(len(x) - 1):
            if x[i] <= x_val <= x[i + 1]:
                dx = x_val - x[i]
                return a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
        return None

    # Выбор количества узлов через слайдер
    n_points = st.slider("Выберите количество узлов интерполяции", 2, 12, 6)

    # Узлы интерполяции и значения функции в этих узлах
    x_points = np.linspace(0, 10, n_points)
    y_points = f(x_points)

    # Рассчитываем коэффициенты сплайнов
    a, b, c, d = cubic_spline(x_points, y_points)

    # Интервал для отображения графика
    x_range = np.linspace(0, 10, 1000)
    y_spline = [spline_value(x, x_points, a, b, c, d) for x in x_range]

    # Визуализация графиков
    fig, ax = plt.subplots()
    ax.plot(x_range, f(x_range), label="Исходная функция")
    ax.plot(x_range, y_spline, label="Приближение сплайнами")
    ax.scatter(x_points, y_points, color='red', label='Узлы интерполяции')
    ax.legend()

    st.pyplot(fig)
