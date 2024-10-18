import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_ace import st_ace
from code_editor import code_editor

from scipy.integrate import odeint
from scipy.optimize import fsolve

import io
import sys


# 3.3 Интерполяция сплайнами
def interpolation_by_splines():
    st.header("""3.3. Интерполяция сплайнами""")
    st.markdown("""
        Интерполяция сплайнами — это метод, который позволяет получить более точную интерполяцию функции по сравнению с 
        многочленами высокой степени, за счёт построения полиномов на отдельных отрезках между узлами. В основе этого
        метода лежит построение так называемых 'плавающих' многочленов, которые определяются для промежутков между 
        соседними узлами. Обычно используются полиномы третьей степени, так называемые кубические сплайны.
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
        Эти условия необходимы для того, чтобы обеспечить непрерывность функции и её первых двух производных в узлах 
        интерполяции. Система линейных уравнений для коэффициентов $$ a_i, b_i, c_i, d_i $$ составляется на основе этих 
        условий и решается для каждого узла.

        В дополнение к условиям стыковки, нужно задать граничные условия. Одним из наиболее распространённых граничных 
        условий является равенство второй производной функции на концах интервала к нулю, что даёт так называемый 
        естественный сплайн:
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
        Теперь, используя эти формулы, можно вычислить значения функции в любом месте интервала между узлами. Давайте
         реализуем это на Python и посмотрим, как работает кубическая интерполяция сплайнами.
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


# 4.1. Численное дифференцирование
def numerical_differentiation():
    st.header("4.1. Численное дифференцирование")

    st.markdown("""
        Численное дифференцирование применяется тогда, когда функцию $$ y(x) $$ невозможно (или трудно) 
        продифференцировать аналитически. Это часто бывает в случае таблично заданных функций или при решении 
        дифференциальных уравнений с конечными разностями.

        Основная идея численного дифференцирования заключается в том, что функцию $$ y(x) $$ аппроксимируют с помощью 
        интерполяционных многочленов или других методов, для которых легко вычислить производную.

        Для численного дифференцирования можно использовать следующий подход: аппроксимировать функцию $$ y(x) $$ 
        некоторой функцией $$ φ(x) $$, производную которой легко вычислить, и затем полагать, что:
    """)

    st.latex(r"""
        y'(x) \approx \varphi'(x)
    """)

    st.markdown("""
        В качестве аппроксимирующих функций широко используются интерполирующие многочлены.
    """)


# 4.1.1. Получение формул дифференцирования на основе многочлена Ньютона
def numerical_differentiation_newton_polynomial():
    st.header("4.1.1. Получение формул дифференцирования на основе многочлена Ньютона")

    st.markdown("""
    Пусть в точках $$ x_1, \dots, x_{n+1} $$ известны значения $$ y_1, \dots, y_{n+1} $$ функции $$ y(x) $$.
    Для этих узлов интерполирующий многочлен в форме Ньютона имеет вид:
    """)

    st.latex(r"""
            P_n(x) = y_1 + (x - x_1)\: ([x_2, x_1] + (x - x_2)\: [x_3, x_2, x_1] + \dots)\tag 1
        """)

    st.markdown("""
    где разделённые разности вычисляются по формулам:
    """)

    st.latex(r"""
        [x_2, x_1] = \frac{y_2 - y_1}{x_2 - x_1};
    """)

    st.latex(r"""
        [x_3, x_2, x_1] = \frac{[x_3, x_2] - [x_2, x_1]}{x_3 - x_1}=\frac1{x_3 - x_1}\left[\frac{y_3 - y_2}{x_3 - x_2} 
        - \frac{y_2 - y_1}{x_2 - x_1}\right];
    """)

    st.latex(r"""
        [x_1, \dots, x_n] = \frac{[x_2, \dots, x_n] - [x_1, \dots, x_{n-1}]}{x_n - x_1}.
    """)

    st.markdown("""
    Из формулы (1) получаем (введем обозначение $$ t_i = x - x_i $$):
    """)

    st.latex(r"""
        P_n'(x) = [x_1, x_2] + (t_1 + t_2) [x_1, x_2, x_3] + t_1 t_2 [x_1, x_2, x_3, x_4] + \dots,
    """)

    st.latex(r"""
        P_n''(x) = 2[x_1, x_2, x_3] + 2(t_1 + t_2 + t_3) [x_1, x_2, x_3, x_4] + \dots,\tag 2
    """)


    st.latex(r"""
        P_n^{(k)}(x) = k! \left( [x_1, \dots, x_{k+1}] +  \left( \sum_{i=1}^{k+1} t_j \right) [x_1, \dots, x_{i+1}]+ 
        \dots \right).
    """)

    st.markdown("""
    Обрезая эти формулы на некотором числе членов, получаем более простые приближённые формулы для производных. Обычно 
    ограничиваются только первыми слагаемыми. Выпишем некоторые простейшие употребительные одночленные формулы:
    """)

    st.markdown("""
    ### Простейшие одночленные формулы:
    """)

    st.latex(r"""
        y'(x) = \frac{y_2 - y_1}{x_2 - x_1};
    """)

    st.latex(r"""
        \frac{1}{2} y''(x) = \frac1{x_3 - x_1}\left[\frac{y_3 - y_2}{x_3 - x_2} - \frac{y_2 - y_1}{x_2 - x_1}\right];
        \tag 3
    """)

    st.latex(r"""
    \frac{1}{k!}y^{(k)}(x)=[x_1,\dots,x_{k+1}].
    """)

    st.markdown("""
    Получим оценки погрешности приближённых формул дифференцирования (2). Очевидно, имеем:
    """)

    st.latex(r"""
        y(x) = P_n(x) + R_n(x),
    """)

    st.markdown("""
    где $$ R_n(x) $$ — погрешность интерполяции. Следовательно:
    """)

    st.latex(r"""
        y^{(k)}(x) = P_n^{(k)}(x) + R_n^{(k)}(x),
    """)

    st.markdown("""
    т.е. погрешность производной равна производной от погрешности интерполяции. Погрешность интерполяционной формулы
     имеет вид (если использованы узлы $$ x_1, \dots, x_{n+1} $$):
    """)

    st.latex(r"""
        R_n(x) = \frac{(x - x_1)(x - x_2) \dots (x - x_n)}{(n+1)!} y^{(n+1)}(\overline x),
    """)

    st.markdown("""
    тогда $$ R_n^{(k)}(x) $$ будет содержать сумму $$ (n+1) n (n-1) \dots [n+1 - (k - 1)] $$ произведений $$ n+1-k $$ 
    множителей $$ t_i $$:
    """)

    st.latex(r"""
        |R_n^{(k)}(x)| \leq \frac{M_{n+1}}{(n+1-k)!}\max_i \left| t_i \right|^{n+1-k},
    """)

    st.markdown("""
    где:
    """)

    st.latex(r"""
        M_{n+1} = \max_x \left| y^{(n+1)}(x) \right|.
    """)

    st.markdown("""
    Если все узлы расположены равномерно, т.е. $$ x_{i+1} - x_i = h $$ для всех $$ i $$, то $$ \max |t_i| = nh $$ и
    """)

    st.latex(r"""
        |R_n^{(k)}(x)| \leq L_{n+1} h^{n+1-k}.
    """)

    st.markdown("""
    Таким образом, порядок точности формул (2) по отношению к шагу $$ h $$ равен числу узлов интерполяции минус порядок
     производной. Поэтому для вычисления $$ k $$-й производной требуется, как минимум, $$ (k+1) $$ узел, при этом 
     получаются формулы вида (3), имеющие первый порядок точности.
    """)

    st.markdown("""
    ### Точка повышенной точности
    """)

    st.markdown("""
    Для одночленных формул вида (3) корни используются из условия:
    """)

    st.latex(r"""
        \sum_{i=1}^{k+1} (x - x_i) = 0,
    """)

    st.markdown("""
    т.е. точка повышенной точности есть:
    """)

    st.latex(r"""
        x = \frac{1}{k+1} (x_1 + \dots + x_{k+1}).
    """)

    st.markdown("""
    В этой точке формулы (3) имеют точность $$ O(h^2) $$, а не $$ O(h) $$.

    Если в формулах (2) оставить по два слагаемых, то для определения точки повышенной точности получим квадратное 
    уравнение. Для большого числа слагаемых найти точку повышенной точности уже сложно. Однако есть один простой частный
     случай, для которого точка повышенной точности в формулах (2) легко указывается.
    """)

    st.markdown("""
    ### Теорема 1
    """)

    st.markdown("""
    Пусть $$ n+1-k $$ — нечетное число, и точки $$ x_1, \dots, x_{k+1} $$ выбраны так, что они лежат симметрично 
    относительно $$ x $$, тогда $$ x $$ — точка повышенной точности для производной $$ k $$-го порядка (2).
    """)

    st.markdown("""
    ### Доказательство
    """)

    st.markdown("""
    Очевидно, величины $$ t_i = x - x_i $$ будут попарно равны по абсолютной величине и будут иметь разные знаки. 
    Остаточный многочлен $$ R_n^{(k)}(x) $$ имеет степень $$ n+1-k $$, по условию нечетную, и имеет вид 
    $$ \sum\prod (t_i) $$.

    Поэтому, если сменить знак у всех $$ t_i $$, то $$ R_n^{(k)}(x) $$ должен также сменить знак. Но так как при этом 
    производные лишь перенумеруются, то они не должны измениться. Это возможно лишь при $$ R_n^{(k)}(x) = 0 $$, что и 
    требовалось доказать.
    """)

    st.markdown("""
    При произвольном расстановке узлов условие симметрии реализуется в исключительных случаях. При равномерной 
    расстановке фактически каждый узел (за исключением близких к краям) окружен симметрично другими узлами. Поэтому для 
    этих узлов можно написать простые формулы хорошей точности.
    """)

    st.markdown("""
    Пусть $$ x_{i+1} - x_i = h $$, напишем формулы по трем узлам $$ x_1, x_2, x_3 $$ для производных в среднем узле:
    """)

    st.latex(r"""
        y'_2 = \frac{y_3 - y_1}{2h} + O(h^2);
    """)

    st.latex(r"""
            y'_2 = \frac{y_3 - 2y_2 + y_1}{h^2} + O(h^2). \tag 4
        """)

    st.markdown("""
    Очевидно, что:
    """)

    st.latex(r"""
        y'(x_{i+1/2}) = y'(x_i + h/2) = \frac{y_{i+1} - y_i}{h}, \quad h = x_{i+1} - x_i
    """)

    st.markdown("""
    также имеет повышенную точность.
    """)

    st.markdown("""
    ### Пример вычисления производной
    """)

    st.markdown("""
    Рассмотрим четыре узла $$ x_1, x_2, x_3, x_4 $$ и вычислим производную при:
    """)

    st.latex(r"""
        x_{5/2} = \frac{x_2 + x_3}{2}
    """)

    st.latex(r"""
        y'_{5/2} = \frac{-y_4 + 27y_3 - 27y_2 + y_1}{24h},
    """)

    st.markdown("""
    Её точность будет $$ O(h^4) $$.

    ### Вторая производная

    Рассмотрим пять узлов $$ x_1, x_2, x_3, x_4, x_5 $$ и вычислим вторую производную $$ y''(x_3) $$ при:
    """)

    st.latex(r"""
        y''_n = \frac{-y_5 + 16y_4 - 30y_3 + 16y_2 - y_1}{12h^2}
    """)

    st.markdown("""
    Её точность также $$ O(h^4) $$.
    """)

    # Функция для разделённых разностей
    def divided_differences(x, y):
        n = len(y)
        coef = [y.copy()]  # Первое значение — это сами y
        for j in range(1, n):
            coef.append([(coef[j - 1][i + 1] - coef[j - 1][i]) / (x[i + j] - x[i]) for i in range(n - j)])
        return [row[0] for row in coef]

    # Функция для вычисления значения многочлена Ньютона в точке x_val
    def newton_polynomial(x, coef, x_val):
        n = len(coef)
        result = coef[-1]
        for i in range(n - 2, -1, -1):
            result = coef[i] + (x_val - x[i]) * result
        return result

    # Производная многочлена Ньютона
    def newton_polynomial_derivative(x, coef, x_val):
        n = len(coef)
        derivative = 0
        for i in range(n - 1):
            term = coef[i]
            for j in range(i):
                term *= (x_val - x[j])
            derivative += term
        return derivative

    #  функция f(x)
    def f(x):
        return x ** 3 * np.sin(x) + 2 * x ** 2

    # Производная функции f'(x) для сравнения
    def f_prime(x):
        return 3 * x ** 2 * np.sin(x) + x ** 3 * np.cos(x) + 4 * x

    # Входные данные: узлы и их значения
    x_nodes = [1, 2, 3, 4, 5]
    y_nodes = [f(x) for x in x_nodes]

    # Получение коэффициентов для многочлена Ньютона
    coef = divided_differences(x_nodes, y_nodes)

    # Точка повышенной точности
    x_central = sum(x_nodes) / len(x_nodes)

    # Вычисление значений для построения графика
    x_vals = [i * 0.05 for i in range(10, 110)]
    y_vals = [f(x) for x in x_vals]
    poly_vals = [newton_polynomial(x_nodes, coef, x) for x in x_vals]
    poly_prime_vals = [newton_polynomial_derivative(x_nodes, coef, x) for x in x_vals]

    # Построение графиков
    plt.figure(figsize=(10, 6))

    # Исходная функция
    plt.plot(x_vals, y_vals, label='Исходная функция $f(x) = x^3 \cdot \sin(x) + 2x^2$', color='blue')

    # Интерполяционный многочлен
    plt.plot(x_vals, poly_vals, label='Интерполяционный многочлен Ньютона', color='orange')

    # Производная
    plt.plot(x_vals, poly_prime_vals, label="Производная многочлена Ньютона", linestyle="--", color='green')

    # Узлы интерполяции
    plt.scatter(x_nodes, y_nodes, color='red', label='Узлы интерполяции')

    # Точка повышенной точности
    plt.axvline(x_central, color='purple', linestyle=':', label='Точка повышенной точности')

    # Настройки графика
    plt.title("Иллюстрация Теоремы 1: Симметричные узлы и точка повышенной точности")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # Отображение графика
    st.pyplot(plt)

    st.markdown("""
    ### Замечание

    Для равноотстоящих узлов при оценке точности формул численного дифференцирования часто используют разложение 
    $$ y(x) $$ в ряд Тейлора вблизи рассматриваемой точки $$ x $$.

    Рассмотрим, например, формулу (4) для второй производной $$ y''(x_2) $$.
    """)

    st.latex(r"""
        y_{2\pm1} = y(x_2 \pm h) = y_2 \pm h y_2' + \frac{h^2}{2} y''(x_2)\pm \frac{h^3}{6} y''' + 
        \frac{h^4}{24} y^{(4)}(\overline x_\pm)\tag{5}
    """)

    st.latex(r"""
     где,\  x_1 < \bar x_+ < x_3,\  x_1 < \bar x_- < x_3 \text { — некоторые точки}.
    """)

    st.markdown("""
    Подставив это разложение в формулу (4), получим:
    """)

    st.latex(r"""
        \frac{y_3 - 2y_2 + y_1}{h^2} = y_2'' + \frac{h^2}{24} \left[y^{(4)}(\bar x_+) + y^{(4)}(\bar x_-) \right]= y_2''
         + \frac{h^2}{12} y^{(4)} 
    """)

    st.markdown("""
    Этот же метод рядов Тейлора применяется и для получения формул дифференцирования с помощью неопределённых 
    коэффициентов.

    Получим, например, формулу (4) для $$ y_2'(x) $$ по узлам $$ x_1, x_2, x_3 $$.

    Предположим, что:
    """)

    st.latex(r"""
    y_2'=a_1 y_1 + a_2 y_2 + a_3 y_3\tag{6}
    """)

    st.markdown("""где, $$a_1, a_2, a_3$$ - неопределенные коэффициенты. Подставляя вместо $$y_1,y_3$$ их выражения 
    через ряд (5), получим
    """)

    st.latex(r"""
        a_1 y_1 + a_2 y_2 + a_3 y_3 = a_1 \left[y_2 - h y_2' + \frac{h^2}{2} y_2'' - \frac{h^3}{6} y_2''' + 
        \frac{h^4}{24} y_2^{(4)}(\bar x_-) \right]+
    """)

    st.latex(r"""
        + a_2 y_2 + a_3 \left[ y_2 + h y_2' + \frac{h^2}{2} y_2'' + \frac{h^3}{6} y_2''' +
        \frac{h^4}{24} y_2^{(4)}(\bar x_+) \right]=
    """)

    st.latex(r"""
        = (a_1 + a_2 + a_3) y_2 + h y_2' (a_3 - a_1) + \frac{h^2}{2} y_2'' (a_3 + a_1) + 
        \frac{h^3}{6} y_2''' (a_3 - a_1)  + O(h^4)
    """)

    st.markdown("""
    Мы хотим обеспечить условия, при которых сумма в правой части (6) приближает $$ y_2' $$ с возможно 
    меньшей погрешностью. Поэтому для $$ a_i $$ получаем уравнения:
    """)

    st.latex(r"""
        a_1 + a_2 + a_3 = 0;
    """)

    st.latex(r"""
        h (a_3 - a_1) = 1;
    """)

    st.latex(r"""
        a_3 + a_1 = 0;
    """)

    st.markdown("""
    Отсюда:
    """)

    st.latex(r"""
        a_3 = -a_1 = \frac{1}{2h}, \quad a_2 = 0.
    """)

    st.markdown("""
    При этом погрешность формулы будет $$ O(h^2) $$:
    """)

    st.latex(r"""
        y_2'' = \frac{y_3 - y_1}{2h} + \frac{h^2}{6} y_2'''
    """)


# 4.1.2. Метод Рунге—Ромберга повышения точности
def runge_romberg_method_of_increasing_accuracy():
    st.header("""4.1.2. Метод Рунге—Ромберга повышения точности
    """)

    st.markdown("""
    Как следует из предыдущего рассмотрения, для вычисления производной с высокой точностью надо привлекать большее 
    число узлов, при этом получаются достаточно громоздкие формулы. Оказывается, что можно, зная порядок погрешности 
    простейших формул дифференцирования вида (3) для равноотстоящих узлов, получить значение производной с повышенной 
    точностью, не прибегая к громоздким формулам.
    """)

    st.latex(r"""
        y^{(k)}(x) = \varphi(x, h) + \psi(x) h^p + O(h^{p+1})\tag 7
    """)

    st.markdown("""
    где $$ p $$ характеризует порядок точности формулы дифференцирования.

    Вычислим ту же производную $$ y^{(k)}(x) $$ по формуле (7), но используя шаг $$ h_1 = r h $$:
    """)

    st.latex(r"""
        y^{(k)}(x) = \varphi(x, rh) + \psi(x)\cdot(rh)^p + O\left[(rh)^{p+1}\right]\tag{8}
    """)

    st.markdown("""
    Из формул (7) и (8) можно получить оценку погрешности формулы:
    """)

    st.latex(r"""
        R \equiv \psi(x) h^p = \frac{\varphi(x, h) - \varphi(x, rh)}{r^p - 1} + O(h^{p+1})\tag {9}
    """)

    st.markdown("""
    (первая формула Рунге). Теперь, зная оценку погрешности (9), можно получить уточненное значение производной:
    """)

    st.latex(r"""
        y^{(k)}(x) = \varphi(x, h) + \frac{\varphi(x, h) - \varphi(x, rh)}{r^p - 1} + O(h^{p+1})\tag {10}
    """)

    st.markdown("""
    (вторая формула Рунге). Рассмотрим примеры.
    """)

    st.markdown("""
    ### Пример 1.

    Пусть функция $$ y(x) = \lg x $$ задана таблицей:
    """)

    st.markdown("""
    | x  | 1  | 2  | 3   | 4   | 5   |
    |----|----|----|-----|-----|-----|
    | y  | 0  | 0.301  | 0.478  | 0.602  | 0.699  |

    И нужно вычислить $$ y'(3) $$.
    """)

    st.markdown("""
    Вычислим эту производную следующим образом:
    """)

    st.latex(r"""
        y'(3) \approx \frac{y(4) - y(2)}{4 - 2} = 0.151 \quad (h = 1);
    """)

    st.latex(r"""
        y'(3) \approx \frac{y(5) - y(1)}{5 - 1} = 0.175 \quad (h = 2).
    """)

    st.markdown("""
    Здесь $$ p = 2 $$, $$ r = 2 $$, поэтому уточненное значение есть:
    """)

    st.latex(r"""
        y'(3) = 0.151 + \frac{0.151 - 0.175}{2^2 - 1} = 0.143.
    """)

    st.markdown("""
    Точное значение $$ y'(3) = 0.145 $$.
    """)
    st.markdown("""
    ##### Пример реализации метода при помощи Python
    """)
    st.code("""
    # Табличные значения функции y(x) = lg(x)
    x_values = [1, 2, 3, 4, 5]
    y_values = [0, 0.301, 0.478, 0.602, 0.699]  # log10 values

    # Вычисление первой производной через конечные разности
    def finite_difference(x1, y1, x2, y2):
        return (y2 - y1) / (x2 - x1)

    # Первое приближение с шагом h = 1 (используем x = 2 и x = 4)
    h1 = 1
    y_prime_h1 = finite_difference(x_values[1], y_values[1], x_values[3], y_values[3])

    # Второе приближение с шагом h = 2 (используем x = 1 и x = 5)
    h2 = 2
    y_prime_h2 = finite_difference(x_values[0], y_values[0], x_values[4], y_values[4])

    # Параметры метода Рунге
    r = h2 / h1
    p = 2  # Порядок точности

    # Уточнение значения производной по методу Рунге
    y_prime_refined = y_prime_h1 + (y_prime_h1 - y_prime_h2) / (r ** p - 1)
    print(f"Производная методом конечных разностей при h = 1: {y_prime_h1:.5f}")
    print(f"Производная методом конечных разностей при h = 2: {y_prime_h2:.5f}")
    print(f"Уточнённая производная по методу Рунге: {y_prime_refined:.5f}")
    """)
    code_editor()

    st.markdown("""

    ### Пример 2.

    Пусть имеются равноотстоящие узлы $$ x_1, x_2, x_3, x_4 $$, причем $$ x_{i+1} - x_i = h $$, и пусть требуется
     вычислить $$ y'_{5/2} = y'(x_{5/2}) $$:
    """)

    st.latex(r"""
        y'_{5/2} = \frac{y_3 - y_2}{h}, \quad y'_{5/2} = \frac{y_4 - y_1}{3h}.
    """)

    st.markdown("""
    Здесь $$ p = 2 $$, $$ r = 3 $$, по второй формуле Рунге получим:
    """)

    st.latex(r"""
        y'_{5/2} = \frac{y_3 - y_2}{h} + \frac{1}{8} \left[ \frac{y_3 - y_2}{h} - \frac{y_4 - y_1}{3h} \right]
        = \frac{y_1 - 27y_2 + 27y_3 - y_4}{24h}
    """)

    st.markdown("""
    Точность этой формулы должна иметь порядок не ниже $$ O(h^3) $$, на самом деле она равна $$ O(h^4) $$.
    """)

    st.markdown("""
    Метод Рунге обходится для достаточно гладких функций на произвольное число сеток узлов. Пусть для некоторой 
    функции $$ Z(x) $$ получено приближенное значение $$ φ(x, h) $$ с использованием сетки узлов с шагом $$ h $$ 
    и известно, что погрешность можно представить в виде:
    """)

    st.latex(r"""
        Z(x) = \varphi(x, h) + \sum_{m \geq p} \psi_m(x) h^m \tag{11}
    """)

    st.markdown("""
    (этот вид устанавливается с помощью разложений $$ φ(x, h) $$ вблизи точки $$ x $$ в ряд Тейлора). Пусть теперь проведены расчёты на $$ q $$ разных сетках с шагами $$ h_j $$, $$ j = 1, 2, ..., q $$. Считая, что в (11) $$ \psi_m(x) $$ не зависят от сеток, можем составить систему уравнений вида:
    """)

    st.latex(r"""
        Z(x) - \sum_{m = p}^{p+q-2} \psi_m(x) h_j^m = \varphi(x, h_j) + O(h_j^{p+q-1}) \quad (j = 1, 2, ..., q)
    """)

    st.markdown("""
    Решая её относительно $$ Z(x) $$ и {$$ \psi_m(x) $$}, получим для $$ Z(x) $$ формулу Ромберга.
    """)

    st.markdown("""
    ### Замечание 1.
    Метод Рунге-Ромберга можно применять только в том случае, если $$ \psi_m(x) $$ одинаковы для всех сеток. Обычно 
    так бывает, если расположение узлов относительно $$ x $$ подобно для всех сеток. Если этого нет, то метод 
    неприменим. Обычно при его использовании прибегают к изменению шага для узлов вдвое.
    """)

    st.markdown("""
    ### Замечание 2.
    Очевидно, что погрешность формул численного дифференцирования тем меньше, чем меньше шаг $$ h $$; с другой стороны, 
    при малых шагах $$ h $$ мы вынуждены вычитать близкие числа, т.е. происходит потеря точности. Поэтому, если значения
    функции имеют невысокую точность (зашумлены), то для вычисления производных лучше не использовать конечные
    разности, а предварительно получить аппроксимирующую функцию с помощью МНК.
    """)
