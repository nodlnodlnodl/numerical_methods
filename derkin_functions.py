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


def linear_algebra_triangular_matrix():
    st.markdown(r"""
        # 6. Решение задач линейной алгебры

        Линейная алгебра включает в себя четыре основные задачи:
        1. Решение систем линейных уравнений вида $$ A \overline{x} = \overline{b} $$,
            где:
            - $$ A $$ — квадратная матрица размерности $$ n \times n $$,
            - $$ \overline{x} $$ и $$ \overline{b} $$ — векторы размерности $$ n $$.
        2. Вычисление определителя матрицы $$ A $$,
        3. Нахождение обратной матрицы $$ A^{-1} $$,
        4. Определение собственных значений и собственных векторов матрицы $$ A $$.

        Первые три задачи тесно связаны и могут быть решены с использованием аналогичных методов. 
        Последняя задача, связанная с определением собственных значений и векторов, требует более специфического подхода.

        В вычислительной математике задачи 1-3 наиболее часто решаются на ЭВМ, однако для больших значений $$ n $$ ($$ n \geq 200 $$) возникает необходимость применения более эффективных численных методов.
    """)

    st.markdown(r"""
        # 6.1.1. Треугольные матрицы

        **Определение:** Матрица $$ A $$ называется правой (левой) или верхней (нижней) треугольной, если для её элементов $$ a_{ij} $$ выполняется одно из следующих условий:
        - Для верхней треугольной матрицы: $$ a_{ij} = 0 $$, если $$ j < i $$,
        - Для нижней треугольной матрицы: $$ a_{ij} = 0 $$, если $$ j > i $$.

        **Свойства треугольных матриц:**
        1. Сумма треугольных матриц одного типа также является треугольной матрицей того же типа.
        2. Произведение двух треугольных матриц одного типа также является треугольной матрицей того же типа.
        3. Определитель треугольной матрицы равен произведению её диагональных элементов.
        4. Обратная матрица для треугольной матрицы также является треугольной матрицей того же типа.

        ## LU-теорема

         **Теорема.** Пусть дана квадратная матрица $$ A $$ порядка $$ n $$, у которой главные миноры $$ A_k $$, составленные из первых $$ k $$ строк и $$ k $$ столбцов, не равны нулю:
    """)

    st.latex(r"\det A_k \neq 0, \, k = 1, 2, \dots, n - 1.")

    st.markdown(r"""
        Тогда существует единственная нижняя треугольная матрица $$ L = (m_{ij}) $$, где $$ m_{11} = $$ $$ m_{22} = $$ $$ ... = $$ $$ m_{nn} = 1 $$ , и единственная верхняя треугольная матрица $$ U = (U_{ij}) $$, такие, что:
    """)

    st.latex(r"LU = A.")

    st.markdown(r"""
        Кроме того,
    """)

    st.latex(r"\det A = U_{11} \cdot U_{22} \dots U_{nn}.")

    st.markdown(r"""
        Доказательство проведём по индукции. Если $$ n = 1 $$, то очевидно, однозначно
    """)

    st.latex(r"a_{11} = 1 \cdot U_{11}")

    st.markdown(r"""
        и $$ det A = U_{11} $$.Пусть теорема верна для $$ n = m-1 $$, для $$ n = m $$ разложим матрицу $$ A $$ на подматрицы:
    """)

    st.latex(r"""
        A = \begin{pmatrix}
        A_{m-1} & C \\
        r & a_{mm}
        \end{pmatrix},
    """)

    st.markdown(r"""
        где $$ r $$ — строка из ($$ m-1 $$) элементов, $$ C $$ — столбец из ($$ m-1 $$) элементов; при этом $$ A_{m-1} \neq 0$$

        Запишем блоковые матрицы $$ L, U $$ для $$ A $$ в виде:
    """)

    st.latex(r"""
        L = \begin{pmatrix}
        L_{m-1} & 0 \\
        p & 1
        \end{pmatrix}, \quad
        U = \begin{pmatrix}
        U_{m-1} & q \\
        0 & U_{mm}
        \end{pmatrix},
    """)

    st.markdown(r"""
        где $$ p $$ и $$ q $$ — столбец и строка, аналогичные $$ r $$ и $$ C $$.

        Тогда по правилам умножения блочных матриц:
    """)

    st.latex(r"""
        LU = \begin{pmatrix}
        L_{m-1} U_{m-1} & L_{m-1} q \\
        p U_{m-1} & p q + U_{mm}
        \end{pmatrix}.
    """)

    st.markdown(r"""
        Согласно предположению индукции $$ L_{m-1}, U_{m-1} $$ можно однозначно определить таким образом, чтобы $$ L_{m-1} U_{m-1} = A_{m-1} $$ при этом $$ \det L_{m-1} \neq 0 $$, $$ \det U_{m-1} \neq 0 $$.

        Тогда условие $$ LU = A $$ приводит нас к следующей системе уравнений:
    """)

    st.latex(r"""
        L_{m-1} q = C, \quad p U_{m-1} = r, \quad pq + U_{mm} = a_{mm}.
    """)

    st.markdown(r"""
        В силу невырожденности матриц этих систем они имеют однозначное решение относительно $$ p, q $$; $$ U_{mm} $$ также определяется однозначно.
        При этом 
    """)

    st.latex(r"""
        \det A = \det L \cdot \det U = 1 \cdot \det U_{m-1} \cdot U_{mm} = U_{11} \dots U_{mm}.
    """)

    st.markdown(r"""
        что и требовалось доказать.
        
        #### Следствие 1. (LDU-теорема)
        В условиях LU-разложения матрица $$ A $$ имеет единственное разложение вида:
    """)

    st.latex(r"A = LDU.")

    st.markdown(r"""
        Где $$ L,U $$ — нижняя и верхняя треугольные матрицы с единицами на диагоналях, а $$ D $$ — диагональная матрица: $$ D = (d_{ij}) $$, $$ d_{ij} = 0, i \neq j $$. При этом $$ \det A = d_{11} \dots d_{nn} $$.
   
        В самом деле, $$ U $$ из LU-теоремы есть $$ DU $$ из LDU-теоремы.

        #### Следствие 2. 
        Пусть $$ A $$ — вещественная симметричная матрица: $$ a_{ij} = a_{ji} $$ и удовлетворяет условиям LU-теоремы. Обозначим через $$ A' $$ матрицу, транспонированную к $$ A $$. Тогда
    """)

    st.latex(r"A' = (LDU)' = (DU)' L' = U'DL' = LDU.")

    st.markdown(r"""
        Таким образом, для симметричной матрицы $$ U = L' $$. Если к тому же $$ A $$ - положительно определенная матрица, то, как известно, для всех главных миноров $$ \det A_{k} > 0 $$, $$ k = 1, \dots , n $$. Следовательно, в $$ LDU $$-теореме $$ d_{kk}>0$$.

        Обозначим через $$ D^{1/2} $$ матрицу вида:
    """)

    st.latex(r"""
        D^{1/2} = \begin{pmatrix}
        \sqrt{d_{11}} & 0 & \dots & 0 \\
        0 & \sqrt{d_{22}} & \dots & 0 \\
        \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & \dots & \sqrt{d_{nn}}
        \end{pmatrix}.
    """)

    st.markdown(r"""
        Тогда можно ввести $$ G = L D^{1/2} $$, и $$ A = G G' $$, то есть симметричная положительно определённая матрица имеет единственное разложение такого вида, где $$ G $$ — нижняя треугольная матрица с положительными диагональными элементами.

        Представление матрицы $$ A $$ в виде $$ LU $$-разложения является основой идей гауссовских схем исключения. Можно подсчитать, что для его получения требуется $$ \sim 2/3n^3 $$ сложений и умножений.
    """)


def linear_algebra_unitary_matrix():
    st.markdown(r"""
        # 6.1.2. Унитарные матрицы

        Матрица $$ A $$ называется унитарной, если она удовлетворяет уравнению:
    """)

    st.latex(r"A \cdot A^H = E, \tag{3}")

    st.markdown(r"""
        где $$ E $$ — единичная матрица, а $$ A^H = (A^*)'$$ — эрмитово сопряжение к матрице $$ A $$.
        Из уравнения $$ (3) $$ следует, что для унитарной матрицы имеет место:
    """)

    st.latex(r"\sum_{j=1}^{n} a_{ij} a_{kj}^* = \delta_{ik}, \quad i,k = 1,\dots, n.")

    st.markdown(r"""
        То есть строки унитарной матрицы ортогональны, если их рассматривать как компоненты векторов 
        $$ \overline{a}_i = (a_{i1}, \dots, a_{in})' $$ и 
        $$ \overline{a}_k = (a_{k1}, \dots, a_{kn})' $$ 
        и ввести скалярное произведение в виде:
    """)

    st.latex(r"(\overline{a}_i, \overline{a}_k) = \sum_{j=1}^{n} a_{ij} a_{kj}^*.")

    st.markdown(r"""
        Аналогично из равенства $$ A^H \cdot A = E $$ следует ортогональность столбцов унитарной матрицы:
    """)

    st.latex(r"\sum_{i=1}^{n} a_{ji} a_{jk}^* = \delta_{ik}.")

    st.markdown(r"""
        Перечислим некоторые свойства унитарных матриц:
        - модуль определителя унитарной матрицы $$ A $$ равен 1; в самом деле, из (3) имеем:
    """)

    st.latex(r"""
        1 = \det (A \cdot A^H) = \det A \cdot \det A^H = \det A\cdot (\det A)^* = |\det A|^2;
    """)

    st.markdown(r"""
        - для унитарной матрицы $$A \qquad A^{-1} = A^H $$, это получается из (3) умножением на $$ A^{-1} $$;
        - если $$ A $$ — унитарная, то и $$ A^H $$ — унитарная:
    """)

    st.latex(r"A^H \cdot (A^H)^H = A^H \cdot A = (A \cdot A^H)^H = E;")

    st.markdown(r"""
        - произведение двух унитарных матриц $$ A $$ и $$ B $$ есть также унитарная матрица:
    """)

    st.latex(r"""
        (A B) \cdot (A B)^H = A B \cdot B^H A^H  = A(B B^H )A^H = E.
    """)

    st.markdown(r"""
        К наиболее употребительным унитарным матрицам относятся ***электромагнитные матрицы***.
    """)

    st.latex(r"""
    R_{ij} = \begin{pmatrix}
    1 & \dots &  &  &  &  \\
     & \ddots &  &  &  &  \\
     &  & \cos \varphi & \cdots & -\sin \varphi e^{i \psi} &  \\
     &  & \vdots &  & \vdots &  \\
     &  & \sin \varphi e^{-i \psi} & \dots & \cos \varphi &  \\
    \dots & \dots & \dots & \dots & \dots & 1
    \end{pmatrix} \substack{i \\[15pt] j} \;,
    """)

    st.markdown(r"""
        которые отличны от $$ E $$ лишь четырьмя элементами: 
        
        $$ a_{ii} = \cos \varphi ; \quad $$ $$ a_{jj} = \cos \varphi ; \quad $$ $$ a_{ij} = -\sin \varphi \cdot e^{i \psi} ; \quad $$ $$ a_{ji} = \sin \varphi \cdot e^{-i \psi} $$.

        Рассмотрим матрицу $$ B = R_{ij} A $$. Очевидно, что:
    """)

    st.latex(r"""
    b_{mp} = a_{mp} \quad, \, m \neq i, j; \quad b_{ip} = \cos \varphi \cdot a_{ip} - \sin \varphi \cdot e^{i \psi} a_{jp};
    """)

    st.latex(r"""
    b_{jp} = \sin \varphi \cdot e^{-i \psi} a_{ip} + \cos \varphi \cdot a_{jp}, \quad p = 1, 2, \dots, n.
    """)

    st.markdown(r"""
        Всегда можно подобрать параметры $$ \varphi, \psi $$ так, чтобы $$ b_{jp} = 0 $$.
        Действительно, положим:
    """)

    st.latex(r"""
    \psi = \arg a_{ip} - \arg a_{jp}.
    """)

    st.markdown(r"""
        Тогда:
    """)

    st.latex(r"""
    \cos \varphi = \frac{|a_{ip}|}{\sqrt{|a_{ip}|^2 + |a_{jp}|^2}}; \quad \sin \varphi = \frac{|a_{jp}|}{\sqrt{|a_{ip}|^2 + |a_{jp}|^2}} \tag{4}
    """)

    st.markdown(r"""
        (если $$ |a_{ip}|^2 + |a_{jp}|^2 = 0 $$, то $$ \cos \varphi = 1 $$, $$ \sin \varphi = 0 $$). Тогда:
    """)

    st.latex(r"""
    b_{jp} = \frac{1}{\sqrt{|a_{ip}|^2 + |a_{jp}|^2}} \left[ - a_{ip}|a_{jp}| e^{-i \arg a_{ip}} e^{i \arg a_{jp}} + |a_{ip}| a_{jp} \right] = 0.
    """)

    st.markdown(r"""
        Поэтому имеет место теорема: любая комплексная матрица преобразуется в верхнюю треугольную матрицу посредством умножения слева на конечную цепочку матриц $$ R_{ij} $$.

        Вместо элементарных матриц при преобразованиях употребляются некоторые другие унитарные матрицы.

        Рассмотрим преобразование векторного пространства, осуществляемое отражением векторов от заданной плоскости $$ Q $$. Пусть $$ \overline{W} $$ — ортогональный к плоскости $$ Q $$ вектор единичной длины. Тогда произвольный вектор $$ \overline{Z} $$ можно представить в виде:
    """)

    st.latex(r"""
    \overline{Z} = \overline{x} + \alpha \overline{W},
    """)

    st.markdown(r"""
        где $$ (\overline{x}, \overline{W}) = \sum_{j=1}^{n} x_j W_{j}^* = 0 $$; $$ \alpha $$ — число. Отраженный от плоскости $$ Q $$ вектор по определению равен:
    """)

    st.latex(r"""
    \tilde{Z}' = \overline{x} - \alpha \overline{W}.
    """)

    st.markdown(r"""
        Матрица преобразования (**отражения**) от $$ \overline{Z} $$ к $$ \tilde{Z} $$ имеет вид:
    """)

    st.latex(r"""
    U = E - 2 \overline{W} \overline{W}^H, \tag{5}
    """)

    st.markdown(r"""
        где $$ (\overline{W}\overline{W}^H)_{ij} = \overline{W}_i \overline{W}_j^* $$, т.е. $$ \overline{W} \overline{W}^H $$ — произведение матрицы-столбца $$ W $$ на матрицу-строку $$ W^H $$.

        В самом деле:
    """)

    st.latex(r"""
    U \overline{Z} = (E - 2 \overline{W} \overline{W}^H)(\overline{x} + \alpha \overline{W}) = \overline{Z} - 2 (\overline{W} \overline{W}^H)\overline{x} - 2\alpha ( \overline{W} \overline{W}^H)\overline{W} = \\
    = \overline{Z} - 2\overline{W}(\overline{x}, \overline{W}) - 2\alpha \overline{W}(\overline{W}, \overline{W}),
    """)

    st.markdown(r"""
        где $$ (\overline{x}, \overline{W}), $$ $$ (\overline{W}, \overline{W}) $$ — скалярное произведение. Учитывая ортогональность $$ \overline{x} $$ и $$ \overline{W} $$, имеем:
    """)

    st.latex(r"""
    U \overline{Z} = \overline{Z} - 2 \alpha \overline{W}(\overline{W}, \overline{W}) = \overline{Z} - 2 \alpha \overline{W} = \overline{x} - \alpha \overline{W} = \tilde{Z}.
    """)

    st.markdown(r"""
        Покажем, что $$ U $$ — унитарная матрица:
    """)

    st.latex(r"""
    UU^H = (E - 2 \overline{W} \overline{W}^H)(E - 2 (\overline{W}^H)^H \overline{W}^H) = \\
    = E - 4 \overline{W} \overline{W}^H + 4 \overline{W} ( \overline{W}^H \overline{W} ) \overline{W}^H = \\ 
    = E - 4 \overline{W} \overline{W}^H + 4 ( \overline{W}, \overline{W}) \overline{W} \overline{W}^H = E.
    """)

    st.markdown(r"""
        Пусть $$ \overline{S}, \overline{e} $$ — два произвольных вектора-столбца, причем $$ \overline{e} $$ — единичной длины: $$ (\overline{e}, \overline{e}) = 1 $$.
        Вектор $$ \overline{W} $$ всегда можно выбрать так, чтобы построенная по нему матрица $$ U $$ переводила $$ \overline{S} $$ в вектор, параллельный $$ \overline{e} $$.

        Положим:
    """)

    st.latex(r"""
    \overline{W} = \frac{1}{\rho} (\overline{S} - \alpha \overline{e}),
    """)

    st.markdown(r"""
        где:
    """)

    st.latex(r"""
    |\alpha| = \sqrt{(\overline{S}, \overline{S})}; \quad \arg \alpha = \arg (\overline{S}, \overline{e}) - \pi; \quad \rho = \sqrt{(\overline{S} - \alpha \overline{e}, \overline{S} - \alpha \overline{e})} = \\
     = \sqrt{2|\alpha|^2 + 2|\alpha| |(\overline{S}, \overline{e})|}.
    """)

    st.markdown(r"""
            тогда:
        """)

    st.latex(r"""
        U \overline{S} = \overline{S} - 2 (\overline{S}, \overline{W}) \overline{W} = \overline{S} - \frac{2} {\rho} (\overline{S}, \overline{S} - \alpha \overline{e}) \overline{W} = \overline{S} - \frac{2} {\rho} [\overline{S}, \overline{S}) - (\overline{S}, \alpha \overline{e}]\overline{W} =\\
        = \overline{S} - \frac{1}{\rho} \left[ 2|\alpha|^2 - 2\alpha^* (\overline{S}, \overline{e})\right] \overline{W} = \\
        = \overline{S} - \frac{1}{\rho} \left[ 2|\alpha|^2 - 2|\alpha| e^{i[\pi - \arg (\overline{S}, \overline{e})]} |(\overline{S}, \overline{e})| e^{i \arg (\overline{S}, \overline{e})} \right] \overline{W} = \\
        = \overline{S} - \frac{1}{\rho} \left[ 2|\alpha|^2 + 2|\alpha| |(\overline{S}, \overline{e})| \right] \overline{W} = \overline{S} - \rho \overline{W} = \alpha \overline{e}.
        """)

    st.markdown(r"""
            Свойства матрицы отражения используются при преобразованиях матриц. Пусть $$ A $$ — некоторая комплексная матрица, умножим ее слева на матрицу $$ U_1 $$, выбрав в качестве $$ \overline{S} $$ и $$ \overline{e} $$ векторы:
        """)

    st.latex(r"""
        \overline{S} = \begin{pmatrix}
        a_{11} \\
        \vdots \\
        a_{n1}
        \end{pmatrix}; \quad \overline{e} = \begin{pmatrix}
        1 \\
        0 \\
        \vdots \\
        0
        \end{pmatrix}.
        """)

    st.markdown(r"""
            Тогда:
        """)

    st.latex(r"""
        A^{(1)} = U_1 A = \begin{pmatrix}
        a_{11}^{(1)} & a_{12}^{(1)} & \dots & a_{1n}^{(1)} \\
        0 & a_{22}^{(1)} & \dots & a_{2n}^{(1)} \\
        0 & a_{32}^{(1)} & \dots & a_{3n}^{(1)} \\
        \dots & \dots & \dots & \dots \\
        0 & a_{n2}^{(1)} & \dots & a_{nn}^{(1)}
        \end{pmatrix}.
        """)

    st.markdown(r"""
            На втором шаге полагаем:
        """)

    st.latex(r"""
        \overline{S} = \begin{pmatrix}
        0 \\
        a_{22}^{(1)} \\
        \vdots \\
        a_{n2}^{(1)}
        \end{pmatrix}, \quad \overline{e} = \begin{pmatrix}
        0 \\
        1 \\
        0 \\
        \vdots \\
        0
        \end{pmatrix}.
        """)

    st.markdown(r"""
            И так далее. В конце процесса получим верхнюю треугольную матрицу.

            Вещественные унитарные матрицы $$ A $$ называются **ортогональными**. Все их свойства определяются равенством $$ A \cdot A' = E $$.
            Вещественные унитарные элементарные матрицы называются матрицами **вращения** $$ T_{ij} $$ или матрицами **простого поворота**. Очевидно, что:
    """)

    st.latex(r"T_{ij} (\varphi) = R_{ij} (\varphi, \psi = 0).")

    st.markdown(r"""
        При этом в формулах $$ (4) $$ полагают:
    """)

    st.latex(r"""
    \cos \varphi = \frac{a_{ip}}{\sqrt{a_{ip}^2 + a_{jp}^2}}, \quad \sin \varphi = \frac{-a_{jp}}{\sqrt{a_{ip}^2 + a_{jp}^2}}.
    """)

    st.markdown(r"""
        Отметим, что если для матрицы $$ A $$ выполнено условие $$ A A^H = A^H A $$, то $$ A $$ называется **нормальной** матрицей. Унитарные матрицы дают примеры нормальных матриц.
    """)

#4.2. Численное интегрирование
def numerical_integration():
    st.markdown(r"""
        ## 4.2. Численное интегрирование

        Численное интегрирование применяется в тех случаях, когда вычислить аналитически интегралы либо сложно, либо вообще невозможно (например, если подынтегральная функция задана таблично). 

        Идея для численного интегрирования такая же, как и для численного дифференцирования: подынтегральную функцию заменяют подходящей аппроксимирующей функцией, для которой и производят вычисления. Наиболее широко применяется многочленная аппроксимация.

        Введем некоторые определения. Пусть требуется вычислить определённый интеграл:
    """)

    st.latex(r"F = \int_a^b \rho(x) y(x) dx,")

    st.markdown(r"""
        где $$ \rho(x) > 0 $$ — весовая функция. Будем считать, что $$ y(x) $$ непрерывна на $$ [a, b] $$, а $$ \rho(x) $$ непрерывна на $$ (a, b) $$. В разделе 3 мы видели, что заменять $$ y(x) $$ многочленом сразу на всём отрезке $$ [a, b] $$ вообще говоря, нецелесообразно. Поэтому обычно исходный отрезок $$ [a, b] $$ разбивают на несколько отрезков, на каждом из которых и производится замена $$ y(x) $$ на некоторый многочлен $$ P_n^{(m)}(x) $$:
    """)

    st.latex(r"""
        F = \sum_{m=1}^{M} \int_{a_m}^{a_{m+1}} \rho(x) y(x) dx 
        \overset{\sim}{=} \sum_{m=1}^{M} \int_{a_m}^{a_{m+1}} \rho(x) P_n^{(m)}(x) dx, \tag{13}
    """)

    st.markdown(r"""
        где $$ a_1 = a $$, $$ a_{M+1} = b $$.

        Рассмотрим интеграл:
    """)

    st.latex(r"f_m = \int_{a_m}^{a_{m+1}} \rho(x) y(x) dx, \tag{14}")

    st.markdown(r"""
        обозначая для краткости $$ a_m = \alpha $$, $$ a_{m+1} = \beta $$ и опуская индекс $$ m $$ у $$ f $$.

        Пусть на отрезке $$ [\alpha, \beta] $$ введены некоторые узлы $$ x_1, \dots, x_n $$, и $$ y(x) $$ заменена на многочлен (в форме Лагранжа):
    """)

    st.latex(r"y(x) = \sum_{i=1}^{n} y_i L_i(x) + r(x), \tag{15}")

    st.markdown(r"""
        где $$ r(x) $$ — погрешность интерполяции, а $$ L_i(x) $$ — многочлены Лагранжа степени $$ (n-1) $$ системы. Если теперь выражение (15) подставить в (14), то получим:
    """)

    st.latex(r"f = \sum_{i=1}^{n} W_i y_i + R. \tag{16}")

    st.markdown(r"""
        Эта формула называется **квадратурной**, $$ W_i $$ — веса, $$ x_i $$ — узлы, $$ R $$ — остаточный член квадратурной формулы:
    """)

    st.latex(r"""
        W_i = \int_{\alpha}^{\beta} L_i(x) \rho(x) dx, \quad R = \int_{\alpha}^{\beta} r \rho(x) dx. \tag{17}
    """)

    st.markdown(r"""
        Многочлены $$ L_i(x) $$ не зависят от $$ y(x) $$, поэтому веса от неё также не зависят. Эти веса можно определять (при заданных узлах $$ x_i $$), построя интерполирующий многочлен. В самом деле, если $$ y(x) $$ есть многочлен степени $$ (n-1) $$, то в формуле (15) $$\, r(x) = 0 $$, следовательно, в (16) $$ R = 0 $$, т.е. квадратурная формула даёт точный ответ для $$ y(x) = P_{n-1}(x) $$.
        Теперь очевидно, что если веса и узлы квадратурной формулы подобраны так, чтобы она давала точные ответы для $$ y(x) = 1, x, \dots, x^{n-1}, $$ то она даст точный ответ и для
    """)

    st.latex(r"y(x) = P_{n-1}(x) = a_{n-1} x^{n-1} + \dots + a_0,")

    st.markdown(r"""
        в силу линейности операции интегрирования. Это свойство можно положить в основу получения квадратурных формул.
    """)

#4.2.1. Фиксированные узлы
def fixed_nodes():
    st.markdown(r"""
        ### 4.2.1. Фиксированные узлы

        Пусть узлы $$ x_1, \dots, x_n $$ на отрезке $$ [\alpha, \beta] $$ известны. Обозначим через $$ m_k $$ интегралы вида:
    """)

    st.latex(r"m_k = \int_\alpha^\beta  \rho x^k dx.")

    st.markdown(r"""
        Они называются **моментами оператора интегрирования**. Определим веса $$ W_i $$ квадратурной формулы так, чтобы она была точной для $$ y(x) = 1, x, \dots, x^{n-1} $$. Отсюда получаем систему уравнений для {$$ W_i $$}:
    """)

    st.latex(r"""
        \begin{aligned}
        m_0 &= W_1 + \dots + W_n, \\
        m_1 &= W_1 x_1 + \dots + W_n x_n, \\
        &\cdots\,\cdots\,\cdots\,\cdots\,\cdots\,\cdots \\
        m_{n-1} &= W_1 x_1^{n-1} + \dots + W_n x_n^{n-1}. \tag{18}
        \end{aligned}
    """)

    st.markdown(r"""
        При $$ x_i \neq x_j $$ определитель этой системы (он называется **определителем Вандермонда**) не равен нулю, следовательно, она имеет решение.

        Запишем эту систему в виде:
    """)

    st.latex(r"X \overline{W} = \overline{m}. \tag{19}")

    st.markdown(r"""
            Тошда решение ее можно представить как
        """)

    st.latex(r"\overline{W} = X^{-1} \overline{m},")

    st.markdown(r"""
        и вся трудность заключается в вычислении обратной матрицы $$ X^{-1} $$. Чтобы её вычислить, введем в рассмотрение многочлены:
    """)

    st.latex(r"""
    \pi_i(x) = (x - x_1) \dots (x - x_{i-1})(x - x_{i+1}) \dots (x - x_n) \equiv \\
    \equiv \sum_{k=0}^{n-1} c_{ik} x^k \quad (i = 1, 2, \dots, n).
    """)

    st.markdown(r"""
        Очевидно, что $$ \pi_i(x_j) = 0, \quad$$ $$ j \neq i, \quad $$, и $$ \pi_i(x_i) \neq 0 $$.

        Покажем теперь, что каждый элемент матрицы $$ X^{-1} $$ имеет вид:
    """)

    st.latex(r"(X^{-1})_{mk} = \frac{c_{mk}}{\pi_m(x_m)}, \quad \text{(20)}")

    st.markdown(r"""
        В самом деле, из условия $$ X^{-1} X = E $$ (единичная матрица) имеем:
    """)

    st.latex(r"\sum_{j=0}^{n-1} (X^{-1})_{ij} (X)_{jk} = \delta_{ik} = \begin{cases} 1, & i = k, \\ 0, & i \neq k. \end{cases}")

    st.markdown(r"""
            С другой стороны,
        """)

    st.latex(r"""
            \sum_{j=0}^{n-1} \frac{c_{ij}}{\pi_i(x_i)} x_{k}^j = \sum_{j=0}^{n-1} (X^{-1})_{ij} (X)_{jk} = \frac{\pi_i(x_k)}{\pi_i(x_i)} = \delta_{ik}.
        """)

    st.markdown(r"""
            Таким образом, формула (20) действительно даёт элементы обратной матрицы.
            Получим с помощью этого подхода некоторые употребительные квадратурные формулы для интеграла (14), полагая в нём $$ \rho(x) = 1 $$.
            Будем предполагать для простоты записи, что узлы на этом отрезке располагаются равномерно:
        """)

    st.latex(r"x_1 = \alpha, \quad x_2 = \alpha + \tilde{h}, \dots, x_n = \beta = \alpha + (n-1)\tilde{h},")

    st.markdown(r"""
            т.е. шаг между узлами
        """)

    st.latex(r"\tilde{h} = \frac{\beta - \alpha}{n - 1}.")

    st.markdown(r"""
            Рассмотрим случай $$ n = 2 $$, когда $$ x_1 = \alpha $$, $$ x_2 = \beta $$.
            Для него имеем
        """)

    st.latex(r"""
            \pi_1 = x - \beta, \quad \pi_2 = x - \alpha, \quad \pi_1(x_1) = -\tilde{h}, \quad \pi_2(x_2) = \tilde{h}.
        """)

    st.markdown(r"""
            Следовательно,
        """)

    st.latex(r"""
            c_{10} = -\beta, \quad c_{11} = 1, \quad c_{20} = -\alpha, \quad c_{21} = 1
        """)

    st.markdown(r"""
            и
        """)

    st.latex(r"""
            X^{-1} = \frac{1}{\tilde{h}} \begin{pmatrix} \beta & -1 \\ -\alpha & 1 \end{pmatrix}.
        """)

    st.markdown(r"""
            Вычислим моменты:
        """)

    st.latex(
        r"m_0 = \int_\alpha^\beta 1 \cdot dx = \tilde{h}; \quad m_1 = \int_\alpha^\beta x dx = \frac{1}{2} (\alpha + \beta)\tilde{h}.")

    st.markdown(r"""
            Таким образом,
        """)

    st.latex(r"""
            \begin{pmatrix} W_1 \\ W_2 \end{pmatrix} = \frac{1}{\tilde{h}} \begin{pmatrix} \beta & -1 \\ -\alpha & 1 \end{pmatrix} \begin{pmatrix} \tilde{h} \\ \frac{(\alpha + \beta)}{2} \tilde{h} \end{pmatrix} = \begin{pmatrix} \tilde{h}/2 \\ \tilde{h}/2 \end{pmatrix},
        """)

    st.latex(r"\tilde{h} = \beta - \alpha,")

    st.markdown(r"""
            и квадратурная формула принимает вид (формула трапеций):
        """)

    st.latex(
        r"\int_\alpha^\beta y(x) = \frac{\beta - \alpha}{2} \left[ y(\alpha) + y(\beta) \right] + R. \quad \text{(21)}")

    st.markdown(r"""
            Пусть теперь $$ n = 3 $$, т.е.
        """)

    st.latex(r"\tilde{h} = (\beta - \alpha)/2,")

    st.markdown(r"""
            $$ x_1 = \alpha, \quad x_2 = (\alpha + \beta)/2, \quad x_3 = \beta. $$

            Для этого случая:
        """)

    st.latex(r"""
            \pi_1(x) = (x - x_2)(x - x_3) = x^2 - (x_2 + x_3)x + x_2 x_3, \quad \pi_1(x_1) = 2\tilde{h}^2;
        """)

    st.latex(r"""
            \pi_2(x) = x^2 - (x_1 + x_3)x + x_1 x_3, \quad \pi_2(x_2) = -\tilde{h}^2;
        """)

    st.latex(r"""
            \pi_3(x) = x^2 - (x_1 + x_2)x + x_1 x_2, \quad \pi_3(x_3) = 2\tilde{h}^2.
        """)

    st.markdown(r"""
            Таким образом,
        """)

    st.latex(r"""
            X^{-1} = \frac{1}{\tilde{h}^2} \begin{pmatrix} \frac{x_2 x_3}{2} & -\frac{x_2 + x_3}{2} & \frac{1}{2} \\ -x_1 x_3 & x_1 + x_3 & -1 \\ \frac{x_1 x_2}{2} & -\frac{x_1 + x_2}{2} & \frac{1}{2} \end{pmatrix}.
        """)

    st.markdown(r"""
            Для моментов находим:
        """)

    st.latex(r"m_0 = \int_\alpha^\beta dx = 2\tilde{h}; \quad m_1 = \int_\alpha^\beta x dx = (\alpha + \beta)\tilde{h};")

    st.latex(r"m_2 = \int_\alpha^\beta x^2 dx = \frac{2}{3} \tilde{h} (\alpha^2 + \alpha\beta + \beta^2).")

    st.markdown(r"""
            Таким образом, для весов имеем:
        """)

    st.latex(r"""
            \begin{pmatrix} W_1 \\ W_2 \\ W_3 \end{pmatrix} = \frac{1}{\tilde{h}} \begin{pmatrix} x_2 x_3 -\frac{(x_1 + x_3)(x_2 + x_3)}{2} + \frac{1}{3} (x_{1}^2 + x_1 x_3 + x_{3}^2 ) \\ 
            -2x_1 x_3 + (x_1 + x_3)^2 - \frac{2}{3} (x_{1}^2 + x_1 x_3 + x_{3}^2 ) \\ 
            x_1 x_2 - \frac{1}{2} (x_1 + x_2)(x_1 + x_3) + \frac{1}{3} (x_{1}^2 + x_1 x_3 + x_{3}^2)  \end{pmatrix}  =\frac{\tilde{h}}{3} \begin{pmatrix} 1 \\ 4 \\ 1 \end{pmatrix}.
        """)

    st.markdown(r"""
            И квадратурная формула принимает вид (формула Симпсона):
        """)

    st.latex(
        r"\int_\alpha^\beta y dx = \frac{\tilde{h}}{3} \left[ y(\alpha) + 4y\left( \frac{\alpha + \beta}{2} \right) + y(\beta) \right] + R. \quad \text{(22)}")

    st.markdown(r"""
            Квадратурные формулы, в которых весь интервал $$ [\alpha, \beta] $$ разбит на равные промежутки и концы интервала входят в число узлов, называются **формулами Ньютона — Котеса**. Формулы трапеций и Симпсона дают примеры этих формул.
        """)

    st.markdown(r"""
            Пусть теперь на интервале $$ [\alpha, \beta] $$ выбраны узлы вида:
        """)

    st.latex(r"x_1 = \alpha + \frac{\tilde{h}}{2}, \dots, x_n = \alpha + \frac{2n - 1}{2} \tilde{h},")

    st.markdown(r"""
            где
        """)

    st.latex(r"\tilde{h} = \frac{\beta - \alpha}{n}.")

    st.markdown(r"""
            Квадратурные формулы с узлами такого типа называются **формулами Маклорена**. Рассмотрим простейший случай $$ n = 1 $$, когда:
        """)

    st.latex(r"x_1 = (\alpha + \beta)/2.")

    st.markdown(r"""
            Для него:
        """)

    st.latex(r"\int_\alpha^\beta y dx = W_1 y \left( \frac{\alpha + \beta}{2} \right) + R,")

    st.markdown(r"""
            где $$ W_1 $$ надо определить из условия, что при $$ y(x) \equiv 1 $$ формула будет точной. Очевидно, что:
        """)

    st.latex(r"W_1 = (\beta - \alpha), \quad \text{т.е.}")

    st.latex(r"\int_\alpha^\beta y(x) dx = (\beta - \alpha) y \left( \frac{\alpha + \beta}{2} \right) + R.")

    st.markdown(r"""
            Это — **формула средних** (или формула центральных прямоугольников).

            Рассмотрим теперь, к каким квадратурным формулам приводит интерполяция по Эрмиту, когда в узлах известны не только значения функции, но и производной.

            Пусть на $$ [\alpha, \beta] $$ заданы два узла: $$ x_1 = \alpha $$, $$ x_2 = \beta $$, и квадратурная формула отыскивается в виде:
        """)

    st.latex(r"""
            \int_\alpha^\beta y(x) dx = W_1 y_1 + W_2 y_2 + W_3 y_1' + W_4 y_2' + R,
        """)

    st.markdown(r"""
            где $$ y_i = y(x_i) $$, $$ y_i' = y'(x_i) $$. Для упрощения выкладок преобразуем интервал $$ [\alpha, \beta] $$ в $$ [0, 1] $$, сделав замену переменной:
        """)

    st.latex(r"t = \frac{x - \alpha}{\beta - \alpha}.")

    st.markdown(r"""
            Очевидно, при этом:
        """)

    st.latex(r"\int_\alpha^\beta y(x) dx = (\beta - \alpha) \int_0^1 \tilde{y}(t) dt.")

    st.markdown(r"""
            Пусть интеграл по отрезку $$ [0, 1] $$ вычисляется по квадратурной формуле
        """)

    st.latex(r"\int_0^1 \tilde{y} dt = \tilde{W}_1 \tilde{y}_1 + \tilde{W}_2 \tilde{y}_2 + \tilde{W}_3 \tilde{y}_1' + \tilde{W}_4 \tilde{y}_2' + \tilde{R},")

    st.markdown(r"""
        где $$ t_1 = 0 $$, $$ t_2 = 1 $$.

        Поскольку:
    """)

    st.latex(r"""
        \tilde{y}_1 = \tilde{y}(0) = y(\alpha) = y_1; \quad \tilde{y}_2 = y_2;
    """)

    st.latex(r"""
        \tilde{y}_1' = \frac{d \tilde{y}(0)}{dt} = (\beta - \alpha) \frac{dy(\alpha)}{dx} = (\beta - \alpha) y_1'; \quad \tilde{y}_2' = (\beta - \alpha) y_2',
    """)

    st.latex(r"""
        \tilde{y}_2' = \frac{d \tilde{y}(1)}{dt} = (\beta - \alpha) y_2',
    """)

    st.markdown(r"""
        то, очевидно:
    """)

    st.latex(r"""
        W_{1,2} = (\beta - \alpha) \tilde{W}_{1,2}; \quad W_{3,4} = (\beta - \alpha)^2 \tilde{W}_{3,4}.
    """)

    st.markdown(r"""
        Сделаем эту формулу точной для $$ \tilde{y}(t) = 1, t, t^2, t^3 $$. Вычисляя моменты, получим систему уравнений:
    """)

    st.latex(r"""
    \left\{
    \begin{array}{l}
    \tilde{W}_1 + \tilde{W}_2 = 1, \\
    \tilde{W}_2 + \tilde{W}_3 + \tilde{W}_4 = 1/2, \\
    \tilde{W}_2 + 2\tilde{W}_4 = 1/3, \\
    \tilde{W}_2 + 3\tilde{W}_4 = 1/4.
    \end{array}
    \right.
    """)

    st.markdown(r"""
        Из которой находим $$ \tilde{W}_1 = \tilde{W}_2 = 1/2 $$, $$ \tilde{W}_3 = -\tilde{W}_4 = 1/12 $$.

        Следовательно:
    """)

    st.latex(r"""
        \int_\alpha^\beta y dx = \frac{\beta - \alpha}{2} \left[ y(\alpha) + y(\beta) \right] + \frac{(\beta - \alpha)^2}{12} \left[ y'(\alpha) - y'(\beta) \right] + R. \quad \text{(23)}
    """)

    st.markdown(r"""
        Эта квадратурная формула называется **формулой Эйлера**. Существуют и другие квадратурные формулы с заданными узлами, но мы на них не будем останавливаться.
    """)