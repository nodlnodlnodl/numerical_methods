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
        1. Решение систем линейных уравнений вида $$ A \bar{x} = \bar{b} $$,
            где:
            - $$ A $$ — квадратная матрица размерности $$ n \times n $$,
            - $$ \bar{x} $$ и $$ \bar{b} $$ — векторы размерности $$ n $$.
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
        $$ \bar{a}_i = (a_{i1}, \dots, a_{in})' $$ и 
        $$ \bar{a}_k = (a_{k1}, \dots, a_{kn})' $$ 
        и ввести скалярное произведение в виде:
    """)

    st.latex(r"(\bar{a}_i, \bar{a}_k) = \sum_{j=1}^{n} a_{ij} a_{kj}^*.")

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

        Рассмотрим преобразование векторного пространства, осуществляемое отражением векторов от заданной плоскости $$ Q $$. Пусть $$ \bar{W} $$ — ортогональный к плоскости $$ Q $$ вектор единичной длины. Тогда произвольный вектор $$ \bar{Z} $$ можно представить в виде:
    """)

    st.latex(r"""
    \bar{Z} = \bar{x} + \alpha \bar{W},
    """)

    st.markdown(r"""
        где $$ (\bar{x}, \bar{W}) = \sum_{j=1}^{n} x_j W_{j}^* = 0 $$; $$ \alpha $$ — число. Отраженный от плоскости $$ Q $$ вектор по определению равен:
    """)

    st.latex(r"""
    \tilde{Z}' = \bar{x} - \alpha \bar{W}.
    """)

    st.markdown(r"""
        Матрица преобразования (**отражения**) от $$ \bar{Z} $$ к $$ \tilde{Z} $$ имеет вид:
    """)

    st.latex(r"""
    U = E - 2 \bar{W} \bar{W}^H, \tag{5}
    """)

    st.markdown(r"""
        где $$ (\bar{W}\bar{W}^H)_{ij} = \bar{W}_i \bar{W}_j^* $$, т.е. $$ \bar{W} \bar{W}^H $$ — произведение матрицы-столбца $$ W $$ на матрицу-строку $$ W^H $$.

        В самом деле:
    """)

    st.latex(r"""
    U \bar{Z} = (E - 2 \bar{W} \bar{W}^H)(\bar{x} + \alpha \bar{W}) = \bar{Z} - 2 (\bar{W} \bar{W}^H)\bar{x} - 2\alpha ( \bar{W} \bar{W}^H)\bar{W} = \\
    = \bar{Z} - 2\bar{W}(\bar{x}, \bar{W}) - 2\alpha \bar{W}(\bar{W}, \bar{W}),
    """)

    st.markdown(r"""
        где $$ (\bar{x}, \bar{W}), $$ $$ (\bar{W}, \bar{W}) $$ — скалярное произведение. Учитывая ортогональность $$ \bar{x} $$ и $$ \bar{W} $$, имеем:
    """)

    st.latex(r"""
    U \bar{Z} = \bar{Z} - 2 \alpha \bar{W}(\bar{W}, \bar{W}) = \bar{Z} - 2 \alpha \bar{W} = \bar{x} - \alpha \bar{W} = \tilde{Z}.
    """)

    st.markdown(r"""
        Покажем, что $$ U $$ — унитарная матрица:
    """)

    st.latex(r"""
    UU^H = (E - 2 \bar{W} \bar{W}^H)(E - 2 (\bar{W}^H)^H \bar{W}^H) = \\
    = E - 4 \bar{W} \bar{W}^H + 4 \bar{W} ( \bar{W}^H \bar{W} ) \bar{W}^H = \\ 
    = E - 4 \bar{W} \bar{W}^H + 4 ( \bar{W}, \bar{W}) \bar{W} \bar{W}^H = E.
    """)

    st.markdown(r"""
        Пусть $$ \bar{S}, \bar{e} $$ — два произвольных вектора-столбца, причем $$ \bar{e} $$ — единичной длины: $$ (\bar{e}, \bar{e}) = 1 $$.
        Вектор $$ \bar{W} $$ всегда можно выбрать так, чтобы построенная по нему матрица $$ U $$ переводила $$ \bar{S} $$ в вектор, параллельный $$ \bar{e} $$.

        Положим:
    """)

    st.latex(r"""
    \bar{W} = \frac{1}{\rho} (\bar{S} - \alpha \bar{e}),
    """)

    st.markdown(r"""
        где:
    """)

    st.latex(r"""
    |\alpha| = \sqrt{(\bar{S}, \bar{S})}; \quad \arg \alpha = \arg (\bar{S}, \bar{e}) - \pi; \quad \rho = \sqrt{(\bar{S} - \alpha \bar{e}, \bar{S} - \alpha \bar{e})} = \\
     = \sqrt{2|\alpha|^2 + 2|\alpha| |(\bar{S}, \bar{e})|}.
    """)

    st.markdown(r"""
            тогда:
        """)

    st.latex(r"""
        U \bar{S} = \bar{S} - 2 (\bar{S}, \bar{W}) \bar{W} = \bar{S} - \frac{2} {\rho} (\bar{S}, \bar{S} - \alpha \bar{e}) \bar{W} = \bar{S} - \frac{2} {\rho} [\bar{S}, \bar{S}) - (\bar{S}, \alpha \bar{e}]\bar{W} =\\
        = \bar{S} - \frac{1}{\rho} \left[ 2|\alpha|^2 - 2\alpha^* (\bar{S}, \bar{e})\right] \bar{W} = \\
        = \bar{S} - \frac{1}{\rho} \left[ 2|\alpha|^2 - 2|\alpha| e^{i[\pi - \arg (\bar{S}, \bar{e})]} |(\bar{S}, \bar{e})| e^{i \arg (\bar{S}, \bar{e})} \right] \bar{W} = \\
        = \bar{S} - \frac{1}{\rho} \left[ 2|\alpha|^2 + 2|\alpha| |(\bar{S}, \bar{e})| \right] \bar{W} = \bar{S} - \rho \bar{W} = \alpha \bar{e}.
        """)

    st.markdown(r"""
            Свойства матрицы отражения используются при преобразованиях матриц. Пусть $$ A $$ — некоторая комплексная матрица, умножим ее слева на матрицу $$ U_1 $$, выбрав в качестве $$ \bar{S} $$ и $$ \bar{e} $$ векторы:
        """)

    st.latex(r"""
        \bar{S} = \begin{pmatrix}
        a_{11} \\
        \vdots \\
        a_{n1}
        \end{pmatrix}; \quad \bar{e} = \begin{pmatrix}
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
        \bar{S} = \begin{pmatrix}
        0 \\
        a_{22}^{(1)} \\
        \vdots \\
        a_{n2}^{(1)}
        \end{pmatrix}, \quad \bar{e} = \begin{pmatrix}
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
