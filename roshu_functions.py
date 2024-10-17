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



def metod_gausa():
    st.write("## 6.4 Прямые методы решения систем линейных уравнений")
    st.write("### 6.4.1 Метод исключения Гаусса")

    st.write("""
    Для систем линейных уравнений $$Ax = b$$ с заполненной матрицей $$A$$, хранящейся в оперативной памяти машины, не найдено алгоритмов решения, лучших по времени и точности, чем метод последовательного исключения Гаусса. Основной его идеей является представление матрицы $$A$$ в виде LU-разложения. Если для $$A$$ известно такое разложение, то решение исходной системы сводится к решению уравнений:
    
    $$
    Ly = b, \quad Ux = y,
    $$
    
    которые легко решаются, поскольку $$L$$ и $$U$$ имеют вид:
    """)

    st.latex(r"""
    L = \begin{pmatrix}
    1 & 0 & 0 & \dots & 0 \\
    m_{21} & 1 & 0 & \dots & 0 \\
    m_{31} & m_{32} & 1 & \dots & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    m_{n1} & m_{n2} & m_{n3} & \dots & 1 \\
    \end{pmatrix},
    \quad
    U = \begin{pmatrix}
    U_{11} & U_{12} & U_{13} & \dots & U_{1n} \\
    0 & U_{22} & U_{23} & \dots & U_{2n} \\
    0 & 0 & U_{33} & \dots & U_{3n} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & 0 & \dots & U_{nn} \\
    \end{pmatrix}
    """)

    st.write("""
    Из первой системы последовательно находим $$y_1, y_2, \dots, y_n$$ (прямое исключение), из второй $$x_n, x_{n-1}, \dots, x_1$$ (обратная подстановка).
    
    Рассмотрим теперь классический метод исключения Гаусса и выведем попутно на его основе выражения для матриц $$L, U$$. Запишем исходную систему в виде:
    """)

    st.latex(r"""
    \begin{aligned}
    a_{11}x_1 + a_{12}x_2 + \dots + a_{1n}x_n &= b_1, \\
    a_{21}x_1 + a_{22}x_2 + \dots + a_{2n}x_n &= b_2, \\
    \vdots \\
    a_{n1}x_1 + a_{n2}x_2 + \dots + a_{nn}x_n &= b_n.
    \end{aligned}
    """)

    st.write("""
    Предположим, что $$a_{11} \neq 0$$, и образуем числа $$m_{i1} = a_{i1}/a_{11}$$, $$i = 2, 3, \dots, n$$. Умножая первое уравнение на $$m_{i1}$$ и вычитая результат из $$i$$-го уравнения $$(i = 2, \dots, n$$), придем к новой системе, в которой уравнения с номерами $$i \geq 2$$ не содержат $$x_1$$. Это преобразование эквивалентно умножению системы $$Ax = b$$ слева на матрицу:
    """)

    st.latex(r"""
    M_1 = \begin{pmatrix}
    1 & 0 & 0 & \dots & 0 \\
    -m_{21} & 1 & 0 & \dots & 0 \\
    -m_{31} & 0 & 1 & \dots & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    -m_{n1} & 0 & 0 & \dots & 1 \\
    \end{pmatrix}
    """)

    st.write("""
    В результате приходим к системе:
    """)

    st.latex(r"""
    A^{(2)} x = b^{(2)}
    """)

    st.write("""
    где
    """)

    st.latex(r"""
    A^{(2)} = \begin{pmatrix}
    a_{11} & a_{12} & \dots & a_{1n} \\
    0 & a^{(2)}_{22} & \dots & a^{(2)}_{2n} \\
    0 & 0 & \dots & a^{(2)}_{nn}
    \end{pmatrix}, \quad
    b^{(2)} = \begin{pmatrix}
    b_1 \\
    b_2 - m_{21}b_1 \\
    b_3 - m_{31}b_1
    \end{pmatrix}
    """)

    st.write("""
    Пусть в (11) $$a^{(2)}_{22} \neq 0$$, составим числа $$m_{i2} = a_{i2}^{(2)} / a_{22}^{(2)}$$, $$i = 3, \dots, n$$, и умножим на них второе уравнение, затем вычтем из каждого последующего. После этих преобразований получится матрица:
    """)

    st.latex(r"""
    M_2 = \begin{pmatrix}
    1 & 0 & 0 & \dots & 0 \\
    0 & 1 & 0 & \dots & 0 \\
    0 & -m_{32} & 1 & \dots & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    0 & -m_{n2} & 0 & \dots & 1
    \end{pmatrix}
    """)

    st.write("""
    Умножая $$ (11) $$ на $$M_2$$ слева, приходим к системе:
    
    $$
    A^{(3)} x = b^{(3)},
    $$
    
    в которой уравнения, начиная с третьего, не содержат $$x_1, x_2$$. На практике не формируют матрицу $$M_2$$, а делают, как и на первом шаге, исключение $$x_2$$ из третьего, четвертого и так далее уравнений с помощью элементарных преобразований. Проделав подобную процедуру $$ (n - 1) $$ раз, придем к системе вида:
    """)

    st.latex(r"""
    \begin{aligned}
    a^{(n)}_{nn}x_n &= b^{(n)}_n \\
    0 &= b^{(n)}_{n-1} \\
    \vdots \\
    0 &= b^{(n)}_2 \\
    0 &= b^{(n)}_1
    \end{aligned}
    """)

    st.write("""
    Решение этой системы очевидно. Сформулируем теперь получение результата $$ (12) $$ в матричном виде. Как следует из предыдущего, $$ (12) $$ возникает при умножении $$ (10) $$ слева на $$ M = M_{n-1} \dots M_1 $$, т.е. $$ M = L^{-1} $$, где $$L$$ — нижняя треугольная матрица для $$A$$. Поэтому:
    """)

    st.latex(r"""
    A = M^{-1}U.
    """)

    st.write("""
    Но $$M^{-1} = L$$, нетрудно убедиться, что $$M^{-1}_k$$ есть матрица $$M_k$$ с измененными знаками у элементов ниже диагонали. В самом деле, пусть $$C = (c_{ij})$$, $$F = (f_{ij})$$ — две нижние треугольные $$ (n \times n) $$-матрицы с единицами на диагонали. Тогда:
    
    $$
    b_{ij} = (CF)_{ij} = \sum_{s \leq i} c_{is} f_{sj}.
    $$
    
    Откуда видно, что $$b_{ij} = 1$$, $$b_{ij} = 0$$ при $$i < j$$, т.е. $$B = CF$$ — нижняя треугольная матрица с единичной диагональю. Пусть теперь $$C = M_k$$, $$F$$ равна матрице $$M_k$$ с элементами противоположного знака в $$k$$-м столбце. Тогда:
    
    $$
    b_{ij} = (CF)_{ij} = \sum_{s \leq i} c_{is} f_{sj} = \delta_{ij}.
    $$
    
    Таким образом, $$b_{ij} = \delta_{ij}$$. Следовательно, $$L = M^{-1}$$ — нижняя треугольная матрица. Если хранить матрицы $$L, U$$ на месте матрицы $$A$$, то (единицы для $$(L$$ не храним!) вместо $$A$$ получим:
    """)

    st.write("""
    Таким образом, метод последовательного исключения Гаусса привел нас к разложению матрицы $$A$$ в $$LU$$-произведение. Сделаем некоторые замечания.
    """)

    st.write("### Замечания:")

    st.write("""
    1. Если разложение (13) получено, то, согласно LU-теореме:
    
    $$
    \det A = U_{11} U_{22} \dots U_{nn}.
    $$
    """)

    st.write("""
    2. Если на каком-то $$k$$-этапе окажется, что $$a^{(k)}_{kk} = 0$$, т.е. на главной диагонали появится нулевой элемент, то исключение Гаусса далее с ним производить нельзя. Но все элементы $$a_{ik}, i > k$$, не могут быть нулями, так как это означает, что $$ \det A = 0 $$. Перестановкой строк к $$k$$-й строке можно добиться, чтобы $$a^{(k)}_{kk} \\neq 0$$, и вычисления можно продолжить. На практике обычно перестановка строк осуществляется выбором $$a^{(k)}_{kk}$$, являющегося максимальным по модулю из элементов ниже диагонали.
    """)

    st.write("""
    Решим систему, выполняя действия на машине с тремя десятичными знаками и плавающей запятой:
    
    $$
    0.000100x_1 + 1.00x_2 = 1.00, \\
    -10000x_2 = -10000 \quad \Rightarrow x_2 = 1.00, \\
    1.00x_1 = 1.00 \quad \Rightarrow x_1 = 1.00.
    $$
    
    Истинное решение этой системы есть:
    """)

    st.latex(r"""
    x_1 = 10000, \quad x_2 = 9999.
    """)

    st.write("""
    Этот метод дает экономичный способ построения обратной матрицы $$A^{-1}$$. Для этого можно использовать матричное уравнение вида:
    
    $$
    AA^{-1} = E.
    $$
    """)
