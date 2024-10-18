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

def metod_otrajeniy():
    st.title("6.4.2 Метод отражений")

    st.write("""
    Напомним, что для произвольного вектора $\\vec{S}$ легко построить матрицу отражения такую, что $U \\cdot \\vec{S} = \\alpha \\cdot \\vec{I}$, где $\\vec{I}$ — единичный вектор заданного направления. Для этого надо взять:
    """)
    st.latex(r"""
    W = \frac{1}{\rho}(\vec{S} - \alpha \vec{I}),
    """)
    st.write("""
    где
    """)
    st.latex(r"""
    \alpha = (\vec{S}, \vec{I}), \quad |\alpha| = |(\vec{S}, \vec{I})|, \quad \text{arg} = \text{arg}(\vec{S}, \vec{I}) - \pi,
    """)
    st.latex(r"""
    \rho = |\vec{S} - \alpha \vec{I}| = \sqrt{2|\alpha|^2 + 2| (\vec{S}, \vec{I}) |}.
    """)

    st.write("""
    Пусть требуется решить систему $A \\cdot \\vec{x} = \\vec{b}$. Обозначим расширенную матрицу со столбцами $\\vec{a}_1, \\vec{a}_2, \\dots, \\vec{a}_n, \\vec{b}$ через $A_0$:
    """)
    st.latex(r"""
    A_0 = \begin{pmatrix}
    a_{11}^{(0)} & a_{12}^{(0)} & \dots & a_{1n}^{(0)} & b_1^{(0)} \\
    a_{21}^{(0)} & a_{22}^{(0)} & \dots & a_{2n}^{(0)} & b_2^{(0)} \\
    \vdots & \vdots & \ddots & \vdots & \vdots \\
    a_{n1}^{(0)} & a_{n2}^{(0)} & \dots & a_{nn}^{(0)} & b_n^{(0)}
    \end{pmatrix}.
    """)

    st.write("""
    Будем преобразовывать ее по правилу:
    """)
    st.latex(r"""
    A_{k+1} = U_{k+1} A_k, \quad k = 0, 1, \dots, n-2,
    """)
    st.write("""
    с помощью последовательного умножения на матрицы $U_1, U_2, \dots, U_{n-1}$. Очевидно, что:
    """)
    st.latex(r"""
    \vec{a}_i^{(k+1)} = U_{k+1} \vec{a}_i^{(k)}.
    """)

    st.write("""
    Для $U_1$ выберем:
    """)
    st.latex(r"""
    \vec{S} = \vec{a}_1^{(0)} = (a_{11}, a_{12}, \dots, a_{1n}, b_1)^T, \quad \vec{I} = (1, 0, 0, \dots, 0)^T.
    """)

    st.write("""
    Тогда у вектора $\\vec{a}_1^{(0)}$ все координаты, кроме первой, будут равны нулю. Пусть уже построена матрица $A_k$, у которой $\\vec{a}_i^{(k)} = 0$ для $i > j$, где $j = 1, 2, \\dots, k$.

    Для построения матрицы $U_{k+1}$ в качестве $\\vec{S}_{k+1}$ выберем:
    """)
    st.latex(r"""
    \vec{S}_{k+1} = (0, 0, \dots, 0, a_{k+1}^{(k+1)}, a_{k+2}^{(k+1)}, \dots, a_n^{(k+1)}, b_{k+1}^{(k+1)})^T,
    """)
    st.latex(r"""
    \vec{I}_{k+1} = (0, 0, \dots, 0, 1, 0, \dots, 0)^T.
    """)

    st.write("""
    В этом случае:
    """)
    st.latex(r"""
    W = (0, 0, \dots, 0, W_{k+1}, W_{k+2}, \dots, W_n).
    """)

    st.write("""
    Для матрицы $U_k$ имеем:
    """)
    st.latex(r"""
    U_k = \begin{pmatrix}
    1 & 0 & 0 & \dots & 0 & 0 \\
    0 & 1 & 0 & \dots & 0 & 0 \\
    0 & 0 & 1 & \dots & 0 & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
    0 & 0 & 0 & \dots & 1 & 0 \\
    0 & 0 & 0 & \dots & 0 & k
    \end{pmatrix}.
    """)

    st.write("""
    Для матрицы $A_{k+1}$ условие (14) будет выполнено и для $j = k+1$. При этом вектор $S$ не будет равен нулю для невырожденной матрицы $A$. После $(n-1)$ шага получим матрицу $A_{n-1}$, первые $n$ столбцов которой образуют верхнюю треугольную матрицу. Тогда:
    """)

    st.latex(r"""
    x_n = -\frac{a_{nn+1}^{(n-1)}}{a_{nn}^{(n-1)}};
    """)

    st.latex(r"""
    x_i = -\frac{a_{in+1}^{(n-1)} + \sum_{j=i+1}^{n} a_{ij}^{(n-1)} x_j}{a_{ii}^{(n-1)}}, \quad i = n-1, n-2, \dots, 1.
    """)

    st.write("""
    Этот метод похож на метод исключения Гаусса. Достоинством его является то, что вычисления делаются единообразно, без изменения порядка исключения. Метод отражений требует $\\frac{2}{3}n^3$ операций умножения, $\\frac{3}{2}n^3$ операций сложения, $-2n$ делений и $(n-1)$ извлечений квадратных корней, т.е. он вдвое дороже метода Гаусса.
    """)