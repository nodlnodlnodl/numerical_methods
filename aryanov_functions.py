import streamlit as st  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from streamlit_ace import st_ace

from scipy.integrate import odeint
from scipy.optimize import fsolve
import time

import time
import io
import sys


def convergence_of_difference_scheme():
    st.markdown("""
    ### 10.1. Сходимость разностной схемы
""")
    
    st.markdown(r"""
        Пусть на некотором отрезке $$ D $$ поставлена дифференциальная задача (задача Коши или краевая задача). 
        Это значит, что задано дифференциальное уравнение (или система уравнений), которому удовлетворяет решение, и 
        некоторые дополнительные условия на одном или двух концах отрезка. Будем дифференциальную задачу записывать 
        в виде символического равенства:
    """)

    st.latex(r"""
        L y = f, \quad (1)
    """)
    st.markdown(r"""
        где $$ L $$ — заданный дифференциальный оператор, $$ f $$ — заданная правая часть.

        **Пример 1:**
    """)

    st.latex(r"""
        \frac{dy}{dx} + \frac{x}{1 + y^2} = \cos x, \quad 0 \leq x \leq 1, \quad y(0) = 3.
    """)

    st.markdown("""
        Здесь:
    """)

    st.latex(r"""
        L y = \frac{dy}{dx} + \frac{x}{1 + y^2}, \quad 0 < x \leq 1, \quad y(0) = 3, \quad f = \cos x.
    """)

    st.markdown("""
        **Пример 2:**
    """)

    st.latex(r"""
        \frac{d^2 y}{dx^2} - (1 + x^2) y = \sqrt{x}, \quad 0 \leq x \leq 1, \quad y(0) = 2, \quad y'(0) = 1.
    """)

    st.markdown("""
        Здесь:
    """)

    st.latex(r"""
        L y = \frac{d^2 y}{dx^2} - (1 + x^2) y, \quad 0 < x \leq 1, \quad f = \sqrt{x}, \quad y(0) = 2, \quad y'(0) = 1.
    """)

    st.markdown("""
        **Пример 3:**
    """)

    st.latex(r"""
        \frac{d^2 y}{dx^2} - (1 + x^2) y = \sqrt{x + 1}, \quad 0 \leq x \leq 1, \quad y(0) = 2, \quad y(1) = 1.
    """)

    st.markdown("""
        Здесь:
    """)

    st.latex(r"""
        L y = \frac{d^2 y}{dx^2} - (1 + x^2) y, \quad 0 < x \leq 1, \quad f = \sqrt{x + 1}, \quad y(0) = 2, \quad y(1) = 1.
    """)

    st.markdown("""
        **Пример 4: Рассмотрим дифференциальную задачу для системы уравнений:**
    """)

    st.latex(r"""
        \frac{dv}{dx} + x v w = x^2 - 3x + 1, \quad 0 \leq x \leq 1, \quad v(0) = 1, 
    """)

    st.latex(r"""
        \frac{dw}{dx} = \frac{1}{1 + x^2} (v + w), \quad w(0) = -3.
    """)

    st.markdown(r"""
        Будем считать $$ v $$ и $$ w $$ компонентами вектор-функции $$ y = (v, w) $$. Тогда:
    """)

    st.latex(r"""
        L y = 
        \begin{cases}
        \frac{dv}{dx} + x v w = x^2 - 3x + 1, \quad 0 \leq x \leq 1, \\
        \frac{dw}{dx} = \frac{1}{1 + x^2} (v + w), \quad v(0) = 1, \quad w(0) = -3.
        \end{cases}
    """)

    st.markdown(r"""
        Предположим, что решение задачи (1) на отрезке $$ D $$ существует и единственно. Для вычисления его с 
        помощью метода конечных разностей (или метода сеток) нужно выбрать конечное число точек (узлов), совокупность которых $$ D_h $$ 
        называется сеткой. Затем можно вычислить значения решения задачи в узлах сетки.
        
        Отметим, что используя разностные схемы, мы найдем в узлах сетки не значения точной функции, а значения другой сеточной функции 
        $$ y^{(h)} $$, которая стремится к $$ [y] $$ при $$ h \rightarrow 0 $$.
        
        Рассмотрим линейное нормированное пространство $$ U_h $$, где каждой функции из этого пространства можно присвоить норму. 
        Например, норма $$ ||v^{(h)}||_{U_h} $$ — это точная верхняя грань значений модуля функции в узлах сетки:
    """)

    st.latex(r"""
        ||v^{(h)}||_{U_h} = \sup_n |v^{(h)}(x_n)|, \quad (2)
    """)

    st.markdown(r"""
        Пусть $$ a^{(h)}(x) $$ и $$ b^{(h)}(x) $$ — две произвольные функции на сетке, тогда отклонение этих функций 
        друг от друга можно записать в следующем виде:
    """)

    st.latex(r"""
        ||a^{(h)} - b^{(h)}||_{U_h}
    """)

    st.markdown("""
        Далее перейдем к строгому определению сходимости разностной схемы.

        Для приближенного решения задачи (1) на сетке, пусть существует система уравнений (разностная схема), которую можно записать как:
    """)

    st.latex(r"""
        L_h y^{(h)} = f^{(h)}, \quad (3)
    """)

    st.markdown("""
        Например, для задачи:
    """)

    st.latex(r"""
        \frac{dy}{dx} + \frac{x}{1 + y^2} = \cos x, \quad 0 \leq x \leq 1, \quad y(0) = 3
    """)

    st.markdown("""
        разностная схема на сетке будет иметь вид:
    """)

    st.latex(r"""
        \frac{1}{h}(y_n - y_{n-1}) + \frac{x_{n-1}}{1 + y_{n-1}^2} = \cos x_{n-1}, \quad n = 1, 2, ..., N, \quad y_0 = 3
    """)

    st.markdown(r"""
        Теперь, определим сходимость разностной схемы. Решение $$ y^{(h)} $$ сходится к решению задачи (1), если
    """)

    st.latex(r"""
        || [y]_h - y^{(h)} ||_{U_h} \to 0, \quad h \to 0 \quad (4)
    """)

    st.markdown("""
        Если дополнительно выполняется неравенство:
    """)

    st.latex(r"""
        || [y]_h - y^{(h)} ||_{U_h} \leq C h^k, \quad (5)
    """)

    st.markdown(r"""
        то разностная схема (3) имеет $$ k $$-й порядок точности.
        
        Сходимость является фундаментальным требованием для разностных схем. Чтобы схема обладала сходимостью, 
        нужно также обеспечить её аппроксимацию и устойчивость.
    """)

    st.markdown(r"""
        Предположим, что решение задачи (1) на отрезке $$ D $$ существует и единственно. 
        Для вычисления его с помощью метода конечных разностей (или метода сеток) надо на отрезке $$ D $$ выбрать конечное число точек (узлов), 
        совокупность которых $$ D_h $$ называется сеткой, а затем насчитать таблицу $$ [y]_h $$ значений решения задачи (1) в узлах сетки $$ D_h $$. 
        Предполагается, что сетка $$ D_h $$ зависит от параметра $$ h > 0 $$, который может принимать сколько угодно малые значения.
    """)

    st.markdown(r"""
        Отметим, что, используя разностные схемы, мы найдем в узлах сетки не значения точной функции, а определим значения другой сеточной функции $$ y^{(h)} $$, 
        которая стремится к $$ [y]_h $$ при $$ h \to 0 $$. Для того чтобы придать этим рассуждениям количественный характер, рассмотрим линейное нормированное пространство $$ U_h $$ 
        функций, определенных на $$ D_h $$. Напомним, что в этом случае каждому элементу $$ y^{(h)} \in U_h $$ можно приписать норму $$ ||y^{(h)}||_{U_h} $$ — некоторое неотрицательное число, 
        характеризующее меру отклонения функции от тождественного нуля.

        Наиболее широко нами будет использоваться норма вида:
    """)

    st.latex(r"""
        ||v^{(h)}||_{U_h} = \sup_n |v^{(h)}(x_n)|, \quad (2)
    """)

    st.markdown(r"""
        — точная верхняя грань модуля значений функции в узлах сетки. Если $$ v^{(h)} $$ — вектор-функция, то за $$ ||v^{(h)}||_{U_h} $$ можно принять верхнюю грань модулей значений её компонент.

        Пусть $$ a^{(h)}, b^{(h)} \in U_h $$ — две произвольные сеточные функции; мерой их отклонения друг от друга назовем
    """)

    st.latex(r"""
        ||a^{(h)} - b^{(h)}||_{U_h}
    """)

    st.markdown(r"""
        После этих понятий перейдем к строгому определению сходимости разностной схемы.

        Пусть для приближенного решения дифференциальной задачи (1), т.е. для приближенного вычисления таблицы $$ [y]_h $$, составлена некоторая система уравнений (разностная схема), 
        которую символически будем записывать в виде
    """)

    st.latex(r"""
        L_h y^{(h)} = f^{(h)}, \quad (3)
    """)

    st.markdown("""
        Например, для дифференциальной задачи
    """)

    st.latex(r"""
        \frac{dy}{dx} + \frac{x}{1 + y^2} = \cos x, \quad 0 \leq x \leq 1, \quad y(0) = 3
    """)

    st.markdown(r"""
        можно написать схему на сетке $$ x_0 = 0, x_1 = h, x_2 = 2h, \dots, x_N = Nh = 1 $$:
    """)

    st.latex(r"""
        \frac{1}{h}(y_n - y_{n-1}) + \frac{x_{n-1}}{1 + y_{n-1}^2} = \cos x_{n-1}, \quad n = 1, 2, \dots, N, \quad y_0 = 3.
    """)

    st.markdown("""
        В этом случае
    """)

    st.latex(r"""
        L_h y^{(h)} = 
        \begin{cases}
        \frac{1}{h}(y_n - y_{n-1}) + \frac{x_{n-1}}{1 + y_{n-1}^2} = \cos x_{n-1}, \quad n = 1, 2, \dots, N, \\
        y_0 = 3.
        \end{cases}
    """)

    st.markdown(r"""
        Система (3) зависит от $$ h $$ и должна быть выписана для всех тех $$ h $$, для которых строится сетка $$ D_h $$ и вычисляется функция $$ [y]_h $$. 
        Будем предполагать, что при каждом рассматриваемом $$ h $$ существует и единственно решение $$ y^{(h)} $$ задачи (3) и $$ y^{(h)} \in U_h $$.

        **Определение.** Решение $$ y^{(h)} $$ разностной задачи (3) сходится к решению дифференциальной задачи (1), если
    """)

    st.latex(r"""
        ||[y]_h - y^{(h)}||_{U_h} \to 0, \quad h \to 0. \quad (4)
    """)

    st.markdown("""
        Если, кроме того, выполнено неравенство
    """)

    st.latex(r"""
        ||[y]_h - y^{(h)}||_{U_h} \leq c h^k, \quad (5)
    """)

    st.markdown(r"""
        где $$ c > 0, k > 0 $$ — не зависящие от $$ h $$ постоянные, то разностная схема (3) имеет $$ k $$-й порядок точности (имеет место сходимость порядка $$ h^k $$).

        Свойство сходимости является фундаментальным требованием, которое предъявляется к разностной схеме (3) для численного решения дифференциальной задачи (1). Если оно имеет место, 
        то с помощью схемы можно вычислять решение дифференциальной задачи с любой наперед заданной точностью.

        Как было видно на примерах, не всякая разностная схема пригодна для счета. Чтобы построить сходящуюся разностную схему, надо обеспечить два её фундаментальных свойства: аппроксимацию и устойчивость.
    """)


def approximation_of_difference_scheme():
    st.markdown("""
    ### 10.2. Аппроксимация дифференциальной задачи разностной схемой
""")

    st.markdown(r"""
    Рассмотрим разностную схему (3), составленную для приближенного решения дифференциальной задачи (1),
    и предположим, что эта система уравнений имеет единственное решение $$ y^{(h)} $$. Как правило, разностная схема не
    удовлетворяет дифференциальной задаче (1) в точности, т.е. $$ L_h [y] \neq f^{(h)} $$, и на самом деле она аппроксимирует
    эту задачу. Степень этой аппроксимации можно количественно оценить с помощью величины невязки:
""")

    st.latex(r"""
    \delta f^{(h)} = L_h [y] - f^{(h)}
""")

    st.markdown(r"""
    Величина $$ \delta f^{(h)} $$ будет зависеть от выбора разностной схемы, величины шага $$ h $$ и точности задачи (1).
    Чтобы величина невязки была как можно меньше, разностная схема должна хорошо аппроксимировать дифференциальную задачу.
    Величина $$ \delta f^{(h)} $$ зависит от некоторых параметров, таких как размер шага сетки, гладкость функций и другие параметры.
""")
    st.markdown("""
    **Пример 1.** Рассмотрим задачу:
""")

    st.latex(r"""
        \frac{d^2 y}{dx^2} + y = 1 + x^2, \quad 0 \leq x \leq 1, \quad y(0) = b.
    """)

    st.markdown("""
        В этой задаче решаем задачу приближенного вычисления значений решения дифференциального уравнения с использованием разностной схемы вида:
    """)

    st.latex(r"""
        \frac{1}{h^2}(y_{n+1} - 2y_n + y_{n-1}) + y_n = 1 + x_n^2, \quad n = 1, 2, \dots, N.
    """)

    st.markdown("""
        Здесь \( y_0 = b \), \( y_N = b \), и сетка на отрезке \( D = [0,1] \) введена сеткой \( D_h \), 
        где \( x_0 = 0 \), \( x_1 = h \), \( x_2 = 2h, \dots, x_N = Mh = 1 \).
    """)

    st.markdown(r"""
        **Определение.** Разностная схема $$ L_h y^{(h)} = f^{(h)} $$ аппроксимирует задачу с порядком $$ h^k $$, если
    """)

    st.latex(r"""
        ||\delta f^{(h)}||_{F_h} \leq c h^k.
    """)

    st.markdown("""
        Кроме того, если имеет место неравенство:
    """)

    st.latex(r"""
        ||\delta f^{(h)}||_{F_h} \leq c h^k,
    """)

    st.markdown(r"""
        где $$ c > 0 $$, $$ k > 0 $$ — постоянные, не зависящие от $$ h $$, то имеет место аппроксимация порядка $$ h^k $$.

        Таким образом, при $$ h \to 0 $$ выполняется:
    """)

    st.latex(r"""
        ||\delta f^{(h)}||_{F_h} \leq M h,
    """)

    st.markdown(r"""
        т.е. схема аппроксимирует задачу с порядком $$ h $$.
    """)

    st.markdown(r"""
        Обратим внимание на тот факт, что разные компоненты вектора невязки имеют разный порядок относительно $$ h $$. Порядок аппроксимации этой системы можно повысить, 
        если более точно задать значение $$ y_1 = y(h) $$. Типичный прием повышения точности задания граничных условий — привлечение дифференциальных уравнений. Положим, 
        вместо условия $$ y_1 = b $$
    """)

    st.latex(r"""
        y_1 = y(0) + h y'(0) + O(h^2) = b + y'(0) h + O(h^2).
    """)

    st.markdown("""
        Из уравнения имеем:
    """)

    st.latex(r"""
        y'(0) = -a y(0) + 1 = 1 - ab.
    """)

    st.markdown("""
        Поэтому
    """)

    st.latex(r"""
        y_1 = b + h (1 - ab) + O(h^2) = b (1 - ha) + h + O(h^2).
    """)

    st.markdown(r"""
        Очевидно, в этом случае все компоненты вектора невязки будут иметь порядок $$ O(h^2) $$ и
    """)

    st.latex(r"""
        ||\delta f^{(h)}||_{F_h} \leq M h^2.
    """)

    st.markdown("""
        Как построить разностную схему, аппроксимирующую дифференциальные уравнения с заданным порядком аппроксимации? Простым, но не всегда самым лучшим способом является замена производных 
        соответствующими разностями. Для этого используется метод неопределенных коэффициентов.
    """)

    st.markdown(r"""
        Пусть имеется производная $$ d^k y/dx^k $$ и надо на сетке с шагом $$ h $$ таким образом заменить эту производную разностным отношением, чтобы для гладкой функции $$ y(x) $$ было выполнено равенство:
    """)

    st.latex(r"""
        \frac{d^k y_i}{dx^k} = h^{-k} \sum_{s_1}^{s_2} a_s y_{i+s} + O(h^p),
    """)

    st.markdown(r"""
        где $$ a_s $$ — подлежащие определению коэффициенты; $$ s_1 \geq 0, s_2 \geq 0 $$.

        По формуле Тейлора имеем:
    """)

    st.latex(r"""
        y_{i+s} = y(x_i + s h) = y_i + s h y'_i + \dots + O(h^{k+p-1}).
    """)

    st.markdown("""
        Отсюда получаем систему уравнений для аппроксимации:
    """)

    st.latex(r"""
        \sum a_s = 0, \quad \sum s^k a_s = k!.
    """)

    st.markdown(r"""
        Если $$ s_1 + s_2 = k + p - 1 $$, то выписанные $$ k + p $$ равенств образуют систему относительно $$ k + p $$ неизвестных.
    """)

    st.markdown(r"""
        **Пример 1:** Существуют единственные разностные формулы для $$ dy/dx $$ порядка $$ O(h) $$, имеющие вид:
    """)

    st.latex(r"""
        \frac{dy}{dx} = \frac{y_{i+1} - y_i}{h} + O(h), \quad \frac{dy}{dx} = \frac{y_i - y_{i-1}}{h} + O(h).
    """)

    st.markdown("""
        Отсюда имеем:
    """)

    st.latex(r"""
        a_1 = \frac{1}{2}(1 - a_0), \quad a_{-1} = \frac{1}{2}(1 + a_0).
    """)

    st.markdown("""
        Таким образом:
    """)

    st.latex(r"""
        \frac{dy_i}{dx} = \frac{1}{2h} \left[ -(a_0 + 1)y_{i-1} + 2a_0 y_i + (1 - a_0) y_{i+1} \right] + O(h)
    """)

    st.markdown(r"""
        при любом $$ a_0 $$.

        Отметим, что соотношение вида:
    """)

    st.latex(r"""
        \frac{dy_i}{dx} = \frac{1}{h} \left[ a_{-1} y_{i-1} + a_0 y_i + a_1 y_{i+1} \right] + O(h^2)
    """)

    st.markdown("""
        существует только одно:
    """)

    st.latex(r"""
        a_0 = 0, \quad a_{-1} = -\frac{1}{2}, \quad a_1 = \frac{1}{2}.
    """)
