import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_ace import st_ace
from code_editor import code_editor


# 4.2.6.1. Кратные интегралы Метод ячеек
def multiple_integrals_cell_method():
    st.header("4.2.6 Кратные интегралы")

    st.markdown("#### 4.2.6.1. Метод ячеек")

    st.markdown("""
    Рассмотрим двойной интеграл по прямоугольнику:
    """)
    st.latex(r"""
    G(a \leq x \leq b, \ a \leq y \leq \beta).
    """)

    st.markdown("""
    Разобьем прямоугольник на малые прямоугольники (ячейки) $$g_i$$ и положим:
    """)
    st.latex(r"""
    I = \int_a^b \int_a^\beta f(x, y) dxdy \approx \sum_i s_i f(\bar{x}_i, \bar{y}_i), \tag{31}
    """)
    st.markdown("""
    где $$s_i$$ — площадь $$i$$-го прямоугольника, $$\\bar{x}_i, \\bar{y}_i$$ — координаты его центра.
    """)

    st.markdown("""
    Оценим погрешность этой формулы. Для каждого малого прямоугольника $$g_i$$ имеем:
    """)
    st.latex(r"""
    f(x, y) = f_i + \xi f'_{xi} + \eta f'_{yi} + \frac{1}{2} \xi^2 f''_{xxi} + \xi \eta f''_{xyi} + \frac{1}{2} \eta^2 
    f''_{yyi} + \dots ,
    """)
    st.markdown("""
    где $$\\xi = x - \\bar{x}_i$$, $$\\eta = y - \\bar{y}_i$$, а производные вычисляются в центрах ячеек.
    """)

    st.markdown("""
    Тогда главный член погрешности равен:
    """)

    st.latex(r"""
        R = \sum_i \left\{\iint_{g_i} \left[f_i + \xi f'_{xi} + \eta f'_{yi} + \frac{1}{2} \xi^2 f''_{xxi} + \xi \eta 
        f''_{xyi} + \frac{1}{2} \eta^2 f''_{yyi}\right] d\eta d\xi - s_i f_i\right\} =
        """)

    st.latex(r"""
        = \sum_i \iint_{g_i} \left[\frac{1}{2} \xi^2 f''_{xxi} + \frac{1}{2} \eta^2 f''_{yyi}\right] d\xi d\eta ,
        """)

    st.markdown("""
        так как интегралы от нечетных степеней $$\\xi, \\eta$$ дадут нулевой вклад.  
        Пусть отрезок $$[a, b]$$ разбит на $$N$$ равных частей, а отрезок $$[\\alpha, \\beta]$$ — на $$M$$ равных 
        частей. Тогда:
        """)
    st.latex(r"""
        \frac{1}{2} \iint_{g_i} \xi^2 f''_{xxi} d\xi d\eta = \frac{1}{2} f''_{xxi}  \frac{\beta - 
        \alpha}{M} \cdot \left(\frac{b - a}{N}\right)^3 \cdot \frac{1}{12};
        """)
    st.latex(r"""
        \frac{1}{2} \iint_{g_i} \eta^2 f''_{yyi} d\xi d\eta = \frac{1}{2} f''_{yyi} \frac{b - a}{N} \cdot 
        \left(\frac{\beta - \alpha}{M}\right)^3 \cdot \frac{1}{12}.
        """)

    st.markdown("""
        Следовательно:
        """)
    st.latex(r"""
        R = \frac{1}{24} \left[\left(\frac{b - a}{N}\right)^2 \iint_G f''_{xx} dxdy + \left(\frac{\beta - \alpha}{M}
        \right)^2 \iint_G f''_{yy} dxdy \right].
        """)

    st.markdown("""
        Т. е. формула (31) имеет второй порядок точности. К формуле (31) можно применить метод Рунге для повышения 
        точности, при этом надо лишь соблюдать постоянное соотношение $$N / M$$ для различных сеток узлов.

        Формулу (31) можно обобщить на область $$G$$ произвольной формы, с произвольным разбиением на ячейки, если при 
        этом под $$s_i$$ понимать площадь ячеек, а под $$\\bar{x}_i, \\bar{y}_i$$ — центры тяжести этих ячеек:
        """)
    st.latex(r"""
        s_i = \iint_{g_i} dxdy; \quad \bar{x}_i = \frac{1}{s_i} \iint_{g_i} x dxdy; \quad \bar{y}_i = \frac{1}{s_i} 
        \iint_{g_i} y dxdy. \tag{32}
        """)

    st.markdown("""
        Очевидно, что практически такой способ годится лишь для ячеек, ограниченных простыми линиями 
        (чаще всего ломаной). 
        Если граница области $$G$$ имеет сложную форму, то формулу (31) можно применить следующим образом.

        Наложим на область $$G$$ прямоугольную сетку. Те ячейки, которые лежат целиком внутри области, назовем 
        внутренними, все остальные будут граничными. 
        Площади внутренних ячеек и $$\\bar{x}_i, \\bar{y}_i$$ для них вычисляются просто, площади граничных ячеек и 
        $$\\bar{x}_i, \\bar{y}_i$$ для них можно вычислить по (32), если криволинейную границу заменить хордой.
        """)


# 4.2.6.2. Последовательное интегрирование
def multiple_integrals_sequential_integration():
    st.markdown("### 4.2.6.2. Последовательное интегрирование")
    st.markdown("""
    Снова вернёмся к интегралу по прямоугольной области и перепишем его в виде:
    """)
    st.latex(r"""
    I = \int_{\alpha}^{\beta}\int_{a}^{b} f(x, y) \, dx \, dy = \int_{\alpha}^{\beta} F(y) \, dy,
    """)
    st.markdown("""
    где
    """)
    st.latex(r"""
    F(y) = \int_{a}^{b} f(x, y) \, dx.
    """)
    st.markdown("""
    Каждый из однократных интегралов можно вычислить по каким-либо квадратурным формулам. Пусть, например:
    """)
    st.latex(r"""
    F(y_j) = \sum_i W_i f(x_i, y_j) = \sum_i W_i f_{ij},
    """)
    st.markdown("""
    а
    """)
    st.latex(r"""
    I = \sum_j \overline{W}_j F(y_j) = \sum_j \overline{W}_j f_{j}.
    """)
    st.markdown("""
    тогда
    """)
    st.latex(r"""
    I = \sum_{i, j} W_i \overline{W}_j f_{ij} = \sum_{i, j} W_{ij} f_{ij}, \quad W_{ij} = W_i \overline{W}_j.
    """)
    st.markdown("""
    Такие квадратурные формулы называются прямым произведением одномерных квадратурных формул. Так, если по каждому 
    направлению сетка узлов равномерная (с шагами $$h_x, h_y$$) и выбрана формула трапеций, то веса будут равны:
    """)
    st.latex(r"""
    W_{ij} = h_x h_y 
    \begin{cases} 
    1, & \text{внутренние узлы,} \\
    1/2, & \text{граничные узлы,} \\
    1/4, & \text{угловые узлы.}
    \end{cases}
    """)
    st.markdown("""
    Для дважды дифференцируемых функций эта формула будет иметь второй порядок точности, к ней можно применять метод 
    Рунге.
    
    По разным направлениям можно использовать квадратурные формулы разных порядков точности $$p, q$$. Тогда главный член
     погрешности будет иметь вид:
    """)
    st.latex(r"""
    R = O(h_x^p + h_y^q);
    """)
    st.markdown("""
    при применении метода Рунге отношение $$h_x^p / h_y^q$$ должно выдерживаться постоянным.

    Для области $$G$$ произвольной формы метод последовательного интегрирования применяется следующим образом. Пусть 
    граница области $$G$$ имеет вид:
    """)

    st.latex(r"""
        y = g(x) \ \text{и} \ \alpha = \min_x g(x) = g(x_0); \quad \beta = \max_x g(x) = g(x_1).
        """)

    st.markdown("""
        Запишем интеграл в виде:
        """)
    st.latex(r"""
        I = \int_{\alpha}^{\beta} F(y) \, dy, \tag{33}
        """)
    st.markdown("""
        где
        """)
    st.latex(r"""
        F(y) = \int_{\varphi_1(y)}^{\varphi_2(y)} f(x, y) \, dx.
        """)

    st.markdown("""
        Проведем через область $$G$$ хорды $$y = y_i$$, $$i = 1, 2, \\dots, M + 1$$, параллельные $$OX$$.

        Сначала вычислим по какой-либо формуле $$F(y)$$ на каждой хорде, а затем используем узлы $$y_i$$ для вычисления
        интеграла $$(33)$$. 
        Отметим, что для гладкой кривой при $$y \\to \\alpha $$: $$\\  F(y) \\sim \\sqrt{y - \\alpha},$$ a при $$y \\to 
        \\beta$$: $$\\  F(y) \\sim \\sqrt {\\beta - y}$$.
        """)

    st.markdown("""
        В самом деле, в окрестности точки $$(x_0, \\alpha)$$ для границы $$y = g(x)$$ имеем:
        """)
    st.latex(r"""
        y = \alpha + \frac{1}{2}g''(x - x_0)^2,
        """)
    st.markdown("""
        откуда:
        """)
    st.latex(r"""
        |x - x_0| = \sqrt{2 (y - \alpha)/g''}.
        """)

    st.markdown("""
        Поэтому по $$y$$ нецелесообразно непосредственно применять квадратурные формулы высокого порядка. Можно из 
        $$F(y)$$ выделить множитель:
        """)
    st.latex(r"""
        \rho = \sqrt{(\beta - y)(y - \alpha)},
        """)
    st.markdown("""
        и использовать его при построении квадратурных формул. Ему соответствуют ортогональные многочлены Чебышева 2-го 
        рода. 
        Поэтому если использовать формулы Гаусса–Кристоффеля, то ординаты $$y_i$$ надо выбирать в соответствии с нулями 
        этих многочленов.
        """)

    st.markdown("""
        **Замечание 1.** Для кратных интегралов применяется также подход, при котором узлы и веса формул для 
        интегрирования 
        (кубаторных формул) подбираются так, чтобы сделать формулы точными для многочленов определённой степени. 
        Эта задача 
        решена для областей простой формы (круг, квадрат, сфера).
        """)

    st.markdown("""
    **Замечание 2.** Для кратных интегралов высокой размерности $$n \\ (n \\geq 3)$$ широко используется метод 
    статистических испытаний (метод Монте-Карло).
    """)

