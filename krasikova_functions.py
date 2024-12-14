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
