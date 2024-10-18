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


# 6.2. Нормы векторов и матриц
def vector_matrix_norm():
    st.markdown(r"""
                ### 6.2. Нормы векторов и матриц

                Пусть $$ \bar{x} $$ означает вектор-столбец 
    """)

    st.latex(r"""
             \overline{x} = \begin{pmatrix} x_1 \\ \vdots \\ x_n \end{pmatrix}
    """)

    st.markdown(r"""
                некоторого линейного пространства $$ R_{n} $$. Поставим в соответствие каждому вектору $$ \bar{x} $$ некоторое число $$ || \bar{x} || $$ - норму вектора $$ \bar{x} $$. Норма вектора обладает обычными для этого понятия свойствами: 
    """)

    st.latex(r"""
             \|\alpha \overline{x}\| = |\alpha| \|\overline{x}\|, \quad \alpha - число;
    """)

    st.latex(r"""
              \|\overline{x}\| > 0, \quad \text{если } \overline{x} \text{ - нулевой вектор, } \quad \|\overline{x}\| = 0;
    """)

    st.latex(r"""
             \|\overline{x} + \overline{y}\| \leq \|\overline{x}\| + \|\overline{y}\|, \quad \text{для любых } \overline{x}, \overline{y}.
    """)

    st.markdown(r"""
        Теперь мы можем определить норму матрицы $$ A $$, подчинённую заданной векторной норме:
    """)

    st.latex(r"""
        \|A\| = \max_{\overline{x} \neq 0} \frac{\|A \overline{x}\|}{\|\overline{x}\|}. \quad \quad (6)
    """)

    st.markdown(r"""
        Покажем, что отсюда следуют следующие свойства $$ \|A\| $$:
    """)

    st.latex(r"""
        ||\alpha A|| = |\alpha| \cdot ||A||;
    """)

    st.latex(r"""
        \|A\| > 0 \quad \text{при} \quad A \neq 0;

    """)

    st.latex(r"""
        \|A + B\| \leq \|A\| + \|B\| \quad \forall \quad n \times n \text{ матриц } A, B.
    """)

    st.markdown(r"""
        Первые два свойства очевидны, докажем последнее:
    """)

    st.latex(r"""
        ||(A + B)\overline{x}|| = \|A\overline{x} + B\overline{x}\| \leq ||A\overline{x}|| + ||B\overline{x}||.
    """)

    st.markdown("""
        Согласно (6),
    """)

    st.latex(r"""
        \|A \overline{x}\| \leq \|A\| \|\overline{x}\|, \|B \overline{x}\| \leq \|B\| \|\overline{x}\|.
    """)

    st.markdown(r"""
        Отсюда
    """)

    st.latex(r"""
        ||(A + B)\overline{x}|| \leq (\|A\| +\|B\|)\|\overline{x}\|.
    """)

    st.markdown(r"""
        Для произведения двух матриц $$ A $$, $$ B $$ имеем:
    """)

    st.latex(r"""
        \|AB \overline{x}\| = \|A(B \overline{x})\| \leq \|A\| \cdot \|B\| \cdot \|\overline{x}\|,
    """)

    st.markdown(r"""
        Потому
    """)

    st.latex(r"""
        \frac{\|AB \overline{x}\|}{\|\overline{x}\|} \leq \|A\| \cdot \|B\|
    """)

    st.markdown(r"""
        для любого $$ \overline{x} $$ и, следовательно, 
    """)

    st.latex(r"""
        \|AB\| \leq \|A\| \cdot \|B\|
    """)

    st.markdown(r"""
        Рассмотрим некоторые нормы векторов и матриц. Определим в пространстве $$ R_{n} $$ скалярное произведение векторов $$ \overline{x}, \overline{y} $$ по формуле: $$ (\overline{x}, \overline{y}) = \sum_{i=1}^{n} x_i y_i^* $$ (что равно $$ x^` $$ $$ y^* $$, если рассматривать $$ \overline{x}, \overline{y} $$ как матрицы-столбцы) и введём евклидову длину (евклидова норма) вектора $$ \overline{x} $$:
    """)

    st.latex(r"""
        ||\overline{x}|| = \sqrt{|x_1|^2 + ... + |x_n|^2} = \sqrt{(\overline{x}, \overline{x})}. \quad \quad (7)
    """)

    st.markdown(r"""
        Покажем, что так определённая $$ \|\overline{x}\| $$ обладает перечисленными выше свойствами. Первые два свойства следуют непосредственно из определения, последняя вытекает из цепочки неравенств
    """)

    st.latex(r"""
        ||\overline{x} + \overline{y}||^2 = (\overline{x}^` + \overline{y}^`)(\overline{x}^* + \overline{y}^*) = ||\overline{x}||^2 + ||\overline{y}||^2 + (\overline{x}^`\overline{y}^*+\overline{y}^`\overline{x}^*),
    """)

    st.markdown(r"""
        с другой стороны, при любом вещественном $$ \alpha $$ имеем:
    """)

    st.latex(r"""
        0 \leq (\alpha \overline{x} + \overline{y}, \alpha \overline{x} + \overline{y}) = \alpha^2 \|\overline{x}\|^2 + 2\alpha(\overline{x}, \overline{y}) + \|\overline{y}\|^2.
    """)

    st.markdown("""
        Потому
    """)

    st.latex(r"""
        [(\overline{x}, \overline{y}) + (\overline{x}, \overline{y})]^2 \leq 4\|\overline{x}\|^2 \cdot \|\overline{y}\|^2
    """)

    st.markdown("""
        и, следовательно,
    """)

    st.latex(r"""
        |\overline{x}^`\overline{y}^* + \overline{y}^`\overline{x}^*| \leq 2\|\overline{x}\| \cdot \|\overline{y\|}.
    """)

    st.markdown("""
        Таким образом,
    """)

    st.latex(r"""
        \|\overline{x} + \overline{y}\|^2 \leq \|\overline{x}\|^2 + 2\|\overline{x}\| \|\overline{y}\| + \|\overline{y}\|^2 = (\|\overline{x}\| + \|\overline{y}\|)^2,
    """)

    st.markdown("""
        что и требовалось доказать.
    """)

    st.markdown(r"""
        Норма матрицы А, подчиненная евклидовой норме вектора, указана в разделе 6.3. Часто используют так называемую сферическую норму $$ \|A\| $$
    """)

    st.latex(r"""
        ||A|| = \left( \sum_{i,j} |a_{i,j}|^2 \right)^{1/2}.
    """)

    st.markdown(r"""
        Она является лишь согласованной с евклидовой нормой вектора. В самом деле, рассмотрим вектор $$ \overline{y} = A \overline{x} $$, используя неравенство Коши, получим, что
    """)

    st.latex(r"""
            |y_i|^2 \leq \sum_{m=1}^{n} |a_{im}|^2 \cdot \sum_{m=1}^{n} |x_m|^2.
        """)

    st.markdown(r"""
            Отсюда следует, что
        """)

    st.latex(r"""
            \sum_{i=1}^{n} |y_i|^2 \leq \sum_{m=1}^{n} |x_m|^2 \cdot \sum_{i,m} |a_{im}|^2
        """)

    st.markdown(r"""
            и
        """)

    st.latex(r"""
            \frac{||A \overline{x}||}{||\overline{x}||} = \left( \sum_{i,m}|a_{im}|^2 \right)^{1/2}
        """)

    st.markdown(r"""
            что и означает согласованность матричной и векторной норм.
        """)

    st.markdown(r"""
            Наряду с евклидовой нормой вектора часто рассматривают две другие нормы:
        """)

    st.latex(r"""
            ||\overline{x}||_l = \sum_{i=1}^{n} |x_i|; \quad ||\overline{x}||_m = \max_i |x_i|.
        """)

    st.markdown(r"""
            Эти нормы векторов приводят, согласно (6), к следующим нормам матриц:
        """)

    st.markdown(r"""
            — для нормы $$ l $$
        """)

    st.latex(r"""
            ||A \overline{x}||_l = \sum_{i=1}^{n} | \sum_{j=1}^{n} a_{ij} x_j| \leq \sum_{i=1}^{n} \sum_{j=1}^{n} |a_{ij}| \cdot |x_j| = \sum_{j=1}^{n} |x_j| \sum_{i=1}^{n} |a_{ij}| \leq \max_j \sum_{i=1}^{n} |a_{ij}| \cdot ||\overline{x}||_l,
        """)

    st.markdown(r"""
            следовательно,
        """)

    st.latex(r"""
            ||A||_l = \max_{1 \leq j \leq n} \sum_{i=1}^{n} |a_{ij}|;
        """)

    st.markdown(r"""
            — для нормы $$ m $$
        """)

    st.latex(r"""
            ||A \overline{x}||_m = \max_{1 \leq i \leq n} |\sum_{j=1}^{n} a_{ij} x_j| \leq \max_{i} \sum_{j=1}^{n} |a_{ij}| \cdot |x_j| \leq \max_{i} \sum_{j=1}^{n} |a_{ij}| \cdot ||\overline{x}||_m,
        """)

    st.markdown(r"""
            следовательно,
        """)

    st.latex(r"""
            ||A||_m = \max_{1 \leq i \leq n} \sum_{j=1}^{n} |a_{ij}|.
        """)
