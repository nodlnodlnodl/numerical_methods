import streamlit as st


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


def conditionality_liner_system():
    st.markdown(r"""
        ### 6.3. Обусловленность линейной системы

        Для системы уравнений (1), как известно, при $det A \neq 0$ существует удинственное решение. Однако на практике важно знать не только факт существования решения, но и насколько оно чувствительно к изменениям элементов матрицы правых частей.

        Пусть матрица А известна точно, а вектор $\overline{b}$ - с некоторой неопределенностью $\overline{\delta b}$. Тогда решением системы $A \overline{x} = \overline{b} + \overline{\delta b}$ будет вектор $\overline{x} = \overline{x_0} + \overline{\delta x}$, где $\overline{x_0}$ - решение системы $A \overline{x} = \overline{b}$. Очевидно, что

        $$
        \overline{\delta x} = A^{-1}; \quad ||A^{-1}|| \leq ||A^{-1}|| \cdot ||\overline{\delta b}||. 
        $$

        Так как $\overline{b} = A \overline{x_0}$, то $||\overline{b}|| \leq ||A|| \cdot ||\overline{x_0}||$; перемножая два последних неравенства, находим 

        $$
        ||\overline{\delta x}|| \cdot ||\overline{b}|| \leq ||A|| \cdot ||A^{-1}|| \cdot ||\overline{x_0}|| \cdot ||\overline{x_0}|| \cdot ||\overline{\delta b}||,
        $$

        откуда при $\overline{b} \neq 0$

        $$
        \|\overline{\delta x}\| /\left\|\bar{x}_0\right\| \leq\|A\| \cdot\left\|A^{-1}\right\| \cdot\|\overline{\delta b}\| /\|\bar{b}\| .
        $$

        Определим теперь число обусловленности любой невырожденной матрицы по формуле

        $$
        \text { cond } A=\|A\| \cdot\left\|A^{-1}\right\| \text {,   (8)}
        $$

        оно, очевидно, зависит от используемой нормы матрицы. В зависимости от величины $\text { cond } A$ система называется хорошо обусловненной $\left(\text { cond } A \leq 10^2\right)$,либо плохо обусловленной $\left(\text { cond } A \geq 10^3\right)$.
        Пусть теперь $\overline{b}$ известен точно, а матрица А имеет неопределенность $\delta A$, тогда 

        $$
        \begin{aligned}
        & \bar{x}_0+\overline{\delta x}=(A+\delta A)^{-1} \bar{b}; \\
        & \overline{\delta x}=\left[(A+\delta A)^{-1}-A^{-1}\right] \bar{b}.
        \end{aligned}
        $$

        Используя тождество $B^{-1}-A^{-1}=A^{-1}(A-B) B^{-1}$, в котором $B=A+\delta A$, получим

        $$
        \overline{\delta x}=-A^{-1} \delta A(A+\delta A)^{-1} \bar{b}=-A^{-1} \delta A\left(\bar{x}_0+\overline{\delta x}\right)
        $$

        Отсюда находим

        $$
        \begin{aligned}
        & \|\overline{\delta x}\| \leq\left\|A^{-1}\right\| \cdot\|\delta A\| \cdot\left\|\bar{x}_0+\overline{\delta x}\right\| ; \\
        & \ \frac{\|\delta x\|}{\left\|\bar{x}_0+\overline{\delta x}\right\|} \leq \operatorname{cond} A \cdot \frac{\|\delta A\|}{\|A\|}.
        \end{aligned}
        $$

        Таким образом, и в случае неопределенности в элементах матрицы $A$ 
        число обусловленности $\mathrm{cond}\, A$ связывает отклонение нормы решения 
        с отклонением нормы матрицы.

        Рассмотрим теперь теорему, на основании которой можно вычислить число обусловленности матрицы.

        **Теорема.** Для любой вещественной $(n \times n)$-матрицы $A$ существуют 
        две вещественные ортогональные $(n \times n)$-матрицы $U, V$, такие, что $U'A V = D$
        — диагональная матрица. При этом можно выбрать $U, V$ так, чтобы диагональные элементы $D$ имели вид

        $$ 
        \mu_1 \geq \mu_2 \geq \dots \geq \mu_r > \mu_{r+1} = \dots = \mu_n = 0, 
        $$

        где $r \neq 0$ — ранг матрицы $A$. Если $A$ невырождена, то 

        $$
        \mu_1 \geq \mu_2 \geq \dots \geq \mu_n > 0. 
        $$

        **Доказательство.** Составим матрицу $B = A \cdot A'$, очевидно, $B' = B$, 
        т. е. $B$ — симметричная матрица. Для любого вектора $\overline{x}$ имеем, 
        используя евклидову норму,

        $$
        \overline{x}' B \overline{x} = \overline{x}' AA' \overline{x} = ((A'\overline{x})', (A' \overline{x})) = \|A \overline{x}\|^2 \geq 0, 
        $$

        таким образом, $B$ — положительно полуопределенная матрица. Обозначим n 
        собственных чисел её через $\mu_1^2, \dots, \mu_n^2$ (они действительны и 
        неотрицательны)

        $$
        \det(B - \mu_i^2 E) = 0, 
        $$

        причём 

        $$
        \mu_1 \geq \mu_2 \geq \dots \geq \mu_n \geq 0. 
        $$

        Пусть $\mu_r > 0$ и либо $r = n$, либо $\mu_{r+1} = \dots = \mu_n = 0$. Из теории 
        вещественных симметричных матриц известно, что можно найти ортогональную 
        матрицу $U$, такую, что

        $$
        U' B U = D^2, 
        $$

        где $D$ — диагональная матрица; $d_{ii} = \mu_i^2 \geq 0$.

        Определим $(n \times n)$ - матрицу $F$, такую, что $F = U'A$. Тогда 

        $$
        FF' = (U' A) \cdot (A' U) = U' B U = D^2, 
        $$

        и, следовательно, для элементов $f_{ij}$ матрицы $F$ имеем $\sum_{j=1}^n f_{ij}^2 = \mu_i^2, \quad \text{т.е. } ||\overline{f_i'}||^2 = \mu_i$, где $\overline{f_i} = (f_{i1}, f_{i2}, \dots, f_{in})$, $\overline{f}_i$ — вектор-столбец. Кроме того, из $FF' = D^2$ следует, что различные строки матрицы $F$ ортогональны друг другу.

        При $\mu_1 \geq \mu_2 \geq \dots \geq \mu_r > 0$ первые $r$ строк матрицы $F$ не нулевые; если $r < n$, то строки $\overline{f}_{r+1}, \dots, \overline{f}_n$ — нулевые. Построим ортонормированную систему вектор-строк $\overline{g}_1, \dots, \overline{g}_n$ следующим образом: для $i = 1, 2, \dots, r$

        $$
        \overline{g}_i = \frac{1}{u_i} \cdot \overline{f}_i;
        $$

        для $i = r+1, \dots, n$ вектор-строки выберем так, чтобы $\overline{g}_1, \dots, \overline{g}_n$ были взаимно ортогональны (это, как известно, всегда можно сделать). Образуем матрицу $V' = \begin{bmatrix} \overline{g}_1 & \dots & \overline{g}_n \end{bmatrix}$, очевидно, что по построению $V'V = E$ и $V'$ — ортогональная матрица. Кроме того, 

        $$
        \overline{f}_i = \mu_i \overline{g}_i, \quad \text{т.е. }
        F = D V' \quad \text{или} \quad U'A = D V'. 
        $$

        Таким образом, $U'AV = D$, что и требовалось доказать.

        Числа $\mu_1, \dots, \mu_n$ называются сингулярными числами матрицы $A$. Если $A$ — невырожденная матрица, то для $A^{-1}$ приведение её к диагональному виду выполняют матрицы $\widetilde{U}'\ = V, \widetilde{V} = U$. В самом деле, обозначим через $P = V' A^{-1} U$, тогда $PD = (V' A^{-1} U) \cdot (U' A V) = E$, т.е. $P = D^{-1}$. Таким образом, сингулярные числа матрицы $A^{-1}$ будут 

        $$
        1 / \mu_1, \dots, 1 / \mu_n. 
        $$

        Сингулярные числа оказываются тесно связанными со спектральной нормой матрицы $A$, подчиненной евклидовой норме вектора. Чтобы установить эту связь, покажем сначала, что ортогональная матрица не меняет норму вектора. Пусть $\overline{y} = U \overline{x}$, где $U$ — ортогональная матрица. Тогда

        $$
        \|\overline{y}\|^2 = \sum_{i=1}^n \left( \sum_{j=1}^n U_{ij} x_j \right)^2 = 
        \sum_{i=1}^n \sum_{j, k}U_{ij} x_j U_{ik} x_k = 
        \sum_{j, k} x_j x_k \sum_{i=1}^n U_{ij} U_{ik}. 
        $$

        Поскольку $U' U = E$, то $\sum_{i=1}^n U_{ji} U_{ik} = \sum_{i=1}^n U_{ij} U_{ik} = \delta_{jk}$, следовательно,

        $$
        \|\overline{y}\|^2 = \|\overline{x}\|^2 \text{, т.е. } (\|U\| = 1).
        $$

        Пусть $U, V$ — ортогональные матрицы, приводящие невырожденную матрицу $A$ к диагональному виду. Тогда

        $$ 
        \|A\| = \max_{V \cdot\overline{x}} \frac{\|AV \cdot \overline{x}\|}{\|V \cdot\overline{x}\|} = 
        \max_{\overline{x}} \frac{\|U' A V \overline{x}\|}{\|\overline{x}\|} = 
        \max_{\overline{x}} \frac{\|D \overline{x}\|}{\|\overline{x}\|}. 
        $$

        Отсюда очевидно, что

        $$
        \|A\| = \max_{x_1, x_2, \dots, x_n} \sqrt{\frac {\mu_1^2 x_1^2 + \mu_2^2 x_2^2 + \dots + \mu_n^2 x_n^2}{x_1^2 + \dots + x_n^2}} \leq \mu_1, 
        $$
        точное равенство достигается при $x_2 = \dots = x_n = 0$, $x_1 = 1$.

        Таким образом, $\|A\| = \mu_1$, аналогично $\|A^{-1}\| = 1 / \mu_n$. Итак, мы получаем:

        $$
        \mathrm{cond}\, A = \frac{\mu_1}{\mu_n}, 
        $$

        где $\mu_1,  \mu_2$ — собственные числа матрицы $A A'$. Поэтому, вопреки обычному заблуждению о плохой обусловленности систем, у которых $\|det A \| \ll 1$, на самом деле критерием обусловленности является отношение наибольшего и наименьшего сингулярных чисел. Отметим, что для симметричных матриц сингулярные числа совпадают с модулями их собственных чисел $\lambda_i$.

        Таким образом, при большом разбросе сингулярных чисел матрицы $A$ в решении системы $A \overline{x} = \overline{b}$ будет проявляться сильное влияние неопределенности в коэффициентах и правой части (в том числе и ошибок округления!).

        **Пример:**

        $$
        A = \begin{pmatrix}
        1 & 0.99 \\
        0.99 & 0.98
        \end{pmatrix}; 
        $$

        $$
        \lambda_1 = 1.98005, \quad \lambda_2 \approx 0.00005 \quad \text{— собственные числа и, следовательно,} 
        $$

        $$
        \mathrm{cond}\, A = \frac{\lambda_1}{\lambda_2} = 3.96 \cdot 10^4. 
        $$

        Если $\overline{b} = \begin{pmatrix} 1.99 \\ 1.97 \end{pmatrix}$, то система $A \overline{x} = \overline{b}$ имеет решение: $x_1 = x_2 = 1$.

        Если $\overline{b} = \begin{pmatrix} 1.989903 \\ 1.970106 \end{pmatrix}$, то $x_1 = 3.0000, \quad x_2 = -1.0203.$

        Таким образом, 

        $$
        \overline{\delta b} = \begin{pmatrix} -0.000097 \\ 0.000106 \end{pmatrix}; \quad \overline{\delta x} = \begin{pmatrix} 2.0000 \\ -2.0203 \end{pmatrix}; \quad \|\overline{x_0}\| = \sqrt{2}; \quad \|\overline{\delta x}\| \cong 2\sqrt{2}; 
        $$

        $$
            \|\overline{\delta b}\| \cong 10^{-4}\sqrt{2}; \quad
            \|\overline{b}\| \cong 2\sqrt{2}; \quad \frac{\|\overline{\delta x}\|}{\|\overline{x_0}\|} \leq 4 \cdot 10^4 \frac{\|\overline{\delta b}\|}{\|\overline{b}\|}. 
        $$

        т. е. мы нашли почти наихудший случай (так как $cond \ A$ также $\approx 4 \cdot 10^{4})$.
    """)
