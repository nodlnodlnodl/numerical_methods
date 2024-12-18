import streamlit as st

import streamlit as st

def mnk():
    st.markdown("### 3.4. Среднеквадратичное приближение (метод наименьших квадратов)")

    st.markdown(
        """
        До сих пор, рассматривая приближение функций, мы для их построения использовали критерий точного 
        прохождения интерполирующей функции через узловые точки. Точность такой аппроксимации, как правило, 
        гарантирована в небольшом интервале вблизи середины множества используемых узлов. Кроме того, нередки ситуации, 
        когда опорные точки сильно искажены шумом (например, экспериментально измеренные значения некоторой величины 
        со случайной ошибкой).
        """
    )

    st.markdown(
        """
        Таким образом, если ставится задача получения достаточно точного приближения для функции 
        $ y(x) $ на заданном отрезке $ a \leq x \leq b $ или же значения $ y $ в узловых точках 
        $ x_1, x_2, \dots, x_N $ сильно зашумлены, то широко используются методы, основанные на приближении в среднем. 
        При этом наиболее распространённым методом является метод, основанный на минимизации среднеквадратичного отклонения 
        (метод **наименьших квадратов** – **МНК**).
        """
    )

def mnogochlen_approksimashion():
    st.markdown("### 3.4.1. Дискретное задание функции, многочленная аппроксимация")

    st.markdown(
        """
        Пусть измеряется некоторая величина $ y $ и сделано $ N $ измерений, в ходе которых получены значения 
        $ y_1, y_2, \dots, y_N $. Предположим, что ошибки измерений
        """
    )
    st.markdown(r"$\varepsilon_i = y - y_i$")

    st.markdown(
        """
        имеют случайный характер, независимы и:
        """
    )
    st.markdown(r"$\sum_{i=1}^N \varepsilon_i = 0.$")

    st.markdown(
        """
        В этом случае в качестве оценки $ \overline{y} $ величины $ y $ выбирается такое значение, которое 
        минимизирует сумму квадратов ошибок:
        """
    )
    st.markdown(r"$f(\overline{y}) = \sum_{i=1}^N \varepsilon_i^2 = \sum_{i=1}^N (\overline{y} - y_i)^2 = \min.$")

    st.markdown(
        """
        Определим это значение $ \overline{y} $. Из условия минимума $ f(y) $ имеем:
        """
    )
    st.markdown(r"$\frac{df}{dy} \Big|_{y = \overline{y}} = 2 \sum_{i=1}^N (\overline{y} - y_i) = 0.$")

    st.markdown("Следовательно:")
    st.markdown(r"$\overline{y} = \frac{1}{N} \sum_{i=1}^N y_i.$")

    st.markdown(
        """
        При этом $ f'' = 2N > 0 $, т. е. $ y = \overline{y} $ – точка минимума для $ f(y) $.

        Таким образом, мы получили хорошо известный результат, когда в качестве оценки измеряемой величины берётся 
        среднее арифметическое по нескольким измерениям.

        На практике обычно измеряется отклик некоторой системы на изменение внешних параметров; в простейшем случае 
        имеют дело с одним входным параметром $ x $, т. е. при измерении получают значения функции $ y(x) $. 
        Одним из наиболее общих случаев применения МНК является приближение многочленом $ N $ наблюдений $ (x_i, y_i) $, 
        при этом
        """
    )
    st.markdown(r"$y(x) = a_0 + a_1 x + \dots + a_M x^M$")

    st.markdown(
        """
        имеет степень $ M < N $, коэффициенты его заранее неизвестны. Коэффициенты $ a_0, a_1, \dots, a_M $, 
        следуя МНК, выбираются таким образом, чтобы доставить минимум функции:
        """
    )
    st.markdown(r"$f(a_0, a_1, \dots, a_M) = \sum_{i=1}^N \rho_i \left[ y_i - y(x_i) \right]^2. \quad (17)$")

    st.markdown(
        """
        Здесь $ \rho_i $ – некоторые положительные веса, приписываемые каждому измерению. В случае равновесных 
        измерений обычно полагают $ \rho_i = 1 $, $ i = 1, \dots, N $. Минимум функции $ f $ достигается 
        при таких значениях $ a_0, a_1, \dots, a_M $, которые удовлетворяют системе уравнений:
        """
    )
    st.markdown(
        r"$\frac{\partial f}{\partial a_k} = -2 \sum_{i=1}^N \rho_i \left[ y_i - y(x_i) \right] x_i^k = 0, \quad k = 0, 1, \dots, M.$"
    )

    st.markdown("Эту систему можно представить в виде:")
    st.markdown(
        r"$\sum_{j=0}^M a_j \left( \sum_{i=1}^N \rho_i x_i^{k+j} \right) = \sum_{i=1}^N \rho_i y_i x_i^k. \quad (18)$"
    )

    st.markdown(
        """
        Покажем, что система (18) относительно $ a_0, a_1, \dots, a_M $ всегда имеет решение; это будет заведомо так, 
        если определитель $ \Delta $ этой системы не равен нулю.

        Пусть $ \Delta = 0 $. Тогда однородная система
        """
    )
    st.markdown(r"$\sum_{j=0}^M a_j \left( \sum_{i=1}^N \rho_i x_i^{k+j} \right) = 0, \quad k = 0, 1, \dots, M$")

    st.markdown(
        """
        имеет ненулевое решение. Умножая $ k $-е уравнение на $ a_k $ и суммируя по всем $ k $, получим:
        """
    )
    st.markdown(
        r"$0 = \sum_{k=0}^M a_k \sum_{j=0}^M a_j \left( \sum_{i=1}^N \rho_i x_i^{k+j} \right) = \sum_{i=1}^N \rho_i \left( \sum_{j=0}^M a_j x_i^j \right)^2.$"
    )

    st.markdown(
        """
        Отсюда следует, что все $ y(x_i) = 0, \ i = 1, \dots, N $. Так как $ N > M $, то этого не может быть в силу 
        фундаментальной теоремы алгебры: многочлен степени $ M $ имеет не более $ M $ действительных корней. 
        Таким образом, для системы (18) $ \Delta \ne 0 $, и она всегда имеет решение.
        На практике систему (18) при $ M \geq 5 $ решают, как правило, неявно, вычисляя непосредственно суммы вида:
        """
    )
    st.markdown(r"$\sum_{i=1}^N \rho_i x_i^{k+j},$")

    st.markdown(
        """
        а некоторые их эквиваленты. Это приводит к понятию ортогональных многочленов на множестве 
        $ x_1, \dots, x_N $, которые рассмотрим позже.

        **Пример.** Дана таблица $ y(x) $ равновесных измерений:
        """
    )

    st.markdown(
        """
        | $x$ | 0 | 1 | 2 | 3 | 4 |
        |-----|---|---|---|---|---|
        | $y$ | 1 | 2 | 1 | 2 | 4 |

        Требуется приблизить $ y(x) $ прямой:
        """
    )
    st.markdown(r"$y = a_0 + a_1 x.$")

    st.markdown("Сосчитаем суммы:")
    st.latex(r"S_0 = \sum_{i=1}^5 x_i^0 = 5; \quad S_1 = \sum_{i=1}^5 x_i = 10;")
    st.latex(
        r"S_2 = \sum_{i=1}^5 x_i^2 = 30; \quad T_0 = \sum_{i=1}^5 y_i x_i^0 = 8; \quad T_1 = \sum_{i=1}^5 y_i x_i = 20.")

    st.markdown("Имеем систему:")
    st.latex(r"S_0 a_0 + S_1 a_1 = T_0;")
    st.latex(r"S_1 a_0 + S_2 a_1 = T_1 \implies a_0 = \frac{4}{5}, \quad a_1 = \frac{2}{5}.")

    st.markdown("Таким образом, искомая прямая есть:")
    st.latex(r"y(x) = \frac{2}{5} (x + 2).")


def linear_approcsimation():
    st.markdown("### 3.4.2. Непрерывное задание функции, линейная аппроксимация")

    st.markdown(
        r"Пусть на отрезке $a \leq x \leq b$ задана некоторая функция $y(x)$ и её надо аппроксимировать "
        r"с помощью линейной комбинации некоторых функций $f_k(x), \ k = 0, 1, \dots, M:$"
    )
    st.markdown(r"$$y_M = a_0 f_0 + a_1 f_1 + \dots + a_M f_M$$")

    st.markdown(
        r"Таким образом, чтобы $f(a_0, a_1, \dots, a_M) = \int_a^b \rho(x) [y(x) - y_M(x)]^2 dx$ "
        r"имела наименьшее значение, где $\rho(x)$ — весовая функция, "
        r"предполагается положительной, а система функций $f_0, f_1, \dots, f_M$ — линейно независимой."
    )

    st.markdown(r"Введём обозначение (скалярное произведение функций $q(x), \phi(x)$) как:")
    st.markdown(r"$$ (q, \phi) = \int_a^b \rho q \phi dx $$")

    st.markdown("Используя это обозначение, получим:")
    st.markdown(r"$$f(a_0, a_1, \dots, a_M) = (y, y) - 2 \sum_{k=0}^M a_k (y, f_k) + \sum_{k,m=0}^M a_k a_m (f_k, f_m)$$")

    st.markdown(
        r"Приравнивая к нулю производные от $f$ по $a_0, \dots, a_M$, получим систему уравнений:"
    )
    st.markdown(r"$$\sum_{m=0}^M (f_k, f_m) a_m = (y, f_k), \quad k = 0, 1, \dots, M.$$")

    st.markdown(
        r"Её определитель $\Delta = \det((f_k, f_m))$, называется определителем Грама функций $f_k$. "
        r"Для линейно независимых систем $\Delta \neq 0$, поэтому система всегда имеет решение."
    )

    st.markdown(
        r"Однако на практике непосредственно систему (22) при $M \geq 5$ решать сложно, так как "
        r"определитель Грама с ростом $M$ быстро убывает, что делает систему (22) малопригодной для вычислений."
    )

    st.markdown(
        r"Положение сильно упрощается, если $f_0, \dots, f_M$ образуют систему ортогональных функций."
    )

def ortogonal():
    st.markdown("### 3.4.3. Ортогональные функции")

    st.markdown("Рассмотрим теперь некоторые общие теоремы, поясняющие свойства ортогональных функций.")

    st.markdown(
        "**Определение 3.** Функции $f_0, f_1, ..., f_M$ называются линейно независимыми на отрезке $[a, b]$, если из равенства"
    )
    st.latex(r"a_0 f_0 + a_1 f_1 + \dots + a_M f_M = 0")
    st.markdown("на всём отрезке следует $a_0 = 0, a_1 = 0, ..., a_M = 0$.")

    st.markdown(
        "**Пример.** Функции $1, x, ..., x^M$ линейно независимы на любом отрезке $[a, b]$, так как многочлен"
    )
    st.latex(r"a_0 + a_1 x + \dots + a_M x^M")
    st.markdown("согласно основной теореме алгебры имеет не более $M$ корней.")

    st.markdown(
        "**Теорема 1.** Любая система непрерывных ортогональных на отрезке $[a, b]$ функций линейно независима."
    )

    st.markdown(
        "В самом деле, если $f_0, f_1, ..., f_M$ — такая система и $a_0 f_0 + \dots + a_M f_M \equiv 0$ при $a \leq x \leq b$,"
    )
    st.markdown("то для коэффициентов имеем")
    st.latex(r"a_j = \frac{1}{\lambda_j} \int_a^b \rho \cdot 0 \cdot f_j dx = 0.")

    st.markdown(
        "**Теорема 2.** Из любой системы линейно независимых функций $f_0, f_1, ..., f_M$ на отрезке $[a, b]$"
    )
    st.markdown("можно получить систему ортогональных функций на этом отрезке.")

    st.markdown("**Доказательство.** Рассмотрим следующий процесс (он называется процессом ортогонализации Шмидта):")

    st.latex(r"\lambda_0 = \int_a^b \rho(x) f_0^2 dx \quad (\rho(x) \geq 0 \text{ по условию})")

    st.markdown("тогда равенство $g_0(x) = \\frac{f_0(x)}{\sqrt{\lambda_0}}$ приводит нас к нормированной функции $g_0(x)$.")

    st.markdown("Пусть построены первые $j$ ортонормированных функций $g_i(x), i = 0, ..., j - 1.$")

    st.markdown("Положим")
    st.latex(r"F_j(x) = a_0 g_0 + \dots + a_{j-1} g_{j-1} + f_j(x).")

    st.markdown(
        "Функция $F_j$ отлична от нуля, так как $f_j$ линейно независимы, а каждая из $g_i$ — линейная комбинация $f_k, k \leq i$."
    )

    st.markdown("Определим $a_0, ..., a_{j-1}$ из условий ортогональности $F_j$ функциям $g_0, ..., g_{j-1}:$")
    st.latex(r"\int_a^b \rho F_j g_i dx = 0, \quad i = 0, 1, \dots, j - 1")

    st.markdown("Отсюда находим, подставляя выражение для $F_j$, что")
    st.latex(r"a_i + \int_a^b \rho f_j g_i dx = 0, \quad i = 0, 1, \dots, j - 1.")

    st.markdown("Следовательно, $F_j(x)$ определена, функцию $g_j$ определим так:")
    st.latex(r"g_j(x) = \frac{F_j(x)}{\sqrt{\lambda_j}}; \quad \lambda_j = \int_a^b \rho F_j^2 dx")

    st.markdown("Таким образом, шаг по индукции выполнен.")

    st.markdown("**Замечание.** На практике процесс ортогонализации Шмидта редко применяется для значений $M \geq 5$.")

    st.markdown(
        "Дело в том, что при численной ортогонализации потребности вычисления быстро приводят к тому, что "
        "очередная ортогональная функция становится малой."
    )

    st.markdown(
        "При нормировке ошибка только возрастает. Поэтому обычно пользуются аналитическими ортогональными системами функций."
    )

    st.markdown("**Теорема 3.** Коэффициенты Фурье $\{a_j\}$ функции $F(x)$ относительно ортонормированного семейства $\{g_j\}$")
    st.markdown("на отрезке $[a, b]$ удовлетворяют неравенству Бесселя:")
    st.latex(r"\sum_{j=0}^M a_j^2 \leq \int_a^b \rho F^2 dx.")

    st.markdown("**Доказательство.** Запишем очевидное неравенство:")
    st.latex(r"\int_a^b \rho \left(F - \sum_{i=0}^M a_i g_i(x)\right)^2 dx \geq 0")

    st.markdown("и раскроем скобки:")
    st.latex(
        r"0 \leq \int_a^b \rho F^2 dx - 2 \int_a^b \rho F \sum_{i=0}^M a_i g_i(x) dx + \int_a^b \rho \sum_{i=0}^M \sum_{j=0}^M a_i a_j g_i g_j dx ="
    )
    st.latex(r"= \int_a^b \rho F^2 dx - 2 \sum_{i=0}^M a_i^2 + \sum_{j=0}^M a_j^2.")

    st.markdown("Таким образом, теорема доказана.")

def orthogonal_polynomials():

    st.markdown("### 3.4.4. Ортогональные многочлены")

    st.markdown("""
    Важным подклассом ортогональных функций являются ортогональные на отрезке $[a, b]$ многочлены, где $k$-й многочлен имеет степень $k$ $(k = 0, 1, \dots)$.

    **Теорема 5.** $k$-й ортогональный на отрезке $[a, b]$ многочлен $P_k(x)$ имеет на $[a, b]$ $k$ действительных и различных корней.

    **Доказательство.** Предположим, что $P_k(x)$ имеет на $[a, b]$ $r < k$ действительных корней $x_1, x_2, \dots, x_r$. Образуем произведение:
    """)
    st.latex(r"\pi(x) = (x - x_1)(x - x_2) \dots (x - x_r).")

    st.markdown("""
    Тогда:
    """)
    st.latex(r"\int_a^b \rho(x) \pi(x) P_k(x) dx = 0, \quad (\rho(x) \geq 0),")

    st.markdown("""
    так как функция $\pi(x)$ может быть разложена по $P_0, P_1, \dots, P_r$, а многочлен $P_k(x)$ ортогонален им всем. Но такое равенство невозможно, так как подынтегральная функция знакопостоянна на $[a, b]$. Таким образом, многочлен $P_k(x)$ имеет $k$ корней.

    Докажем, что они не кратные. Пусть какой-либо корень, например $x_1$, имеет кратность $n > 1$. Составим многочлен:
    """)
    st.latex(
        r"\tilde{\pi}(x) = (x - x_2)(x - x_3) \dots (x - x_k) \begin{cases} 1, & n \ \text{четное}, \\ (x - x_1), & n \ \text{нечетное}. \end{cases}")

    st.markdown("""
    Тогда:
    """)
    st.latex(r"\int_a^b \rho(x) P_k(x) \tilde{\pi}(x) dx = 0.")

    st.markdown("""
    из условий ортогональности. С другой стороны, подынтегральная функция знакопостоянна, что и доказывает теорему.

    **Замечание.** Корни ортогонального многочлена не совпадают с концами отрезка. В самом деле, пусть $x_1, \dots, x_k$ – корни $P_k$ в порядке возрастания. Предположим, что $x_1 = a$. Тогда, если составить член:
    """)
    st.latex(r"\pi(x) = (x - x_2)(x - x_3) \dots (x - x_k),")

    st.markdown("""
    то будет нарушено условие ортогональности, так как:
    """)
    st.latex(r"\int_a^b \rho(x) P_k(x) \pi(x) dx \neq 0,")

    st.markdown("""
    поскольку $\rho(x) P_k(x) \pi(x) \geq 0 \ (x - a > 0)$. Аналогичное доказательство при $x_k = b$.

    **Теорема 6.** Ортогональные многочлены удовлетворяют трехчленному рекуррентному соотношению вида:
    """)
    st.latex(r"a_k P_{k+1}(x) + (b_k - x)P_k(x) + c_k P_{k-1}(x) = 0, \quad (k \geq 1). \ \ \ (25)")

    st.markdown("""
    **Доказательство.** Пусть:
    """)
    st.latex(r"P_j(x) = \alpha_j x^j + \dots, \ \alpha_j > 0.")

    st.markdown("""
    Определим в (25) коэффициенты $a_k$, $b_k$, $c_k$. Разность $a_k P_{k+1} - xP_k$ есть многочлен степени $k$ при условии, что:
    """)
    st.latex(r"a_k = \alpha_k / \alpha_{k+1}.")

    st.markdown("""
    Следовательно:
    """)
    st.latex(r"a_k P_{k+1} - x P_k = \gamma_k P_{k-1} + \gamma_{k-1} P_{k-2} + \dots + \gamma_0 P_0. \ \ \ (26)")

    st.markdown("""
    Умножим это равенство на $\rho(x) P_m$, $m \leq k$, и проинтегрируем по отрезку $[a, b]$:
    """)
    st.latex(r"\int_a^b \rho(x) \left(a_k P_{k+1} - x P_k\right) P_m dx = \gamma_m \lambda_m, \ \ \ (27)")

    st.markdown("""
    где:
    """)
    st.latex(r"\lambda_m = \int_a^b \rho(x) P_m^2 dx.")

    st.markdown("Для $m = 0, 1, \dots, k$ $P_{k+1}(x)$ ортогонален $P_m(x)$, следовательно, для этих $m$:")
    st.latex(r"a_k \int_a^b P_{k+1} P_m dx = 0.")

    st.markdown(
        "Для $m = 0, 1, \dots, k - 2$ произведение $x P_m$ есть многочлен степени, меньшей $k$, и, следовательно, для этих $m$:")
    st.latex(r"\int_a^b \rho x P_k P_m dx = 0.")

    st.markdown("""
    Таким образом, $\gamma_0 = \gamma_1 = \dots = \gamma_{k-2} = 0.$

    Для $m = k - 1$ имеем, учитывая (26):
    """)
    st.latex(
        r"\gamma_k \lambda_k - \lambda_{k-1} = - \int_a^b \rho P_k P_{k-1} dx = - \int_a^b \rho P_k \left( P_{k-1} - \frac{\alpha_k}{\alpha_{k-1}} P_{k-1} \right) dx = - \frac{\alpha_k}{\alpha_{k-1}} \lambda_{k-1}.")

    st.markdown("Следовательно:")
    st.latex(r"\gamma_k \lambda_k = - \frac{\alpha_k}{\alpha_{k+1}} \lambda_{k-1} \neq 0. \ \ \ (28)")

    st.markdown("Таким образом, (26) принимает вид:")
    st.latex(r"\frac{\alpha_k}{\alpha_{k+1}} P_{k+1} - x P_k = \gamma_k P_{k-1}.")

    st.markdown("т. е. имеет вид (25), где:")
    st.latex(
        r"a_k = \frac{\alpha_k}{\alpha_{k+1}}, \quad b_k = -\gamma_k = \frac{1}{\lambda_k} \int_a^b x \rho P_k^2 dx, \quad c_k = -\frac{\alpha_k \lambda_{k-1}}{\lambda_k}.")

    st.markdown("что и требовалось доказать. Перепишем формулу (30) в виде:")
    st.latex(
        r"P_{k+1} = \left( \frac{\alpha_k}{\alpha_{k+1}} x + \frac{\gamma_k}{\alpha_{k+1}} \right) P_k - \frac{\alpha_k \lambda_{k-1}}{\alpha_{k+1} \lambda_k} P_{k-1}. \ \ \ (31)")

    st.markdown("Отсюда получается несколько важных следствий.")

    st.markdown("**Следствие 1.** При известных коэффициентах рекуррентных соотношений и известных $P_0$, $P_1$ легко получить все последующие многочлены, не прибегая к процедуре ортогонализации Шмидта.")

    st.markdown("""
    **Следствие 2.** $k$ нулей многочлена $P_k(x)$ разделены $(k - 1)$ нулями многочлена $P_{k-1}(x)$ при условии $\rho(x) \geq 0$.  
    Доказательство проведем по индукции. На первом шаге имеем:
    """)
    st.latex(r"P_0(x) = \alpha_0 > 0,")
    st.latex(r"P_1(x) = \alpha_1 (x + \gamma_0).")

    st.markdown("Поскольку:")
    st.latex(r"\int_a^b \rho P_1 P_0 dx = \int_a^b \rho (x + \gamma_0) \alpha_1 \alpha_0 dx = 0,")
    st.markdown("""
    то $P_1(x)$ имеет действительный корень на $[a, b]$.  
    Пусть нули $P_1(x)$ разделены нулями $P_0(x)$, т. е. между двумя соседними нулями $P_1(x)$ лежит один нуль $P_0(x)$. Пусть $x_1, x_2, \dots, x_k$ — упорядоченные нули $P_k(x)$, тогда из (31) имеем:
    """)
    st.latex(r"P_{k+1}(x) = \frac{1}{\alpha_k} \left( x P_k(x) - \lambda_k P_{k-1}(x) \right). \ \ (32)")

    st.markdown("""
    При $x = b$ $P_{k+1}(b) > 0$, $P_k(b) > 0$, так как старшие коэффициенты положительны, а все нули многочленов лежат внутри $[a, b]$.  
    При $x = x_k$ $P_k(x_k) > 0$, следовательно, $P_{k+1}(x_k) < 0$ в силу (32).  
    При $x = x_{k-1}$ $P_{k-1}(x_{k-1}) < 0$ (по предположению индукции), а $P_{k+1}(x_{k-1}) > 0$.
    """)

    st.markdown("""
    Таким образом, $x_k$ оказывается между двумя нулями $P_{k+1}(x)$ и т. д.  
    Последний нуль $P_{k+1}(x)$ должен быть меньше $x_1$, так как мы в предыдущем процессе нашли $k$ нулей $P_k(x)$, что и требовалось доказать.
    """)

    st.markdown("Классические примеры ортогональных многочленов — многочлены Лежандра $P_n(x)$:")
    st.latex(r"[a, b] = [-1, 1], \quad \rho(x) = 1; \quad \int_{-1}^1 P_m P_n dx = \begin{cases} 0, & m \neq n, \\ \frac{2}{2n + 1}, & m = n. \end{cases}")

    st.latex(r"\frac{n + 1}{2n + 1} P_{n+1} - x P_n + \frac{n}{2n + 1} P_{n-1} = 0.")

    st.latex(r"P_0 = 1; \quad R(x) = x; \quad P_2(x) = \frac{1}{2} (3x^2 - 1), \quad P_3(x) = \frac{1}{2} (5x^3 - 3x).")

    st.markdown("**– многочлены Лагерра** $L_n(x)$:")
    st.latex(r"[a, b] = [0, \infty); \quad \rho(x) = e^{-x}; \quad \int_0^\infty e^{-x} L_m L_n dx = \begin{cases} 0, & m \neq n, \\ 1, & m = n. \end{cases}")
    st.latex(r"L_{n+1} - (2n + 1 - x)L_n + n L_{n-1} = 0.")

    st.markdown("**– многочлены Эрмита** $H_n(x)$:")
    st.latex(r"[a, b] = (-\infty, \infty); \quad \rho(x) = e^{-x^2}; \quad \int_{-\infty}^\infty e^{-x^2} H_m H_n dx = \begin{cases} 0, & m \neq n, \\ 2^n n! \sqrt{\pi}, & m = n. \end{cases}")
    st.latex(r"\frac{1}{2} H_{n+1} - x H_n + n H_{n-1} = 0.")

    st.latex(r"H_0 = 1, \quad H_1 = 2x, \quad H_2 = 4x^2 - 2, \quad H_3 = 8x^3 - 12x.")

    st.markdown("**– многочлены Чебышёва** (разд. 3.2.5).")

    st.markdown("### Замечания")
    st.markdown("1. Для получения ортогональных многочленов можно не использовать процесс ортогонализации Шмидта, а воспользоваться трехчленным соотношением. Пусть:")
    st.latex(r"P_0(x) = 1; \quad P_1(x) = x - \alpha_1;")
    st.latex(r"P_{k+1}(x) = x P_k(x) - \alpha_{k+1} P_k - \beta_{k+1} P_{k-1}, \quad k > 1. \ \ (33)")

    st.markdown("где коэффициенты $\\alpha_{k+1}, \\beta_{k+1}$ требуется определить.")

    st.markdown("Найдем $\\alpha_1$. Должно быть:")
    st.latex(r"\int_a^b \rho P_0 P_1 dx = 0.")

    st.markdown("Отсюда имеем:")
    st.latex(r"\alpha_1 = \frac{\int_a^b x \rho dx}{\int_a^b \rho dx}.")

    st.markdown("""
    Пусть известны ортогональные многочлены $P_0, P_1, \dots, P_k(x)$, вычислим $P_{k+1}(x)$.  
    Потребуем сначала, чтобы:
    """)

    st.latex(r"\int_a^b \rho P_k P_{k+1} dx = 0; \quad \int_a^b \rho P_{k+1} P_{k-1} dx = 0.")

    st.markdown("Этих соотношений достаточно для определения $\\alpha_{k+1}$ и $\\beta_{k+1}$:")
    st.latex(
        r"\int_a^b \rho x P_k^2 dx = \alpha_{k+1} \int_a^b \rho P_k^2 dx + \beta_{k+1} \int_a^b \rho P_{k-1} P_k dx.")

    st.latex(
        r"\int_a^b \rho x P_k P_{k+1} dx = \alpha_{k+1} \int_a^b \rho P_k P_k dx + \beta_{k+1} \int_a^b \rho P_{k-1} P_k dx.")

    st.markdown("**Отсюда имеем:**")
    st.latex(
        r"\alpha_{k+1} = \frac{\int_a^b \rho x P_k^2 dx}{\int_a^b \rho P_k^2 dx}, \quad \beta_{k+1} = \frac{\int_a^b \rho x P_k P_{k-1} dx}{\int_a^b \rho P_{k-1}^2 dx}.")

    st.markdown(
        "Ортогональность $P_{k+1}(x)$ всем многочленам степени $i < k-1$ видна непосредственно из (33)."
    )

    st.markdown(
        "**2. Для многочленов, заданных на дискретном множестве точек $x_i$, $i=1,2,...,N$, можно построить лишь $N$ ортогональных многочленов, при этом все интегралы заменяются суммами.**")

    st.markdown("**Примеры:**")
    st.markdown(
        "1. **Ортогонализовать функции $1, x, x^2$ на $[0, 1]$, $\rho(x)=1$, используя трехчленное соотношение.**"
    )

    st.markdown("**Решение:**")
    st.latex(r"P_0 = 1; \quad P_1 = x - \alpha_1; \quad \int_0^1 P_1 P_0 dx = 0.")
    st.markdown("**Отсюда:**")
    st.latex(r"\int_0^1 (x - \alpha_1) dx = 0.")
    st.markdown("**Следовательно:**")
    st.latex(r"\alpha_1 = \int_0^1 x dx = \frac{1}{2}.")

    st.latex(r"P_1(x) = x - \frac{1}{2};")
    st.latex(r"P_2(x) = x P_1 - \alpha_2 P_1 - \beta_2 P_0;")
    st.markdown("**Отсюда имеем:**")
    st.latex(r"\int_0^1 P_2 P_0 dx = 0, \quad \text{отсюда имеем} \quad \beta_2 = \int_0^1 x P_1 dx;")
    st.latex(
        r"\int_0^1 P_2 P_1 dx = 0, \quad \text{отсюда имеем} \quad \alpha_2 \int_0^1 P_1^2 dx = \int_0^1 x P_1^2 dx.")

    st.markdown("**После вычислений находим:**")
    st.latex(r"P_2(x) = \left( x - \frac{1}{2} \right)^2 - \frac{1}{12}.")

    st.markdown(
        "**2. Ортогонализовать многочлены $1, x, x^2$ на множестве узлов $x_i$: $x_1 = 0; x_2 = \\frac{1}{2}; x_3 = 1.$**")

    st.markdown("**Решение:**")
    st.markdown("Определим $\\alpha_1$ из условия:")
    st.latex(r"\sum_{i=1}^3 x_i P_1(x_i) = 0.")
    st.latex(
        r"\sum_{i=1}^3 x_i - 3 \alpha_1 = 0; \quad \alpha_1 = \frac{1}{3} \left( 0 + \frac{1}{2} + 1 \right) = \frac{1}{2}.")

    st.markdown("**Таким образом:**")
    st.latex(r"P_1(x) = x - \frac{1}{2};")

    st.markdown("**Уравнение для $\\alpha_2$:**")
    st.latex(r"\sum_{i=1}^3 x_i P_1^2(x_i) = \alpha_2 \sum_{i=1}^3 P_1^2(x_i).")

    st.markdown("**Уравнение для $\\beta_2$:**")
    st.latex(r"\sum_{i=1}^3 x_i P_1(x_i) P_0(x_i) = \beta_2 \sum_{i=1}^3 P_0^2(x_i).")

    st.markdown("**Отсюда находим:**")
    st.latex(r"\alpha_2 = \frac{1}{2}; \quad \beta_2 = \frac{1}{6}.")
    st.markdown("**Таким образом:**")
    st.latex(r"P_2(x) = \left( x - \frac{1}{2} \right)^2 - \frac{1}{6}.")

    st.markdown("**Попытаемся построить $P_3(x)$, ортогональный на указанном множестве $P_0, P_1, P_2$. Имеем:**")
    st.latex(r"P_3 = x P_2 - \alpha_3 P_2 - \beta_3 P_1.")

    st.markdown("**Из условий ортогональности $P_3$ к $P_2$ и $P_3$ к $P_1$ получим:**")
    st.latex(r"\alpha_3 = \frac{1}{2}; \quad \beta_3 = \frac{1}{12}.")

    st.markdown("**Поэтому:**")
    st.latex(r"P_3(x) = x \left( x - \frac{1}{2} \right)(x - 1)")

    st.markdown("**и $P_3 \equiv 0$ на заданном множестве $\\{x_i\\}$:**")

    st.markdown("### 3. Пусть задана таблица равновесных измерений:")
    st.table({
        "x": ["0", "1/2", "1"],
        "y": ["1", "2", "1"]
    })

    st.markdown("**Требуется приблизить эти данные прямой, используя МНК.**")
    st.markdown("**Решение:**")
    st.latex(r"y = a_0 P_0 + a_1 P_1,")
    st.markdown("где $P_0, P_1$ — ортогональные многочлены из примера 2:")

    st.latex(r"P_0 = 1, \quad P_1 = x - \frac{1}{2}.")

    st.latex(r"a_0 = \sum_{i=1}^3 y_i P_0(x_i) \Big/ \sum_{i=1}^3 P_0^2(x_i) = 1;")
    st.latex(r"a_1 = \sum_{i=1}^3 y_i P_1(x_i) \Big/ \sum_{i=1}^3 P_1^2(x_i) = -1.")

    st.markdown("**Таким образом:**")
    st.latex(r"y = -x + \frac{3}{2}.")

    st.markdown("""
        В заключение этого раздела отметим следующее свойство алгебраических многочленов наилучшего среднеквадратичного приближения: разность $y(x) - P_n(x)$ на $[a, b]$ имеет не менее $(n+1)$ нулей.
        """)

    st.markdown("**В самом деле, пусть $P_n = \sum_{k=0}^n a_k x^k$ — многочлен, дающий наилучшее среднеквадратичное приближение для $y(x)$ на $[a, b]$. Тогда коэффициенты его удовлетворяют системе уравнений:**")
    st.latex(r"\sum_{m=0}^n f_{km} a_m = (y, x^k), \quad k = 0, 1, \dots, n,")
    st.markdown("**где**")
    st.latex(r"f_{km} = (x^k, x^m),")
    st.markdown("**а**")
    st.latex(r"(\varphi_1, \varphi_2) = \int_a^b \rho \varphi_1 \varphi_2 dx, \quad (\rho(x) \geq 0).")

    st.markdown("""
    Пусть нули разности $y(x) - P_n(x)$ есть $x_1, \dots, x_r$, $r \leq n$. Составим многочлен $Q_r(x) = (x - x_1)(x - x_2) \dots (x - x_r) = \sum_{k=0}^r b_k x^k.$ 
    """)

    st.markdown("**Тогда произведение $(y - P_n) Q_r$ имеет на $[a, b]$ постоянный знак и**")
    st.latex(r"\int_a^b \rho (y - P_n) Q_r dx \neq 0.")

    st.markdown("**С другой стороны, этот интеграл равен**")
    st.latex(r"\int_a^b \rho (y - P_n) Q_r dx = \sum_{j=0}^n (y, x^j) b_j - \sum_{m=0}^n a_m (x^m, x^j) b_j = 0")
    st.markdown("**в силу определения $\\{a_m\\}$. Противоречие доказывает сделанные утверждения.**")

    st.markdown("### Дополнительная литература по МНК")
    st.markdown("""
    1. Худсон Д. **Статистика для физиков**. – М.: Мир, 1970.  
    2. Яноши Л. **Теория и практика обработки результатов измерений**. – М.: Мир, 1965.
    """)