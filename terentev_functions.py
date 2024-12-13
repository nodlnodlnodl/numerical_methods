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

def uravnenie_1z():
    st.markdown("""
    ### 5. ВЫЧИСЛЕНИЕ КОРНЕЙ УРАВНЕНИЙ

    #### 5.1. Уравнения с одним неизвестным
    Пусть задана непрерывная функция \( f(x) \) и требуется найти все (или некоторые) корни уравнения
    $$ f(x) = 0. $$

    Эта задача включает несколько этапов:

    1. Необходимо исследовать количество, характер (простые или кратные корни, действительные или комплексные) и расположение корней.
    2. Необходимо найти приближенные значения корней.
    3. Для интересующих нас значений корней провести их вычисление с требуемой точностью.

    Первый и второй этапы рассматриваются аналитически и графически. Когда требуется найти только действительные корни, очень часто оказывается полезным составить таблицу значений и найти зоны смены знака \( f(x) \).
    """)

def metod_dihotomii():
    st.markdown(r"""
            ### 5.1.1. Метод дихотомии (половинного деления)

            Пусть известно, что функция \( f(x) \) непрерывна на отрезке \( [a, b] \), и выполнено условие \( f(a) \cdot f(b) < 0 \), 
            то есть функция \( f(x) \) принимает на концах отрезка значения разных знаков. В таком случае отрезок \( [a, b] \) содержит 
            нечётное число корней уравнения:
        """)

    st.latex(r"f(x) = 0 \tag{1}")

    st.markdown(r"""
            Положим \( x_0 = a \), \( x_1 = b \), найдём середину отрезка \( x_2 \) следующим образом:
        """)

    st.latex(r"x_{2} = \frac{x_{0} + x_{1}}{2}")

    st.markdown(r"""
            Вычислим значение функции \( f(x_2) \). Если \( f(x_2) \cdot f(x_1) \leq 0 \), то новый отрезок для дальнейшего рассмотрения будет 
            \( [x_0, x_2] \). В противном случае, если выполнено \( f(x_2) \cdot f(x_0) \leq 0 \), выбираем отрезок \( [x_2, x_1] \).

            Повторяя этот процесс, мы последовательно уменьшаем размер отрезка, который содержит корень уравнения (1). В результате мы получаем
            систему вложенных отрезков:
        """)

    st.latex(r"[a_0, b_0], [a_1, b_1], [a_2, b_2], \dots [a_n, b_n], \dots")

    st.markdown(r"""
            где:
        """)

    st.latex(r"""
        a_0 = a, \quad b_0 = b, \quad a_1 = x_2, \quad b_1 = b, \quad a_2 = \frac{a_0 + x_2}{2}, \quad b_2 = x_2,
        """)

    st.markdown(r"""
            При этом выполняется:
        """)

    st.latex(r"""
        f(a_n) \cdot f(b_n) < 0, \quad b_n - a_n = \frac{b - a}{2^n}.
        """)

    st.markdown(r"""
            После 10 шагов такого процесса область локализации корня будет уменьшена примерно в \( 10^3 \) раз.

            Метод дихотомии применим и к недифференцируемым функциям, он прост и надёжен, и гарантирует точность ответа. Однако у метода есть несколько недостатков:

            1. Необходимо заранее знать отрезок \( [a, b] \), на котором функция \( f(x) \) меняет знак.
            2. Если на отрезке \( [a, b] \) несколько корней, метод может сойтись к любому из них, не зная заранее, к какому.
            3. Метод неприменим для корней чётной кратности, так как для таких корней метод может быть чувствителен к ошибкам округления.
            4. Метод не обобщается на системы уравнений.

            Для точного нахождения всех корней необходимо исключать уже найденные корни, чтобы избежать возможности замены нескольких близко лежащих корней уравнения \( f(x) = 0 \) на один приближённый корень. 
            После нахождения одного корня, нужно исключить его область локализации из исходного отрезка \( [a, b] \) для поиска остальных корней.
        """)

def metod_hord():
    st.markdown(r"""
            ### 5.1.2. Метод хорд

            Иногда вместо метода половинного деления используют метод хорд для более быстрого нахождения корня. Пусть $$ x^* $$ — корень уравнения (1), и он находится на отрезке $$ [a, b] $$ (при этом других корней на этом отрезке нет), причём $$ f(a) \cdot f(b) < 0 $$.

            Проведем прямую через точки $$ (a, f(a)) $$ и $$ (b, f(b)) $$ и найдём её пересечение с осью $$ x $$:
        """)

    st.latex(r"""
        y = \frac{x - a}{b - a} f(b) + \frac{b - x}{b - a} f(a)
        """)

    st.latex(r"""
        x_1 = \frac{a f(b) - b f(a)}{f(b) - f(a)} = a - \frac{f(a)(b - a)}{f(b) - f(a)}
        """)

    st.markdown(r"""
            Далее, как и в методе дихотомии, вычислим $$ f(x_1) $$, выберем новый отрезок $$ [a_1, b_1] $$ и повторим процедуру. 
            Пусть для определенности $$ f'(x) > 0 $$ на всём отрезке $$ [a, b] $$ (то есть хорда лежит выше кривой).
            Возможны два случая:

            - Если $$ f(a) > 0 $$, то конечной точкой становится $$ x = a $$;
            - Если $$ f(a) < 0 $$, то конечной точкой становится $$ x = b $$.

            В обоих случаях хорда $$ x_n $$ будет приближаться к корню.

            На каждом шаге приближения к корню вычисляется новое значение:
        """)

    st.latex(r"""
        x_{n+1} = x_n - \frac{f(x_n)(x_n - a)}{f(x_n) - f(a)} \tag{1'}
        """)

    st.markdown(r"""
            Это образует монотонно убывающую ограниченную последовательность. Аналогичным образом, если $$ f(a) < 0 $$, то конечной точкой становится $$ b $$, и последовательность приближений принимает вид:
        """)

    st.latex(r"""
        x_{n+1} = x_n - \frac{f(x_n)(x_n - b)}{f(x_n) - f(b)} \tag{1''}
        """)

    st.markdown(r"""
            В общем случае можно предсказать конечную точку, исходя из знака производной $$ f'(x) $$. Метод хорд не всегда быстрее метода дихотомии, и его точность зависит от поведения производной $$ f'(x) $$ на отрезке $$ [a, b] $$.

            Для оценки точности $$ n $$-приближений можно воспользоваться следующей формулой:
        """)

    st.latex(r"""
        x_n = x_{n-1} - \frac{f(x_{n-1})(x_{n-1} - a)}{f(x_{n-1}) - f(a)}
        """)

    st.markdown(r"""
            Используя теорему Лагранжа, для дифференцируемой функции $$ f(x) $$ найдём:
        """)

    st.latex(r"""
        f(x^*) - f(x_{n-1}) = f'(x_n^*)(x^* - x_{n-1})
        """)

    st.markdown(r"""
            Таким образом, оценка точности приближений зависит от поведения функции $$ f'(x) $$ на всём отрезке $$ [a, b] $$.

            Если известно, что производная $$ f'(x) $$ знакопостоянна, и для некоторых констант выполняется $$ 0 < m \leq |f'(x)| \leq M $$, то оценка приближения к корню $$ x^* $$ принимается в виде:
        """)

    st.latex(r"""
        |x^* - x_n| \leq \frac{M - m}{m} |x_n - x_{n-1}|
        """)

    st.markdown(r"""
            Этот факт необходимо учитывать при окончании итераций метода хорд. Если требуется точность $$ \varepsilon $$, то условие прекращения итераций:
        """)

    st.latex(r"""
        |x_n - x_{n-1}| \leq \varepsilon \frac{m}{M - m}
        """)

def simple_iterecia():
    st.markdown("""
    ### 5.1.3. Метод простой итерации

    Сущность этого метода заключается в замене исходного уравнения вида
    """)
    st.latex(r"x = \varphi(x)")
    st.markdown("""
    (например, можно положить, что \( \varphi(x) = x + \psi(x)f(x) \), где \( \psi(x) \) — непрерывная знакопостоянная функция). Метод простой итерации реализуется следующим образом. Выберем из каких-либо соображений приближенное (может быть, очень грубое) значение корня \( x_0 \) и подставим в правую часть (2), тогда получим некоторое число
    """)
    st.latex(r"x_1 = \varphi(x_0)")
    st.markdown("""
    Подставляя в правую часть (2) \( x_1 \), получим \( x_2 \) и т. д. Таким образом, возникает некоторая последовательность чисел \( x_0, x_1, x_2, \dots, x_n, \dots \), где
    """)
    st.latex(r"x_n = \varphi(x_{n-1})")
    st.markdown("""
    Если эта последовательность сходится, т. е. существует
    """)
    st.latex(r"x^* = \lim_{n \to \infty} x_n,")
    st.markdown("""
    где \(x^*\) будет корнем уравнения (2), который, используя (3), можно вычислить с любой степенью точности.

    В приложениях часто весьма полезно представить графическую интерпретацию итерационного процесса (3), начертив графики функций \(y = x\) и \(y = \varphi(x)\).

    Установим достаточные условия сходимости.
    """)
    st.markdown("#### Теорема 1")
    st.markdown("""
    Пусть функция \( \varphi(x) \) определена и дифференцируема на отрезке \([a, b]\) и все её значения \( \varphi(x) \in [a, b]\). Тогда, если выполнено условие
    """)
    st.latex(r"| \varphi'(x) | \leq q < 1 \tag{4}")
    st.markdown("""
    при \(x \in [a, b]\), то:
    1) итерационный процесс (3) сходится независимо от начального приближения \(x_0 \in [a, b]\);
    2) \( x^* = \lim_{n \to \infty} x_n \) — единственный корень уравнения \(x = \varphi(x)\) на \([a, b]\).

    #### Доказательство
    Пусть
    """)
    st.latex(r"x_{n+1} = \varphi(x_n), \quad x_n = \varphi(x_{n-1})")
    st.markdown("""
    Тогда
    """)
    st.latex(r"x_{n+1} - x_n = \varphi(x_n) - \varphi(x_{n-1}).")
    st.markdown("""
    По теореме Лагранжа имеем
    """)
    st.latex(r"x_{n+1} - x_n = (x_n - x_{n-1}) \varphi'(\tilde{x}_n), \quad \tilde{x}_n \in (x_{n-1}, x_n).")
    st.markdown("""
    Поэтому в силу (4)
    """)
    st.latex(r"|x_{n+1} - x_n| \leq q |x_n - x_{n-1}|.")
    st.markdown("""
    Отсюда, полагая последовательно \(n = 1, 2, \dots\), находим
    """)
    st.latex(r"|x_{n+1} - x_n| \leq q^n |x_1 - x_0|. \tag{5}")
    st.markdown("""
    Рассмотрим ряд
    """)
    st.latex(r"x_0 + (x_1 - x_0) + (x_2 - x_1) + \dots + (x_n - x_{n-1}) + \dots \tag{6}")
    st.markdown("""
    Очевидно, частные суммы этого ряда равны
    """)
    st.latex(r"s_{n+1} = x_n.")
    st.markdown("""
    Для этого ряда имеем, согласно (5), что его члены меньше по абсолютной величине членов геометрической прогрессии со знаменателем \(q < 1\). Поэтому этот ряд сходится абсолютно и
    """)
    st.latex(r"\lim_{n \to \infty} s_n = x^* \in [a, b].")
    st.markdown("""
    Переходя к пределу в (3), в силу непрерывности \( \varphi(x) \) получим
    """)
    st.latex(r"x^* = \varphi(x^*),")
    st.markdown("""
    т. е. \( x^* \) — действительно корень уравнения (2).

    Покажем, что этот корень единственный. Пусть есть еще один корень \( x_1^* \), \( x_1^* = \varphi(x_1^*) \), тогда
    """)
    st.latex(r"x_1^* - x^* = \varphi(x_1^*) - \varphi(x^*) = \varphi'(c)(x_1^* - x^*), \quad c \in [x_1^*, x^*].")
    st.markdown("""
    Отсюда имеем
    """)
    st.latex(r"(x_1^* - x^*)(1 - \varphi'(c)) = 0")
    st.markdown("""
    и
    """)
    st.latex(r"x_1^* = x^*,")
    st.markdown("""
    что и требовалось доказать.

    ##### Замечание.
    Теорема остается справедливой для функции, удовлетворяющей на \([a, b]\) условию Липшица:
    """)
    st.latex(r"| \varphi(x) - \varphi(y) | \leq L | x - y |, \quad x, y \in [a, b].")
    st.markdown("""
    #### Оценим точность \(n\)-приближения.
    Для этого рассмотрим последовательность неравенств
    """)
    st.latex(r"|x_{n+p} - x_n| \leq |x_{n+p} - x_{n+p-1}| + |x_{n+p-1} - x_{n+p-2}| + \dots + |x_{n+1} - x_n|")
    st.latex(r"\leq q^n (q^{p-1} + \dots + 1)(x_1 - x_0) = q^n(x_1 - x_0) \frac{1 - q^p}{1 - q}.")
    st.markdown("""
    При \( p \to \infty \) для \( q < 1 \) имеем
    """)
    st.latex(r"|x^* - x_n| \leq \frac{q^n}{1 - q}(x_1 - x_0).")
    st.markdown("""
    Отсюда видно, что чем меньше \( q \), тем лучше сходимость.

    Можно дать немного отличную по форме оценку точности \(n\)-приближения. Рассмотрим \( g(x) = x - \varphi(x) \), тогда \( g'(x) = 1 - \varphi'(x) \geq 1 - q \); учитывая, что \( g(x^*) = 0 \), находим
    """)
    st.latex(r"|x_n - \varphi(x_n)| = |g(x_n) - g(x^*)| = |x_n - x^*||g'(\tilde{x}_n)| \geq |x_n - x^*|(1 - q),")
    st.markdown("""
    где \( \tilde{x}_n \in [x_{n-1}, x^*] \). Отсюда имеем
    """)
    st.latex(r"|x_n - x^*| \leq \frac{q}{1 - q} |x_n - x_{n-1}|. \tag{7}")
    st.markdown("""
    Из формулы (7) следует, что вопреки распространенному мнению из условия
    """)
    st.latex(r"|x_n - x_{n-1}| \leq \varepsilon")
    st.markdown("""
    не следует
    """)
    st.latex(r"|x_n - x^*| < \varepsilon,")
    st.markdown("""
    если \( q \) достаточно близко к 1. Поэтому для достижения заданной точности в процессе итераций надо проводить до выполнения условия
    """)
    st.latex(r"|x_n - x_{n-1}| \leq \varepsilon(1 - q)/q. \tag{8}")
    st.markdown("""
    Отметим, что
    """)
    st.latex(
        r"|x^* - x_n| = |\varphi(x^*) - \varphi(x_{n-1})| = |x^* - x_{n-1}||\varphi'(\tilde{x}_{n-1})| \leq q|x^* - x_{n-1}|,")
    st.markdown("""
    т. е. процесс сходимости монотонный.

    Условию (8) можно придать другую форму, используемую на практике. Начиная с некоторого \(n\), параметр \(q\) можно определить по формуле
    """)
    st.latex(r"q = \frac{x_n - x_{n-1}}{x_{n-1} - x_{n-2}}.")
    st.markdown("""
    Тогда (7) можно переписать в виде
    """)
    st.latex(
        r"\left| \frac{q}{1 - q} (x_n - x_{n-1}) \right| = \frac{(x_n - x_{n-1})^2}{|2x_{n-1} - x_n - x_{n-2}|} \leq \varepsilon. \tag{9}")
    st.markdown("""
    При выполнении этого условия итерации можно прекращать. Отметим, что поскольку \(x_n = s_{n+1}\) — частичной сумме ряда (6), члены которого ведут себя с некоторого \(n\) как геометрическая прогрессия, то к последовательности \( \{x_n\} \) обычно бывает выгодно на поздних этапах применять нелинейное преобразование (процесс Эйткена)
    """)
    st.latex(r"\tilde{x}_n = \frac{x_{n+1}x_{n-1} - x_n^2}{x_{n+1} + x_{n-1} - 2x_n},")
    st.markdown("""
    которое сокращает число итераций.

    Как видно из предыдущего, успех итерационного метода зависит от выбора \( \varphi(x) \) и, вообще говоря, от выбора \( x_0 \).
    """)
    st.markdown("#### Пример")
    st.markdown("""
    Рассмотрим уравнение \( f(x) = x^2 - a = 0 \), его можно записать в разной форме \( x = \varphi_\alpha(x) \), \( \alpha = 1, 2 \):
    """)
    st.latex(r"\varphi_1(x) = \frac{a}{x}, \quad \varphi_2(x) = \frac{1}{2} \left(x + \frac{a}{x} \right).")
    st.markdown("""
    С функцией \( \varphi_1(x) \) итерационный процесс вообще не сходится
    """)
    st.latex(r"x_1 = \frac{a}{x_0}, \quad x_2 = \frac{a}{x_1} = x_0; \quad x_3 = x_1 = x_0, \dots")
    st.markdown("""
    С функцией \( \varphi_2(x) \) итерационный процесс сходится при любом \( x_0 > 0 \) и очень быстро, так как
    """)
    st.latex(r"\varphi_2'(x) = \frac{1}{2} - \frac{1}{2} \frac{a}{x^2} = 0.")
    st.markdown("""
    Отметим, что \( |\varphi_2'(x)| < 1 \) только при \( x > \sqrt{\frac{a}{3}} \), тем не менее итерационный процесс сходится при любом \( x_0 > 0 \).
    """)

def newton_method():
    st.markdown(r"""
            ## 5.1.4. Метод Ньютона

            Этот метод иногда ещё называют методом касательных или методом линеаризации.

            Пусть корень $$ x^* $$ уравнения $$ f(x) = 0 $$ отделён на отрезке $$ [a,b] $$, и $$ f'(x) $$, $$ f''(x) $$ непрерывны и сохраняют определённые знаки на $$ [a,b] $$. Если известно какое-либо приближение $$ x_n $$ для $$ x^* $$, мы можем уточнить его, используя метод Ньютона. Положим
        """)

    st.latex(r"h_n = x^* - x_n")

    st.markdown(r"""
            Тогда
        """)

    st.latex(r"0 = f(x_n + h_n) = f(x_n) + h_n f'(x_n)")

    st.markdown(r"""
            и, следовательно,
        """)

    st.latex(r"h_n = - \frac{f(x_n)}{f'(x_n)}")

    st.markdown(r"""
            Таким образом, можно ожидать, что
        """)

    st.latex(r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} \quad \text{(10)}")

    st.markdown(r"""
            окажется лучшим приближением для $$ x^* $$.

            Можно дать другой вывод формулы (10). Напишем уравнение касательной, проходящей через точку $$ (x_n, f(x_n)) $$ и имеющей наклон $$ f'(x_n) $$:
        """)

    st.latex(r"y = f(x_n) + (x - x_n) f'(x_n)")

    st.markdown(r"""
            и найдём её точку пересечения с осью абсцисс:
        """)

    st.latex(r"0 = f(x_n) + (x_{n+1} - x_n) f'(x_n)")

    st.markdown(r"""
            т.е. получилась та же формула (10).

            Метод Ньютона является частным случаем метода простой итерации, для него
        """)

    st.latex(r"\varphi(x) = x - \frac{f(x)}{f'(x)}")

    st.markdown(r"""
            Вычислим $$ \varphi'(x) $$, которая определяет сходимость итерационного процесса:
        """)

    st.latex(r"\varphi'(x) = 1 - \frac{f'(x)^2 - f(x)f''(x)}{f'(x)^2}")

    st.markdown(r"""
            Если $$ x^* $$ — простой корень уравнения $$ f(x) = 0 $$, то $$ \varphi'(x^*) = 0 $$; если $$ x^* $$ — $$ p $$-кратный корень, то вблизи $$ x^* $$:
        """)

    st.latex(r"f(x) = a(x - x^*)^p, \quad \varphi'(x^*) = \frac{p - 1}{p} < 1.")

    st.markdown(r"""
            Таким образом, если начальное приближение выбрано близким к $$ x^* $$, то метод Ньютона приведёт к сходящимся итерациям.

            Как выбрать это приближение?

            Теорема 2. Если $$ f(a)f(b) < 0 $$ и $$ f'(x) $$ отлично от нуля и знакопостоянно на $$ [a,b] $$, а начальное приближение удовлетворяет условию
        """)

    st.latex(r"f(x_0) f'(x_0) > 0 \quad \text{(11)}")

    st.markdown(r"""
            то итерационный процесс (10) сходится.

            Доказательство. Пусть, например, $$ f(a) < 0 $$, $$ f(b) > 0 $$, $$ f'(x) > 0 $$, $$ f''(x) > 0 $$ при $$ a \leq x \leq b $$ (остальные случаи рассматриваются аналогично). Положим $$ x_0 = b $$ и покажем, что все $$ x_n > x^* $$.

            В самом деле, пусть при некотором $$ n $$, $$ x_n > x^* $$, тогда
        """)
    st.latex(r"x^* = x_n + (x^* - x_n)")

    st.markdown(r"""
            и
        """)

    st.latex(r"""
            0 = f(x^*) = f(x_n) + f'(x_n)(x^* - x_n) + \frac{1}{2} f''(c_n)(x^* - x_n)^2,
        """)

    st.markdown(r"""
            где $$ x^* < c_n < x_n $$.

            Так как $$ f''(c_n) > 0 $$, то
        """)

    st.latex(r"f(x_n) + f'(x_n)(x^* - x_n) < 0.")

    st.markdown(r"""
            Откуда
        """)

    st.latex(r"x^* < x_n - \frac{f(x_n)}{f'(x_n)} = x_{n+1}.")

    st.markdown(r"""
            Учитывая, что $$ f(x_n) > 0 $$, $$ f'(x_n) > 0 $$, имеем $$ x_{n+1} > x^* $$.

            Таким образом, в условиях теоремы формула (10) даёт монотонно убывающую ограниченную последовательность, имеющую предел $$ x^* $$.

            Переходя в (10) к пределу, находим
        """)

    st.latex(r"\bar{x} = x^* - \frac{f(x^*)}{f'(x^*)} = x^*,")

    st.markdown(r"""
            т.е. $$ \bar{x} = x^* $$, что и требовалось доказать.

            Получим оценку скорости сходимости вблизи простого корня. По определению простых итераций
        """)

    st.latex(r"\frac{x_{n+1} - x^*}{x_n - x^*} = \varphi'(c_n) = 0")

    st.markdown(r"""
            т.е. погрешность очередного приближения равна квадрату погрешности предыдущего приближения (квадратичная сходимость). Если $$ n = 9 $$ итераций дало 3 верных знака, то $$ n = 10 $$ даст 6, а $$ n = 11 $$ — 12 верных знаков. Конечно, это верно лишь вблизи от корня, поэтому обычно метод Ньютона используется для уточнения корней.

            Пример. Рассмотрим уравнение
        """)

    st.latex(r"f(x) = x^2 - a, \quad a > 0,")

    st.markdown(r"""
            исходя из метода Ньютона
        """)

    st.latex(r"\varphi(x) = x - \frac{f(x)}{f'(x)} = \frac{1}{2} \left( x + \frac{a}{x} \right)")

    st.markdown(r"""
            — известная формула.
        """)
    st.markdown(r"""
            Замечание 1. Метод Ньютона при вычислении кратных корней, когда $$ f'(x^*) = 0 $$, надо применять, обеспечивая высокую точность вычисления $$ f'(x) $$.

            Замечание 2. Если производная $$ f'(x) $$ на рассматриваемом отрезке меняется мало, то в методе Ньютона её можно считать не на каждой итерации, сохраняя для нескольких (а то и для всех) итераций значение $$ f'(x) $$ неизменным.

            Замечание 3. Если известна кратность корня $$ p $$, то итерации по Ньютону выгоднее вести по формуле:
        """)

    st.latex(r"x_{n+1} = x_n - \frac{p f(x_n)}{f'(x_n)},")

    st.markdown(r"""
            что повышает скорость сходимости.
    """)

def metod_sekushchih():
    st.markdown("""
    ### 5.1.5. Метод секущих

    В методе Ньютона требуется вычислять производную, что не всегда удобно. Иногда применяется метод, в котором точная производная заменена на первую разделённую разность, вычисленную по двум последним итерациям, т.е. касательная заменена на секущую. Для этого метода надо задать формулу:

    $$ x_{n+1} = x_n - \frac{(x_n - x_{n-1}) f(x_n)}{f(x_n) - f(x_{n-1})}. \tag{12} $$

    Для оценки скорости сходимости разложим все функции в ряд Тейлора в окрестности простого корня \( x^* \):

    $$ f(x_n) = f'(x^*)(x_n - x^*) + \frac{1}{2} f''(x^*)(x_n - x^*)^2, $$

    $$ f(x_{n-1}) = f'(x^*)(x_{n-1} - x^*) + \frac{1}{2} f''(x^*)(x_{n-1} - x^*)^2. $$

    Тогда

    $$ x_{n+1} = x_n - \frac{(x_n - x_{n-1}) f'(x^*)(x_n - x^*) \left(1 + \frac{1}{2} \frac{f''(x^*)}{f'(x^*)}(x_n + x_{n-1} - 2x^*) \right)}{f'(x^*)(x_n - x_{n-1}) \left( 1 + \frac{1}{2} \frac{f''(x^*)}{f'(x^*)} (x_n + x_{n-1} - 2x^*) \right)}. $$

    Введём обозначения: \( z_n = x_n - x^*, a = \frac{f''(x^*)}{2 f'(x^*)} \), тогда предыдущее соотношение запишется следующим образом:

    $$ z_{n+1} = a z_n z_{n-1}. $$

    Будем искать решение этого уравнения в виде

    $$ z_n+1 = a z_n^\alpha z_{n-1}^\beta. $$

    Из соотношения \( z_{n+1} = a z_n z_{n-1} \) имеем

    $$ z_{n-1} = z_n^{\frac{\alpha}{\beta}}, $$

    следовательно,

    $$ a z_n^\alpha z_n^{\frac{\alpha}{\beta}} = a z_n z_{n-1}, $$

    Отсюда для \( \alpha, \beta \) получаем следующие уравнения:

    $$ \beta^2 - \beta - 1 = 0, $$

    $$ \alpha \beta = 1. $$

    Нам надо выбрать корень \( \beta > 0 \), поэтому

    $$ \beta = \frac{1}{2} ( \sqrt{5} + 1) \approx 1.62. $$

    Таким образом, метод секущих хотя и имеет сверхлинейную сходимость, но скорость её меньше, чем в методе Ньютона. Однако в методе Ньютона при вычислениях надо знать и функцию, и производную, т.е. делать вдвое больше работы, чем в методе секущих. Поэтому для одного объёма вычислений в методе секущих можно сделать вдвое больше итераций и получить большую точность, чем в методе Ньютона.

    В \( (12) \) в знаменателе стоит разность значений функции; вблизи корня (особенно кратного) значения \( f(x_n), f(x_{n-1}) \) малы и близки, следовательно, при вычитании возможна потеря значащих цифр. Это ограничивает точность, с которой можно найти корень. Для повышения точности при малых значениях разности применяется приём Гарвика: задаётся не слишком малое \( \varepsilon \), и итерации ведутся до выполнения условия

    $$ |x_{n+1} - x_n| < \varepsilon, $$

    затем они продолжаются до тех пор, пока \( |x_{n+1} - x_n| \) убывает; первое возрастание свидетельствует о начале потери точности, и вычисления прекращают.
    """)

def metod_rybakova():
    st.markdown("""
    ### 5.1.6. Метод Рыбакова Л.М.

    Этот метод позволяет вычислить все действительные корни уравнения \( f(x) = 0 \) ( \( f \) — дифференцируемая функция) на \( [a,b] \), если известна оценка модуля \( f'(x) \):

    $$ \max_{a \leq x \leq b} |f'(x)| \leq M. \tag{13} $$

    Пусть \( M \) в (13) известно, возьмем \( x_0 = a \) и построим последовательность

    $$ x_{k+1} = x_k + \frac{f(x_k)}{M}, \quad k = 0,1, \dots $$

    Обозначим через \( x^* \) ближайший к \( a \) корень \( f(x) = 0 \), тогда

    $$ x_k < x_{k+1} \leq x^*. $$

    В самом деле, первое неравенство очевидно, второе неравенство доказывается так:

    $$ f(x_k) = f(x_k) - f(x^*) = f'(x_n) (x_k - x^*), \quad x_n \in [x_k, x^*]. $$

    Отсюда, при \( x_k < x^* \), имеем

    $$ \frac{f(x_k)}{f'(x_n)} = x_k - x^* \leq \frac{1}{M} |f'(x_n)| (x_k - x^*), $$

    или

    $$ x_{k+1} = x_k + \frac{f(x_k)}{M} = x^* - \left(x^* - x_k \right)\left(1 - \frac{1}{M}|f'(x_n)| \right). $$

    Таким образом, процесс (14) дает монотонно возрастающую ограниченную последовательность, в силу непрерывности \( f(x) \) её предел будет совпадать с \( x^* \). После того как корень \( x^* \) найден с нужной точностью \( \varepsilon \), положим в качестве начального приближения \( x_0 = x^* \), и по формуле (14) получим следующий корень и т.д.

    Метод Рыбакова, хотя и требует большого объёма вычислений для получения корней с высокой точностью, является единственным методом, гарантированно находящим все корни на \( [a,b] \) с заданной точностью.

    Описание усовершенствованного метода Рыбакова можно найти в книге В. Л. Загускина "Численные методы решения плохо обусловленных задач". Изд-во Ростовского университета, 1976 г.
    """)

def korni_mnogochlenov():
    st.markdown("""
    ### 5.1.7. Вычисление корней многочленов

    Задача вычисления корней многочленов часто встречается в приложениях, и вопрос этот давно привлекает внимание вычислителей. Как правило, в математобеспечении ЭВМ имеются программы для решения этой задачи, основанные на том или ином методе. Существуют методы, позволяющие найти все действительные и комплексные корни многочленов и не требующие предварительного отделения корней. Естественно, что если корни отделены, то к многочленам можно применить любой из рассмотренных выше методов. По способам вычисления корней многочленов существует обширная литература (см., например, [1]). Поэтому ограничимся некоторыми замечаниями.

    Сначала обсудим вопрос устойчивости значений корней к погрешностям в коэффициентах многочлена (корректность задачи отделения корней). Рассмотрим широко известный пример (см. книгу Д. Форсайта, М. Малькольма, К. Мойлер. "Машинные методы математических вычислений". М., Мир, 1980 г.).

    Пусть требуется найти корни многочлена

    $$ P(x) = (x - 1)(x - 2)\dots(x - 20) = x^{20} - 210x^{19} + \dots $$

    и корни хорошо разделены. Теперь предположим, что нам задали многочлен

    $$ P(x) = x^{20} - (210 - 2^{-23}) x^{19} + \dots, $$

    у которого "испортен" коэффициент при \( x^{19} \). Вычисления корней этого многочлена, проведённые с высокой точностью, показали, что 10 корней стали комплексными с мнимой частью от \( \pm 0.6 \) до \( \pm 2.8 \)! Сравнительно малое изменение (в шестом знаке после запятой) лишь одного коэффициента сильно изменило действительные корни, что является примером чувствительности задачи отделения корней. Таким образом, задача оказалась некорректной.

    Установим причину некорректности: малая ошибка породила большую ошибку в корнях. Эти задачи называются некорректными.

    Рассмотрим причину такой некорректности. Рассмотрим многочлен

    $$ P(x, \alpha) = x^{20} - (210 - \alpha) x^{19} + \dots $$

    и найдём зависимость корней уравнения \( P(x, \alpha) = 0 \) от \( \alpha \). Имеем:

    $$ \frac{\partial P}{\partial x} = 20 x^{19}, \quad \frac{\partial P}{\partial \alpha} = x^{19}, $$

    отсюда при \( x_i = i \) находим

    $$ \frac{\partial x_i}{\partial \alpha} = \frac{20}{\prod_{j = 1, j \neq i}^{20} (i - j)}. $$

    Для корня \( x_i = i \) получим

    $$ \frac{\partial x_i}{\partial \alpha} = \frac{20}{\prod_{j = 1, j \neq i}^{20} (i - j)}. $$

    Наибольшие значения производной \( (\sim 10^7 - 10^9) \) приходятся на значения \( i \geq 10 \). В этом и кроется причина некорректности этой задачи: очень велик коэффициент усиления ошибки!

    Теперь кратко обсудим метод поиска корней многочлена, не требующий, вообще говоря, предварительного определения границ корней. Этот метод — **метод парабол** — на самом деле применим не только к многочленам, но широкое распространение он получил именно для многочленов. Сходимость этого метода теоретически не доказана, но в практике вычислений корней многочленов не известны примеры, когда бы он не сходился или сходился медленно.

    Пусть требуется определить все корни алгебраического многочлена

    $$ P_n(z) = a_n z^n + a_{n-1} z^{n-1} + \dots + a_0 z^0, \quad a_0 \neq 0 $$

    с комплексными коэффициентами. Идея метода парабол заключается в следующем. Пусть заданы три попарно различных комплексных числа \( z_0, z_1, z_2 \) (т.е. метод 3-шаговый). Построим по этим числам, как по узлам, интерполяционный многочлен Лагранжа для \( P_n(z) \). Это будет многочлен 2-го порядка, корни которого несложно определить. Выберем в качестве \( z_3 \) тот его корень, который находится ближе к \( z_2 \), и в качестве следующей тройки чисел для интерполяции возьмём \( z_1, z_2, z_3 \) и т.д.

    Эмпирически установлено, что последовательность точек \( \{z_i\} \) всегда сходится к какому-либо корню. После того как этот корень найден, он выделяется, и вся процедура применяется к многочлену меньшей степени.

    **Замечание 1.** Перед удалением найденного корня его надо уточнить по исходному многочлену (например, методом Ньютона).

    **Замечание 2.** В качестве начальных точек выбирают \( z_1, z_2, z_3 \) так, чтобы итерации сходились к наименьшему по модулю корню.
    """)

    st.markdown("""
    Оценим потерю точности при понижении степени многочлена. Пусть корень (простой) \( x_1 \) вычислен с погрешностью \( \Delta x_1 \). Тогда в точке \( \tilde{x_1} = x_1 + \Delta x_1 \) многочлен \( P_n(\tilde{x_1}) \equiv P_n(x_1) + P'_n(x_1) \Delta x_1 = P'_n(x_1) \Delta x_1 \).

    При делении \( P_n(x) \) на \( (x - \tilde{x_1}) \) получим:

    $$ P_n(x) = P_{n-1}(x) (x - \tilde{x_1}) + P_n(\tilde{x_1}). $$

    Пусть \( x_2 \) — один из оставшихся корней (простых) \( P_n(x), \) который мы будем вычислять с помощью \( P_{n-1}(x). \) Обозначим его погрешность через \( \Delta x_2 \) и из (15) находим

    $$ P_{n-1}(x_2 + \Delta x_2) = P_{n-1}(x_2) + P'_{n-1}(x_2) \Delta x_2 = P_n(x_2) + P_n(\tilde{x_1}). $$

    Отсюда, так как

    $$ | \Delta x_2 | = | \Delta x_1 | \frac{| P'_n(x_1) |}{| P'_n(x_2) |}, $$

    Таким образом, потеря точности зависит от соотношения производных при значениях корней и может быть значительной.

    **Пример**:

    $$ P_{25}(x) = (x - 10^{-2})(x - 10^{-4})(x^2 - 16^{-2})(x^2 - 16^{-2})(x - 1). $$

    Для него

    $$ P_{25}(1) \equiv 1; \quad P_{25}(1/4) \equiv 1.4 \cdot 10^{-10}. $$

    Если корень \( x = 1 \) найден первым, то его исключение приведёт к большой потере точности для корней с \( |x| = 1/4. $$
    """)

def systemi_nonliniar_uravn():
    st.markdown("""
    ### 5.2. Системы нелинейных уравнений

    Системы нелинейных уравнений практически решают только итерационными методами. Пусть система имеет вид:

    $$ f_1(x_1, x_2, \dots, x_n) = 0, $$

    $$ f_2(x_1, x_2, \dots, x_n) = 0, $$

    $$ \dots $$

    $$ f_n(x_1, x_2, \dots, x_n) = 0. $$

    Для систем с большим числом переменных одним из первых шагов является отделение отдельных корней. Для двух нелинейных уравнений отделение можно произвести графически, построив на плоскости \( (x_1, x_2) \) графики кривых \( f_1(x_1, x_2) = 0 \) и \( f_2(x_1, x_2) = 0 \). Для большего числа переменных удовлетворительных способов подбора нулевых приближений нет.
    """)

def simple_iteracia_method():
    st.markdown("### 5.2.1. Метод простой итерации")

    st.markdown(r"""
    Пусть имеется система \( \bar{f}(\bar{x}) = 0 \) и нам удалось переписать её в эквивалентной форме:
    """)
    st.latex(r"\bar{x} = \bar{\varphi}(\bar{x}), \tag{17}")
    st.markdown(r"""
    где функции \( \varphi_i \), образующие вектор \( \varphi \), действительно определены и непрерывны в некоторой окрестности изолированного решения \( \bar{x}^* \) системы (16). Например, таким преобразованием может быть (при подходящих \( f_1, \dots, f_n \)) линейное преобразование
    """)
    st.latex(r"\bar{x} = \bar{x} + A \bar{f}(x), \tag{18}")

    st.markdown(r"""
    с несингулярной матрицей \( A \).

    Выбирая для системы (17) некоторое приближение \( \bar{x}_0 \), проводим итерации по формуле
    """)
    st.latex(r"\bar{x}_k = \bar{\varphi}(\bar{x}_{k-1}), \quad k = 1, 2, \dots. \tag{19}")

    st.markdown(r"""
    Если итерации сходятся, то они приводят к решению системы (17).
    """)

    st.markdown(r"""
    Для того чтобы установить условия сходимости этого процесса, дадим некоторые определения. Рассмотрим вектор-функцию
    """)
    st.latex(r"\bar{y} = \varphi(\bar{x}), \tag{20}")
    st.markdown(r"""
    для которой все компоненты вектора \( \varphi = (\varphi_1, \dots, \varphi_n) \) определены и непрерывны в известной области \( G \) действительного \( n \)-мерного пространства \( E_n \). Пусть значения \( \bar{y} = \varphi(\bar{x}) \) при \( \bar{x} \in G \) заполняют некоторую область \( G' \subset E_n \). В этом случае говорят, что уравнение (20) устанавливает отображение в области \( G \). Введем в пространстве \( E_n \) норму \( ||\bar{x}|| \) для вектора \( \bar{x} \). Например, можно выбрать такие нормы:
    """)
    st.latex(r"||\bar{x}|| = \max |x_i|, \quad ||\bar{x}|| = \sqrt{\sum x_i^2}.")
    st.markdown(r"""
    Отображение (20) называется сжимающим в области \( G \), если существует \( 0 \leq q < 1 \), такое, что для любых \( x_1, x_2 \in G \) их образы \( \bar{y}_1 = \varphi(x_1), \bar{y}_2 = \varphi(x_2) \) удовлетворяют условию:
    """)
    st.latex(r"||\bar{y}_1 - \bar{y}_2|| = ||\varphi(\bar{x}_1) - \varphi(\bar{x}_2)|| \leq q ||\bar{x}_1 - \bar{x}_2||. \tag{21}")

    st.markdown(r"""
    Вернемся теперь к итерационному процессу (19).

    #### Теорема 3.
    Пусть область \( G \) замкнута, \( \varphi(\bar{x}) \in G \) для \( \bar{x} \in G \) и отображение (17) является сжимающим в \( G \), т.е.
    """)
    st.latex(r"||\varphi(\bar{x}_1) - \varphi(\bar{x}_2)|| \leq q ||\bar{x}_1 - \bar{x}_2||.")
    st.markdown(r"""
    Тогда, независимо от выбора \( \bar{x}_0 \in G \), процесс (19) сходится, т.е. существует
    """)
    st.latex(r"\bar{x}^* = \lim_{k \to \infty} \bar{x}_k,")
    st.markdown(r"""
    вектор \( \bar{x}^* \) — единственное решение (17) в \( G \) и справедлива оценка нормы
    """)
    st.latex(r"||\bar{x}^* - \bar{x}_k|| \leq \frac{q^k}{1 - q} ||\bar{x}_1 - \bar{x}_0||. \tag{22}")

    st.markdown(r"""
    #### Доказательство.
    Рассмотрим цепочку неравенств:
    """)
    st.latex(r"||\bar{x}_{k+p} - \bar{x}^*|| = ||\bar{x}_{k+p} - \bar{x}_{k+p-1} + \dots + \bar{x}_k - \bar{x}^*||")
    st.latex(r"\leq ||\bar{x}_{k+p} - \bar{x}_{k+p-1}|| + \dots + ||\bar{x}_k - \bar{x}^*||. \tag{23}")
    st.markdown(r"""
    Для каждого слагаемого в правой части имеем
    """)
    st.latex(r"||\bar{x}_{s+1} - \bar{x}_s|| = ||\varphi(\bar{x}_s) - \varphi(\bar{x}_{s-1})|| \leq q ||\bar{x}_s - \bar{x}_{s-1}||.")
    st.markdown(r"""
    Поэтому
    """)
    st.latex(r"||\bar{x}_{k+p} - \bar{x}^*|| \leq ||\bar{x}_1 - \bar{x}_0|| \cdot \left( q^k + q^{k+1} + \dots + q^{k+p-1} \right) = \frac{q^k(1 - q^p)}{1 - q} ||\bar{x}_1 - \bar{x}_0||. \tag{24}")
    st.markdown(r"""
    Так как \( q < 1 \), то для любого \( \varepsilon > 0 \) существует \( N(\varepsilon) > 0 \), что при \( k > N \) и любом \( p > 0 \) будет
    """)
    st.latex(r"||\bar{x}_{k+p} - \bar{x}_k|| \leq \varepsilon.")
    st.markdown(r"""
    Таким образом, для последовательности \( \{x_k\} \) выполнен критерий Коши, поэтому существует
    """)
    st.latex(r"\bar{x}^* = \lim_{k \to \infty} \bar{x}_k,")
    st.markdown(r"""
    и \( \bar{x}^* \in G \) в силу замкнутости \( G \). Вектор \( \bar{x}^* \) — решение уравнения (17), что следует из непрерывности \( \varphi \) после перехода к пределу в (19).

    Покажем, что \( \bar{x}^* \) — единственное решение уравнения (17) в \( G \). Пусть имеется другое решение \( \bar{x}_1^* \) и \( \bar{x}_2^* = \varphi(\bar{x}_2^*) \). Тогда
    """)
    st.latex(r"||\bar{x}_1^* - \bar{x}_2^*|| \leq q ||\bar{x}_1^* - \bar{x}_2^*||.")
    st.markdown(r"""
    Откуда, поскольку \( -q + 1 > 0 \), \( \bar{x}_1^* = \bar{x}_2^* \) или \( \bar{x}_1^* = \bar{x}_2^* \). Из (24) при \( p \to \infty \) получим
    """)
    st.latex(r"||\bar{x}^* - \bar{x}_k|| \leq \frac{q^k}{1 - q} ||\bar{x}_1 - \bar{x}_0||,")
    st.markdown(r"""
    что и требовалось доказать.

    #### Замечание.
    Из (23) имеем:
    """)
    st.latex(r"||\bar{x}_{k+p} - \bar{x}_k|| \leq (q + q^2 + \dots + q^p) ||\bar{x}_k - \bar{x}_{k-1}|| \leq \frac{q}{1-q} ||\bar{x}_k - \bar{x}_{k-1}||.")
    st.markdown(r"""
    При \( p \to \infty \) получим
    """)
    st.latex(r"||\bar{x}^* - \bar{x}_k|| \leq \frac{q}{1 - q} ||\bar{x}_k - \bar{x}_{k-1}||.")
    st.markdown(r"""
    Этим условием надо пользоваться для окончания итераций:
    """)
    st.latex(r"||\bar{x}_k - \bar{x}_{k-1}|| \leq \varepsilon (1-q)/q.")

def metod_newtona():
    st.markdown("""
    ### 5.2.2. Метод Ньютона

    Пусть известно некоторое приближение \( \bar{x}_p \) к корню \( \bar{x}^* \) уравнения \( \bar{f}(\bar{x}) = 0 \). Положим \( \Delta x = \bar{x} - \bar{x}_p \) и определим \( \Delta x^{(p)} \) из условия

    $$ \bar{f}(\bar{x}_p + \Delta x^{(p)}) = 0. $$

    Разложив функцию \( f_k \) в ряды и ограничиваясь первыми дифференциалами (т.е. производя линеаризацию), находим:

    $$ \sum_{i=1}^{n} \frac{\partial f_k (\bar{x}_p)}{\partial x_i} \Delta x_i^{(p)} = -f_k (\bar{x}_p), \quad 1 \leq k \leq n, $$

    где \( \Delta x_i^{(p)} = x_i^* - x_{i,p}. \)

    Мы получили линейную систему относительно \( \{ \Delta x_i^{(p)} \}. \) Решая её, находим эти приращения и определяем новое приближение по формуле:

    $$ \bar{x}_{p+1} = \bar{x}_p + \Delta x^{(p)}. $$

    Этот метод аналогичен методу простой итерации с

    $$ \varphi(\bar{x}) = \bar{x} - \left( \frac{\partial f}{\partial x} \right)^{-1} f(\bar{x}). $$

    Условия сходимости в координатном виде имеют сложный вид, и их редко удается проверить.

    Отметим только следующий результат: в окрестности корня итерации сходятся, если \( \det f'(x) \neq 0 \), причем сходимость квадратичная. 

    Критерий окончания итераций:

    $$ ||\bar{x}_p - \bar{x}_{p+1}|| \leq \varepsilon. $$

    #### Замечание. 
    Иногда матрицу производных вычисляют для сокращения работы, лишь при \( \bar{x}_0 \) и используют потом для всех \( \bar{x}_p \); скорость сходимости при этом может упасть до линейной, если \( \bar{x}_p \) далеко от \( \bar{x}^* \).

    В заключение этого раздела отметим, что метод простой итерации и метод Ньютона можно применять для решения уравнения \( f(z) = 0 \) в комплексной области, сводя его к системе двух уравнений, либо задавая начальное приближение для комплексных \( z \) (для вещественной функции \( f' \)).
    """)