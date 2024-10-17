import streamlit as st


def primer():
    # Заголовок
    st.title("10.3 Устойчивость разностной схемы")

    # Обычный текст
    st.write("Пусть для приближенного вычисления решения у дифференциальной задачи")

    # Первая формула
    st.latex(r"L y = f \tag{8}")

    # Обычный текст
    st.write("составлена разностная схема")

    # Формула
    st.latex(r"L_h y^{(h)} = f^{(h)} \tag{9}")

    # Обычный текст
    st.write("""
    которая аппроксимирует задачу (8) на решении у с порядком h^k. 
    Это означает, что невязка
    """)

    # Формула
    st.latex(r"\delta f^{(h)} = L_h [y]_h - f^{(h)}")

    # Обычный текст
    st.write("""
    возникающая при подстановке таблицы [y]_h точного решения в уравнение (9), удовлетворяет оценке вида
    """)

    # Формула
    st.latex(r"\|\delta f^{(h)}\|_{F_h} \leq c_1 h^k, \tag{10}")

    st.write("""
    где c_1 — не зависящая от h константа. Как было показано на примерах ранее, одной аппроксимации для сходимости недостаточно, требуется ещё устойчивость схемы.
    """)

    # Определение 1
    st.subheader("Определение 1")
    st.write("""
    Разностная схема (9) называется устойчивой, если существуют h_0 > 0, \delta > 0, такие, что при любом h \leq h_0 и любом \epsilon^{(h)} \in F_h, 
    выполняется неравенство:
    """)

    # Формула
    st.latex(r"\| \epsilon^{(h)} \|_{F_h} \leq \delta")

    # Обычный текст
    st.write("""
    Разностная задача
    """)

    # Формула
    st.latex(r"L_h z^{(h)} = f^{(h)} + \epsilon^{(h)} \tag{11}")

    # Обычный текст
    st.write("имеет единственное решение z^{(h)} \in U_h, причем")

    # Формула
    st.latex(r"\| z^{(h)} - y^{(h)} \|_{U_h} \leq c \| \epsilon^{(h)} \|_{F_h},")

    st.write("где c — не зависящая от h константа.")

    # Определение 2
    st.subheader("Определение 2")
    st.write("""
    Разностная схема (9) с линейным оператором L_h называется устойчивой, если существует h_0 > 0, такое, что при любом h \leq h_0 уравнение (9) имеет единственное решение
    """)

    # Формула
    st.latex(r"\| y^{(h)} \|_{U_h} \leq c \| f^{(h)} \|_{F_h}, \tag{12}")

    st.write("где c — не зависящая от h константа.")

    # Теорема
    st.subheader("Теорема")
    st.write("Для линейных операторов L_h определения 1 и 2 устойчивости схемы равносильны.")

    # Доказательство
    st.subheader("Доказательство")
    st.write(
        "Пусть имеет место устойчивость схемы (9) в смысле определения 2. Покажем, что схема (9) устойчива в смысле определения 1.")

    # Формула
    st.latex(r"L_h z^{(h)} = f^{(h)} + \epsilon^{(h)}")

    st.write("""
    с \epsilon^{(h)} \in F_h. По условиям теоремы (определение 2) эта задача имеет единственное решение при любом рассматриваемом h \leq h_0 и произвольных f^{(h)}, \epsilon^{(h)} \in F_h.
    """)

    # Формула
    st.latex(r"L_h (z^{(h)} - y^{(h)}) = \epsilon^{(h)}")

    st.write("""
    В силу (13) получим
    """)

    # Формула
    st.latex(r"\| z^{(h)} - y^{(h)} \|_{U_h} \leq c \| \epsilon^{(h)} \|_{F_h}, \tag{13}")

    st.write("т.е. задача (9) устойчива в смысле определения 1.")

    st.write("""
    Пусть теперь задача (9) устойчива в смысле определения 1. Тогда при некоторых h_0 > 0, \delta > 0 и при произвольных h \leq h_0, \epsilon^{(h)} \in F_h существует единственное решение уравнений
    """)

    # Формула
    st.latex(r"L_h z^{(h)} = f^{(h)} + \epsilon^{(h)} \tag{14}")

    st.write("и")

    # Формула
    st.latex(r"L_h y^{(h)} = f^{(h)}.")

    st.write("Положим W^{(h)} = z^{(h)} - y^{(h)}, тогда из предыдущих равенств получим")

    # Формула
    st.latex(r"L_h W^{(h)} = \epsilon^{(h)}")

    st.write("""
    и
    """)

    # Формула
    st.latex(r"\| W^{(h)} \|_{U_h} \leq c \| \epsilon^{(h)} \|_{F_h}.")

    st.write("""
    Таким образом, мы получили, что при любом h \leq h_0 задача (9) имеет единственное решение, и оно удовлетворяет оценке
    """)

    # Формула
    st.latex(r"\| y^{(h)} \|_{U_h} \leq c \| f^{(h)} \|_{F_h},")

    st.write("""
    Однако в таком случае уравнение L_h y^{(h)} = f^{(h)} будет иметь единственное решение при любом f^{(h)} \in F_h, и оно будет удовлетворять оценке
    """)

    # Формула
    st.latex(r"\| y^{(h)} \|_{U_h} \leq c \| f^{(h)} \|_{F_h}.")

    st.write("В самом деле, пусть \| f^{(h)} \|_{F_h} > \delta, положим")

    # Формула
    st.latex(r"f^{(h)} = \frac{\delta}{2} \frac{f^{(h)}}{\| f^{(h)} \|_{F_h}}.")

    st.write("Тогда для")

    # Формула
    st.latex(r"y^{(h)}")

    st.write("получим уравнение (в силу линейности L_h)")

    # Формула
    st.latex(r"L_h y^{(h)} = f^{(h)}, ")

    st.write("причем, очевидно,")

    # Формула
    st.latex(r"\| f^{(h)} \|_{F_h} = \frac{\delta}{2} < \delta.")

    st.write("""
    Поэтому это уравнение однозначно разрешимо, и
    """)

    # Формула
    st.latex(r"\| y^{(h)} \|_{U_h} \leq c \| f^{(h)} \|_{F_h}.")

    st.write("""
    Отсюда следует однозначная разрешимость (9) при любом f^{(h)} \in F_h и оценка (13), что и требовалось доказать.
    """)

    st.subheader("Теорема о сходимости")
    st.write("""
    Теперь мы имеем возможность, введя определение устойчивости, доказать важнейшую теорему в теории разностных схем, устанавливающую связь между сходимостью и аппроксимацией и устойчивостью.
    """)

    # Продолжение текста и формулы
    st.write(
        "Пусть разностная схема L_h y^{(h)} = f^{(h)} аппроксимирует задачу Ly = f на решении y с порядком h^k и устойчива.")

    st.latex(r"\|[y]_{h} - y^{(h)}\|_{U_h} \leq (c_1 c_2) h^k,")

    st.write("где c_1, c_2 — константы, входящие в оценку для аппроксимации и устойчивости.")

    # Доказательство
    st.subheader("Доказательство")
    st.write("""
    Так как разностная схема аппроксимирует дифференциальную задачу, то невязка
    """)

    # Формула
    st.latex(r"\delta f^{(h)} = L_h [y]_h - f^{(h)} \tag{15}")

    st.write(
        "которая возникает при подстановке таблицы [y]_h точного решения дифференциальной задачи в разностную, удовлетворяет условию")

    # Формула
    st.latex(r"\|\delta f^{(h)}\|_{F_h} \leq c_1 h^k.")

    st.write(
        "Очевидно, что при заданном \delta > 0 всегда можно указать шаг h_0, что при h \leq h_0 будет выполнено условие")

    # Формула
    st.latex(r"\|\delta f^{(h)}\|_{F_h} \leq \delta.")

    st.write("""
    Так как схема устойчива, то при h \leq h_0 разностная задача имеет одно и только одно решение [y]_h и
    """)

    # Финальная формула
    st.latex(r"\|[y]_{h} - y^{(h)}\|_{U_h} \leq c \|\delta f^{(h)}\|_{F_h}.")

    st.write("Объединяя эти оценки, находим")

    st.latex(r"\|[y]_{h} - y^{(h)}\|_{U_h} \leq (c_1 c_2) h^k,")

    st.write("что и требовалось доказать.")

    # Добавляем последние два абзаца

    st.write("""
    Таким образом, доказательство сходимости решения разностной задачи к решению дифференциальной задачи сведено нами к проверке аппроксимации и устойчивости разностной задачи. 
    Отметим, что проверка аппроксимации обычно проводится гораздо легче, чем проверка устойчивости, для которой, если исходить из ее определения, надо доказать однозначную разрешимость возмущенной задачи 
    и получить оценку отклонения решений исходной и возмущенной задач. Даже для простых уравнений – это довольно сложная процедура.
    """)

    st.write("""
    Замечание. Схема доказательства сходимости решения задачи L_h y^{(h)} = f путем проверки аппроксимации и устойчивости носит общий характер. Под Ly = f можно понимать любое функциональное уравнение, 
    а не только ОДУ. Оно используется лишь для конструирования разностной задачи L_h y^{(h)} = f^{(h)}.
    """)
