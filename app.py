from functions import *
from code_editor import *
from derkin_functions import *
from volkov_functions import *
from terentev_functions import *
from roshu_functions import *
from archipkin_functions import *
from zinkin_functions import *

st.set_page_config(page_title="Методическое пособие по численным методам", page_icon=None)

# Основной заголовок приложения
st.title("Численные методы")

# Темы для каждого раздела
section1_themes = ["1. Общие правила вычислительной работы"]
section2_themes = ["2. Источники и классификация погрешностей"]
section3_themes = ["3.1. Общие правила приближения", "3.2.1. Метод Лагранжа", "3.2.2. Метод Ньютона",
                   "3.2.3. Погрешность многочленной аппроксимации", "3.2.4. Трудности приближения многочленом",
                   "3.2.5. Многочлены Чебышева", "3.3 Интерполяция сплайнами", "Тема 3.4", "Тема 3.4.1", "Тема 3.4.2",
                   "Тема 3.4.3", "Тема 3.4.4"]
section4_themes = ["4.1 Численное дифференцирование",
                   "4.1.1. Получение формул дифференцирования на основе многочлена Ньютона",
                   "4.1.2. Метод Рунге—Ромберга повышения точности", "Тема 4.2.", "Тема 4.2.1.", "Тема 4.2.2.",
                   "Тема 4.2.3.", "Тема 4.2.4.", "Тема 4.2.5.", "Тема 4.2.5.1.", "Тема 4.2.5.2.", "Тема 4.2.5.3.",
                   "Тема 4.2.5.4.", "Тема 4.2.6.", "Тема 4.2.6.1.", "Тема 4.2.6.2.", "Тема 4.2.7"]
section5_themes = ["5.1. Уравнения с одним неизвестным", "5.1.1. Метод дихотомии (половинного деления)",
                   "5.1.2. Метод хорд", "5.1.3. Метод простой итерации", "5.1.4. Метод Ньютона",
                   "5.1.5. Метод секущих", "5.1.6. Метод Рыбакова Л.М.", "5.1.7. Вычисление корней многочленов",
                   "5.2. Системы нелинейных уравнений", "5.2.1. Метод простой итерации", "5.2.2. Метод Ньютона"]
section6_themes = ["6.1.1. Треугольные матрицы", "6.1.2. Унитарные матрицы", "6.2. Нормы векторов и матриц",
                   "6.3. Обусловленность линейной системы", "6.4.1. Метод исключения Гаусса",
                   "6.4.2. Метод отражений", "6.4.3. ",
                   "6.4.4. ", "6.4.5. ",
                   "6.5. "]
section7_themes = ["Тема 7.1", "Тема 7.2"]
section8_themes = ["Тема 8.1", "Тема 8.2"]
section9_themes = ["Тема 9.1", "Тема 9.2"]
section10_themes = ["Тема 10.1", "Тема 10.2", "10.3. Устойчивость разностной схемы", "10.4. О выборе норм"]
section11_themes = ["Тема 11.1", "Тема 11.2"]
section12_themes = ["Тема 12.1", "Тема 12.2"]
section13_themes = ["Тема 13.1", "Тема 13.2"]

# Переменные для хранения текущего выбора раздела и темы
selected_section = st.sidebar.radio("Выберите раздел:",
                                    ["1. Общие правила вычислительной работы",
                                     "2. Источники и классификация погрешностей",
                                     "3. Приближение функций", "4. Численное дифференцирование и интегрирование",
                                     "5. Вычисление корней уравнений", "6. Решение задач линейной алгебры",
                                     "7. Алгебраическая проблема собственных значений",
                                     "8. Задача Коши для ОДУ. Методы решения",
                                     "9. Элементарные примеры разностных схем",
                                     "10. Сходимость, аппроксимация, устойчивость разностных схем",
                                     "11. Употребительные численные методы решения ОДУ",
                                     "12. Некоторые методы решения жестких систем ОДУ", "13. Специальные методы"])
selected_theme = None

# В зависимости от выбранного раздела показываем соответствующий expander и выбираем первую тему
if selected_section == "1. Общие правила вычислительной работы":
    selected_theme = "1. Общие правила вычислительной работы"  # Автоматически выбрана первая тема
elif selected_section == "2. Источники и классификация погрешностей":
    selected_theme = "2. Источники и классификация погрешностей"  # Автоматически выбрана первая тема
elif selected_section == "3. Приближение функций":
    with st.sidebar.expander("3. Приближение функций", expanded=True):
        selected_theme = st.radio("Выберите тему:", section3_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "4. Численное дифференцирование и интегрирование":
    with st.sidebar.expander("4. Численное дифференцирование и интегрирование", expanded=True):
        selected_theme = st.radio("Выберите тему:", section4_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "5. Вычисление корней уравнений":
    with st.sidebar.expander("5. Вычисление корней уравнений", expanded=True):
        selected_theme = st.radio("Выберите тему:", section5_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "6. Решение задач линейной алгебры":
    with st.sidebar.expander("6. Решение задач линейной алгебры", expanded=True):
        selected_theme = st.radio("Выберите тему:", section6_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "7. Алгебраическая проблема собственных значений":
    with st.sidebar.expander("7. Алгебраическая проблема собственных значений", expanded=True):
        selected_theme = st.radio("Выберите тему:", section7_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "8. Задача Коши для ОДУ. Методы решения":
    with st.sidebar.expander("8. Задача Коши для ОДУ. Методы решения", expanded=True):
        selected_theme = st.radio("Выберите тему:", section8_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "9. Элементарные примеры разностных схем":
    with st.sidebar.expander("9. Элементарные примеры разностных схем", expanded=True):
        selected_theme = st.radio("Выберите тему:", section9_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "10. Сходимость, аппроксимация, устойчивость разностных схем":
    with st.sidebar.expander("10. Сходимость, аппроксимация, устойчивость разностных схем", expanded=True):
        selected_theme = st.radio("Выберите тему:", section10_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "11. Употребительные численные методы решения ОДУ":
    with st.sidebar.expander("11. Употребительные численные методы решения ОДУ", expanded=True):
        selected_theme = st.radio("Выберите тему:", section11_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "12. Некоторые методы решения жестких систем ОДУ":
    with st.sidebar.expander("12. Некоторые методы решения жестких систем ОДУ", expanded=True):
        selected_theme = st.radio("Выберите тему:", section12_themes, index=0)  # Автоматически выбрана первая тема4
elif selected_section == "13. Специальные методы":
    with st.sidebar.expander("13. Специальные методы", expanded=True):
        selected_theme = st.radio("Выберите тему:", section13_themes, index=0)  # Автоматически выбрана первая тема

# 1. Общие правила вычислительной работы вычислительной работы
if selected_theme == "1. Общие правила вычислительной работы":
    general_rules()

# 2. Источники и классификация погрешностей
elif selected_theme == "2. Источники и классификация погрешностей":
    inaccuracy()

# 3.1. Общие правила приближения функций
elif selected_theme == "3.1. Общие правила приближения":
    general_rules_for_approximating_functions()

# 3.2.1. Метод Лагранжа для интерполяции
elif selected_theme == "3.2.1. Метод Лагранжа":
    lagrange_for_interpolation()
    code_editor()

# 3.2.2. Метод Ньютона для интерполяции
elif selected_theme == "3.2.2. Метод Ньютона":
    newton_for_interpolation()
    code_editor()

# 3.2.3. Погрешность многочленной аппроксимации
elif selected_theme == "3.2.3. Погрешность многочленной аппроксимации":
    error_of_polynomial_interpolation()
    code_editor()

# 3.2.4. Трудности приближения многочленом
elif selected_theme == "3.2.4. Трудности приближения многочленом":
    polynomial_approximation_difficulties()

# 3.2.5. Многочлены Чебышева
elif selected_theme == "3.2.5. Многочлены Чебышева":
    chebyshev_polynomials()
    code_editor()

# 3.3 Интерполяция сплайнами
elif selected_theme == "3.3 Интерполяция сплайнами":
    interpolation_by_splines()
    code_editor()

# 4.1 Численное дифференцирование
elif selected_theme == "4.1 Численное дифференцирование":
    numerical_differentiation()

# 4.1.1. Получение формул дифференцирования на основе многочлена Ньютона
elif selected_theme == "4.1.1. Получение формул дифференцирования на основе многочлена Ньютона":
    numerical_differentiation_newton_polynomial()

# 4.1.2. Метод Рунге—Ромберга повышения точности
elif selected_theme == "4.1.2. Метод Рунге—Ромберга повышения точности":
    runge_romberg_method_of_increasing_accuracy()

# 5.1. Уравнения с одним неизвестным
elif selected_theme == "5.1. Уравнения с одним неизвестным":
    uravnenie_1z()

# 5.1.1. Метод дихотомии (половинного деления)
elif selected_theme == "5.1.1. Метод дихотомии (половинного деления)":
    metod_dihotomii()

# 5.1.2. Метод хорд
elif selected_theme == "5.1.2. Метод хорд":
    metod_hord()

# 5.1.3. Метод простой итерации
elif selected_theme == "5.1.3. Метод простой итерации":
    simple_iterecia()

# 5.1.4. Метод Ньютона
elif selected_theme == "5.1.4. Метод Ньютона":
    newton_method()

# 5.1.5. Метод секущих
elif selected_theme == "5.1.5. Метод секущих":
    metod_sekushchih()

# 5.1.6. Метод Рыбакова Л.М.
elif selected_theme == "5.1.6. Метод Рыбакова Л.М.":
    metod_rybakova()

# 5.1.7. Вычисление корней многочленов
elif selected_theme == "5.1.7. Вычисление корней многочленов":
    korni_mnogochlenov()

# 5.2. Системы нелинейных уравнений
elif selected_theme == "5.2. Системы нелинейных уравнений":
    systemi_nonliniar_uravn()

# 5.2.1. Метод простой итерации
elif selected_theme == "5.2.1. Метод простой итерации":
    simple_iteracia_method()

# 5.2.2. Метод Ньютона
elif selected_theme == "5.2.2. Метод Ньютона":
    metod_newtona()

# 6.1.1. Треугольные матрицы
elif selected_theme == "6.1.1. Треугольные матрицы":
    linear_algebra_triangular_matrix()
    code_editor()

# 6.1.2. Унитарные матрицы
elif selected_theme == "6.1.2. Унитарные матрицы":
    linear_algebra_unitary_matrix()
    code_editor()

# 6.2. Нормы векторов и матриц
elif selected_theme == "6.2. Нормы векторов и матриц":
    vector_matrix_norm()

# 6.4.1 Метод исключения Гаусса
elif selected_theme == "6.4.1. Метод исключения Гаусса":
    metod_gausa()

# 6.4.2 Метод отражений
elif selected_theme == "6.4.2. Метод отражений":
    metod_otrajeniy()

# 10.3 Устойчивость разностной схемы
elif selected_theme == "10.3. Устойчивость разностной схемы":
    ustoichivost_raznostnoi_scheme()

# 10.4. О выборе норм
elif selected_theme == "10.4. О выборе норм":
    about_norm_choose()

else:
    st.write(f"Тут в дальнейшем будет {selected_theme}...")
