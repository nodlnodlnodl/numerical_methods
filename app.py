import streamlit as st
from streamlit_ace import st_ace

from scipy.integrate import odeint
from scipy.optimize import fsolve
import time

import time
import io
import sys
from functions import *
from code_editor import *



# Основной заголовок приложения
st.title("Численные методы")

# Темы для каждого раздела
section1_themes = ["1. Общие правила вычислительной работы"]
section2_themes = ["2. Источники и классификация погрешностей"]
section3_themes = ["3.1. Общие правила приближения", "3.2.1. Метод Лагранжа", "3.2.2. Метод Ньютона", "Тема 3.2.3", "Тема 3.2.4", "Тема 3.2.5", "Тема 3.3", "Тема 3.4", "Тема 3.4.1", "Тема 3.4.2", "Тема 3.4.3", "Тема 3.4.4"]
section4_themes = ["Тема 4.1", "Тема 4.1.1", "Тема 4.1.2.", "Тема 4.2.", "Тема 4.2.1.", "Тема 4.2.2.", "Тема 4.2.3.", "Тема 4.2.4.", "Тема 4.2.5.", "Тема 4.2.5.1.", "Тема 4.2.5.2.", "Тема 4.2.5.3.", "Тема 4.2.5.4.", "Тема 4.2.6.", "Тема 4.2.6.1.", "Тема 4.2.6.2.", "Тема 4.2.7"]
section5_themes = ["Тема 5.1", "Тема 5.2"]
section6_themes = ["Тема 6.1", "Тема 6.2"]
section7_themes = ["Тема 7.1", "Тема 7.2"]
section8_themes = ["Тема 8.1", "Тема 8.2"]
section9_themes = ["Тема 9.1", "Тема 9.2"]
section10_themes = ["Тема 10.1", "Тема 10.2"]
section11_themes = ["Тема 11.1", "Тема 11.2"]
section12_themes = ["Тема 12.1", "Тема 12.2"]
section13_themes = ["Тема 13.1", "Тема 13.2"]

# Переменные для хранения текущего выбора раздела и темы
selected_section = st.sidebar.radio("Выберите раздел:",
                                    ["1. Общие правила вычислительной работы вычислительной работы", "2. Источники и классификация погрешностей",
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
if selected_section == "1. Общие правила вычислительной работы вычислительной работы":
    selected_theme = "1. Общие правила вычислительной работы"  # Автоматически выбрана первая тема
elif selected_section == "2. Источники и классификация погрешностей":
    selected_theme = "2. Источники и классификация погрешностей"  # Автоматически выбрана первая тема
elif selected_section == "3. Приближение функций":
    with st.sidebar.expander("3. Приближение функций", expanded=True):
        selected_theme = st.radio("Выберите тему:", section3_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "4. Численное дифференцирование и интегрирование":
    with st.sidebar.expander("4. Численное дифференцирование и нтегрирование", expanded=True):
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

# Общие правила приближения функций
elif selected_theme == "Общие правила приближения":
    general_rules_for_approximating_functions()

# Метод Лагранжа для интерполяции
elif selected_theme == "Метод Лагранжа":
    lagrange_for_interpolation()
    code_editor()
# Метод Ньютона для интерполяции
elif selected_theme == "Метод Ньютона":
    newton_for_interpolation()
    code_editor()
else:
    st.write(f"мяу {selected_theme}")

