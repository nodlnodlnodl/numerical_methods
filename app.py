import streamlit as st
from streamlit_ace import st_ace

from scipy.integrate import odeint
from scipy.optimize import fsolve
import time

import time
import io
import sys
from functions import *

# Основной заголовок приложения
st.title("Численные методы")

# Темы для каждого раздела
section1_themes = ["Общие правила"]
section2_themes = ["Погрешности"]
section3_themes = ["Общие правила приближения", "Метод Лагранжа", "Метод Ньютона"]
section4_themes = ["Тема 4.1", "Тема 4.2"]
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
                                    ["Общие правила вычислительной работы", "Источники и классификации погрешностей",
                                     "Приближение функций", "4. Численное дифференцирование и интегрирование",
                                     "5. Вычисление корней уравнений", "6. Решение задач линейной алгебры",
                                     "7. Алгебраическая проблема собственных значений",
                                     "8. Задача Коши для ОДУ. Методы решения",
                                     "9. Элементарные примеры разностных схем",
                                     "10. Сходимость, аппроксимация, устойчивость разностных схем",
                                     "11. Употребительные численные методы решения ОДУ",
                                     "12. Некоторые методы решения жестких систем ОДУ", "13. Специальные методы"])
selected_theme = None

# В зависимости от выбранного раздела показываем соответствующий expander и выбираем первую тему
if selected_section == "Общие правила вычислительной работы":
    selected_theme = "Общие правила"  # Автоматически выбрана первая тема
elif selected_section == "Источники и классификации погрешностей":
    selected_theme = "Погрешности"  # Автоматически выбрана первая тема
elif selected_section == "Приближение функций":
    with st.sidebar.expander("Приближение функций", expanded=True):
        selected_theme = st.radio("Выберите тему:", section3_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "4. Численное дифференцирование и нтегрирование":
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

# Общие правила вычислительной работы
if selected_theme == "Общие правила":
    general_rules()

# Источники и классификация погрешности
elif selected_theme == "Погрешности":
    inaccuracy()

# Общие правила приближения функций
elif selected_theme == "Общие правила приближения":
    general_rules_for_approximating_functions()

# Метод Лагранжа для интерполяции
elif selected_theme == "Метод Лагранжа":
    lagrange_for_interpolation()

# Метод Ньютона для интерполяции
elif selected_theme == "Метод Ньютона":
    newton_for_interpolation()
else:
    st.write(f"мяу {selected_theme}")

# ----------------------------------8-----------------------------------------------------------------------------------------------------------------
# Блок для интерактивного ввода кода пользователем
st.header("Интерактивный ввод кода")
# Настройка st_ace для автоматического обновления с начальным комментарием
user_code = st_ace(language='python', theme='monokai', height=300, auto_update=True,
                   value="print('Вашкод работает хорошо')")

# Контейнер для вывода результатов
result_container = st.empty()

if user_code:
    # Создаем строковый буф12р для перехвата вывода
    buffer = io.StringIO()
    sys.stdout = buffer  # Перенаправляем stdout на буфе13

    with result_container.container():
        try:
            exec_start_time = time.time()
            exec(user_code)
            exec_end_time = time.time()

            # Получаем содержимое буфера
            output = buffer.getvalue()

            # Отображаем результат выполнения кода и время
            st.write("### Результат выполнения:")
            st.text(output)
            st.write(f"Код выполнен за {exec_end_time - exec_start_time:.6f} секунд")
        except Exception as e:
            st.error(f"Ошибка выполнения кода: {e}")
        finally:
            sys.stdout = sys.__stdout__  # Восстанавливаем стандартный stdout
