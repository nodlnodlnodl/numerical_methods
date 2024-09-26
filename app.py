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
section2_themes = ["Погрешности", "Тема 2.2"]
section3_themes = ["Тема 3.1", "Тема 3.2"]
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
                                    ["Общие правила вычислительной работы", "Источники и классификации разделов",
                                     "Раздел 3", "Раздел 4", "Раздел 5", "Раздел 6", "Раздел 7", "Раздел 8", "Раздел 9",
                                     "Раздел 10", "Раздел 11", "Раздел 12", "Раздел 13"])
selected_theme = None

# В зависимости от выбранного раздела показываем соответствующий expander и выбираем первую тему
if selected_section == "Общие правила вычислительной работы":
    selected_theme = "Общие правила"  # Автоматически выбрана первая тема
elif selected_section == "Источники и классификации погрешностей":
    with st.sidebar.expander("Источники и классификации погрешностей", expanded=True):
        selected_theme = st.radio("Выберите тему:", section2_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "Раздел 3":
    with st.sidebar.expander("Раздел 3", expanded=True):
        selected_theme = st.radio("Выберите тему:", section3_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "Раздел 4":
    with st.sidebar.expander("Раздел 4", expanded=True):
        selected_theme = st.radio("Выберите тему:", section4_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "Раздел 5":
    with st.sidebar.expander("Раздел 5", expanded=True):
        selected_theme = st.radio("Выберите тему:", section5_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "Раздел 6":
    with st.sidebar.expander("Раздел 6", expanded=True):
        selected_theme = st.radio("Выберите тему:", section6_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "Раздел 7":
    with st.sidebar.expander("Раздел 7", expanded=True):
        selected_theme = st.radio("Выберите тему:", section7_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "Раздел 8":
    with st.sidebar.expander("Раздел 8", expanded=True):
        selected_theme = st.radio("Выберите тему:", section8_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "Раздел 9":
    with st.sidebar.expander("Раздел 9", expanded=True):
        selected_theme = st.radio("Выберите тему:", section9_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "Раздел 10":
    with st.sidebar.expander("Раздел 10", expanded=True):
        selected_theme = st.radio("Выберите тему:", section10_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "Раздел 11":
    with st.sidebar.expander("Раздел 11", expanded=True):
        selected_theme = st.radio("Выберите тему:", section11_themes, index=0)  # Автоматически выбрана первая тема
elif selected_section == "Раздел 12":
    with st.sidebar.expander("Раздел 12", expanded=True):
        selected_theme = st.radio("Выберите тему:", section12_themes, index=0)  # Автоматически выбрана первая тема4
elif selected_section == "Раздел 13":
    with st.sidebar.expander("Раздел 13", expanded=True):
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
