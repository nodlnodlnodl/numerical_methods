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
theme_list = ["Общие правила", "Погрешности", "Общие правила приближения", "Метод Лагранжа"]
theme_list.append("Метод Ньютона")
# Боковое меню с выбором темы
st.sidebar.header("Численные методы")
st.sidebar.markdown("---")
st.sidebar.header("Темы:")
task = st.sidebar.radio("", theme_list)

# Общие правила вычислительной работы
if task == "Общие правила":
    general_rules()

# Источники и классификация погрешности
elif task == "Погрешности":
    inaccuracy()

# Общие правила приближения функций
elif task == "Общие правила приближения":
    general_rules_for_approximating_functions()

# Метод Лагранжа для интерполяции
elif task == "Метод Лагранжа":
    lagrange_for_interpolation()

# Метод Ньютона для интерполяции
elif task == "Метод Ньютона":
    newton_for_interpolation()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Блок для интерактивного ввода кода пользователем
st.header("Интерактивный ввод кода")

# Настройка st_ace для автоматического обновления с начальным комментарием
user_code = st_ace(language='python', theme='monokai', height=300, auto_update=True,
                   value="print('Ваш код работает хорошо')")

# Контейнер для вывода результатов
result_container = st.empty()

if user_code:
    # Создаем строковый буфер для перехвата вывода
    buffer = io.StringIO()
    sys.stdout = buffer  # Перенаправляем stdout на буфер

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
