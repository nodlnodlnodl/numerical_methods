import streamlit as st
from streamlit_ace import st_ace

import time
import io
import sys


def code_editor():
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
