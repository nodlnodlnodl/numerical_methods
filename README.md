# numerical_methods
Численные методы

Создаем дочернюю ветку от main, пушить в main конечно можно но опасно лучше дочернюю

делаем просто черед elif дописываем свой метод сверху комментарий метода и фамилия как у меня сделано
ну то есть 
# Метод Лагранжа для интерполяции Кожаев
elif task == "Метод Лагранжа":
  ...

то что ниже # ----------------------------------------------------------------------------------------------------------------------------------------------------
не трогаем!


когда создаете elif task == "НАЗВАНИЕ":


theme_list = ["Общие правила", "Погрешности", "Общие правила приближения", "Метод Лагранжа"]
добавляет поптом в task = st.sidebar.radio("", theme_list)

чтобы всё хорошо было пишем через theme_list.add("НАЗВАНИЕ")
а не в главный список
