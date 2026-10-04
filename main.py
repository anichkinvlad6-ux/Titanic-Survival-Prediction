import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ==========================================
# 1. Загрузка датасета
# ==========================================

data = pd.read_csv("Titanic-Dataset.csv")


# ==========================================
# 2. Выбираем признаки
# ==========================================

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = data[features]
y = data["Survived"]


# ==========================================
# 3. Преобразуем текстовые данные
# ==========================================

X = pd.get_dummies(
    X,
    columns=["Sex", "Embarked"],
    dtype=float
)


# ==========================================
# 4. Заполняем пропуски
# ==========================================

X = X.fillna(X.median())


# Запоминаем названия столбцов
model_columns = X.columns


# ==========================================
# 5. Разделяем данные
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. Нормализация данных
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================
# 7. Создаём нейронную сеть
# ==========================================

model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42,
    early_stopping=True
)


# ==========================================
# 8. Обучение
# ==========================================

print("================================")
print("Обучение нейронной сети...")
print("================================")

model.fit(X_train, y_train)

print("Обучение завершено!")


# ==========================================
# 9. Проверяем точность
# ==========================================

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print()
print("Точность нейронной сети:",
      round(accuracy * 100, 2), "%")


# ==========================================
# 10. Ввод пассажира
# ==========================================

print()
print("================================")
print("Введите данные пассажира")
print("================================")

pclass = int(input("Класс (1/2/3): "))

sex = input(
    "Пол (male/female): "
).lower()

age = float(input("Возраст: "))

sibsp = int(
    input("Братья/сёстры/супруги (SibSp): ")
)

parch = int(
    input("Родители/дети (Parch): ")
)

fare = float(
    input("Стоимость билета: ")
)

embarked = input(
    "Порт посадки (C/Q/S): "
).upper()


# ==========================================
# 11. Создаём данные пассажира
# ==========================================

new_passenger = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked]
})


# ==========================================
# 12. Преобразуем данные так же,
#     как тренировочные
# ==========================================

new_passenger = pd.get_dummies(
    new_passenger,
    columns=["Sex", "Embarked"],
    dtype=float
)

new_passenger = new_passenger.reindex(
    columns=model_columns,
    fill_value=0
)

new_passenger = new_passenger.fillna(
    X.mean(axis=0)
)


# ==========================================
# 13. Нормализация
# ==========================================

new_passenger = scaler.transform(
    new_passenger
)


# ==========================================
# 14. Предсказание
# ==========================================

result = model.predict(new_passenger)

probability = model.predict_proba(
    new_passenger
)[0]


# ==========================================
# 15. Результат
# ==========================================

print()
print("================================")

if result[0] == 1:
    print("ПРЕДСКАЗАНИЕ: ВЫЖИЛ")
else:
    print("ПРЕДСКАЗАНИЕ: ПОГИБ")

print()

print(
    "Вероятность погибнуть:",
    round(probability[0] * 100, 2),
    "%"
)

print(
    "Вероятность выжить:",
    round(probability[1] * 100, 2),
    "%"
)

print("================================")