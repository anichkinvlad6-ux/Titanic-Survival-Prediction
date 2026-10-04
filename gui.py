import math
import tkinter as tk
import tkinter.font as tkfont
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image
import os
import pandas as pd
import threading

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import ollama


# ============================================================
# НАСТРОЙКИ
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "Titanic-Dataset.csv")
TITANIC_IMAGE = os.path.join(BASE_DIR, "titanic_ship.png")
SEAL_IMAGE = os.path.join(BASE_DIR, "seal_signature.png")
OLLAMA_MODEL = "qwen2.5:7b"

# В датасете Titanic Fare указана в фунтах.
# Для интерфейса вводим стоимость в долларах и переводим её
# обратно в фунты перед передачей в нейронную сеть.
GBP_TO_USD_1912 = 4.8665


# ============================================================
# ЦВЕТА И ШРИФТЫ (чёрные чернила на старой бумаге)
# ============================================================

DESK = "#CFCABB"
PAPER = "#F6F3EA"
ENTRY_BG = "#FBF9F2"

INK = "#141414"
INK_HOVER = "#3A3A3A"
INK_BLUE = "#1F2F7A"   # синяя ручка для «вписанных» данных
MUTED = "#4A4A4A"
LINE = "#7A7A7A"
BAR_BG = "#CFC9B8"

GREEN = "#2F5D3A"
RED = "#8B1E26"

SERIF = "Times New Roman"

# Рукописные шрифты (первый найденный будет использован) и поправка размера.
# Можно положить свой файл handwriting.ttf рядом со скриптом —
# например Marck Script, Caveat или Bad Script с Google Fonts (есть кириллица).
# Масштаб для файла handwriting.ttf (Great Vibes мелкий — увеличиваем).
# Если текст слишком крупный/мелкий, меняйте это число.
HANDWRITING_SCALE = 1.35

HAND_CANDIDATES = [
    ("Segoe Script", 0.80),
    ("Monotype Corsiva", 1.15),
    ("Ink Free", 1.00),
    ("Lucida Handwriting", 0.80),
    ("Segoe Print", 0.85),
    ("Comic Sans MS", 0.90),
]


# ============================================================
# ЗАГРУЗКА DATASET
# ============================================================

data = pd.read_csv(DATA_FILE)

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = data[features].copy()
y = data["Survived"]

X = pd.get_dummies(
    X,
    columns=["Sex", "Embarked"],
    dtype=float
)

X = X.fillna(X.median())


# ============================================================
# TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# SCALER
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# НЕЙРОННАЯ СЕТЬ
# ============================================================

model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42,
    early_stopping=True
)

model.fit(
    X_train_scaled,
    y_train
)

accuracy = model.score(
    X_test_scaled,
    y_test
)


# ============================================================
# QWEN
# ============================================================

def ask_qwen(pclass, sex, age, sibsp, parch, fare, embarked, survived_probability, died_probability):
    prediction = 'выжил' if survived_probability > died_probability else 'погиб'
    sex_ru = {'male':'мужчина','female':'женщина'}.get(sex, sex)
    embarked_ru = {'S':'Саутгемптон','C':'Шербур','Q':'Квинстаун'}.get(embarked, embarked)
    prompt = f"""Проанализируй результат прогнозирования выживания пассажира Титаника.

Класс: {pclass}
Пол: {sex_ru}
Возраст: {age}
Братья, сёстры или супруги: {sibsp}
Родители или дети: {parch}
Стоимость билета: {fare}
Порт посадки: {embarked_ru}

Вероятность выживания: {survived_probability:.2f}%
Вероятность гибели: {died_probability:.2f}%
Прогноз: пассажир скорее всего {prediction}.

Объясни результат в 3-5 коротких предложениях и назови 2-4 важных фактора.
ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. Не используй китайские иероглифы. Не изменяй вероятности."""
    fallback = (f'Нейронная сеть оценивает вероятность выживания в {survived_probability:.2f}%, '
                f'а вероятность гибели — {died_probability:.2f}%. Прогноз модели: пассажир скорее всего {prediction}. '
                'На результат влияют класс пассажира, пол, возраст, стоимость билета и семейный состав.')
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {'role':'system','content':'Ты русскоязычный аналитик Titanic. Отвечай только на русском языке. Китайские иероглифы и китайский язык запрещены. Ответ короткий.'},
            {'role':'user','content':prompt}
        ], options={'temperature':0.0,'top_p':0.8})
        answer = response['message']['content'].strip()
        answer = ''.join(c for c in answer if not ('\u3400' <= c <= '\u4DBF' or '\u4E00' <= c <= '\u9FFF' or '\uF900' <= c <= '\uFAFF')).strip()
        return answer if answer else fallback
    except Exception:
        return fallback


# ============================================================
# ГЛАВНОЕ ОКНО
# ============================================================

ctk.set_appearance_mode("light")

root = ctk.CTk()
root.title("R.M.S. Титаник — Пассажирский договор-билет")
root.geometry("1280x880")
root.minsize(1150, 800)
root.configure(fg_color=DESK)


# ============================================================
# РУКОПИСНЫЙ ШРИФТ
# ============================================================

def pick_hand_font():
    # 1) свой файл handwriting.ttf / handwriting.otf рядом со скриптом
    for fn in ("handwriting.ttf", "handwriting.otf"):
        path = os.path.join(BASE_DIR, fn)
        if os.path.exists(path):
            try:
                from PIL import ImageFont
                ctk.FontManager.load_font(path)
                return ImageFont.truetype(path, 20).getname()[0], HANDWRITING_SCALE
            except Exception:
                pass

    # 2) системные рукописные шрифты
    families = set(tkfont.families())
    for name, scale in HAND_CANDIDATES:
        if name in families:
            return name, scale

    return SERIF, 1.0


HAND, HAND_SCALE = pick_hand_font()


def F(size, weight="normal", slant="roman", family=None, underline=False):
    return ctk.CTkFont(
        family=family or HAND,
        size=max(8, round(size * HAND_SCALE)),
        weight=weight,
        slant=slant,
        underline=underline
    )


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОФОРМЛЕНИЯ
# ============================================================

def draw_arc_text(canvas, text, cx, cy, radius, font, spacing=8, fill=INK):
    """Рисует текст по дуге окружности (выпуклость вверх)."""
    widths = [font.measure(ch) + spacing for ch in text]
    theta = -(sum(widths) / radius) / 2

    for ch, w in zip(text, widths):
        step = w / radius
        mid = theta + step / 2
        x = cx + radius * math.sin(mid)
        y = cy - radius * math.cos(mid)
        canvas.create_text(
            x, y,
            text=ch,
            font=font,
            fill=fill,
            angle=-math.degrees(mid)
        )
        theta += step


def make_border(parent, side):
    """Боковая рамка из «бусинок», как на старом билете."""
    c = tk.Canvas(
        parent,
        width=22,
        bg=PAPER,
        highlightthickness=0
    )
    c.pack(side=side, fill="y", padx=(4, 4))

    def redraw(event):
        c.delete("all")
        for yy in range(6, event.height - 10, 15):
            c.create_oval(4, yy, 17, yy + 12, outline=INK, width=2)

    c.bind("<Configure>", redraw)
    return c


def fill_slot(parent, var, width):
    """Вписанное «от руки» слово с линией под ним."""
    box = ctk.CTkFrame(parent, fg_color="transparent")
    ctk.CTkLabel(
        box,
        textvariable=var,
        width=width,
        font=F(16, "bold"),
        text_color=INK_BLUE
    ).pack()
    ctk.CTkFrame(box, height=1, fg_color=INK).pack(fill="x")
    box.pack(side="left", padx=4)


def printed(parent, text):
    ctk.CTkLabel(
        parent,
        text=text,
        font=F(13),
        text_color=INK
    ).pack(side="left", padx=(6, 2), pady=(0, 2))


# ============================================================
# ГЛАВНЫЙ БИЛЕТ
# ============================================================

ticket = ctk.CTkFrame(
    root,
    fg_color=PAPER,
    corner_radius=0,
    border_width=1,
    border_color=INK
)
ticket.pack(fill="both", expand=True, padx=20, pady=16)

make_border(ticket, "left")
make_border(ticket, "right")

content = ctk.CTkFrame(ticket, fg_color="transparent")
content.pack(side="left", fill="both", expand=True)


# ============================================================
# HEADER
# ============================================================

header = ctk.CTkFrame(content, fg_color="transparent")
header.pack(fill="x", padx=14, pady=(6, 0))

header.grid_columnconfigure(0, weight=1, uniform="h")
header.grid_columnconfigure(1, weight=0)
header.grid_columnconfigure(2, weight=1, uniform="h")


# ---------- Левый блок ----------

header_left = ctk.CTkFrame(header, fg_color="transparent")
header_left.grid(row=0, column=0, sticky="nw")

ctk.CTkLabel(
    header_left,
    text="Мужская койка.................\n"
         "Женская койка.................\n"
         "Семейная койка...............",
    font=F(11),
    text_color=INK,
    justify="left"
).pack(anchor="w", pady=(0, 10))

ctk.CTkLabel(
    header_left,
    text="ИСМЭЙ, ИМРИ и К°",
    font=F(16, "bold"),
    text_color=INK
).pack(anchor="w")

ctk.CTkLabel(
    header_left,
    text="Джеймс-стрит, 30, Ливерпуль\nи Кэнют-роуд, Саутгемптон",
    font=F(11),
    text_color=INK,
    justify="left"
).pack(anchor="w")

ctk.CTkLabel(
    header_left,
    text="Агент в Париже —",
    font=F(12),
    text_color=INK
).pack(anchor="w", pady=(6, 0))

ctk.CTkLabel(
    header_left,
    text="Никола Мартен, рю Скриб, 9.",
    font=F(12, "bold"),
    text_color=INK
).pack(anchor="w")


# ---------- Центр: название дугой + корабль ----------

header_center = ctk.CTkFrame(header, fg_color="transparent")
header_center.grid(row=0, column=1, padx=20)

CW, CH, RADIUS = 560, 132, 1100

arc_font = tkfont.Font(
    family=HAND,
    size=max(12, round(24 * HAND_SCALE)),
    weight="bold"
)

title_canvas = tk.Canvas(
    header_center,
    width=CW,
    height=CH,
    bg=PAPER,
    highlightthickness=0
)
title_canvas.pack()

draw_arc_text(
    title_canvas,
    "УАЙТ СТАР ЛАЙН",
    cx=CW / 2,
    cy=40 + RADIUS,
    radius=RADIUS,
    font=arc_font,
    spacing=10
)

title_canvas.create_text(
    CW / 2, 112,
    text="Королевские почтовые пароходы Соединённых Штатов.",
    font=(HAND, max(9, round(12 * HAND_SCALE)), "bold"),
    fill=INK
)
title_canvas.create_line(
    CW / 2 - 200, 128,
    CW / 2 + 200, 128,
    fill=INK
)

if os.path.exists(TITANIC_IMAGE):
    titanic_image = ctk.CTkImage(
        light_image=Image.open(TITANIC_IMAGE),
        dark_image=Image.open(TITANIC_IMAGE),
        size=(200, 147)
    )
    ctk.CTkLabel(
        header_center,
        text="",
        image=titanic_image
    ).pack(pady=(2, 0))


# ---------- Правый блок ----------

header_right = ctk.CTkFrame(header, fg_color="transparent")
header_right.grid(row=0, column=2, sticky="ne")

ctk.CTkLabel(
    header_right,
    text="УАЙТ СТАР ЛАЙН,",
    font=F(16, "bold"),
    text_color=INK
).pack(anchor="e")

ctk.CTkLabel(
    header_right,
    text="Генуя — Виа алла Нунциата, 18\n"
         "Неаполь — Пьяцца делла Борса, 21\n"
         "Бостон — Стейт-стрит, 84\n"
         "Нью-Йорк — Бродвей, 9\n"
         "Квебек — Далхузи-стрит, 53\n"
         "Монреаль — Нотр-Дам-стрит Вест, 118",
    font=F(11),
    text_color=INK,
    justify="right"
).pack(anchor="e", pady=(2, 4))

ctk.CTkLabel(
    header_right,
    text="Джеймс Скотт и К°, агенты,\nКвинстаун.",
    font=F(12, "bold"),
    text_color=INK,
    justify="right"
).pack(anchor="e")


# ============================================================
# ДВОЙНАЯ ЛИНИЯ
# ============================================================

ctk.CTkFrame(content, height=1, fg_color=INK).pack(fill="x", padx=24, pady=(6, 2))
ctk.CTkFrame(content, height=3, fg_color=INK).pack(fill="x", padx=24, pady=(0, 4))


# ============================================================
# КОНТРАКТ
# ============================================================

ctk.CTkLabel(
    content,
    text="Океанская пароходная компания, Лимитед, Великобритания.",
    font=F(15, "bold"),
    text_color=INK
).pack()

contract_title = ctk.CTkLabel(
    content,
    text="ТРЕТИЙ КЛАСС (общая палуба) — пассажирский договор-билет.",
    font=F(18, "bold"),
    text_color=INK
)
contract_title.pack(pady=(2, 0))

ctk.CTkLabel(
    content,
    text="(Не передаётся другому лицу.)",
    font=F(10),
    text_color=INK
).pack()


# Переменные «вписанных» полей контракта

slot_ship = ctk.StringVar(value="Р.М.С. «Титаник»")
slot_tons = ctk.StringVar(value="45 000")
slot_port = ctk.StringVar(value="Саутгемптон")
slot_dest = ctk.StringVar(value="Нью-Йорк")
slot_day = ctk.StringVar(value="10")
slot_month = ctk.StringVar(value="апреля")


line1 = ctk.CTkFrame(content, fg_color="transparent")
line1.pack(pady=(8, 0))

printed(line1, "Судно")
fill_slot(line1, slot_ship, 230)
printed(line1, "регистровым тоннажем")
fill_slot(line1, slot_tons, 100)
printed(line1, "тонн,")


line2 = ctk.CTkFrame(content, fg_color="transparent")
line2.pack(pady=(4, 8))

printed(line2, "принимает пассажиров в порту")
fill_slot(line2, slot_port, 150)
printed(line2, "для рейса в")
fill_slot(line2, slot_dest, 120)
printed(line2, "отправление")
fill_slot(line2, slot_day, 36)
printed(line2, "числа,")
fill_slot(line2, slot_month, 90)
printed(line2, "1912 года.")


# ============================================================
# ОСНОВНАЯ ЧАСТЬ
# ============================================================

body = ctk.CTkFrame(content, fg_color="transparent")
body.pack(fill="both", expand=True, padx=24, pady=(0, 4))

body.grid_columnconfigure(0, weight=1, uniform="b")
body.grid_columnconfigure(1, weight=1, uniform="b")
body.grid_rowconfigure(0, weight=1)


# ============================================================
# ЛЕВАЯ ЧАСТЬ — ТАБЛИЦА ПАССАЖИРА + ПЕЧАТЬ И ПОДПИСЬ
# ============================================================

left_col = ctk.CTkFrame(body, fg_color="transparent")
left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 18))

class_var = ctk.StringVar(value="3")
sex_var = ctk.StringVar(value="Мужчина")
embarked_var = ctk.StringVar(value="Саутгемптон")


# Таблица как на оригинальном билете: линии и заголовки колонок.
# Занимает ровно столько места, сколько нужно (без чёрной пустоты).
table = ctk.CTkFrame(left_col, fg_color=INK, corner_radius=0)
table.pack(fill="x")

for c in range(4):
    table.grid_columnconfigure(c, weight=1, uniform="f")

LAST_ROW = 6


def tcell(row, col, span, bg):
    cell = ctk.CTkFrame(table, fg_color=bg, corner_radius=0)
    cell.grid(
        row=row,
        column=col,
        columnspan=span,
        sticky="nsew",
        padx=(1, 1 if col + span == 4 else 0),
        pady=(1, 1 if row == LAST_ROW else 0)
    )
    return cell


def head_cell(text, row, col, span=1):
    cell = tcell(row, col, span, PAPER)
    ctk.CTkLabel(
        cell,
        text=text,
        font=F(11, "bold"),
        text_color=MUTED
    ).pack(anchor="w", padx=8, pady=(3, 1))


def input_cell(row, col, span, factory):
    cell = tcell(row, col, span, ENTRY_BG)
    widget = factory(cell)
    widget.pack(fill="x", padx=4, pady=3)
    return widget


def t_entry(parent, default="", placeholder=""):
    e = ctk.CTkEntry(
        parent,
        height=32,
        corner_radius=0,
        border_width=0,
        fg_color=ENTRY_BG,
        text_color=INK_BLUE,
        placeholder_text=placeholder,
        font=F(16)
    )
    if default:
        e.insert(0, default)
    return e


def t_combo(parent, var, values, command=None):
    return ctk.CTkComboBox(
        parent,
        variable=var,
        values=values,
        height=32,
        corner_radius=0,
        border_width=0,
        fg_color=ENTRY_BG,
        button_color=BAR_BG,          # светлая кнопка — стрелка видна
        button_hover_color="#B9B29C",
        text_color=INK_BLUE,
        dropdown_fg_color=ENTRY_BG,
        dropdown_text_color=INK,
        dropdown_hover_color=BAR_BG,
        font=F(16),
        dropdown_font=F(14),
        state="readonly",
        command=command
    )


# Строка 1: имя | возраст | класс
head_cell("Фамилия и имя", 0, 0, 2)
head_cell("Возраст", 0, 2)
head_cell("Класс", 0, 3)

name_entry = input_cell(1, 0, 2, lambda p: t_entry(p, placeholder="Например: Иванов Иван"))
age_entry = input_cell(1, 2, 1, lambda p: t_entry(p, "30"))
class_box = input_cell(
    1, 3, 1,
    lambda p: t_combo(p, class_var, ["1", "2", "3"],
                      command=lambda v: update_ticket())
)

# Строка 2: пол | порт | стоимость
head_cell("Пол", 2, 0)
head_cell("Порт посадки", 2, 1, 2)
head_cell("Стоимость билета, $", 2, 3)

sex_box = input_cell(3, 0, 1, lambda p: t_combo(p, sex_var, ["Мужчина", "Женщина"]))
embarked_box = input_cell(
    3, 1, 2,
    lambda p: t_combo(p, embarked_var, ["Саутгемптон", "Шербур", "Квинстаун"],
                      command=lambda v: update_ticket())
)
fare_entry = input_cell(3, 3, 1, lambda p: t_entry(p, "30.00"))

# Строка 3: родственники
head_cell("Братья, сёстры и супруги", 4, 0, 2)
head_cell("Родители и дети", 4, 2, 2)

sibsp_entry = input_cell(5, 0, 2, lambda p: t_entry(p, "0"))
parch_entry = input_cell(5, 2, 2, lambda p: t_entry(p, "0"))

# Строка кнопок
buttons_cell = tcell(LAST_ROW, 0, 4, PAPER)
buttons = ctk.CTkFrame(buttons_cell, fg_color="transparent")
buttons.pack(fill="x", padx=8, pady=8)


# Печать компании и подпись агента (под таблицей) + точность нейросети справа
seal_row = ctk.CTkFrame(left_col, fg_color="transparent")
seal_row.pack(fill="x", pady=(8, 0))

if os.path.exists(SEAL_IMAGE):
    seal_pil = Image.open(SEAL_IMAGE)
    seal_w = 340
    seal_h = round(seal_w * seal_pil.height / seal_pil.width)
    seal_image = ctk.CTkImage(
        light_image=seal_pil,
        dark_image=seal_pil,
        size=(seal_w, seal_h)
    )
    ctk.CTkLabel(
        seal_row,
        text="",
        image=seal_image
    ).pack(side="left")

accuracy_block = ctk.CTkFrame(seal_row, fg_color="transparent")
accuracy_block.pack(side="left", padx=(14, 0), pady=(0, 6))

ctk.CTkLabel(
    accuracy_block,
    text="Точность прогноза нейросети",
    font=F(14, "bold"),
    text_color=INK
).pack(anchor="w")

ctk.CTkLabel(
    accuracy_block,
    text=f"{accuracy * 100:.2f}%",
    font=F(30, "bold"),
    text_color=INK_BLUE
).pack(anchor="w", pady=(0, 0))

ctk.CTkLabel(
    accuracy_block,
    text=f"сеть {X.shape[1]} → 32 → 16 → 2",
    font=F(10),
    text_color=MUTED
).pack(anchor="w")


# Маленькая таблица «о модели» справа — в стиле билета (линии, как у таблицы пассажира)
model_info = ctk.CTkFrame(seal_row, fg_color=INK, corner_radius=0)
model_info.pack(side="right", padx=(10, 4), pady=(14, 14))

MODEL_ROWS = [
    ("Обучено на", f"{len(X_train)} чел."),
    ("Проверено на", f"{len(X_test)} чел."),
    ("Скрытые слои", "32 · 16"),
]

for i, (k, v) in enumerate(MODEL_ROWS):
    row = ctk.CTkFrame(model_info, fg_color=PAPER, corner_radius=0)
    row.pack(fill="x", padx=1, pady=(1, 1 if i == len(MODEL_ROWS) - 1 else 0))

    ctk.CTkLabel(
        row,
        text=k,
        font=F(12),
        text_color=MUTED
    ).pack(side="left", padx=(8, 14), pady=1)

    ctk.CTkLabel(
        row,
        text=v,
        font=F(13, "bold"),
        text_color=INK_BLUE
    ).pack(side="right", padx=(0, 8))


# ============================================================
# ОБНОВЛЕНИЕ НАДПИСЕЙ БИЛЕТА (класс, порт, дата)
# ============================================================

CLASS_TITLES = {
    "1": "ПЕРВЫЙ КЛАСС",
    "2": "ВТОРОЙ КЛАСС",
    "3": "ТРЕТИЙ КЛАСС (общая палуба)"
}

PORT_INFO = {
    "Саутгемптон": ("Саутгемптон", "10"),
    "Шербур": ("Шербур", "10"),
    "Квинстаун": ("Квинстаун", "11")
}


def update_ticket():
    contract_title.configure(
        text=f"{CLASS_TITLES.get(class_var.get(), 'ТРЕТИЙ КЛАСС (общая палуба)')}"
             f" — пассажирский договор-билет."
    )

    port_name, day = PORT_INFO.get(embarked_var.get(), ("Саутгемптон", "10"))
    slot_port.set(port_name)
    slot_day.set(day)


# ============================================================
# ОЧИСТКА
# ============================================================

def clear_form():

    name_entry.delete(0, "end")

    class_var.set("3")
    sex_var.set("Мужчина")
    embarked_var.set("Саутгемптон")

    age_entry.delete(0, "end")
    age_entry.insert(0, "30")

    sibsp_entry.delete(0, "end")
    sibsp_entry.insert(0, "0")

    parch_entry.delete(0, "end")
    parch_entry.insert(0, "0")

    fare_entry.delete(0, "end")
    fare_entry.insert(0, "30.00")

    update_ticket()

    result_label.configure(text="Ожидание", text_color=INK)
    stamp.configure(border_color=INK)

    probability_label.configure(text="Введите данные пассажира")

    survival_bar.set(0)

    qwen_text.configure(state="normal")
    qwen_text.delete("1.0", "end")
    qwen_text.insert("1.0", "Здесь появится анализ нейронной сети.")
    qwen_text.configure(state="disabled")


# ============================================================
# ПРЕДСКАЗАНИЕ
# ============================================================

def predict():

    try:

        pclass = int(class_var.get())

        sex_display = sex_var.get()
        sex = {
            "Мужчина": "male",
            "Женщина": "female"
        }[sex_display]

        age = float(age_entry.get())

        sibsp = int(sibsp_entry.get())

        parch = int(parch_entry.get())

        fare_usd = float(fare_entry.get())

        if fare_usd < 0:
            raise ValueError

        # Нейросеть обучена на Fare из исходного датасета (фунты).
        fare = fare_usd / GBP_TO_USD_1912

        port_display = embarked_var.get()
        embarked = {
            "Саутгемптон": "S",
            "Шербур": "C",
            "Квинстаун": "Q"
        }[port_display]

    except (ValueError, KeyError):

        messagebox.showerror(
            "Ошибка",
            "Проверьте введённые данные."
        )

        return


    passenger = pd.DataFrame([{

        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked

    }])


    passenger = pd.get_dummies(
        passenger,
        columns=["Sex", "Embarked"],
        dtype=float
    )


    passenger = passenger.reindex(
        columns=X.columns,
        fill_value=0
    )


    passenger_scaled = scaler.transform(passenger)


    probabilities = model.predict_proba(passenger_scaled)[0]


    died_probability = probabilities[0] * 100
    survived_probability = probabilities[1] * 100


    prediction = model.predict(passenger_scaled)[0]


    # ========================================================
    # РЕЗУЛЬТАТ
    # ========================================================

    if prediction == 1:

        result_label.configure(text="Выжил", text_color=GREEN)
        stamp.configure(border_color=GREEN)

    else:

        result_label.configure(text="Погиб", text_color=RED)
        stamp.configure(border_color=RED)


    probability_label.configure(
        text=(
            f"Вероятность выживания: "
            f"{survived_probability:.2f}%\n"
            f"Вероятность гибели: "
            f"{died_probability:.2f}%"
        )
    )


    survival_bar.set(survived_probability / 100)


    # ========================================================
    # QWEN
    # ========================================================

    qwen_text.configure(state="normal")
    qwen_text.delete("1.0", "end")
    qwen_text.insert("1.0", "Qwen анализирует данные пассажира...")
    qwen_text.configure(state="disabled")


    def run_qwen():

        explanation = ask_qwen(
            pclass,
            sex,
            age,
            sibsp,
            parch,
            fare_usd,
            embarked,
            survived_probability,
            died_probability
        )

        root.after(0, lambda: show_qwen(explanation))


    threading.Thread(
        target=run_qwen,
        daemon=True
    ).start()


# ============================================================
# QWEN РЕЗУЛЬТАТ
# ============================================================

def show_qwen(explanation):

    qwen_text.configure(state="normal")
    qwen_text.delete("1.0", "end")
    qwen_text.insert("1.0", explanation)
    qwen_text.configure(state="disabled")


# ============================================================
# КНОПКИ
# ============================================================

predict_button = ctk.CTkButton(
    buttons,
    text="Прогнозировать",
    height=34,
    corner_radius=0,
    fg_color=INK,
    hover_color=INK_HOVER,
    text_color=PAPER,
    font=F(14, "bold"),
    command=predict
)

predict_button.pack(side="left", fill="x", expand=True, padx=(0, 4))


clear_button = ctk.CTkButton(
    buttons,
    text="Очистить",
    height=34,
    width=110,
    corner_radius=0,
    fg_color=PAPER,
    hover_color=BAR_BG,
    border_width=1,
    border_color=INK,
    text_color=INK,
    font=F(13, "bold"),
    command=clear_form
)

clear_button.pack(side="right", padx=(4, 0))


# ============================================================
# ПРАВАЯ ЧАСТЬ — ПРОГНОЗ (на месте расписания питания)
# ============================================================

result_panel = ctk.CTkFrame(body, fg_color="transparent")
result_panel.grid(row=0, column=1, sticky="nsew")


# Верхняя строка: заголовок и вероятности слева, «штамп» с результатом справа

top_row = ctk.CTkFrame(result_panel, fg_color="transparent")
top_row.pack(fill="x")
top_row.grid_columnconfigure(0, weight=1)

top_text = ctk.CTkFrame(top_row, fg_color="transparent")
top_text.grid(row=0, column=0, sticky="w")

ctk.CTkLabel(
    top_text,
    text="Прогноз выживания",
    font=F(18, "bold"),
    text_color=INK
).pack(anchor="w")

probability_label = ctk.CTkLabel(
    top_text,
    text="Введите данные пассажира",
    font=F(14),
    text_color=MUTED,
    justify="left"
)
probability_label.pack(anchor="w", pady=(2, 0))

stamp = ctk.CTkFrame(
    top_row,
    fg_color="transparent",
    corner_radius=0,
    border_width=3,
    border_color=INK
)
stamp.grid(row=0, column=1, sticky="e", padx=(10, 0))

result_label = ctk.CTkLabel(
    stamp,
    text="Ожидание",
    font=F(26, "bold"),
    text_color=INK
)
result_label.pack(padx=22, pady=4)


survival_bar = ctk.CTkProgressBar(
    result_panel,
    height=10,
    corner_radius=0,
    border_width=1,
    border_color=INK,
    progress_color=INK,
    fg_color=BAR_BG
)
survival_bar.set(0)
survival_bar.pack(fill="x", pady=(8, 8))


ctk.CTkFrame(result_panel, height=1, fg_color=INK).pack(fill="x", pady=(0, 4))


note_row = ctk.CTkFrame(result_panel, fg_color="transparent")
note_row.pack(fill="x", pady=(0, 3))

ctk.CTkLabel(
    note_row,
    text="Примечание.— анализ Qwen (локальный ИИ)",
    font=F(13, "bold"),
    text_color=INK
).pack(side="left")



# Окно анализа занимает всё оставшееся место
qwen_text = ctk.CTkTextbox(
    result_panel,
    height=240,
    corner_radius=0,
    fg_color=ENTRY_BG,
    border_width=1,
    border_color=INK,
    text_color=INK_BLUE,
    font=F(18),
    wrap="word"
)
qwen_text.pack(fill="both", expand=True)

qwen_text.insert("1.0", "Здесь появится анализ Qwen.")
qwen_text.configure(state="disabled")


# ============================================================
# ЗАПУСК
# ============================================================

update_ticket()

root.mainloop()
