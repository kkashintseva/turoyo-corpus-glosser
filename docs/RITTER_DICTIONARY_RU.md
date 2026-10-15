# Подсказки из словаря Ritter

Этот набор добавляет в Turoyo glosser отдельный **read-only** слой словаря
Ritter/Betzold. Он не импортируется в LIFT и ничего автоматически не создаёт
в основном лексиконе.

## Как устроен поиск

Есть два уровня:

1. `exact` — точное совпадение формы;
2. `vowel_variant` — согласный каркас полностью совпадает, различаться могут
   только гласные.

Согласные не нормализуются друг к другу:

- `qəbəl` и `qabila` → `qbl`, совпадение допустимо;
- `šuro` и `suro` → `šr` и `sr`, совпадения нет;
- `ʕabdo` и `ʕavdo` → `ʕbd` и `ʕvd`, совпадения нет;
- `w` и `y` считаются согласными.

Для слов с согласным каркасом длины 1 подбор по вокализму по умолчанию
отключён: таких кандидатов слишком много.

## Куда положить файлы

Скопировать с сохранением путей:

```text
src/maalula_glosser/ritter_dictionary.py
src/maalula_glosser/resources/__init__.py
src/maalula_glosser/resources/dictionaries/__init__.py
src/maalula_glosser/resources/dictionaries/ritter.txt
src/maalula_glosser/cli.py
tests/unit/test_ritter_dictionary.py
```

`data/` для этого не нужна.

## Проверка

```bash
cd ~/Desktop/projects/turoyo-corpus-glosser
source .venv/bin/activate
python3 -m pip install -e .
pytest tests/unit/test_ritter_dictionary.py
```

Статистика:

```bash
turoyo-glosser ritter-status
```

Проверка конкретной формы:

```bash
turoyo-glosser ritter-lookup 'qəbəl'
```

Только точное совпадение:

```bash
turoyo-glosser ritter-lookup 'qəbəl' --exact-only
```

Полная исходная строка Ritter сохраняется в `raw_entry`. Код намеренно не
пытается автоматически перепарсить немецкие значения, этимологии, корни и
ссылки: они остаются в исходном виде и могут быть показаны пользователю как
словарная подсказка.
