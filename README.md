# Turoyo Corpus Glosser

Редактор интерлинейной разметки Turoyo, работающий от двух основных источников:

- **FieldWorks LIFT** — основной лексикон: отдельные lexical entries, senses, variants, GUID и точное `grammatical-info`;
- **FLEx `.flextext`** — корпус: готовые разборы, частоты, морфемные глоссы и контексты.

Программа использует уже проверенный корпус как evidence. Известная словоформа получает засвидетельствованные разборы с частотами; неоднозначные формы остаются неоднозначными; неизвестным словам не приписывается грамматика из другого языка. Ручные reusable-правки имеют приоритет над импортированным корпусом.

## Запуск на macOS

```bash
cd ~/Desktop/projects/turoyo-corpus-glosser
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
turoyo-glosser edit
```

Открыть: `http://127.0.0.1:8000`.

Можно также запустить `turoyo.command` двойным кликом после однократной установки окружения.

## Импорт

Через стартовую страницу можно загрузить:

- `.lift` и при необходимости `.lift-ranges`;
- `.flextext` / XML;
- DOCX с уже размеченными IGT-строками.

CLI:

```bash
turoyo-glosser import-lift ~/path/Dropbox.lift --ranges ~/path/Dropbox.lift-ranges
turoyo-glosser import-flextext ~/path/allcorpus.flextext
turoyo-glosser lift-status
turoyo-glosser flex-status
```

Порядок LIFT/FLEXTEXT-импорта не критичен. Повторный импорт LIFT заменяет snapshot FieldWorks, но не стирает локально добавленные формы и значения.

## Лексикон

Отдельные FieldWorks entries не сливаются по сходству формы. Например, разные глагольные основы с категориями `Preterite` и `Infectum` остаются отдельными статьями. Variant наследует identity и `grammatical-info` родительской entry.

Неизвестную форму можно привязать к существующей LIFT-entry или создать как локальную entry. Новое значение сохраняется поверх импортированного словаря и затем предлагается снова.

В окне **Словник** доступен экспорт текущего resolved lexicon в CSV.

## Редактор

Сохранены функции, нужные для работы с Turoyo:

- редактирование текста, сегментации и глосс;
- границы `-`, `=` и `+` с сохранением исходного знака;
- альтернативные корпусные разборы и частоты;
- reusable morphology и lexical senses;
- concordance/контексты;
- comments/discussions;
- undo/redo и edit lock;
- аудио и таймкоды;
- экспорт DOCX/JSON/CSV;
- отметка отдельных строк и всего текста как проверенного.

DOCX экспортирует классический двухстрочный IGT: морфемы + глоссы. FieldWorks `grammatical-info` остаётся в данных, лексиконе и CSV/JSON, но отдельной третьей строкой DOCX не выводится.

## Данные

Рабочие данные находятся под `data/` и не должны заменяться при обновлении кода. Этот ZIP намеренно поставляется **без `data/`**. При первом импорте нужные подкаталоги создаются автоматически.

Подробнее: `docs/FLEXTEXT_TUROYO_RU.md` и `docs/USER_GUIDE_RU.md`.

### Примечание о внутреннем имени пакета

Часть Python-модулей всё ещё находится в namespace `maalula_glosser`. Это внутреннее имя унаследованного редакторного кода; пользовательская команда и профиль проекта — только `turoyo-glosser` / Turoyo FLEX.
