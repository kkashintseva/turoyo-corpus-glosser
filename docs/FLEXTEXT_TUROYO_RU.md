# Turoyo / FLEx corpus profile

Эта ветка сохраняет исходный `maalula-glosser-morphology-v2` как основную
кодовую базу, но добавляет независимый профиль `flex` для корпуса Turoyo.
Профиль `flex` является профилем по умолчанию в браузерном редакторе.

## Что сохранено из исходного парсера

Не переписывались и остаются доступными существующие механизмы редактора:

- браузерный редактор с отдельной правкой исходной строки, сегментации и глосс;
- редактируемые тексты и persistent editor sessions;
- reusable morphology: подтверждённый человеком разбор формы может иметь
  приоритет над автоматическим;
- reusable lexical gloss и работа со словником/лексемами;
- варианты разбора, выбор варианта и повторный анализ;
- occurrences / concordance;
- инвентарь глосс и определения;
- comments и discussions;
- undo/redo;
- блокировка одного текста одним «пером»;
- URL аудио и таймкоды;
- DOCX partial-IGT import;
- JSON, CSV и DOCX export;
- ZIP backup persistent-state;
- старый Maalula compositional morphology / morphology-v4 и связанная
  инфраструктура остаются в репозитории и могут использоваться отдельным
  профилем `maalula` при наличии его runtime-data.

## Что добавлено для `.flextext`

### Импорт

FLEx Interlinear Text XML (`.flextext` или `.xml`) можно импортировать двумя
способами.

В браузере: стартовая страница → **Импорт FLEx `.flextext`**.

В CLI:

```bash
turoyo-glosser import-flextext /path/to/allcorpus.flextext
```

Импорт потоковый: весь XML не загружается одной строкой в память. Каждый
`<interlinear-text>` становится отдельным редактируемым документом.

Сохраняются, насколько они присутствуют в FLEx XML:

- baseline/source line;
- перевод фразы;
- `segnum`, GUID и атрибуты текста/фразы/слова/морфемы;
- все FLEx `<item>` с `type`, `lang`, value и attributes;
- морфемные `txt` и `gls`;
- тип морфемы (`stem`, `suffix`, `prefix`, `proclitic`, `enclitic`, …);
- `cf`/lemma и `msa`/POS, если конкретный экспорт их действительно содержит;
- исходные boundary markers `-` / `=` / `+`.

Оригинальный файл архивируется byte-for-byte в persistent state. Редактор при
этом работает с собственной editable session, а не меняет FLEx XML на месте.

### Исходный и текущий анализ

Импортированная ручная FLEx-морфология считается человеческим анализом:
токен `locked`, tier `FLEX`. Его исходное состояние сохраняется в
`original_analysis` и `imported_igt`; текущие клетки можно править обычными
средствами редактора.

Для одиночного FLEx-proclitic вида `m=` редактор не создаёт фиктивную пустую
следующую морфему. На экране форма представлена редактируемой клеткой `m`, но
`flex_raw_form`, `flex_raw_gloss` и исходный trailing boundary сохраняются в
provenance без потери.

`+` — полноценный морфемный разделитель. Он ведёт себя как `=` для разбиения
на клетки, но никогда автоматически не переписывается в `=`. Например,
`b=u+atro` становится тремя клетками `b | u | atro`, а строка остаётся
`b=u+atro`; смешанная запись `a+b=c-d` должна так же сохраниться при повторном
открытии, reusable-разборе и экспорте.

## Как работает corpus parser

Профиль `flex` намеренно **не применяет Maalula-specific morphology к Turoyo**.
Первая надёжная ступень — exact-form corpus parser:

```text
surface wordform
    ↓
подтверждённые FLEx analyses + human reusable corrections
    ↓
ранжирование по частоте
    ↓
best analysis + alternatives
```

Если одна словоформа в импортированном корпусе имеет один анализ, новый текст
получает его с `HIGH` confidence. Если есть несколько реально разных анализов,
самый частотный становится текущим предложением, токен получает `MEDIUM`, а
остальные сохраняются как alternatives вместе с частотами. Неизвестная форма
остаётся `LOW` и пустой: система не придумывает маалульский разбор.

Reusable morphology, введённая человеком в редакторе, имеет приоритет над
исходной статистикой FLEx.

Агрегированный индекс хранится отдельно от документов, один evidence shard на
каждый FLEx text. Повторный импорт того же text GUID заменяет shard и не должен
удваивать частоты.

## Морфемный словарь и выбор вариантов

Помимо exact whole-word анализа, FLEX-профиль строит индекс
`morpheme form → attested glosses` по всем импортированным морфемам. Поэтому
редактор не заставляет принимать глоссу выбранного whole-word разбора. При
клике по ячейке линии глосс для её формы показываются все корпусные варианты
с частотами; первоначально список не фильтруется текущим значением ячейки.
После начала ввода список фильтруется по префиксу.

На `allcorpus.flextext` для морфемы `gab` индекс содержит:

- `chez` — 60 употреблений;
- `lat` — 60 употреблений;
- `side` — 1 употребление.

Список не является закрытым vocabulary. Внизу явно показано
`+ новый вариант`: любое новое значение можно набрать вручную и сохранить.
Если законченный разбор помечен reusable, его морфемные пары также попадают в
словарь как авторские варианты и при следующих правках показываются выше
исходных FLEx-свидетельств.

## Лемматический словник и новые значения

В FLEX/Turoyo-профиле основная единица словаря — **лемма**, а значения являются
дочерними senses. Поэтому одна статья может выглядеть так:

```text
gab
  ├─ chez
  ├─ lat
  └─ side
```

Это работает и со старыми импортами: прежний importer технически создавал
разные `lexeme_id` для `lemma + gloss + POS`. Словник теперь группирует такие
идентификаторы по самой лемме, поэтому существующую `data/` не требуется
удалять или переимпортировать только ради новой модели отображения.

Ручное глоссирование нового текста одновременно обучает словарь. Если автор
вводит лексическое значение именно в host/stem-морфему неизвестного слова,
редактор:

1. находит существующую лемму либо создаёт новую авторскую лемму;
2. привязывает текущий токен к ней;
3. записывает введённое значение как новый sense;
4. сохраняет `stem → meaning` как авторскую подсказку для морфемной ячейки;
5. показывает лемму и новое значение в окне «Словник» сразу после сохранения.

Для этого **не требуется** помечать морфологический разбор reusable. Reusable
morphology по-прежнему отвечает за распространение целого
`surface → segmentation + gloss` разбора; лексическое знание сохраняется
отдельно. Если лемма уже имеет значение и в другом контексте введено новое,
старое значение не стирается. Оба остаются в статье, и редактор предлагает их
при последующем глоссировании.

`+ значение` в окне «Словник» позволяет добавить sense напрямую. Авторское
значение можно удалить отдельно, не удаляя лемму и остальные senses.

## Инвентарь глосс

У Turoyo-профиля отдельный `config/gloss_inventory.flex.json`. Он не
подмешивает curated Maalula inventory (`NC`, `IPV` и т. п.). После импорта
редактор консервативно добавляет повторяющиеся corpus-derived label components;
определения можно редактировать обычным окном инвентаря.

## Профили

По умолчанию браузерный редактор использует:

```text
GLOSSER_PROFILE=flex
```

Старый движок можно включить явно:

```bash
GLOSSER_PROFILE=maalula turoyo-glosser edit
```

Для него по-прежнему нужны исходные Maalula runtime-data из первоначального
проекта. Их отсутствие не мешает запуску `flex`-профиля.

CLI для обычного текста также умеет выбирать профиль:

```bash
turoyo-glosser gloss-text "Kətwa latwa." --profile flex --format igt
```

## Где лежат данные

Без `MAALULA_STATE_DIR` persistent state остаётся рядом с checkout. Для
отдельного каталога данных:

```bash
export MAALULA_STATE_DIR="$HOME/turoyo-glosser-state"
```

Основные новые файлы:

```text
data/flex_corpus/sources/       archived original .flextext
data/flex_corpus/evidence/      evidence per imported FLEx text
data/flex_corpus/analyses.jsonl aggregated exact-form index
data/flex_corpus/manifest.json  corpus counts
data/editor_sessions/           editable documents
config/gloss_inventory.flex.json Turoyo/FLEX label inventory
```

`data/flex_corpus/` добавлен в `.gitignore`: это пользовательские corpus data,
а не код.

## Запуск на macOS

Из папки проекта:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
turoyo-glosser edit
```

Затем открыть `http://127.0.0.1:8000`.

При следующих запусках:

```bash
cd ~/Desktop/projects/turoyo-corpus-glosser
source .venv/bin/activate
turoyo-glosser edit
```

## Проверенный реальный импорт

На переданном `allcorpus.flextext` текущий importer получил:

- 55 interlinear texts;
- 8,984 segments;
- 66,849 non-punctuation word tokens;
- 65,030 analysed tokens;
- 13,412 distinct surface-form keys;
- 15,710 distinct analyses.

Например, после импорта новый `Kətwa latwa unknownword.` даёт:

```text
Kətwa        kət=wa        exist=pst       HIGH
latwa        lat=wa        neg.exist=pst   MEDIUM  (есть конкурирующий анализ)
unknownword  —             —               LOW
```

## Следующий слой парсера

Сейчас новый Turoyo parser намеренно ограничен exact-form evidence. Следующий
логичный этап — corpus-induced morpheme/stem models для невиденных словоформ.
Их стоит строить поверх сохранённых FLEx `morphemes`, не перенимая правила
Maalula автоматически.

## LIFT из FieldWorks

FLEXTEXT не обязан быть источником lexical-entry identity. Отдельный импорт
`.lift` делает FieldWorks основным лексиконом: каждая LIFT entry сохраняет
свой GUID/id, формы, variants, senses и точное `grammatical-info`. Похожие
глагольные entries не объединяются автоматически. Variant наследует категорию
своей entry; омонимичная surface form может предлагать несколько независимых
entries для ручного выбора. FLEXTEXT остаётся источником употреблений,
частот и контекстов. Авторские новые senses/entries сохраняются отдельным
слоем поверх LIFT.

Импорт из CLI:

```bash
turoyo-glosser import-lift Dropbox.lift --ranges Dropbox.lift-ranges
```

## FieldWorks LIFT как лексический слой

Для Turoyo рекомендуется импортировать также FieldWorks `.lift` (и, при
наличии, companion `.lift-ranges`). LIFT не заменяет FLEXTEXT: источники имеют
разные роли.

- **LIFT** задаёт стабильную identity lexical entry (`id`/GUID), citation или
  lexical-unit, variants, senses и точный `grammatical-info`.
- **FLEXTEXT** даёт реальные токены, морфемные разборы, частоты, сегменты,
  переводы и контексты.
- **Локальный overlay** хранит новые формы и senses, добавленные в редакторе;
  исходный LIFT остаётся read-only.

Разные FieldWorks entries никогда не сливаются автоматически по одинаковому
написанию, корню или похожему значению. Поэтому, например, отдельные entries
`abəʕ — Intransitive Preterite` и `obaʕ — Infectum` остаются отдельными. Variant
`әbʕ`, принадлежащий `obaʕ`, наследует именно identity этой entry и её
`Infectum`.

При ручной привязке ранее неизвестной словоформы к существующей entry связь
`surface → lexeme_id` записывается в локальный overlay. Она действует сразу и
после перезапуска: повторное употребление формы получает ту же entry и
`grammatical-info`, хотя морфологический разбор всё равно остаётся LOW, пока не
подтверждена собственно сегментация/глосса.

Импорт через UI находится на стартовой странице. CLI:

```bash
turoyo-glosser import-lift FILE.lift --ranges FILE.lift-ranges
turoyo-glosser lift-status
```

В «Словнике» FieldWorks entries read-only на уровне citation form и identity;
добавлять локальные senses можно. Клик по статье показывает контексты только
этой entry, не контексты омонимичной другой entry. Точная категория FieldWorks
передаётся в token metadata, подсказки, concordance, CSV/JSON и отдельную
компактную строку DOCX под IGT-глоссой.
