// Lexical-gloss window: confirmed meanings by lemma, with a per-lexeme form
// list and edit/retract. Sibling of labels.js, but keyed by lexeme instead of
// by morphological label. No external dependencies.

let ROWS = [];
let TEXT_SOURCES = [];
// One filter for the whole window (§12) — every text title plus corpus and
// dictionary, checked by default. Both the row list and every open
// concordance panel read this same state; it is never per-lexeme.
const FILTER = { corpus: true, dictionary: true, texts: new Set() };
// lexeme_id -> {card, data}, so a filter change can re-render every panel
// that is currently open without re-fetching it.
const OPEN_PANELS = new Map();

async function api(path, options) {
  const res = await fetch(`/api/senses${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await res.json();
  if (!res.ok) throw new Error(body.detail || res.status);
  return body;
}

// ---------------------------------------------------------------------------
// Concordance: three sources, filterable, each clickable through to where the
// form actually lives.
// ---------------------------------------------------------------------------

function openText(documentId, tokenId) {
  window.location.href = `/doc/${documentId}?focus=${encodeURIComponent(tokenId)}`;
}

// A corpus-sourced form is not tied to one saved document — reuse the same
// lookup labels.js already offers there: jump to an open text if one has it,
// else offer to start a document from a corpus sentence.
async function openCorpusForm(surface) {
  const res = await fetch(`/api/occurrences/${encodeURIComponent(surface)}`);
  const { occurrences, corpus_sentences: corpus } = await res.json();
  if (occurrences.length) {
    openText(occurrences[0].document_id, occurrences[0].token_id);
    return;
  }
  if (!corpus.length) {
    alert(`«${surface}» не встречается ни в открытых текстах, ни в корпусных предложениях.`);
    return;
  }
  const listed = corpus.map((s, i) => `${i + 1}. ${s.transcription.slice(0, 90)}`).join("\n");
  const pick = prompt(
    `«${surface}» нет в открытых текстах.\n` +
    `Создать текст из корпусного предложения? Введите номер:\n\n${listed}`, "1");
  if (pick === null) return;
  const chosen = corpus[Number(pick) - 1];
  if (!chosen) return;
  const created = await fetch("/api/documents/from-corpus", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ example_id: chosen.example_id }),
  }).then((r) => r.json());
  window.location.href = `/doc/${created.document_id}`;
}

function renderCorpusRow(form, excluded) {
  const group = document.createElement("span");
  group.className = "usage-group";
  const item = document.createElement("span");
  item.className = "usage-item";
  item.title = "Клик — перейти к тексту с этой формой";
  const surface = document.createElement("b");
  surface.textContent = form.surface_canonical;
  item.append(surface);
  if (form.segmentation) {
    item.append(document.createTextNode(` ${form.segmentation} / ${form.gloss}`));
  } else if (form.lexical_gloss) {
    // No per-form analysis joined to this surface, but the lexeme's own
    // meaning is known from other confirmed corpus analyses — show that
    // lexical gloss alone rather than leaving the row blank.
    item.append(document.createTextNode(` — ${form.lexical_gloss}`));
  }
  if (form.context) {
    if (form.context.context_before) {
      item.prepend(document.createTextNode(`…${form.context.context_before} `));
    }
    if (form.context.context_after) {
      item.append(document.createTextNode(` ${form.context.context_after}…`));
    }
    item.title = form.context.sentence;
  }
  item.addEventListener("click", () => openCorpusForm(form.surface_canonical));
  group.append(item);
  if (form.grammatical_info) {
    const gi = document.createElement("small");
    gi.className = "usage-source lexical-gi";
    gi.textContent = form.grammatical_info;
    group.append(gi);
  }
  if (form.frequency) {
    const freq = document.createElement("small");
    freq.className = "usage-source";
    freq.textContent = `частота ${form.frequency}` + (form.in_document ? ", в этом тексте" : "");
    group.append(freq);
  } else if (form.in_document) {
    const freq = document.createElement("small");
    freq.className = "usage-source";
    freq.textContent = "в этом тексте";
    group.append(freq);
  }
  if (excluded.has(form.surface_canonical)) {
    const mark = document.createElement("small");
    mark.className = "caveat";
    mark.textContent = "исключена";
    group.append(mark);
  }
  return group;
}

function renderTextRow(occurrence) {
  const group = document.createElement("div");
  group.className = "usage-context";

  const meta = document.createElement("div");
  meta.className = "usage-context-meta";
  const location = [occurrence.title, occurrence.segnum ? `§ ${occurrence.segnum}` : ""]
    .filter(Boolean).join(" · ");
  meta.textContent = location;
  if (occurrence.grammatical_info) {
    meta.textContent += `${location ? " · " : ""}${occurrence.grammatical_info}`;
  }
  group.append(meta);

  const item = document.createElement("button");
  item.type = "button";
  item.className = "usage-context-line";
  item.title = "Открыть это место в тексте";
  if (occurrence.context_before) {
    item.append(document.createTextNode(`${occurrence.context_before} `));
  }
  const surface = document.createElement("b");
  surface.textContent = occurrence.surface_canonical;
  item.append(surface);
  if (occurrence.context_after) {
    item.append(document.createTextNode(` ${occurrence.context_after}`));
  }
  item.addEventListener("click", () => openText(occurrence.document_id, occurrence.token_id));
  group.append(item);

  // The full source line matters especially for long FLEx segments where a
  // three-word KWIC alone can conceal the syntactic context.
  if (occurrence.line) {
    const full = document.createElement("div");
    full.className = "usage-context-full";
    full.textContent = occurrence.line;
    group.append(full);
  }
  if (occurrence.translation) {
    const translation = document.createElement("div");
    translation.className = "usage-context-translation";
    translation.textContent = occurrence.translation;
    group.append(translation);
  }
  const analysisBits = [];
  if (occurrence.segmentation) analysisBits.push(occurrence.segmentation);
  if (occurrence.gloss) analysisBits.push(occurrence.gloss);
  if (analysisBits.length) {
    const analysis = document.createElement("div");
    analysis.className = "usage-context-analysis";
    analysis.textContent = analysisBits.join("  /  ");
    group.append(analysis);
  }

  const reassign = document.createElement("button");
  reassign.className = "rename usage-context-reassign";
  reassign.textContent = "→ др. лексема";
  reassign.title = "Переназначить только этот токен на другую лексему";
  reassign.addEventListener("click", (event) => {
    event.stopPropagation();
    reassignOccurrence(occurrence);
  });
  group.append(reassign);
  return group;
}

function renderDictionaryRow(dictionary) {
  const group = document.createElement("span");
  group.className = "usage-group";
  const link = document.createElement("a");
  link.href = dictionary.url;
  link.target = "_blank";
  link.rel = "noopener";
  link.textContent = `словарь Арнольда, стр. ${dictionary.page} ↗`;
  group.append(link);
  if (dictionary.match === "surface") {
    const note = document.createElement("small");
    note.className = "caveat";
    note.textContent = "по написанию, не по лемме";
    group.append(note);
  }
  return group;
}

function joinRows(container, nodes) {
  nodes.forEach((node, index) => {
    container.append(node);
    if (index < nodes.length - 1) container.append(document.createTextNode("; "));
  });
}

// Renders one lexeme's concordance against the CURRENT global filter (§12).
// Called on first fetch and again, for every open panel, whenever the
// filter bar changes — never builds its own checkboxes.
function sameSense(actual, wanted) {
  if (!wanted) return true;
  return String(actual || "").trim().toLocaleLowerCase() ===
    String(wanted || "").trim().toLocaleLowerCase();
}

function renderConcordanceList(list, data, senseGloss = null) {
  list.textContent = "";

  const heading = document.createElement("div");
  heading.className = "usage-heading";
  heading.textContent = senseGloss
    ? `Контексты: ${data.lemma || ""} — ${senseGloss}`
    : `Все контексты: ${data.lemma || ""}`;
  list.append(heading);

  const excluded = new Set((data.active && data.active.excluded_surfaces) || []);
  const nodes = [];
  if (!senseGloss && FILTER.dictionary && data.dictionary) {
    nodes.push(renderDictionaryRow(data.dictionary));
  }
  data.text_occurrences.forEach((occurrence) => {
    if (FILTER.texts.has(occurrence.title) && sameSense(occurrence.lexical_gloss, senseGloss)) {
      nodes.push(renderTextRow(occurrence));
    }
  });
  if (FILTER.corpus) {
    data.forms.forEach((form) => {
      const corpusSense = form.lexical_gloss || "";
      if (sameSense(corpusSense, senseGloss)) nodes.push(renderCorpusRow(form, excluded));
    });
  }
  if (!nodes.length) {
    const empty = document.createElement("div");
    empty.textContent = senseGloss
      ? `Нет сохранённых контекстов со значением «${senseGloss}» при выбранных источниках.`
      : "Ничего не найдено при выбранных источниках.";
    empty.className = "muted";
    list.append(empty);
    return;
  }
  const rows = document.createElement("div");
  rows.className = "usage-contexts";
  nodes.forEach((node) => rows.append(node));
  list.append(rows);
  if (data.truncated) {
    const note = document.createElement("div");
    note.className = "muted";
    note.textContent = `Показаны не все корпусные формы: ${data.forms.length} из ${data.total_forms}.`;
    list.append(note);
  }
}

function refreshOpenPanels() {
  OPEN_PANELS.forEach(({ list, data, senseGloss }) =>
    renderConcordanceList(list, data, senseGloss));
}

async function toggleForms(row, card, senseGloss = null) {
  const existing = card.querySelector(".usage");
  const existingSense = existing ? (existing.dataset.senseGloss || "") : null;
  const wantedSense = senseGloss || "";
  if (existing && existingSense === wantedSense) {
    existing.remove();
    OPEN_PANELS.delete(row.lexeme_id);
    return;
  }
  if (existing) existing.remove();

  const data = await api(`/${encodeURIComponent(row.lexeme_id)}/concordance`);
  const list = document.createElement("div");
  list.className = "usage";
  list.dataset.senseGloss = wantedSense;
  card.append(list);
  OPEN_PANELS.set(row.lexeme_id, { list, data, senseGloss });
  renderConcordanceList(list, data, senseGloss);
}

async function editGloss(row) {
  const next = prompt(`Значение для «${row.lemma}»:`, row.gloss);
  if (next === null || next === row.gloss) return;
  try {
    await api(`/${encodeURIComponent(row.lexeme_id)}`, {
      method: "POST",
      body: JSON.stringify({ gloss: next, gloss_language: row.gloss_language || "en" }),
    });
  } catch (error) {
    alert(`Не записано: ${error.message}`);
    return;
  }
  load();
}

async function addSense(row, initial = "") {
  const next = prompt(`Новое значение для «${row.lemma}»:`, initial);
  if (next === null || !next.trim()) return;
  try {
    await api(`/${encodeURIComponent(row.lexeme_id)}`, {
      method: "POST",
      body: JSON.stringify({ gloss: next.trim(), gloss_language: "en" }),
    });
  } catch (error) {
    alert(`Не записано: ${error.message}`);
    return;
  }
  load();
}

async function confirmSense(row, sense) {
  try {
    await api(`/${encodeURIComponent(row.lexeme_id)}`, {
      method: "POST",
      body: JSON.stringify({
        gloss: sense.gloss,
        gloss_language: sense.gloss_language || "en",
      }),
    });
  } catch (error) {
    alert(`Не подтверждено: ${error.message}`);
    return;
  }
  load();
}

async function retractSense(row, sense) {
  const ids = sense.addition_ids || [];
  if (!ids.length) return;
  if (!confirm(`Убрать значение «${sense.gloss}» из статьи «${row.lemma}»?`)) return;
  try {
    for (const additionId of ids) {
      await api(`/${encodeURIComponent(row.lexeme_id)}/readings/${encodeURIComponent(additionId)}/retract`,
        { method: "POST" });
    }
  } catch (error) {
    alert(`Не отменено: ${error.message}`);
    return;
  }
  load();
}

async function retractRow(row) {
  if (!confirm(`Отменить подтверждённое значение «${row.gloss}» для «${row.lemma}»? ` +
    "Уже сохранённые глоссы в текстах не изменятся.")) return;
  await api(`/${encodeURIComponent(row.lexeme_id)}/retract`, { method: "POST" });
  load();
}

// A row from a static tier (corpus_analysis, exact_stem_gloss, …) has not
// been reviewed by a human yet — confirming it as-is is what promotes it
// into the reusable layer, same call editGloss makes for a changed value.
async function confirmAsIs(row) {
  await api(`/${encodeURIComponent(row.lexeme_id)}`, {
    method: "POST",
    body: JSON.stringify({ gloss: row.gloss, gloss_language: row.gloss_language || "en" }),
  });
  load();
}

// ---------------------------------------------------------------------------
// Manual lexicon edits (merge / delete / rename an established lexeme, and
// reassign specific tokens) — new `/api/lexicon/...` endpoints, not
// `/api/senses/...`. Simple prompt-based controls, same pattern as
// editGloss/confirmAsIs above: no polished UI, just enough for an author to
// trigger each action from the vocabulary window.
// ---------------------------------------------------------------------------

// Detail formatting for the rare disambiguation prompt below — the server
// sends each candidate's root/form_count (same shape as candidate_details
// on a queue row), so an author who hits an ambiguous citation form sees a
// real hint, not a bare id.
function describeCandidates(candidates) {
  return candidates
    .map((c) => {
      const hint = c.root ? `корень ${c.root}, ` : "";
      return `${c.lexeme_id} (${hint}${c.form_count} форм)`;
    })
    .join("\n");
}

async function mergeLexeme(row) {
  const targetWord = prompt(
    `Слить «${row.lemma}» в другую лексему.\n` +
    "Введите словарную форму (слово) лексемы, В КОТОРУЮ сливаем:");
  if (!targetWord) return;
  let body;
  try {
    const res = await fetch("/api/lexicon/merge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_lexeme_id: row.lexeme_id, target_lemma: targetWord.trim() }),
    });
    body = await res.json();
    if (res.status === 409 && body.detail && body.detail.candidates) {
      const targetId = prompt(
        `Эта словарная форма есть у нескольких лексем — уточните lexeme_id:\n` +
        describeCandidates(body.detail.candidates));
      if (!targetId) return;
      const res2 = await fetch("/api/lexicon/merge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_lexeme_id: row.lexeme_id, target_lexeme_id: targetId.trim() }),
      });
      body = await res2.json();
      if (!res2.ok) throw new Error(body.detail || res2.status);
    } else if (!res.ok) {
      throw new Error((body.detail && body.detail.error) || body.detail || res.status);
    }
  } catch (error) {
    alert(`Не слито: ${error.message}`);
    return;
  }
  alert(`Слито. Код правки (для отмены): ${body.merge.edit_id}`);
  load();
}

async function deleteLexeme(row) {
  if (!confirm(
    `Удалить лексему «${row.lemma}» (${row.lexeme_id}) из словаря?\n` +
    "Все токены, ссылавшиеся на неё, вернутся в очередь на переразбор."))
    return;
  let body;
  try {
    const res = await fetch(`/api/lexicon/${encodeURIComponent(row.lexeme_id)}`,
      { method: "DELETE" });
    body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.status);
  } catch (error) {
    alert(`Не удалено: ${error.message}`);
    return;
  }
  alert(`Удалено. Код правки (для отмены): ${body.deletion.edit_id}`);
  load();
}

async function renameLexemeCitationForm(row) {
  const next = prompt(`Словарная форма для «${row.lemma}»:`, row.lemma);
  if (!next || next === row.lemma) return;
  let body;
  try {
    const res = await fetch(`/api/lexicon/${encodeURIComponent(row.lexeme_id)}/lemma`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lemma_canonical: next }),
    });
    body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.status);
  } catch (error) {
    alert(`Не записано: ${error.message}`);
    return;
  }
  alert(`Записано. Код правки (для отмены): ${body.rename.edit_id}`);
  load();
}

// Any of the three edits above prints its `edit_id` in its success alert;
// this is the one control that undoes it, by that id, via the tombstone
// retraction endpoint (`POST /api/lexicon/edits/{edit_id}/retract`).
async function undoEditById() {
  const editId = prompt("Код правки для отмены (показан после слияния/удаления/переименования):");
  if (!editId) return;
  try {
    const res = await fetch(`/api/lexicon/edits/${encodeURIComponent(editId.trim())}/retract`,
      { method: "POST" });
    if (!res.ok) throw new Error((await res.json()).detail || res.status);
  } catch (error) {
    alert(`Не отменено: ${error.message}`);
    return;
  }
  load();
}

// Reassign one specific occurrence (a saved-document token) to a different,
// already-established lexeme — for a single mis-resolved token, not every
// occurrence of its current lexeme (that would be a merge instead).
async function reassignOccurrence(occurrence) {
  const targetWord = prompt(
    `Переназначить токен «${occurrence.surface_canonical}» на другую лексему.\n` +
    "Введите словарную форму (слово) лексемы-цели:");
  if (!targetWord) return;
  const targets = [{ document_id: occurrence.document_id, token_id: occurrence.token_id }];
  try {
    let res = await fetch("/api/lexicon/reassign", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ targets, target_lemma: targetWord.trim() }),
    });
    let body = await res.json();
    if (res.status === 409 && body.detail && body.detail.candidates) {
      const targetId = prompt(
        `Эта словарная форма есть у нескольких лексем — уточните lexeme_id:\n` +
        describeCandidates(body.detail.candidates));
      if (!targetId) return;
      res = await fetch("/api/lexicon/reassign", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ targets, target_lexeme_id: targetId.trim() }),
      });
      body = await res.json();
    }
    if (!res.ok) throw new Error((body.detail && body.detail.error) || body.detail || res.status);
  } catch (error) {
    alert(`Не переназначено: ${error.message}`);
    return;
  }
  alert("Переназначено. Откройте формы заново, чтобы увидеть изменение.");
}

const SOURCE_LABELS = {
  author: "автор",
  corpus_analysis: "корпус",
  exact_stem_gloss: "словарь автора",
  translated_sense: "Арнольд",
  queue: "очередь",
  text: "из текста",
  lift: "FieldWorks",
  lemma: "лексикон",
};

const POS_OPTIONS = [["", "часть речи?"], ["verb", "гл."], ["adjective", "прил."], ["noun", "сущ."]];

async function setPartOfSpeech(row, value) {
  try {
    await api(`/${encodeURIComponent(row.lexeme_id)}/part-of-speech`, {
      method: "POST",
      body: JSON.stringify({ part_of_speech: value || null }),
    });
  } catch (error) {
    alert(`Не записано: ${error.message}`);
  }
  load();
}

function renderPos(row) {
  const select = document.createElement("select");
  select.className = "folds";
  select.title = "Часть речи: по умолчанию угадана по граммемам в подтверждённых " +
    "разборах этой лексемы; можно поправить вручную";
  POS_OPTIONS.forEach(([value, label]) => select.add(new Option(label, value)));
  select.value = row.part_of_speech || "";
  select.addEventListener("click", (event) => event.stopPropagation());
  select.addEventListener("change", () => setPartOfSpeech(row, select.value));
  return select;
}

function renderRow(row) {
  const card = document.createElement("div");
  card.className = "label-card";
  card.dataset.lexemeId = row.lexeme_id;

  const head = document.createElement("div");
  head.className = "label-head";

  const name = document.createElement("button");
  name.className = "label-name";
  name.textContent = row.lemma || row.lexeme_id;
  name.title = "Показать все контексты этой леммы";
  name.addEventListener("click", () => toggleForms(row, card));
  head.append(name);

  if (row.form_count !== null && row.form_count !== undefined) {
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = String(row.form_count);
    head.append(count);
  }

  const grammaticalInfos = (row.grammatical_infos || []).filter(Boolean);
  if (grammaticalInfos.length) {
    const categories = document.createElement("span");
    categories.className = "grammatical-infos";
    grammaticalInfos.forEach((value) => {
      const tag = document.createElement("span");
      tag.className = "grammatical-info";
      tag.textContent = value;
      categories.append(tag);
    });
    head.append(categories);
  }

  if (Array.isArray(row.senses)) {
    const senses = document.createElement("span");
    senses.className = "sense-list";
    if (!row.senses.length) {
      const empty = document.createElement("span");
      empty.className = "definition muted";
      empty.textContent = "значение не задано";
      senses.append(empty);
    }
    row.senses.forEach((sense) => {
      const chip = document.createElement("span");
      chip.className = "sense-chip";
      const value = document.createElement("button");
      value.type = "button";
      value.className = "definition sense-value";
      value.textContent = sense.gloss
        + (sense.grammatical_info ? ` · ${sense.grammatical_info}` : "")
        + (sense.count ? ` ×${sense.count}` : "");
      value.title = `Показать контексты значения «${sense.gloss}». Источники: ${(sense.sources || []).join(", ") || "—"}`;
      value.addEventListener("click", (event) => {
        event.stopPropagation();
        toggleForms(row, card, sense.gloss);
      });
      chip.append(value);
      const action = document.createElement("button");
      action.type = "button";
      action.className = "rename sense-action";
      if (sense.author) {
        action.textContent = "×";
        action.title = "Убрать это авторское значение, не удаляя лемму и другие значения";
        action.addEventListener("click", () => retractSense(row, sense));
      } else {
        action.textContent = "✓";
        action.title = "Подтвердить это значение как авторское";
        action.addEventListener("click", () => confirmSense(row, sense));
      }
      chip.append(action);
      senses.append(chip);
    });
    const add = document.createElement("button");
    add.type = "button";
    add.className = "rename";
    add.textContent = "+ значение";
    add.title = "Добавить ещё одно значение этой леммы";
    add.addEventListener("click", () => addSense(row));
    head.append(senses, add);
    if (row.source !== "lift") head.append(renderPos(row));
  } else {
    const gloss = document.createElement("span");
    gloss.className = row.gloss ? "definition" : "definition muted";
    gloss.textContent = row.gloss
      ? row.gloss + (row.gloss_language === "de" ? " (нем.)" : "")
      : "значение не задано";
    gloss.title = "Изменить значение";
    gloss.addEventListener("click", () => editGloss(row));
    head.append(gloss);
    if (row.source !== "lift") head.append(renderPos(row));
  }

  const source = document.createElement("small");
  source.className = "usage-source";
  source.textContent = row.source === "lift"
    ? `FieldWorks${row.lift_id ? ` · ${row.lift_id}` : ""} · ${(row.senses || []).length} знач.`
    : (Array.isArray(row.senses)
      ? `${row.senses.length} знач.`
      : (SOURCE_LABELS[row.source] || row.source));
  head.append(source);

  if (row.excluded_count) {
    const folds = document.createElement("span");
    folds.className = "folds";
    folds.textContent = `искл.: ${row.excluded_count}`;
    folds.title = "Столько форм исключено из этого решения";
    head.append(folds);
  }

  if (row.needs_citation_form) {
    const promoteButton = document.createElement("button");
    promoteButton.className = "rename";
    promoteButton.textContent = "в словарик";
    promoteButton.title = "Добавить как настоящую лексему по словарной форме";
    promoteButton.addEventListener("click", async () => {
      const lemma = prompt(
        `Словарная форма для «${row.lemma}» (например, app(i) yapp(i)):`, row.lemma);
      if (!lemma) return;
      const res = await fetch("/api/lexicon/additions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          lexeme_id: row.lexeme_id, lemma_canonical: lemma,
          gloss: row.gloss, gloss_language: row.gloss_language || "en",
        }),
      });
      if (!res.ok) {
        const body = await res.json();
        const detail = body.detail;
        const message = detail && typeof detail === "object"
          ? `«${detail.lemma_canonical}» уже принадлежит нескольким лексемам ` +
            `(${(detail.candidates || []).join(", ")}) — выберите одну вручную`
          : (detail || res.status);
        alert(`Не добавлено: ${message}`);
        return;
      }
      load();
    });
    head.append(promoteButton);
  } else if (!Array.isArray(row.senses) && row.source === "author") {
    const retract = document.createElement("button");
    retract.className = "rename";
    retract.textContent = "отменить";
    retract.addEventListener("click", () => retractRow(row));
    head.append(retract);
  } else if (!Array.isArray(row.senses)) {
    const confirmBtn = document.createElement("button");
    confirmBtn.className = "rename";
    confirmBtn.textContent = "подтвердить";
    confirmBtn.title = "Записать как есть в reusable-слой для всех форм";
    confirmBtn.addEventListener("click", () => confirmAsIs(row));
    head.append(confirmBtn);
  }

  // Merge/delete/rename act on an ESTABLISHED lexeme (already in
  // `analyzer.lexemes`) — a queue row (`needs_citation_form`) is not one
  // yet, so these three controls only apply to the other rows.
  if (!row.needs_citation_form && row.source !== "lift") {
    const lemmaBtn = document.createElement("button");
    lemmaBtn.className = "rename";
    lemmaBtn.textContent = "словарная форма";
    lemmaBtn.title = "Исправить словарную форму (леммy) этой лексемы";
    lemmaBtn.addEventListener("click", () => renameLexemeCitationForm(row));
    head.append(lemmaBtn);

    const mergeBtn = document.createElement("button");
    mergeBtn.className = "rename";
    mergeBtn.textContent = "слить";
    mergeBtn.title = "Слить эту лексему в другую (дубликат)";
    mergeBtn.addEventListener("click", () => mergeLexeme(row));
    head.append(mergeBtn);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "rename";
    deleteBtn.textContent = "удалить";
    deleteBtn.title = "Удалить эту лексему из словаря";
    deleteBtn.addEventListener("click", () => deleteLexeme(row));
    head.append(deleteBtn);
  }

  card.append(head);

  if (row.candidate_details && row.candidate_details.length) {
    const hint = document.createElement("div");
    hint.className = "usage-source";
    hint.textContent = "возможно, уже в словаре: " + row.candidate_details
      .map((c) => `${c.lemma}${c.root ? ` (корень ${c.root})`
                    : ` (…${c.lexeme_id.slice(-6)})`}`)
      .join("; ") + " — сравните и слейте вручную, если это то же слово";
    card.append(hint);
  }

  return card;
}

// A row passes the GLOBAL source filter if it has some attestation in at
// least one checked source — corpus (it has any attested form at all),
// dictionary (an Arnold entry exists), or one of the checked saved texts.
function rowMatchesFilter(row) {
  if (document.getElementById("only-new")?.checked && !row.is_new) return false;
  if (FILTER.corpus && row.form_count > 0) return true;
  if (FILTER.dictionary && row.has_dictionary) return true;
  return row.text_titles.some((title) => FILTER.texts.has(title));
}

// Hides/shows already-built row cards in place — never rebuilds them, so an
// open concordance panel survives a filter change (only render() rebuilds).
function applyRowVisibility() {
  const host = document.getElementById("inventory");
  const empty = document.getElementById("filter-empty");
  let anyVisible = false;
  host.querySelectorAll("[data-lexeme-id]").forEach((card) => {
    const row = ROWS.find((r) => r.lexeme_id === card.dataset.lexemeId);
    const visible = !row || rowMatchesFilter(row);
    card.hidden = !visible;
    if (visible) anyVisible = true;
  });
  empty.hidden = anyVisible || !host.children.length;
}

function render() {
  const host = document.getElementById("inventory");
  const needle = document.getElementById("filter").value.trim().toLowerCase();
  host.textContent = "";
  OPEN_PANELS.clear();
  const matchesNeedle = (r) => {
    if (!needle) return true;
    const fields = [r.lemma, ...(r.grammatical_infos || []),
      ...(r.senses || []).map((sense) => sense.gloss),
      ...(r.senses || []).map((sense) => sense.grammatical_info)];
    return fields.some((value) => String(value || "").toLowerCase().includes(needle));
  };
  const matched = ROWS.filter(matchesNeedle);
  if (!matched.length) {
    host.textContent = "Ничего не найдено.";
    document.getElementById("filter-empty").hidden = true;
    return;
  }
  const limit = needle ? 800 : 350;
  const rows = matched.slice(0, limit);
  if (matched.length > rows.length) {
    const note = document.createElement("div");
    note.className = "muted inventory-limit";
    note.textContent = `Показаны первые ${rows.length} из ${matched.length}. Начните печатать лемму, значение или категорию FieldWorks.`;
    host.append(note);
  }
  const fragment = document.createDocumentFragment();
  rows.forEach((r) => fragment.append(renderRow(r)));
  host.append(fragment);
  applyRowVisibility();
}

// ---------------------------------------------------------------------------
// Global source-filter bar (§12): one set of checkboxes for the whole
// window — every saved text by title, plus "корпус" and "словарь" — rather
// than one per lexeme. A change here re-shows/hides rows in place and
// refreshes whatever concordance panels are currently open; it never
// re-fetches or rebuilds the row list itself.
// ---------------------------------------------------------------------------
function buildSourceFilterBar() {
  const bar = document.getElementById("source-filter");
  bar.textContent = "";

  const sourceDefs = TEXT_SOURCES.map((title) => ({ key: `text:${title}`, label: title }));
  sourceDefs.push({ key: "corpus", label: "корпус" });
  sourceDefs.push({ key: "dictionary", label: "словарь" });

  const allLabel = document.createElement("label");
  const allBox = document.createElement("input");
  allBox.type = "checkbox";
  allBox.checked = true;
  allLabel.append(allBox, document.createTextNode(" все вместе"));
  bar.append(allLabel);

  const boxes = sourceDefs.map(({ key, label }) => {
    const wrapLabel = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = true;
    box.dataset.key = key;
    wrapLabel.append(box, document.createTextNode(` ${label}`));
    bar.append(wrapLabel);
    return box;
  });

  function syncFilterFromBoxes() {
    FILTER.corpus = boxes.some((b) => b.dataset.key === "corpus" && b.checked);
    FILTER.dictionary = boxes.some((b) => b.dataset.key === "dictionary" && b.checked);
    FILTER.texts = new Set(boxes.filter((b) => b.dataset.key.startsWith("text:") && b.checked)
      .map((b) => b.dataset.key.slice("text:".length)));
    applyRowVisibility();
    refreshOpenPanels();
  }

  allBox.addEventListener("change", () => {
    boxes.forEach((b) => { b.checked = allBox.checked; });
    syncFilterFromBoxes();
  });
  boxes.forEach((box) => box.addEventListener("change", () => {
    allBox.checked = boxes.every((b) => b.checked);
    syncFilterFromBoxes();
  }));

  syncFilterFromBoxes();
}

async function load() {
  const { rows, text_sources: textSources } = await api("");
  ROWS = rows;
  TEXT_SOURCES = textSources;
  buildSourceFilterBar();
  render();
}

async function seedFromCorpus() {
  const pending = ROWS.reduce((total, row) => {
    if (Array.isArray(row.senses)) {
      return total + row.senses.filter((sense) => !sense.author).length;
    }
    return total + (row.source !== "author" ? 1 : 0);
  }, 0);
  if (!pending) { alert("Всё, что уже есть в корпусе, уже подтверждено."); return; }
  if (!confirm(`Загрузить ${pending} значений из корпуса в reusable-слой разом? ` +
    "Каждое можно будет отменить отдельно.")) return;
  const { written } = await api("/seed-from-corpus", { method: "POST" });
  alert(`Записано: ${written}.`);
  load();
}

document.getElementById("filter").addEventListener("input", render);
document.getElementById("only-new").addEventListener("change", applyRowVisibility);
document.getElementById("seed-all").addEventListener("click", seedFromCorpus);
document.getElementById("undo-edit").addEventListener("click", undoEditById);
load();
