// Label inventory window: categories, definitions, global rename, concordance.
// No external dependencies — the editor must run with the network off.

const CATEGORY_TITLES = {
  "conjugation/aspect": "Спряжение и аспект",
  "syntax/object": "Синтаксис и объект",
  "nominal state": "Именное состояние",
  "person/gender/number": "Лицо, род, число",
  other: "Прочее",
};

let ROWS = [];

async function api(path, options) {
  const res = await fetch(`/api/labels${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await res.json();
  if (!res.ok && res.status !== 409) throw new Error(body.detail || res.status);
  return { status: res.status, body };
}

// Render one concordance entry, highlighting the label being inspected.
// The gloss is split on morpheme and lexical separators so only the whole
// component matches: SC must not light up inside a lexical `discs`.
// `spellings` covers every form that folds to this label, so a row listed under
// NC because it is stored as PRS highlights the PRS it actually contains.
function renderGloss(gloss, spellings) {
  const span = document.createElement("span");
  const wanted = new Set(spellings.map((s) => s.toUpperCase()));
  const parts = gloss.split(/([-+=.])/);
  parts.forEach((piece) => {
    if (wanted.has(piece.toUpperCase())) {
      const mark = document.createElement("mark");
      mark.textContent = piece;
      span.append(mark);
    } else {
      span.append(document.createTextNode(piece));
    }
  });
  return span;
}

// Fix a reading in place. The correction is written to the curated layer, so it
// applies to every text containing the form, not just the row on screen.
async function editEntry(row, item, spellings) {
  const segmentation = prompt(
    `Сегментация для «${row.surface}»:`, row.segmentation);
  if (segmentation === null) return;
  const gloss = prompt("Глоссы (столько же морфем):", row.gloss);
  if (gloss === null) return;

  const res = await fetch("/api/corrections", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      surface: row.surface, segmentation, gloss,
      note: "исправлено в окне инвентаря",
    }),
  });
  const body = await res.json();
  if (!res.ok) {
    alert(`Не записано: ${body.detail}`);
    return;
  }
  row.segmentation = segmentation;
  row.gloss = gloss;
  item.textContent = "";
  fillEntry(item, row, spellings);
  const where = body.occurrences || [];
  alert(`Записано и применено ко всем текстам.` +
    (where.length ? `\nВстречается в текстах: ${where.length}.` : ""));
}

// Jump to a text containing this form, so the gloss can be fixed in context.
async function openOccurrence(row) {
  // A row contributed by a saved editor text already knows its exact token.
  // Do not jump to some other occurrence of the same surface.
  if (row.document_id && row.token_id) {
    window.location.href =
      `/doc/${row.document_id}?focus=${encodeURIComponent(row.token_id)}`;
    return;
  }
  const res = await fetch(`/api/occurrences/${encodeURIComponent(row.surface)}`);
  const { occurrences, corpus_sentences: corpus } = await res.json();

  if (occurrences.length) {
    const first = occurrences[0];
    window.location.href =
      `/doc/${first.document_id}?focus=${encodeURIComponent(first.token_id)}`;
    return;
  }
  // Not in any open text. The corpus still has sentences using the form, so
  // offer to start a document from one rather than leaving a dead end.
  if (!corpus.length) {
    alert(`«${row.surface}» не встречается ни в открытых текстах, ` +
      "ни в корпусных предложениях.");
    return;
  }
  const listed = corpus
    .map((s, i) => `${i + 1}. ${s.transcription.slice(0, 90)}`)
    .join("\n");
  const pick = prompt(
    `«${row.surface}» нет в открытых текстах.\n` +
    `Создать текст из корпусного предложения? Введите номер:\n\n${listed}`, "1");
  if (pick === null) return;
  const chosen = corpus[Number(pick) - 1];
  if (!chosen) return;

  const created = await fetch("/api/documents/from-corpus", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ example_id: chosen.example_id }),
  }).then((r) => r.json());
  window.location.href = `/doc/${created.document_id}`;
}

function fillEntry(item, row, spellings) {
  const seg = document.createElement("b");
  seg.textContent = row.segmentation;
  item.append(seg, document.createTextNode(" "),
              renderGloss(row.gloss, spellings));
}

function renderUsage(rows, spellings) {
  const box = document.createElement("div");
  box.className = "usage";
  if (!rows.length) {
    box.textContent = "В корпусе не встречается.";
    box.classList.add("muted");
    return box;
  }
  rows.forEach((row, index) => {
    const group = document.createElement("span");
    group.className = "usage-group";
    const item = document.createElement("span");
    item.className = "usage-item";
    item.title = "Клик — перейти к тексту с этой формой";
    fillEntry(item, row, spellings);
    item.addEventListener("click", () => openOccurrence(row));

    const edit = document.createElement("button");
    edit.className = "usage-edit";
    edit.textContent = "исправить";
    edit.title = "Записать общую сегментацию и глоссу для этой формы";
    edit.addEventListener("click", (event) => {
      event.stopPropagation();
      editEntry(row, item, spellings);
    });

    const source = document.createElement("small");
    source.className = "usage-source";
    source.textContent = row.source === "text"
      ? `текст: ${row.title || "без названия"}${row.user_edited ? ", правка" : ""}`
      : "корпус";

    group.append(item, edit, source);
    box.append(group);
    if (index < rows.length - 1) box.append(document.createTextNode("; "));
  });
  return box;
}

async function toggleConcordance(row, card) {
  const existing = card.querySelector(".usage");
  if (existing) {
    existing.remove();
    return;
  }
  const { body } = await api(`/${encodeURIComponent(row.label)}/concordance`);
  card.append(renderUsage(body.rows, body.spellings || [row.label]));
}

async function renameLabel(row) {
  const next = prompt(`Переименовать ${row.label} во что?`, row.label);
  if (!next || next === row.label) return;
  let res = await api(`/${encodeURIComponent(row.label)}/rename`, {
    method: "POST",
    body: JSON.stringify({ new: next }),
  });
  if (res.status === 409) {
    const ok = confirm(
      `У лейбла ${row.label} есть оговорка:\n\n${res.body.note}\n\n` +
        `Всё равно переименовать в ${next}?`);
    if (!ok) return;
    res = await api(`/${encodeURIComponent(row.label)}/rename`, {
      method: "POST",
      body: JSON.stringify({ new: next, confirm: true }),
    });
  }
  const { merged, documents_updated: docs } = res.body;
  alert(
    `${row.label} → ${res.body.label}` +
      (merged ? " (объединён с существующим)" : "") +
      `\nТекстов обновлено: ${docs}`);
  load();
}

async function editDefinition(row, card) {
  const next = prompt(`Определение для ${row.label}:`, row.definition || "");
  if (next === null) return;
  await api(`/${encodeURIComponent(row.label)}/definition`, {
    method: "POST",
    body: JSON.stringify({ text: next }),
  });
  load();
}

function renderRow(row) {
  const card = document.createElement("div");
  card.className = "label-card";

  const head = document.createElement("div");
  head.className = "label-head";

  const name = document.createElement("button");
  name.className = "label-name";
  name.textContent = row.label;
  name.title = "Показать конкорданс по корпусу";
  name.addEventListener("click", () => toggleConcordance(row, card));

  const count = document.createElement("span");
  count.className = "count";
  count.textContent = row.usage_count ? `${row.usage_count}` : "0";

  const def = document.createElement("span");
  def.className = row.needs_definition ? "definition missing" : "definition";
  def.textContent = row.needs_definition ? "?? нет определения" : row.definition;
  def.title = "Изменить определение";
  def.addEventListener("click", () => editDefinition(row, card));

  head.append(name, count, def);

  if (row.folds_in.length) {
    const folds = document.createElement("span");
    folds.className = "folds";
    folds.textContent = `← ${row.folds_in.join(", ")}`;
    folds.title = "Эти написания сворачиваются в данный лейбл";
    head.append(folds);
  }

  const rename = document.createElement("button");
  rename.className = "rename";
  rename.textContent = "переименовать";
  rename.addEventListener("click", () => renameLabel(row));
  head.append(rename);

  card.append(head);

  if (row.note) {
    const note = document.createElement("div");
    note.className = "caveat";
    note.textContent = row.note;
    card.append(note);
  }
  return card;
}

function render() {
  const host = document.getElementById("inventory");
  const needle = document.getElementById("filter").value.trim().toUpperCase();
  host.textContent = "";
  const groups = new Map();
  ROWS.filter((r) => !needle || r.label.toUpperCase().includes(needle)).forEach(
    (r) => {
      if (!groups.has(r.category)) groups.set(r.category, []);
      groups.get(r.category).push(r);
    });
  if (!groups.size) {
    host.textContent = "Ничего не найдено.";
    return;
  }
  groups.forEach((rows, category) => {
    const section = document.createElement("section");
    const title = document.createElement("h2");
    title.textContent = CATEGORY_TITLES[category] || category;
    section.append(title);
    rows.forEach((r) => section.append(renderRow(r)));
    host.append(section);
  });
}

async function load() {
  const { body } = await api("");
  ROWS = body.rows;
  render();
}

document.getElementById("filter").addEventListener("input", render);
load();
