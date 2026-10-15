"use strict";
// Interactive gloss editor front-end. The browser holds the live document and
// PATCHes single cells; the server persists and re-glosses. No external deps.

const DOC_ID = document.body.dataset.documentId;

// Identifies this browser tab to the single-editor lock — session-scoped, so
// a reload of the same tab keeps its place but a new tab is a new reader.
// See docs/superpowers/specs/2026-09-12-discussion-threads-and-editor-lock-design.md.
const CLIENT_ID = (() => {
  let id = null;
  try { id = sessionStorage.getItem("editorClientId"); } catch (e) { /* private mode etc. */ }
  if (!id) {
    id = crypto.randomUUID ? crypto.randomUUID() : `c${Date.now()}-${Math.random()}`;
    try { sessionStorage.setItem("editorClientId", id); } catch (e) { /* ignore */ }
  }
  return id;
})();

function editorName() {
  try { return sessionStorage.getItem("editorDisplayName") || ""; }
  catch (e) { return ""; }
}
function setEditorName(name) {
  try { sessionStorage.setItem("editorDisplayName", name); } catch (e) { /* ignore */ }
}

// Every mutating call carries who is asking, and a 423 (someone else holds
// the pen, or nobody granted it to this tab) always resolves the same way:
// tell the reader, refresh, drop into read-only — see `handleLostLock` below.
const api = (path, opts = {}) => fetch(`/api/documents/${DOC_ID}${path}`, {
  ...opts,
  headers: { ...(opts.headers || {}), "X-Editor-Client": CLIENT_ID },
}).then(async (r) => {
  if (r.status === 423) {
    const body = await r.json().catch(() => ({}));
    await handleLostLock(body.holder || null);
    throw new Error("read-only");
  }
  if (!r.ok) throw new Error(r.status);
  return r.json();
});

let DOC = null;
let PREFERENCES = { reusable_default: true };
let LOCK = { holder: null, you_are_holder: false };
let HISTORY_STATE = { can_undo: false, can_redo: false };
const pop = document.getElementById("popover");

async function projectRequest(url, method = "GET", body) {
  const response = await fetch(url, {
    method, headers: { "Content-Type": "application/json" },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

function effectiveReusable(token, preferences) {
  if (token.reuse_mode === "local") return false;
  if (token.reuse_mode === "reusable") return true;
  // this text's own setting, if any, is stronger than the project default
  if (DOC && typeof DOC.reuse_default === "boolean") return DOC.reuse_default;
  return preferences.reusable_default;
}

// The stem's meaning inside a whole gloss: `say.SC=3ms` -> `say`.
function lexicalHead(gloss) {
  const pieces = String(gloss || "").split(/[-=+]/);
  const piece = pieces.find((p) => p && !/^[A-Z0-9.]+$/.test(p) && !/^[0-9][a-zA-Z.]*$/.test(p)) || "";
  const parts = piece.split(".");
  while (parts.length > 1 && /^[A-Z0-9]+$/.test(parts[parts.length - 1])) parts.pop();
  return parts.join(".");
}

// ---------------------------------------------------------------------------
// «не предлагать для …»: words this editor does not want the
// "same reading elsewhere?" panel for (stored per editor name on the server)
// ---------------------------------------------------------------------------
let MUTED = new Map();

// What the panel is about: a whole word ("bess, — ещё 11 мест") or one cell
// ("l — ещё 29 мест … HD"). The key and the label are that form, as shown.
const bareForm = (form) => String(form || "").replace(/^[^\p{L}ʔʕ]+|[^\p{L}\p{M}ʔʕ]+$/gu, "");
const wordMuteKey = (form) => (bareForm(form) ? `form:${bareForm(form).toLowerCase()}` : "");
const cellMuteKey = (form) => (bareForm(form) ? `cell:${bareForm(form).toLowerCase()}` : "");

async function loadMutes() {
  try {
    const res = await projectRequest(`/api/editor-mutes?name=${encodeURIComponent(editorName())}`);
    MUTED = new Map(res.muted.map((m) => [m.key, m.label]));
  } catch (error) { MUTED = new Map(); }
}

async function setMuted(key, label, muted) {
  const res = await projectRequest("/api/editor-mutes", "POST",
    { name: editorName(), key, label, muted });
  MUTED = new Map(res.muted.map((m) => [m.key, m.label]));
  reuseFeedback(muted
    ? `Для ${label} перенос разбора больше не предлагается${editorName() ? ` (редактор ${editorName()})` : ""}. Вернуть — в меню «Слово».`
    : `Для ${label} перенос разбора снова предлагается.`);
}

function muteButton(key, label) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ghost";
  button.textContent = `не предлагать для ${label}`;
  button.title = "Больше не показывать это окно для этой формы (только для вас)";
  button.disabled = !key;
  button.addEventListener("click", async () => {
    try { await setMuted(key, label, true); } catch (error) { reuseFeedback(error.message, true); }
    closeBulk();
  });
  return button;
}

function reuseFeedback(message, error = false) {
  const box = document.getElementById("reuse-feedback");
  box.textContent = message;
  box.classList.toggle("error", error);
}

function reportReuse(res) {
  if (res.correction) reuseFeedback(`Записано как reusable: ${res.correction.segmentation} / ${res.correction.gloss}. Правило доступно для новых разборов во всех текстах.`);
  else if (res.reusable.requested) reuseFeedback("Глосса сохранена в тексте. Правило reusable пока не записано: заполните все глоссы и проверьте границы -, = и +.");
  else reuseFeedback("Глосса сохранена только в этом месте. Ранее записанные правила не отменены.");
}

function renderWordActions(token) {
  const menu = document.createElement("details");
  menu.className = "word-actions";
  const summary = document.createElement("summary");
  summary.textContent = token.reusable_record?.active ? "Слово · reusable" : "Слово";
  menu.append(summary);
  const panel = document.createElement("div");
  panel.className = "word-actions-panel";
  const add = (text, action) => {
    const button = document.createElement("button");
    button.type = "button"; button.textContent = text;
    button.addEventListener("mousedown", (event) => event.stopPropagation());
    button.onclick = async () => {
      button.disabled = true;
      try { await action(); }
      catch (error) { reuseFeedback(`Не удалось выполнить действие: ${error.message}`, true); }
      finally { button.disabled = false; }
    };
    panel.append(button);
    return button;
  };
  const remember = add("Запомнить для всех текстов", async () => {
    await commitCell();
    const cell = token.morphemes[token.morphemes.length - 1];
    const res = await api(`/morphemes/${cell.morph_id}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gloss: cell.gloss, reusable: true }),
    });
    Object.assign(token, res.token);
    reportReuse(res); setHistoryButtons(res.history); render();
  });
  remember.disabled = !token.morphemes.length || token.morphemes.some((m) => !m.gloss?.trim() || /[?*]/.test(m.gloss));
  remember.title = "Записывает весь законченный разбор слова; существующие тексты автоматически не переписываются";
  const meaning = add("Значение (для всех форм основы)",
    () => confirmLexicalGloss(token));
  meaning.disabled = !token.lemma_id;
  meaning.title = "Записывает лексическое значение для всех форм той же основы, а не только для этой словоформы";
  const lemmaLine = document.createElement("small");

  const cellLemmas = (token.morphemes || []).map((cell) => {
    const shownForm = `${cell.sep || ""}${cell.form || ""}`;
    return cell.lemma
      ? `${shownForm} → ${cell.lemma}`
      : `${shownForm} → —`;
  });

  // A morphologically complex word does not have to have one privileged
  // lemma. Each component may point to its own lexical entry.
  lemmaLine.textContent = cellLemmas.length
    ? `леммы: ${cellLemmas.join(" · ")}`
    : "леммы: —";

  panel.append(lemmaLine);
  const giValues = token.grammatical_info
    ? [token.grammatical_info]
    : (token.grammatical_infos || []);
  const categoryLine = document.createElement("small");
  categoryLine.textContent = `FieldWorks-категория: ${giValues.length ? giValues.join(" / ") : "—"}`;
  panel.append(categoryLine);
  add("Ritter…", async () => loadRitterForWord(token, panel));
  const verbParser = add("Парсер…", async () => loadVerbSuggestions(token, panel));
  verbParser.disabled = !verbParserLocallyEligible(token);
  verbParser.title = verbParser.disabled
    ? "Парсер не меняет уже существующий анализ FLEx и ручной разбор"
    : "Показать варианты парсера";
  add("Привязать к лемме…", async () => openLemmaPicker(token, panel));
  const keys = [wordMuteKey(token.surface), ...(token.morphemes || []).map((m) => cellMuteKey(m.form))];
  [...new Set(keys)].filter((k) => k && MUTED.has(k)).forEach((k) => {
    add(`Снова предлагать перенос для ${MUTED.get(k)}`, async () => {
      await setMuted(k, MUTED.get(k), false); render();
    });
  });
  const label = document.createElement("label");
  label.textContent = "При следующих правках этого слова: ";
  const mode = document.createElement("select");
  for (const [value, text] of [["inherit", "Общая настройка"], ["reusable", "Для всех текстов"], ["local", "Только здесь"]]) {
    mode.add(new Option(text, value));
  }
  mode.value = token.reuse_mode || "inherit";
  mode.onchange = async () => {
    try {
      const res = await api(`/tokens/${token.token_id}/reuse-mode`, {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: mode.value }),
      });
      Object.assign(token, res.token);
      reuseFeedback("Режим слова сохранён. Он не отменяет ранее записанное правило.");
    } catch (error) { mode.value = token.reuse_mode || "inherit"; reuseFeedback(error.message, true); }
  };
  label.append(mode); panel.append(label);
  if (token.reusable_record?.active) {
    const record = document.createElement("small");
    record.textContent = `${token.reusable_record.segmentation} / ${token.reusable_record.gloss}`;
    panel.append(record);
    add("Отменить reusable, оставить правку здесь", async () => {
      if (!window.confirm("Отключить это правило для будущих разборов во всех текстах? Сохранённые глоссы в текстах останутся.")) return;
      const res = await api(`/tokens/${token.token_id}/reusable/retract`, { method: "POST" });
      Object.assign(token, res.token); render();
      reuseFeedback("Правило reusable отключено. Локальные глоссы сохранены, история правила не удалена.");
    });
  }
  if (token.lexical_record?.active) {
    const record = document.createElement("small");
    record.textContent = `значение: ${token.lexical_record.gloss}`;
    panel.append(record);
    add("Отменить значение, оставить как есть", async () => {
      if (!window.confirm("Отключить подтверждённое значение для всех форм этой основы? Уже сохранённые глоссы в текстах останутся.")) return;
      const res = await api(`/tokens/${token.token_id}/lexical/retract`, { method: "POST" });
      Object.assign(token, res.token); render();
      reuseFeedback("Значение отменено. Локальные глоссы сохранены, история не удалена.");
    });
  }
  menu.append(panel);
  menu.addEventListener("toggle", () => {
    if (!menu.open) return;
    panel.style.left = "0";
    const bounds = panel.getBoundingClientRect();
    panel.style.left = `${Math.max(8 - bounds.left, Math.min(0, window.innerWidth - 8 - bounds.right))}px`;
  });
  return menu;
}

const GLOSS_HAS_MEANING = (gloss) => {
  const head = lexicalHead(gloss);
  return !!head && !/^[A-Z0-9.]+$/.test(head) && !/^[0-9]/.test(head);
};

// The host: the last cell with a meaning (proclitics come before it,
// enclitics and endings carry labels only): b=lōġ-ṯ-a -> lōġ 'language'.
function hostCell(token) {
  const cells = token.morphemes || [];
  for (let i = cells.length - 1; i >= 0; i -= 1) {
    if (GLOSS_HAS_MEANING(cells[i].gloss)) return i;
  }
  return -1;
}

const VERBAL_LABELS = new Set(["NC", "NC1", "NC2", "PC", "SC", "PRF", "IPV"]);
const CITATION_ENDINGS = new Set(["F", "M", "FS", "MS", "SG", "FREE", "DEF", "INDF", "INDEF", "EMPH"]);

// A non-verb's dictionary form: host + gender/state endings, no clitics,
// no plural; a feminine ending without its free-state -a gets it.
function citationGuess(token) {
  const cells = token.morphemes || [];
  const at = hostCell(token);
  if (at < 0) return "";
  const labels = (g) => String(g || "").split(".").filter(Boolean).map((l) => l.toUpperCase());
  if (labels(cells[at].gloss).some((l) => VERBAL_LABELS.has(l))) return "";
  let word = cells[at].form || "";
  let last = labels(cells[at].gloss);
  for (const cell of cells.slice(at + 1)) {
    const own = labels(cell.gloss);
    if (cell.sep !== "-" || !own.length || !own.every((l) => CITATION_ENDINGS.has(l))) break;
    word += cell.form || "";
    last = own;
  }
  if (last.length === 1 && last[0] === "F" && word !== cells[at].form) word += "a";
  return word;
}

// «Привязать к лемме…»: search the lexicon, the word's own candidates first
function openLemmaPicker(token, panel) {
  panel.querySelector(".lemma-picker")?.remove();
  const box = document.createElement("div");
  box.className = "lemma-picker";
  const input = document.createElement("input");
  input.type = "search";
  input.placeholder = "лемма или √корень";
  input.addEventListener("mousedown", (event) => event.stopPropagation());
  const list = document.createElement("ul");
  list.className = "lemma-results";
  const show = (rows) => {
    list.innerHTML = "";
    if (!rows.length) {
      const li = document.createElement("li");
      li.className = "muted"; li.textContent = "ничего не найдено";
      list.append(li); return;
    }
    rows.forEach((row) => {
      const li = document.createElement("li");
      const pick = document.createElement("button");
      pick.type = "button";
      const categories = (row.grammatical_infos || []).filter(Boolean);
      const senseText = (row.senses || []).map((sense) => {
        const gi = sense.grammatical_info ? ` · ${sense.grammatical_info}` : "";
        return `${sense.gloss || "—"}${gi}`;
      }).join(" / ");
      const glossText = senseText || row.gloss || "";
      pick.textContent = row.lemma
        + (row.root ? `  √${row.root}` : "")
        + (categories.length ? ` · ${categories.join(" / ")}` : "")
        + (glossText ? ` — ${glossText}` : "")
        + (row.source === "lift" ? " · FieldWorks" : "");
      pick.addEventListener("mousedown", (event) => event.stopPropagation());
      pick.addEventListener("click", async () => {
        try {
          const res = await api(`/tokens/${token.token_id}/lemma`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lexeme_id: row.lexeme_id }),
          });
          Object.assign(token, res.token);
          setHistoryButtons(res.history); render();
          reuseFeedback(`«${token.surface}» привязано к лемме «${row.lemma}».`);
        } catch (error) { reuseFeedback(`Не удалось привязать: ${error.message}`, true); }
      });
      li.append(pick); list.append(li);
    });
  };
  let timer = null;
  const search = async () => {
    const q = input.value.trim();
    if (!q) { list.innerHTML = ""; return; }
    try { show((await projectRequest(`/api/lexicon/search?q=${encodeURIComponent(q)}`)).results); }
    catch (error) { reuseFeedback(error.message, true); }
  };
  input.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(search, 250); });
  box.append(input, list);
  panel.append(box);
  if ((token.lexeme_candidates || []).length) {
    // the word's own candidates first (from «Леммы…» or the resolver)
    projectRequest(`/api/lexicon/search?ids=${encodeURIComponent(token.lexeme_candidates.join(","))}`)
      .then((res) => show(res.results)).catch(() => {});
  } else {
    const at = hostCell(token);
    if (at >= 0) { input.value = token.morphemes[at].form || ""; search(); }
  }
  const create = document.createElement("button");
  create.type = "button";
  create.className = "ghost";
  create.textContent = "Новая лемма…";
  create.addEventListener("mousedown", (event) => event.stopPropagation());
  create.addEventListener("click", () => openNewLemmaForm(token, box));
  box.append(create);
  input.focus();
}

// «Новая лемма»: citation form, root (√ already typed), meaning from the gloss
function openNewLemmaForm(token, box) {
  box.querySelector(".new-lemma")?.remove();
  const form = document.createElement("form");
  form.className = "new-lemma";
  const field = (label, value, placeholder) => {
    const wrap = document.createElement("label");
    wrap.textContent = label;
    const input = document.createElement("input");
    input.value = value; input.placeholder = placeholder || "";
    input.addEventListener("mousedown", (event) => event.stopPropagation());
    wrap.append(input); form.append(wrap);
    return input;
  };
  const at = hostCell(token);
  const lemma = field("словарная форма", citationGuess(token), "напр. afhem yafhem / lōġṯa");
  const root = field("корень", "√", "");
  const meaning = field("значение", at >= 0 ? lexicalHead(token.morphemes[at].gloss) : "", "");
  const category = field("грамматическая категория", token.grammatical_info || "",
    "напр. Preterite / Infectum / Imperative");
  const categoryList = document.createElement("datalist");
  categoryList.id = `gi-${token.token_id}`;
  category.setAttribute("list", categoryList.id);
  form.append(categoryList);
  projectRequest("/api/lift-lexicon").then((status) => {
    (status.grammatical_infos || []).forEach((value) => categoryList.append(new Option(value, value)));
  }).catch(() => {});
  const save = document.createElement("button");
  save.type = "submit"; save.textContent = "Создать и привязать";
  save.addEventListener("mousedown", (event) => event.stopPropagation());
  form.append(save);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!lemma.value.trim()) { lemma.focus(); return; }
    save.disabled = true;
    try {
      const response = await fetch(`/api/documents/${DOC_ID}/tokens/${token.token_id}/new-lemma`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Editor-Client": CLIENT_ID },
        body: JSON.stringify({ lemma: lemma.value, root: root.value.replace(/^√\s*$/, ""),
                               gloss: meaning.value, grammatical_info: category.value }),
      });
      const body = await response.json().catch(() => ({}));
      if (response.status === 409 && body.detail && body.detail.candidates) {
        const found = body.detail.candidates.map((c) => c.lemma
          + ((c.grammatical_infos || []).length ? ` · ${(c.grammatical_infos || []).join("/")}` : "")
          + (c.root ? ` √${c.root}` : "")).join(", ");
        reuseFeedback(`Такая лемма уже есть: ${found}. Выберите её в поиске выше.`, true);
        return;
      }
      if (!response.ok) throw new Error(body.detail || response.status);
      Object.assign(token, body.token);
      setHistoryButtons(body.history); render();
      reuseFeedback(`Создана лемма «${token.lemma}»; слово привязано к ней. Она уже в Словнике.`);
    } catch (error) {
      reuseFeedback(`Не удалось создать лемму: ${error.message}`, true);
    } finally { save.disabled = false; }
  });
  box.append(form);
  lemma.focus();
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------
function glossText(cell) {
  // A unique corpus/LIFT proposal is visible immediately, but remains only
  // a proposal. renderCell gives it the `proposed` class, and nothing is
  // persisted until the author explicitly accepts/edits the cell.
  if (cell.gloss) return cell.gloss;
  if (cell.suggested) return cell.suggested;
  return "??";
}

// ---------------------------------------------------------------------------
// the ᴬ…ᴬ brackets of an Arabic insert, raised
//
// Mirrors src/maalula_glosser/editor/arabic_marks.py; the export raises the same
// characters, so what is on screen is what lands in the document. Found in pairs
// only: Maalula has ordinary words beginning with a capital A (Ana, Amērca) and
// no rule about a single letter can tell them from an opening bracket.
// ---------------------------------------------------------------------------
const isLetter = (c) => !!c && /\p{L}/u.test(c);
const opensMark = (w) => w.length > 1 && w[0] === "A" && isLetter(w[1]);

function closingOffset(word) {
  let end = word.length;
  while (end > 0 && /\p{P}/u.test(word[end - 1])) end -= 1;
  return end > 1 && word[end - 1] === "A" && isLetter(word[end - 2]) ? end - 1 : null;
}

function markOffsets(text) {
  const marks = new Set();
  let pending = null;
  let at = 0;
  const words = text.split(/(\s+)/);
  words.forEach((piece) => {
    if (!/\s/.test(piece) && piece) {
      let closing = closingOffset(piece);
      if (closing === null && pending !== null) {
        let end = piece.length;
        while (end > 0 && /\p{P}/u.test(piece[end - 1])) end -= 1;
        if (end === 1 && piece[0] === "A") closing = 0;
      }
      if (opensMark(piece) && closing !== null && piece.length > 2) {
        marks.add(at); marks.add(at + closing); pending = null;
      } else if (opensMark(piece)) {
        pending = at;
      } else if (closing !== null && pending !== null) {
        marks.add(pending); marks.add(at + closing); pending = null;
      }
    }
    at += piece.length;
  });
  return marks;
}

// Appends `text` to `host`, raising whichever characters `marks` names. `base`
// is where this text starts inside the string the marks were measured on.
function appendMarked(host, text, marks, base) {
  let plain = "";
  const flush = () => { if (plain) { host.append(plain); plain = ""; } };
  for (let i = 0; i < text.length; i += 1) {
    if (marks && marks.has(base + i)) {
      flush();
      const sup = document.createElement("sup");
      sup.className = "arabic-mark";
      sup.textContent = text[i];
      host.append(sup);
    } else {
      plain += text[i];
    }
  }
  flush();
}

function morphologyText(text) { return (text || "").replaceAll("‿", " "); }

function renderCell(cell, marks, base) {
  cell = {...cell, sep: morphologyText(cell.sep), form: morphologyText(cell.form),
    gloss: morphologyText(cell.gloss), suggested: morphologyText(cell.suggested)};
  const wrap = document.createElement("span");
  wrap.className = "cell";

  const morph = document.createElement("span");
  morph.className = "morph";
  if (cell.sep) morph.innerHTML = `<span class="sep">${cell.sep}</span>`;
  appendMarked(morph, cell.form || "", marks, (base || 0) + (cell.sep ? 1 : 0));
  wrap.append(morph);

  const gloss = document.createElement("span");
  gloss.className = `gloss ${cell.status}` + (cell.origin === "user" ? " user" : "");
  if (cell.pending) gloss.classList.add("pending");
  // The gap keeps reading ?? — a proposal is not an answer — but it says so,
  // because a suggestion nobody can see until they click is no proposal at all.
  if (!cell.gloss && cell.suggested) {
    gloss.classList.add("proposed");
    gloss.title = `Программа предлагает: ${cell.suggested} — нажмите, чтобы принять`;
  }
  const otherLabels = (cell.alternatives || []).map((item) => item.gloss).filter(Boolean);
  if (otherLabels.length) {
    gloss.title += `${gloss.title ? ". " : ""}Другие разборы: ${otherLabels.join(", ")}`;
  }
  gloss.textContent = glossText(cell);
  gloss.dataset.morphId = cell.morph_id;
  gloss.tabIndex = 0;
  wrap.append(gloss);
  return wrap;
}

function isSpeaker(token) {
  return (token.morphemes || []).some((m) => m.gloss === "#speaker");
}


function renderToken(token) {
  const el = document.createElement("div");
  // a speaker tag keeps its column but is dimmed: it is transcription, not
  // language, and it is dropped from the exported gloss rows
  el.className = "token" + (isSpeaker(token) ? " speaker" : "")
    + (token.locked ? " unbound" : "");
  el.dataset.tokenId = token.token_id;
  el.dataset.origin = token.imported_igt ? "source" : token.locked ? "local" : "auto";
  const provenance = token.imported_igt ? "IGT из источника" : token.locked
    ? "Ручная правка; режим запоминания — в меню «Слово»" : "Автоматический разбор";
  el.title = provenance;
  if (token.locked) {
    el.title += ". Строка 2 защищена — правка строки 1 её не меняет";
  }
  if ((token.uncertainty || []).length) el.title += ". Требует проверки";

  const cells = document.createElement("div");
  cells.className = "cells";
  let at = token.__base || 0;
  (token.morphemes || []).forEach((m) => {
    cells.append(renderCell(m, token.__marks, at));
    at += (m.sep ? 1 : 0) + (m.form || "").length;
  });
  el.append(cells);

  // Parser suggestions appear only on hover.

  const visibleGi = token.grammatical_info
    || ((token.grammatical_infos || []).length === 1 ? token.grammatical_infos[0] : "");
  if (visibleGi) {
    const category = document.createElement("div");
    category.className = "lexical-category";
    category.textContent = visibleGi;
    category.title = token.lift_entry_id
      ? `FieldWorks entry: ${token.lift_entry_id}`
      : "Грамматическая категория лексической записи";
    el.append(category);
  }
  el.append(renderColumnTools(token));
  // room for the always-visible variants button (editor.css)
  if ((token.alternatives || []).length) el.classList.add("has-variants");
  if (!isSpeaker(token)) el.append(renderWordActions(token));
  return el;
}

// Joining and splitting columns. `la` and `ču` attach to some hosts and not
// others, and an Arabic insert is a phrase glossed as a unit — neither is
// something the program can work out, so both are one click.
function tokenHasYellow(token) {
  return (token.morphemes || []).some((cell) =>
    cell.status === "uncertain" ||
    (!String(cell.gloss || "").trim() && String(cell.suggested || "").trim())
  );
}

async function approveYellow(token) {
  if (!LOCK.you_are_holder) {
    reuseFeedback(
      "Правка требует права редактирования.",
      true
    );
    return;
  }

  try {
    const res = await api(
      `/tokens/${token.token_id}/approve-yellow`,
      { method: "POST" }
    );

    Object.assign(token, res.token);
    setHistoryButtons(res.history);
    render();

    reuseFeedback(
      res.approved
        ? `Подтверждено: ${res.approved}`
        : "В этом слове нет жёлтых предложений."
    );
  } catch (error) {
    reuseFeedback(
      `Не удалось подтвердить анализ: ${error.message}`,
      true
    );
  }
}

function renderColumnTools(token) {
  const bar = document.createElement("div");
  bar.className = "column-tools";
  const add = (label, title, run, className) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.title = title;
    if (className) b.className = className;
    b.addEventListener("click", (e) => { e.stopPropagation(); run(); });
    bar.append(b);
  };
  if (tokenHasYellow(token)) {
    add(
      "✓",
      "Подтвердить жёлтые анализы этого слова",
      () => approveYellow(token),
      "approve-yellow"
    );
  }

  if ((token.alternatives || []).length) {
    // The hover popover (`showAlternatives`) is easy to miss — clicking a
    // `.gloss` cell instead opens its own plain text editor (`openCellEditor`,
    // which also closes any open popover), so an author who clicks a word
    // stuck on "several competing analyses" never sees the picker at all.
    // This button is the click-discoverable path to the same popover. It
    // comes first and stays in sight, faint, without the column being
    // hovered (`.variants` in editor.css): a word with several readings
    // shows that it has them.
    add("варианты", "Слово можно прочитать по-разному — выбрать разбор",
        () => {
          const anchor = document.querySelector(
            `.token[data-token-id="${CSS.escape(token.token_id)}"] .gloss`);
          if (anchor) showAlternatives(anchor);
        }, "variants");
  }
  add("=→", "Приклеить следующее слово через = (la=iṯken)",
      () => rearrange(token.token_id, "/merge", { count: 2, joiner: "=" }));
  add("ᴬ→", "Слить со следующим в одну глоссу — арабская вставка",
      () => rearrange(token.token_id, "/merge", { count: 2, joiner: " " }));
  if ((token.surface || "").match(/[\s‿]/)) {
    add("⊣", "Разделить обратно на слова", () => rearrange(token.token_id, "/split"));
  }
  if (!isSpeaker(token)) {
    const origin = document.createElement("span");
    origin.className = "token-origin";
    origin.textContent = token.imported_igt ? "IGT" : token.locked ? "правка" : "авто";
    origin.title = "Источник разбора; не показатель правильности";
    bar.append(origin);
  }
  return bar;
}

async function rearrange(tokenId, path, body) {
  try {
    const res = await api(`/tokens/${tokenId}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const sentence = DOC.sentences.find((s) => s.sentence_id === res.sentence_id);
    if (sentence) sentence.tokens = res.tokens;
    render();
    setHistoryButtons(res.history);
    if (res.topology) await offerReadingAgain(null, null, "", res.topology);
  } catch (err) {
    window.alert("Здесь склеивать или делить нечего.");
  }
}

// ---------------------------------------------------------------------------
// line 1: running text, not a row of columns
//
// Lining line 1 up over line 2 stretched every source word to the width of its
// widest gloss and tore the sentence apart with white space. It reads as a
// paragraph instead, and the link to the block below is shown on hover.
// ---------------------------------------------------------------------------
function renderWord(token, marks, base) {
  const w = document.createElement("span");
  w.className = "word" + (isSpeaker(token) ? " speaker" : "");
  w.contentEditable = "true";
  w.spellcheck = false;
  appendMarked(w, token.surface || "", marks, base || 0);
  w.dataset.original = token.surface;
  w.dataset.tokenId = token.token_id;
  return w;
}

function punctuation(text) {
  const span = document.createElement("span");
  span.className = "punct";
  span.textContent = text;
  return span;
}

// A sandhi group is one written word analysed as several. Its pieces are
// printed tied together, so line 1 keeps the spelling of the transcription —
// including the punctuation, which lives on the group and not on any piece.
// Line 1 is the stored line, with the editable words located inside it. Built
// out of the token surfaces instead, it silently lost whatever the tokenizer
// discards — a lone `?` typed between two words disappeared off the screen
// while staying in the file. Walking the line keeps every character, and the
// ties and stray punctuation of a sandhi group come for free.
function sourcePieces(sentence) {
  const line = sentence.source_line || "";
  const tokens = sentence.tokens || [];
  const out = [];
  let at = 0;
  for (const token of tokens) {
    const surface = token.surface || "";
    const found = surface ? line.indexOf(surface, at) : -1;
    if (found < 0) {
      // Never reconstruct the transcription from a lossy tokenisation. Keep
      // the literal source; Alt still edits the complete line in this case.
      return [{ text: line, kind: "unaligned" }];
    }
    if (found > at) out.push({ text: line.slice(at, found), kind: "punct" });
    out.push({ text: surface, kind: "word", token });
    at = found + surface.length;
  }
  if (at < line.length) out.push({ text: line.slice(at), kind: "punct" });
  return out;
}

function renderSourceLine(sentence) {
  const p = document.createElement("p");
  p.className = "source";
  const pieces = sourcePieces(sentence);
  if (pieces.some((piece) => piece.kind === "unaligned")) {
    p.dataset.alignment = "review";
    p.title = "Привязка слов требует проверки. Исходный текст сохранён. Alt — правка строки.";
  }
  // the brackets are decided over the whole line, not over one word
  const marks = markOffsets(pieces.map((x) => x.text).join(""));
  let at = 0;
  pieces.forEach((piece) => {
    if (piece.kind === "word") p.append(renderWord(piece.token, marks, at));
    else if (piece.text.trim() === "") p.append(document.createTextNode(piece.text));
    else p.append(punctuation(piece.text));
    at += piece.text.length;
  });
  return p;
}

// ---------------------------------------------------------------------------
// Alt: line 1 as a line, not as a row of words
//
// Punctuating a finished text, or slipping a word into the middle of it, is a
// line-level edit — doing it word by word means deciding in advance which word
// the comma belongs to. Holding Alt selects the whole line; the per-word
// editing is untouched and stays the default.
// ---------------------------------------------------------------------------
let editingLine = null;

function setLineMode(on) {
  document.body.classList.toggle("line-mode", on && !editingLine);
}
document.addEventListener("keydown", (e) => { if (e.key === "Alt") setLineMode(true); });
document.addEventListener("keyup", (e) => { if (e.key === "Alt") setLineMode(false); });
window.addEventListener("blur", () => setLineMode(false));

document.addEventListener("mousedown", (e) => {
  if (!e.altKey || editingLine) return;
  const source = e.target.closest(".source");
  if (!source) return;
  // the words are contenteditable, so the caret has to be refused first
  e.preventDefault();
  setLineMode(false);
  openLineEditor(source);
}, true);

function openLineEditor(sourceEl) {
  const block = sourceEl.closest(".sentence");
  const sentence = DOC.sentences.find(
    (s) => s.sentence_id === block.dataset.sentenceId);
  if (!sentence) return;

  const area = document.createElement("textarea");
  area.className = "source-input";
  area.value = sentence.source_line || "";
  area.rows = Math.max(1, Math.ceil(area.value.length / 90));
  sourceEl.replaceWith(area);
  area.focus();
  editingLine = { sentence, area, sourceEl };

  const finish = async (save) => {
    if (!editingLine) return;
    const value = area.value.trim();
    editingLine = null;
    if (save && value && value !== (sentence.source_line || "")) {
      try {
        const res = await api(`/sentences/${sentence.sentence_id}/source`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: value }),
        });
        Object.assign(sentence, res.sentence);
        setHistoryButtons(res.history);
        render();
        return;
      } catch (err) {
        window.alert("Строка не может быть пустой.");
      }
    }
    area.replaceWith(renderSourceLine(sentence));
  };
  area.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); finish(true); }
    if (e.key === "Escape") { e.preventDefault(); finish(false); }
  });
  area.addEventListener("blur", () => finish(true));
}

// Line 2 is one string for the whole block as far as the brackets are
// concerned; each token remembers where it starts in it.
function measureBlock(sentence) {
  const tokens = sentence.tokens || [];
  const line = tokens.map(segmentationOf).join(" ");
  const marks = markOffsets(line);
  let at = 0;
  tokens.forEach((token) => {
    token.__marks = marks;
    token.__base = at;
    at += segmentationOf(token).length + 1;
  });
}

// Hovering either half lights up the other, which is what replaces the column
// alignment as the way of seeing which block a word belongs to.
let linked = null;
function setLinked(tokenId) {
  if (linked === tokenId) return;
  document.querySelectorAll(".linked").forEach((el) => el.classList.remove("linked"));
  linked = tokenId;
  if (!tokenId) return;
  const q = `[data-token-id="${CSS.escape(tokenId)}"]`;
  document.querySelectorAll(`.word${q}, .token${q}`)
    .forEach((el) => el.classList.add("linked"));
}

const blockOf = (tokenId) =>
  document.querySelector(`.token[data-token-id="${CSS.escape(tokenId)}"]`);

function renderComments(sentence) {
  if (!(sentence.comments || []).length) return null;
  const ul = document.createElement("ul");
  ul.className = "comments";
  sentence.comments.forEach((c) => {
    const li = document.createElement("li");
    const hit = findToken(c.anchor?.token_id);
    const word = hit ? hit.t.surface : "";
    li.innerHTML =
      `<span class="anchor">✎</span> ` +
      (word ? `<span class="word">${escapeHtml(word)}:</span> ` : "") +
      `<span class="text">${escapeHtml(c.text)}</span> ` +
      `<span class="actions">` +
      `<a data-act="edit">изменить</a> <a data-act="delete">удалить</a>` +
      `</span>`;
    li.querySelector('[data-act="edit"]').addEventListener("click", () => {
      const anchorEl = blockOf(c.anchor?.token_id) || li;
      openCommentComposer(anchorEl, c.anchor?.token_id, c.anchor?.morph_id, c);
    });
    li.querySelector('[data-act="delete"]').addEventListener("click", async () => {
      if (!window.confirm("Удалить примечание?")) return;
      await api(`/comments/${c.comment_id}`, { method: "DELETE" });
      DOC = await api("");
      render();
    });
    ul.append(li);
  });
  return ul;
}

// The export links point at the live document; the server rebuilds all three
// files on request, so a download always reflects the latest edit.
function wireExportLinks() {
  ["docx", "json", "csv"].forEach((fmt) => {
    document.getElementById(`dl-${fmt}`).href =
      `/api/documents/${DOC_ID}/export/${fmt}`;
  });
}


// A translation belongs to one line, so it is edited on that line. An empty one
// still gets a slot: otherwise there is nothing to click to add it.
//
// A translator sometimes carries the same ᴬ…ᴬ bracket into the English gloss
// itself (an aside standing in for the source's own switch into Arabic), so
// this needs the same raised-marker rendering as the source line and the
// morpheme block, not a plain textContent assignment.
function renderTranslation(sentence) {
  const tr = document.createElement("p");
  tr.className = "translation" + (sentence.translation ? "" : " empty");
  if (sentence.translation) {
    appendMarked(tr, sentence.translation, markOffsets(sentence.translation), 0);
  } else {
    tr.textContent = "+ перевод строки";
  }
  tr.title = "Нажмите, чтобы изменить перевод этой строки";
  tr.addEventListener("click", () => openTranslationEditor(sentence, tr));
  return tr;
}

// Deleting a line takes the whole block with it — glosses, translation,
// comments. It is on the undo stack like every other edit, but a stray click on
// a control that erases a paragraph deserves the extra keystroke.
function renderDropLine(sentence) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "drop-line";
  btn.title = "Удалить строку";
  btn.textContent = "✕";
  btn.addEventListener("click", async () => {
    if (!LOCK.you_are_holder) {
      reuseFeedback("Читателям доступны только обсуждения — правки требуют права редактирования.", true);
      return;
    }
    if (!window.confirm("Удалить эту строку целиком?")) return;
    const res = await api(`/sentences/${sentence.sentence_id}`,
                          { method: "DELETE" });
    DOC.sentences = DOC.sentences.filter(
      (s) => s.sentence_id !== sentence.sentence_id);
    render();
    setHistoryButtons(res.history);
  });
  return btn;
}

// "проверено": marks every gloss on this line as checked, auto-glossed ones
// included, and credits whoever ticked it at REVIEW_RATE rubles/word — the
// running total is shown after every toggle, not kept on screen permanently.
// No visible label: it sits left of "Блок N" in the locator row, where a
// word next to it would crowd that line, and the title tooltip says what
// it does.
function renderReviewCheckbox(sentence) {
  const label = document.createElement("label");
  label.className = "review-checkbox";
  label.title = sentence.reviewed_by
    ? `Проверено: ${sentence.reviewed_by}`
    : "Отметить строку как полностью проверенную";
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = !!sentence.reviewed_by;
  box.addEventListener("change", () => setReviewed(sentence, box));
  label.append(box);
  return label;
}

async function setReviewed(sentence, checkbox) {
  const reviewed = checkbox.checked;
  if (!LOCK.you_are_holder) {
    reuseFeedback("Читателям доступны только обсуждения — правки требуют права редактирования.", true);
    checkbox.checked = !reviewed;
    return;
  }
  const name = editorName();
  if (reviewed && !name.trim()) {
    window.alert("Сначала укажите своё имя вверху страницы.");
    checkbox.checked = false;
    return;
  }
  try {
    const res = await api(`/sentences/${sentence.sentence_id}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reviewed, name }),
    });
    sentence.reviewed_by = res.reviewed_by;
    sentence.reviewed_word_count = res.reviewed_word_count;
    checkbox.closest("label").title = sentence.reviewed_by
      ? `Проверено: ${sentence.reviewed_by}` : "Отметить строку как полностью проверенную";
    const mine = res.totals[name] || 0;
    reuseFeedback(`${name}: ${mine}₽`);
    renderLockBar();
  } catch (err) {
    checkbox.checked = !reviewed;
    reuseFeedback("Не удалось сохранить отметку: " + err.message, true);
  }
}


function unreviewedSentenceCount() {
  return (DOC?.sentences || []).filter(
    (sentence) => !sentence.reviewed_by
  ).length;
}


async function reviewWholeDocument(button) {
  if (!LOCK.you_are_holder) {
    reuseFeedback(
      "Читателям доступны только обсуждения — правки требуют права редактирования.",
      true
    );
    return;
  }

  const name = editorName().trim();

  if (!name) {
    window.alert("Сначала укажите своё имя вверху страницы.");
    return;
  }

  const pending = unreviewedSentenceCount();

  if (!pending) {
    reuseFeedback("Весь текст уже отмечен как проверенный.");
    return;
  }

  const message = pending === 1
    ? "Отметить последнюю непроверенную строку как проверенную?"
    : `Отметить все ${pending} непроверенных строк как проверенные?`;

  if (!window.confirm(message)) {
    return;
  }

  button.disabled = true;

  try {
    const res = await api("/review-all", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        reviewed: true,
        name,
      }),
    });

    const byId = new Map(
      (res.sentences || []).map(
        (row) => [row.sentence_id, row]
      )
    );

    for (const sentence of DOC.sentences || []) {
      const saved = byId.get(sentence.sentence_id);

      if (!saved) {
        continue;
      }

      sentence.reviewed_by = saved.reviewed_by;
      sentence.reviewed_word_count =
        saved.reviewed_word_count;
    }

    render();
    renderLockBar();

    const mine = res.totals?.[name] || 0;
    const lines = res.changed_sentences || 0;
    const words = res.changed_words || 0;

    reuseFeedback(
      `Проверено: ${lines} строк, ${words} слов. ${name}: ${mine}₽`
    );
  } catch (err) {
    reuseFeedback(
      "Не удалось отметить весь текст: " + err.message,
      true
    );

    button.disabled = false;
  }
}


async function openTranslationEditor(sentence, el) {
  const input = document.createElement("input");
  input.className = "translation-input";
  input.value = sentence.translation || "";
  input.placeholder = "перевод этой строки";
  el.replaceWith(input);
  input.focus();

  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    if (save) {
      const res = await api(
        `/sentences/${sentence.sentence_id}/translation`,
        { method: "PATCH", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: input.value }) });
      sentence.translation = res.translation;
    }
    input.replaceWith(renderTranslation(sentence));
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    if (e.key === "Escape") { finish(false); }
  });
  input.addEventListener("blur", () => finish(true));
}

// ---------------------------------------------------------------------------
// discussion threads — distinct from примечания: named replies, open to
// every reader regardless of who holds the pen, never on the Undo stack
// ---------------------------------------------------------------------------
function relativeTime(iso) {
  if (!iso) return "";
  const diffSec = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diffSec < 30) return "только что";
  if (diffSec < 3600) return `${Math.max(1, Math.round(diffSec / 60))} мин назад`;
  if (diffSec < 86400) return `${Math.round(diffSec / 3600)} ч назад`;
  return new Date(iso).toLocaleDateString("ru-RU");
}

function jsonPost(path, body) {
  return api(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function renderDiscussions(sentence) {
  const threads = sentence.discussions || [];
  if (!threads.length) return null;
  const ul = document.createElement("ul");
  ul.className = "discussions";
  threads.forEach((thread) => {
    const li = document.createElement("li");
    li.className = "thread" + (thread.resolved ? " resolved" : "");
    const hit = findToken(thread.anchor?.token_id);
    if (hit) {
      const head = document.createElement("div");
      head.className = "word";
      head.textContent = `${hit.t.surface}:`;
      li.append(head);
    }
    (thread.messages || []).forEach((m) => {
      const row = document.createElement("div");
      row.className = "msg";
      row.innerHTML =
        `<span class="author">${escapeHtml(m.author || "?")}:</span> ` +
        `<span class="text">${escapeHtml(m.text)}</span> ` +
        `<span class="when">${relativeTime(m.created)}</span>`;
      li.append(row);
    });

    const actions = document.createElement("div");
    actions.className = "thread-actions";
    const toggle = document.createElement("a");
    toggle.textContent = thread.resolved ? "вернуть" : "решено";
    toggle.addEventListener("click", async () => {
      const res = await jsonPost(
        `/discussions/${thread.thread_id}/${thread.resolved ? "reopen" : "resolve"}`, {});
      Object.assign(thread, res.thread);
      render();
    });
    actions.append(toggle);
    li.append(actions);

    const replyRow = document.createElement("div");
    replyRow.className = "reply-row";
    const input = document.createElement("input");
    input.placeholder = "ответить…";
    const send = document.createElement("button");
    send.type = "button";
    send.textContent = "Ответить";
    const submitReply = async () => {
      const text = input.value.trim();
      if (!text) return;
      const res = await jsonPost(`/discussions/${thread.thread_id}/messages`,
        { author: editorName(), text });
      Object.assign(thread, res.thread);
      input.value = "";
      render();
    };
    send.addEventListener("click", submitReply);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); submitReply(); }
    });
    replyRow.append(input, send);
    li.append(replyRow);

    ul.append(li);
  });
  return ul;
}

// Start a new thread in the same popover the hover/autocomplete UI and the
// примечание composer already use — see `openCommentComposer` above, which
// this mirrors, except a discussion always needs a name and is never gated
// by the pen.
function openDiscussionComposer(anchorEl, tokenId) {
  pop.innerHTML = "";
  const box = document.createElement("div");
  box.className = "comment-compose";
  const nameField = document.createElement("input");
  nameField.value = editorName();
  nameField.placeholder = "ваше имя";
  const textarea = document.createElement("textarea");
  textarea.placeholder = "Начать обсуждение этого места…";
  box.append(nameField, textarea);
  const row = document.createElement("div");
  row.className = "row";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.textContent = "Отмена";
  cancel.addEventListener("click", closePopover);
  const save = document.createElement("button");
  save.type = "button";
  save.textContent = "Отправить";
  save.addEventListener("click", submit);
  row.append(cancel, save);
  box.append(row);
  pop.append(box);
  placePopover(anchorEl);
  nameField.value ? textarea.focus() : nameField.focus();

  async function submit() {
    const text = textarea.value.trim();
    const author = nameField.value.trim();
    if (!text) return;
    setEditorName(author);
    await jsonPost("/discussions", { token_id: tokenId, author, text });
    closePopover();
    DOC = await api("");
    render();
  }

  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
    else if (e.key === "Escape") { e.preventDefault(); closePopover(); }
    e.stopPropagation();
  });
}

// ---------------------------------------------------------------------------
// the single-editor lock: read-only by default, "take the pen" to edit
// ---------------------------------------------------------------------------
function renderLockBar() {
  const bar = document.getElementById("lock-bar");
  if (!bar) return;
  bar.innerHTML = "";
  bar.classList.toggle("editing", LOCK.you_are_holder);
  document.body.classList.toggle("read-only-mode", !LOCK.you_are_holder);
  document.getElementById("undo").disabled = !HISTORY_STATE.can_undo || !LOCK.you_are_holder;
  document.getElementById("redo").disabled = !HISTORY_STATE.can_redo || !LOCK.you_are_holder;

  const status = document.createElement("span");
  const nameInput = document.createElement("input");
  nameInput.className = "editor-name";
  nameInput.placeholder = "ваше имя";
  nameInput.value = editorName();
  nameInput.addEventListener("change", () => setEditorName(nameInput.value.trim()));
  const button = document.createElement("button");
  button.type = "button";

  if (LOCK.you_are_holder) {
    status.textContent = "Вы редактируете";
    button.textContent = "Отдать перо";
    button.onclick = async () => {
      LOCK = await jsonPost("/lock/release", { client_id: CLIENT_ID });
      renderLockBar();
    };
  } else if (LOCK.holder) {
    status.textContent =
      `Редактирует: ${LOCK.holder.name || "аноним"} · последняя правка ${relativeTime(DOC?.updated)}`;
    button.textContent = "Забрать перо";
    button.onclick = async () => {
      setEditorName(nameInput.value.trim());
      LOCK = await jsonPost("/lock/take-over", { client_id: CLIENT_ID, name: editorName() });
      DOC = await api("");
      render();
      renderLockBar();
    };
  } else {
    status.textContent = "Документ свободен для правки";
    button.textContent = "Начать редактирование";
    button.onclick = async () => {
      setEditorName(nameInput.value.trim());
      LOCK = await jsonPost("/lock/claim", { client_id: CLIENT_ID, name: editorName() });
      renderLockBar();
    };
  }
  const reviewAll = document.createElement("button");
  reviewAll.type = "button";
  reviewAll.className = "review-all";

  const pendingReviews = unreviewedSentenceCount();

  reviewAll.textContent = pendingReviews
    ? `✓ Проверить весь текст (${pendingReviews})`
    : "✓ Весь текст проверен";

  reviewAll.title = pendingReviews
    ? "Отметить все ещё непроверенные строки текущего текста как проверенные"
    : "Все строки этого текста уже отмечены как проверенные";

  reviewAll.disabled =
    !LOCK.you_are_holder ||
    pendingReviews === 0;

  reviewAll.addEventListener(
    "click",
    () => reviewWholeDocument(reviewAll)
  );

  bar.append(
    status,
    nameInput,
    button,
    reviewAll
  );
}

// A reader whose 423 just arrived, and a fresh page load finding the
// document already held, both land here: the pen is not theirs, so the
// document is re-fetched and the UI drops into read-only.
async function handleLostLock(holder) {
  LOCK = { holder, you_are_holder: false };
  try { DOC = await api(""); render(); } catch (e) { /* best effort */ }
  renderLockBar();
  reuseFeedback(holder
    ? `Права редактирования у ${holder.name || "другого участника"}. Страница переключена в режим чтения.`
    : "Документ больше не заблокирован.", true);
}

async function refreshLock() {
  const status = await fetch(
    `/api/documents/${DOC_ID}/lock?client_id=${encodeURIComponent(CLIENT_ID)}`
  ).then((r) => r.json());
  if (LOCK.you_are_holder && !status.you_are_holder) {
    await handleLostLock(status.holder);
    return;
  }
  // Rebuilding the bar recreates the name <input>, discarding an unfinished
  // edit and the caret with it — skip that on every silent poll tick and only
  // redraw when the lock actually changed hands.
  const changed = JSON.stringify(LOCK) !== JSON.stringify(status);
  LOCK = status;
  if (changed) renderLockBar();
}

// The opener of a free document becomes its editor; a document already held
// stays read-only for everyone else until they explicitly take the pen.
async function claimLockIfFree() {
  const status = await fetch(
    `/api/documents/${DOC_ID}/lock?client_id=${encodeURIComponent(CLIENT_ID)}`
  ).then((r) => r.json());
  LOCK = status.holder ? status
    : await jsonPost("/lock/claim", { client_id: CLIENT_ID, name: editorName() });
  renderLockBar();
}

setInterval(() => { refreshLock().catch(() => { /* try again next tick */ }); }, 15000);

function render() {
  document.getElementById("doc-title").textContent = DOC.title || "(без названия)";
  wireExportLinks();
  const root = document.getElementById("doc");
  root.innerHTML = "";
  (DOC.sentences || []).forEach((sentence, index) => {
    const block = document.createElement("section");
    block.className = "sentence";
    block.dataset.sentenceId = sentence.sentence_id;
    measureBlock(sentence);
    const locator = document.createElement("div");
    locator.className = "sentence-locator";
    locator.append(renderReviewCheckbox(sentence));
    const time = sentence.source_locator?.timecode || sentence.source_locator?.segment_marker || "";
    const locatorLabel = document.createElement("span");
    locatorLabel.textContent = `Блок ${index + 1}${time ? " · " + time : ""}`;
    locator.append(locatorLabel);
    block.append(locator);
    block.append(renderDropLine(sentence));
    block.append(renderSourceLine(sentence));
    const igt = document.createElement("div");
    igt.className = "igt";
    (sentence.tokens || []).forEach((t) => igt.append(renderToken(t)));
    block.append(igt);
    block.append(renderTranslation(sentence));
    const comments = renderComments(sentence);
    if (comments) block.append(comments);
    const discussions = renderDiscussions(sentence);
    if (discussions) block.append(discussions);
    root.append(block);
  });
  markAnnotatedWords();
  renderCounts();
}

// Words with a note or an open discussion get a lilac background, in line 1
// and in their IGT column, so they can be found at a glance.
function markAnnotatedWords() {
  (DOC.sentences || []).forEach((sentence) => {
    const ids = new Set();
    (sentence.comments || []).forEach((c) => { if (c.anchor?.token_id) ids.add(c.anchor.token_id); });
    (sentence.discussions || []).forEach((t) => {
      if (t.anchor?.token_id && !t.resolved) ids.add(t.anchor.token_id);
    });
    ids.forEach((id) => {
      document.querySelectorAll(`.word[data-token-id="${CSS.escape(id)}"], .token[data-token-id="${CSS.escape(id)}"]`)
        .forEach((el) => el.classList.add("annotated"));
    });
  });
}

function renderCounts() {
  const tally = { confident: 0, uncertain: 0, unresolved: 0 };
  DOC.sentences.forEach((s) => s.tokens.filter((t) => !isSpeaker(t)).forEach((t) => t.morphemes.forEach(
    (m) => { tally[m.status] = (tally[m.status] || 0) + 1; })));
  document.getElementById("doc-counts").innerHTML =
    `<span class="chip green" title="Уверенные ячейки, без меток говорящих">Уверенно: ${tally.confident}</span>` +
    `<span class="chip yellow" title="Ячейки, требующие проверки">Проверить: ${tally.uncertain}</span>` +
    `<span class="chip red" title="Ячейки без установленной глоссы">Не разобрано: ${tally.unresolved}</span>`;
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// ---------------------------------------------------------------------------
// model helpers
// ---------------------------------------------------------------------------
function findCell(morphId) {
  for (const s of DOC.sentences) for (const t of s.tokens)
    for (const m of t.morphemes) if (m.morph_id === morphId) return { s, t, m };
  return null;
}
function findToken(tokenId) {
  for (const s of DOC.sentences) for (const t of s.tokens)
    if (t.token_id === tokenId) return { s, t };
  return null;
}

// ---------------------------------------------------------------------------
// editing a gloss cell (step 3)
// ---------------------------------------------------------------------------
let editing = null; // { morphId, input }

async function openCellEditor(glossEl) {
  closePopover();
  const morphId = glossEl.dataset.morphId;
  const hit = findCell(morphId);
  const current = hit.m.gloss || hit.m.suggested || "";

  const input = document.createElement("input");
  input.className = "gloss-input";
  input.value = current;
  glossEl.replaceWith(input);
  input.focus();
  input.select();
  editing = { morphId, input, glossEl, lexicalChoice: null };

  const refresh = async (typed = false) => {
    // On opening, show every attested reading for this exact morpheme even if
    // one of them is already in the cell.  After the author starts typing,
    // the same list is filtered by what they type.
    const prefix = input.value;
    const { labels, morpheme_options: morphOptions = [] } = await api(
      `/suggest?morph_id=${encodeURIComponent(morphId)}` +
      `&prefix=${encodeURIComponent(prefix)}&all_morpheme=${typed ? "false" : "true"}`);
    showOptions(input, labels, (label, item) => {
      input.value = label;
      if (editing) editing.lexicalChoice = item?.lexeme_id ? item : null;
      commitCell();
    },
      { allowNew: true });
  };
  input.addEventListener("input", () => {
    if (editing) editing.lexicalChoice = null;
    refresh(true);
  });
  input.addEventListener("keydown", (e) => {
    if (navigatePopover(e)) return;
    if (e.key === "Enter") { e.preventDefault(); commitCell(); }
    if (e.key === "Escape") { cancelCell(); }
  });
  input.addEventListener("blur", () => setTimeout(commitCell, 120));
  refresh(false);
}

async function commitCell() {
  if (!editing) return;
  const { morphId, input, glossEl, lexicalChoice } = editing;
  const value = input.value.trim();
  editing = null;
  closePopover();
  const hit = findCell(morphId);
  try {
  // Read the project setting again so another editor tab can change it.
  PREFERENCES = await projectRequest("/api/editor-preferences");
  document.getElementById("default-reuse").value = String(PREFERENCES.reusable_default);
  const res = await api(`/morphemes/${morphId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      gloss: value,
      reusable: effectiveReusable(hit.t, PREFERENCES),
      lexeme_id: lexicalChoice?.lexeme_id || null,
      sense_id: lexicalChoice?.sense_id || null,
      grammatical_info: lexicalChoice?.grammatical_info || null,
    }),
  });
  const speakerChanged = isSpeaker(res.token) || isSpeaker(hit.t);
  if (speakerChanged) {
    Object.assign(hit.t, res.token);
    redrawToken(hit.t.token_id, hit.t);
  } else {
    Object.assign(hit.m, res.morpheme);
    const { morphemes: _cells, ...metadata } = res.token;
    Object.assign(hit.t, metadata);
    input.replaceWith(renderCell(hit.m, hit.t.__marks, hit.t.__base).querySelector(".gloss"));
    const owner = blockOf(hit.t.token_id);
    owner?.querySelector(".word-actions")?.replaceWith(renderWordActions(hit.t));
    renderCounts();
  }
  reportReuse(res); setHistoryButtons(res.history);
  // a meaning typed into the stem cell of a lexeme that had none is the
  // lexeme's meaning: the server has already put it into the other words of
  // the lexeme in this text, so show them
  if (res.lexical?.recorded) {
    if (res.lexical.updated) { DOC = await api(""); render(); }
    reuseFeedback(`Значение «${res.lexical.gloss}» записано для леммы «${hit.t.lemma || ""}».` +
      (res.lexical.updated ? ` В этом тексте обновлено слов: ${res.lexical.updated}.` : "") +
      " Отменить: «Отменить значение» в меню слова.");
  }
  offerReadingAgain(hit.t.token_id, speakerChanged ? null : morphId, value)
    .catch((error) => reuseFeedback(error.message, true));
  } catch (error) {
    if (!editing) { editing = { morphId, input, glossEl, lexicalChoice }; input.focus(); }
    input.addEventListener("focus", () => {
      editing = { morphId, input, glossEl, lexicalChoice };
    }, { once: true });
    reuseFeedback("Не удалось сохранить глоссу. Введённый текст оставлен в поле: " + error.message, true);
  }
}

// ---------------------------------------------------------------------------
// "and the other N like it"
//
// The same word is glossed the same way over and over. The offer is a proposal,
// never an action: every match is listed with its line, ticked by default, and
// the author unticks whatever is not really the same word.
// ---------------------------------------------------------------------------
let readingOfferSerial = 0;

function readingText(reading) {
  if (Array.isArray(reading)) return reading.map(readingText).join(" | ");
  const cells = reading.cells || [];
  return reading.surface + " · " + cells.map(m => m.sep + m.form).join("")
    + " · " + cells.map(m => m.sep + (m.gloss || "??")).join("");
}

async function offerReadingAgain(tokenId, fallbackMorphId = null, gloss = "", topology = null) {
  const serial = ++readingOfferSerial;
  closeBulk();
  const sourceToken = tokenId ? findToken(tokenId)?.t : null;
  if (!topology && sourceToken && MUTED.has(wordMuteKey(sourceToken.surface))) return;
  const offer = topology || await api(`/tokens/${tokenId}/similar-reading`);
  if (serial !== readingOfferSerial) return;
  if (!topology && MUTED.has(wordMuteKey(offer.form))) return;
  if (!offer.matches.length) {
    if (fallbackMorphId && gloss) await offerTheSameAgain(fallbackMorphId, gloss, serial);
    return;
  }
  const panel = document.getElementById("bulk");
  panel.hidden = false;
  const head = document.createElement("div");
  head.className = "bulk-head";
  head.textContent = topology
    ? `${topology.operation === "merge" ? "Объединение" : "Разделение"} — ещё ${offer.matches.length} совпадающих последовательностей. Применить вместе с разбором?`
    : `${offer.form} — ещё ${offer.matches.length} мест. Перенести весь разбор в отмеченные места?`;
  panel.append(head);
  const list = document.createElement("ul");
  list.className = "bulk-list";
  offer.matches.forEach(match => {
    const row = document.createElement("li");
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox"; box.checked = true; box.value = match.token_id;
    label.append(box, document.createTextNode(
      `${match.context.before} [${match.context.focus}] ${match.context.after}`));
    const change = document.createElement("div");
    change.className = "bulk-now";
    change.textContent = readingText(match.current) + " → " + readingText(match.proposed);
    row.append(label, change); list.append(row);
  });
  const bar = document.createElement("div");
  bar.className = "bulk-bar";
  const all = document.createElement("button");
  all.type = "button"; all.textContent = "снять все";
  all.addEventListener("click", () => {
    const boxes = [...list.querySelectorAll("input")];
    const checked = boxes.some(box => box.checked);
    boxes.forEach(box => { box.checked = !checked; });
    all.textContent = checked ? "отметить все" : "снять все";
  });
  const skip = document.createElement("button");
  skip.type = "button"; skip.textContent = "не надо";
  skip.addEventListener("click", closeBulk);
  const apply = document.createElement("button");
  apply.type = "button"; apply.textContent = "Применить";
  apply.addEventListener("click", async () => {
    const selected = new Set([...list.querySelectorAll("input:checked")].map(box => box.value));
    if (!selected.size) { closeBulk(); return; }
    apply.disabled = true;
    try {
      const targets = Object.fromEntries(offer.matches.filter(m => selected.has(m.token_id))
        .map(m => [m.token_id, m.version]));
      const payload = topology ? {proposal_id: offer.proposal_id, targets}
        : {source_id: offer.source_id, source_version: offer.source_version, targets};
      const res = await api(topology ? "/apply-topology" : "/apply-reading", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      DOC = await api("");
      closeBulk(); render(); setHistoryButtons(res.history);
    } catch (error) {
      reuseFeedback("Не удалось перенести разбор: " + error.message, true);
    } finally { apply.disabled = false; }
  });
  bar.append(all, skip);
  if (!topology && offer.form) bar.append(muteButton(wordMuteKey(offer.form), bareForm(offer.form)));
  bar.append(apply); panel.append(list, bar);
}

async function offerTheSameAgain(morphId, gloss, serial = readingOfferSerial) {
  const { form, matches } = await api(`/similar/${morphId}`);
  if (serial !== readingOfferSerial || !matches.length) return;
  if (MUTED.has(cellMuteKey(form))) return;

  const panel = document.getElementById("bulk");
  panel.innerHTML = "";
  panel.hidden = false;

  const head = document.createElement("div");
  head.className = "bulk-head";
  head.innerHTML = `<b>${escapeHtml(form)}</b> — ещё ` +
    `${matches.length} ${plural(matches.length, "место", "места", "мест")} ` +
    `в этом тексте. Поставить везде <b>${escapeHtml(gloss)}</b>?`;
  panel.append(head);

  const list = document.createElement("ul");
  list.className = "bulk-list";
  matches.forEach((match) => {
    const li = document.createElement("li");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = true;
    box.value = match.morph_id;
    const line = document.createElement("label");
    line.append(box);
    const ctx = document.createElement("span");
    ctx.className = "bulk-context";
    ctx.append(document.createTextNode(match.context.before + " "));
    const focus = document.createElement("b");
    focus.textContent = match.context.focus;
    ctx.append(focus, document.createTextNode(" " + match.context.after));
    line.append(ctx);
    const now = document.createElement("span");
    now.className = `bulk-now ${match.status}`;
    now.textContent = match.gloss || match.suggested || "??";
    line.append(now);
    li.append(line);
    list.append(li);
  });
  panel.append(list);

  const bar = document.createElement("div");
  bar.className = "bulk-bar";
  const all = document.createElement("button");
  all.type = "button";
  all.className = "ghost";
  all.textContent = "снять все";
  all.addEventListener("click", () => {
    const boxes = [...list.querySelectorAll("input")];
    const anyOn = boxes.some((b) => b.checked);
    boxes.forEach((b) => { b.checked = !anyOn; });
    all.textContent = anyOn ? "отметить все" : "снять все";
  });
  const apply = document.createElement("button");
  apply.type = "button";
  apply.textContent = "Применить";
  apply.addEventListener("click", async () => {
    const ids = [...list.querySelectorAll("input:checked")].map((b) => b.value);
    if (!ids.length) { closeBulk(); return; }
    const res = await api("/apply-gloss", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ morph_ids: ids, gloss }),
    });
    res.morphemes.forEach((m) => {
      const hit = findCell(m.morph_id);
      if (hit) Object.assign(hit.m, m);
    });
    closeBulk();
    render();
    setHistoryButtons(res.history);
  });
  const skip = document.createElement("button");
  skip.type = "button";
  skip.className = "ghost";
  skip.textContent = "не надо";
  skip.addEventListener("click", closeBulk);
  bar.append(all, skip);
  bar.append(muteButton(cellMuteKey(form), bareForm(form)));
  bar.append(apply);
  panel.append(bar);
}

function closeBulk() {
  const panel = document.getElementById("bulk");
  panel.hidden = true;
  panel.classList.remove("relink");
  panel.innerHTML = "";
}

// ---------------------------------------------------------------------------
// "a meaning for the whole lexeme"
//
// Same checklist idiom as offerTheSameAgain above, but scoped to one lexeme
// across the whole corpus (data/final/forms.jsonl) rather than to same-surface
// repeats inside this document — see docs/LEXICAL_REUSABLE_SPEC_RU.md §5.
// ---------------------------------------------------------------------------
async function confirmLexicalGloss(token) {
  const current = lexicalHead(token.lexical_gloss || "");
  const gloss = prompt(
    `Значение для «${token.lemma || token.surface}» ` +
    "(применится ко всем формам этой основы):", current);
  if (gloss === null || !gloss.trim()) return;
  let data;
  try {
    data = await api(`/lexeme-forms/${token.token_id}`);
  } catch (error) {
    reuseFeedback(`Не удалось получить формы этой основы: ${error.message}`, true);
    return;
  }
  offerLexemeChecklist(token, gloss.trim(), data);
}

function offerLexemeChecklist(token, gloss, data) {
  const panel = document.getElementById("bulk");
  panel.innerHTML = "";
  panel.hidden = false;

  const head = document.createElement("div");
  head.className = "bulk-head";
  head.innerHTML = `<b>${escapeHtml(data.lemma || token.lemma || "")}</b> — ` +
    `${data.forms.length} ${plural(data.forms.length, "форма", "формы", "форм")} ` +
    `этой основы. Записать значение <b>${escapeHtml(gloss)}</b> для всех, кроме снятых?`;
  panel.append(head);

  const list = document.createElement("ul");
  list.className = "bulk-list";
  data.forms.forEach((form) => {
    const li = document.createElement("li");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = true;
    box.value = form.surface_canonical;
    const line = document.createElement("label");
    line.append(box);
    const surface = document.createElement("b");
    surface.textContent = form.surface_canonical;
    line.append(surface, document.createTextNode(" "));
    const ctx = document.createElement("span");
    ctx.className = "bulk-context";
    // Most forms carry no analysis (spec §2: ~5% do) — a bare surface and
    // frequency is the normal case here, not a missing-data error.
    ctx.textContent = form.segmentation ? `${form.segmentation} / ${form.gloss}`
      : (form.context ? form.context.sentence.slice(0, 90) : "");
    line.append(ctx);
    const freq = document.createElement("span");
    freq.className = "bulk-now";
    freq.textContent = `× ${form.frequency}`;
    line.append(freq);
    li.append(line);
    list.append(li);
  });
  panel.append(list);
  if (data.truncated) {
    const note = document.createElement("div");
    note.className = "bulk-head";
    note.textContent = `Показаны не все формы: ${data.forms.length} из ${data.total_forms}.`;
    panel.append(note);
  }

  const bar = document.createElement("div");
  bar.className = "bulk-bar";
  const all = document.createElement("button");
  all.type = "button";
  all.className = "ghost";
  all.textContent = "снять все";
  all.addEventListener("click", () => {
    const boxes = [...list.querySelectorAll("input")];
    const anyOn = boxes.some((b) => b.checked);
    boxes.forEach((b) => { b.checked = !anyOn; });
    all.textContent = anyOn ? "отметить все" : "снять все";
  });
  const apply = document.createElement("button");
  apply.type = "button";
  apply.textContent = "Применить";
  apply.addEventListener("click", async () => {
    const boxes = [...list.querySelectorAll("input")];
    const confirmed = boxes.filter((b) => b.checked).map((b) => b.value);
    const excluded = boxes.filter((b) => !b.checked).map((b) => b.value);
    try {
      const res = await api(`/tokens/${token.token_id}/lexical/confirm`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gloss, confirmed, excluded }),
      });
      Object.assign(token, res.token);
      if (res.updated) DOC = await api("");
      if (res.history) setHistoryButtons(res.history);
      reuseFeedback(`Значение «${gloss}» записано для всех форм «${data.lemma || ""}».` +
        (res.updated ? ` В этом тексте обновлено слов: ${res.updated}.` : "") +
        (res.kept_checked ? ` Проверенных вручную не трогал: ${res.kept_checked}.` : ""));
    } catch (error) {
      reuseFeedback(`Не удалось записать значение: ${error.message}`, true);
    }
    closeBulk();
    render();
  });
  const skip = document.createElement("button");
  skip.type = "button";
  skip.className = "ghost";
  skip.textContent = "не надо";
  skip.addEventListener("click", closeBulk);
  bar.append(all, skip, apply);
  panel.append(bar);
}

function plural(n, one, few, many) {
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 === 1 && mod100 !== 11) return one;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20)) return few;
  return many;
}

function cancelCell() {
  if (!editing) return;
  const { morphId, input } = editing;
  editing = null;
  closePopover();
  const hit = findCell(morphId);
  input.replaceWith(renderCell(hit.m).querySelector(".gloss"));
}

// ---------------------------------------------------------------------------
// editing line 2: the morphemes and where the boundaries fall
//
// The whole token is edited as one string, because a boundary is not a property
// of a cell — moving one changes how many cells there are.
// ---------------------------------------------------------------------------
function segmentationOf(token) {
  return (token.morphemes || [])
    .map((m) => (m.sep || "") + (m.form || "")).join("");
}

let segmenting = null;

function openSegmentationEditor(tokenEl) {
  if (segmenting) return;
  closePopover();
  const tokenId = tokenEl.dataset.tokenId;
  const hit = findToken(tokenId);
  if (!hit) return;
  const cells = tokenEl.querySelector(".cells");

  const input = document.createElement("input");
  input.className = "seg-input";
  input.value = segmentationOf(hit.t);
  input.title = "Морфемы и границы: -, = и + ; + разделяет морфемы и сохраняется как + ; строка 1 не меняется";
  input.size = Math.max(input.value.length + 1, 6);
  cells.replaceWith(input);
  input.focus();
  input.select();
  segmenting = { tokenId, input, tokenEl };

  const finish = async (save) => {
    if (!segmenting) return;
    const value = input.value.trim();
    segmenting = null;
    if (save && value && value !== segmentationOf(hit.t)) {
      try {
        const res = await api(`/tokens/${tokenId}/segmentation`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: value }),
        });
        Object.assign(hit.t, res.token);
        setHistoryButtons(res.history);
        offerReadingAgain(tokenId).catch(error => reuseFeedback(error.message, true));
      } catch (err) {
        window.alert("Такое членение принять нельзя: "
          + "первая морфема не может начинаться с -, = или +");
      }
    }
    const fresh = renderToken(hit.t);
    tokenEl.replaceWith(fresh);
    renderCounts();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    if (e.key === "Escape") { e.preventDefault(); finish(false); }
  });
  input.addEventListener("blur", () => finish(true));
}

// ---------------------------------------------------------------------------
// Ritter: read-only dictionary hints
// ---------------------------------------------------------------------------
function ritterKindLabel(row) {
  return row.kind === "exact" ? "точное совпадение" : "совпадают согласные";
}

function ritterQueryLabel(row) {
  if (row.query_kind === "stem") return `по основе ${row.query}`;
  if (row.query_kind === "lemma") return `по лемме ${row.query}`;
  if (row.query_kind === "normalized") return `по нормализованной форме ${row.query}`;
  return "по словоформе";
}

function buildRitterResults(ritter, { compact = false } = {}) {
  const box = document.createElement("div");
  box.className = compact ? "ritter-hints compact" : "ritter-hints";

  const head = document.createElement("div");
  head.className = "ritter-title";
  head.textContent = "Ritter";
  box.append(head);

  if (!ritter?.available) {
    const empty = document.createElement("div");
    empty.className = "ritter-empty";
    empty.textContent = "Словарь Ritter недоступен.";
    box.append(empty);
    return box;
  }

  const rows = ritter.results || [];
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "ritter-empty";
    empty.textContent = "Совпадений с тем же согласным каркасом нет.";
    box.append(empty);
    return box;
  }

  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = `ritter-entry ${row.kind || ""}`;

    const top = document.createElement("div");
    top.className = "ritter-entry-head";

    const form = document.createElement("strong");
    form.textContent = row.matched_form || row.headword || "";
    top.append(form);

    const badge = document.createElement("span");
    badge.className = `ritter-badge ${row.kind || ""}`;
    badge.textContent = ritterKindLabel(row);
    top.append(badge);

    item.append(top);

    const meta = document.createElement("div");
    meta.className = "ritter-meta";
    meta.textContent = row.kind === "exact"
      ? ritterQueryLabel(row)
      : `${ritterQueryLabel(row)} · C: ${row.skeleton || row.query_skeleton || "—"}`;
    item.append(meta);

    const raw = document.createElement("div");
    raw.className = "ritter-raw";
    raw.textContent = row.raw_entry || "";
    item.append(raw);

    box.append(item);
  });
  return box;
}

async function loadRitterForWord(token, panel) {
  panel.querySelector(".ritter-word-results")?.remove();
  const holder = document.createElement("div");
  holder.className = "ritter-word-results";
  holder.textContent = "Ищу в Ritter…";
  panel.append(holder);

  const ritter = await api(`/tokens/${token.token_id}/ritter?limit=12`);
  holder.textContent = "";
  holder.append(buildRitterResults(ritter));
}

// ---------------------------------------------------------------------------
// conservative Turoyo verb parser
// ---------------------------------------------------------------------------
function verbParserLocallyEligible(token) {
  if (!token) return false;

  // Настоящий исходный анализ из FLEx неприкосновенен.
  // Само token.flextext НЕ означает, что слово было проанализировано.
  if (token.original_analysis) return false;

  // Импортированный человеческий IGT тоже не переразбираем.
  if (token.imported_igt) return false;

  // Уже подтверждённый человеком токен не трогаем.
  if (token.locked) return false;

  const cells = token.morphemes || [];

  // Любая пользовательская морфема означает ручной анализ.
  if (cells.some((m) => m.origin === "user")) return false;

  // AUTO/LOW-заглушка с ?? считается непроанализированным словом.
  // Такое слово разрешаем отправлять в глагольный parser.
  return true;
}

function verbEvidenceLabel(row) {
  const e = row.evidence;

  if (!e) {
    return "нет подтверждения в FLEx · разбор предложен по морфологическому правилу";
  }

  const match =
    e.match === "exact_stem"
      ? "точная основа"
      : e.match === "consonant_skeleton"
        ? "тот же согласный каркас"
        : "FLEx";

  const form = e.form || e.stem || "";
  const gloss = e.gloss || "";

  const details = [form, gloss].filter(Boolean).join(" / ");

  return details
    ? `${match}: ${details}`
    : match;
}

function buildVerbSuggestions(data, token) {
  const box = document.createElement("div");
  box.className = "verb-parser-hints";

  const head = document.createElement("div");
  head.className = "verb-parser-title";
  head.textContent = "Глагольный парсер";
  box.append(head);

  if (!data?.eligible) {
    const note = document.createElement("div");
    note.className = "verb-parser-empty";
    note.textContent = data?.reason || "Для этого слова парсер отключён.";
    box.append(note);
    return box;
  }

  const rows = data.suggestions || [];
  if (!rows.length) {
    const note = document.createElement("div");
    note.className = "verb-parser-empty";
    note.textContent = data?.reason || "Подходящих глагольных разборов не найдено.";
    box.append(note);
    return box;
  }

  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "verb-parser-entry";

    const seg = document.createElement("div");
    seg.className = "verb-parser-seg";
    seg.textContent = row.segmentation || "";

    const gls = document.createElement("div");
    gls.className = "verb-parser-gloss";
    gls.textContent = row.gloss || "";

    const meta = document.createElement("div");
    meta.className = "verb-parser-meta";
    const confidence = row.confidence === "LOW" ? "LOW" : (row.confidence || "");
    meta.textContent = `${row.rule_label || row.rule || "правило"} · ${confidence}${confidence ? " · " : ""}${verbEvidenceLabel(row)}`;

    const apply = document.createElement("button");
    apply.type = "button";
    apply.className = "verb-parser-apply";
    apply.textContent = "Применить этот разбор";
    apply.disabled = !LOCK.you_are_holder;
    apply.title = LOCK.you_are_holder
      ? "Записать этот разбор только для данного слова"
      : "Чтобы применить разбор, сначала возьмите перо";
    apply.addEventListener("click", async () => {
      apply.disabled = true;
      try {
        const res = await api(`/tokens/${token.token_id}/verb-suggestions/accept`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ suggestion_id: row.suggestion_id }),
        });
        Object.assign(token, res.token);
        setHistoryButtons(res.history);
        reuseFeedback(`Применён глагольный разбор: ${row.segmentation} / ${row.gloss}. Он сохранён только здесь.`);
        render();
      } catch (error) {
        reuseFeedback(`Не удалось применить глагольный разбор: ${error.message}`, true);
        apply.disabled = false;
      }
    });

    item.append(seg, gls, meta, apply);
    box.append(item);
  });
  return box;
}

async function loadVerbSuggestions(token, panel) {
  panel.querySelector(".verb-parser-word-results")?.remove();
  const holder = document.createElement("div");
  holder.className = "verb-parser-word-results";
  holder.textContent = "Проверяю глагольные правила…";
  panel.append(holder);

  const data = await api(`/tokens/${token.token_id}/verb-suggestions?limit=12`);
  holder.textContent = "";
  holder.append(buildVerbSuggestions(data, token));
}

// ---------------------------------------------------------------------------
// popover: autocomplete list and hover alternatives
// ---------------------------------------------------------------------------
function placePopover(anchor) {
  const r = anchor.getBoundingClientRect();
  pop.style.left = `${window.scrollX + r.left}px`;
  pop.style.top = `${window.scrollY + r.bottom + 4}px`;
  pop.hidden = false;
}
function closePopover() { pop.hidden = true; pop.innerHTML = ""; pop.classList.remove("has-ritter"); }
// The comment composer lives in the same #popover the hover/autocomplete UI
// owns; nothing else marks it occupied, so every place that would otherwise
// overwrite or close the popover has to check this first.
const composerOpen = () => !!pop.querySelector(".comment-compose");

function showOptions(anchor, labels, pick, { allowNew = false } = {}) {
  pop.innerHTML = "";
  if (!labels.length && !allowNew) { closePopover(); return; }
  labels.forEach((item, i) => {
    const opt = document.createElement("div");
    opt.className = "opt" + (i === 0 ? " active" : "");
    opt.innerHTML = escapeHtml(item.label) +
      (item.note ? `<span class="note">${escapeHtml(item.note)}</span>` : "");
    opt._choice = item;
    opt.addEventListener("mousedown", (e) => { e.preventDefault(); pick(item.label, item); });
    pop.append(opt);
  });
  if (allowNew) {
    const hint = document.createElement("div");
    hint.className = "new-gloss-hint";
    hint.textContent = "+ новый вариант — введите его в поле и нажмите Enter";
    // Keep the input focused when the hint itself is clicked.
    hint.addEventListener("mousedown", (e) => { e.preventDefault(); anchor.focus(); anchor.select(); });
    pop.append(hint);
  }
  placePopover(anchor);
}

function navigatePopover(e) {
  const opts = [...pop.querySelectorAll(".opt")];
  if (!opts.length || pop.hidden) return false;
  let idx = opts.findIndex((o) => o.classList.contains("active"));
  if (e.key === "ArrowDown") { idx = Math.min(idx + 1, opts.length - 1); }
  else if (e.key === "ArrowUp") { idx = Math.max(idx - 1, 0); }
  else if (e.key === "Tab") { /* accept active */ }
  else return false;
  e.preventDefault();
  opts.forEach((o) => o.classList.remove("active"));
  opts[idx].classList.add("active");
  if (e.key === "Tab" && editing) {
    const choice = opts[idx]._choice || null;
    editing.input.value = choice?.label || opts[idx].textContent.split("\n")[0];
    editing.lexicalChoice = choice?.lexeme_id ? choice : null;
  }
  return true;
}

let hoverTimer = null;
async function showAlternatives(glossEl) {
  const hit = findCell(glossEl.dataset.morphId);
  const { alternatives, dictionary, ritter } = await api(
    `/tokens/${hit.t.token_id}/alternatives`);
  pop.innerHTML = "";
  const tokenEl = glossEl.closest(".token");
  (alternatives || []).forEach((a, index) => {
    const row = document.createElement("div");
    row.className = "alt";
    const frequency = Number.isFinite(Number(a.count)) && Number(a.count) > 0
      ? ` · ×${Number(a.count)}` : "";
    row.innerHTML = `<span class="g">${escapeHtml(a.gloss || "")}</span> ` +
      `<span class="muted">${escapeHtml(a.segmentation || "")}` +
      (a.grammatical_info ? ` · ${escapeHtml(a.grammatical_info)}` : "") +
      (a.lemma ? ` · ${escapeHtml(a.lemma)}` : "") +
      (a.tier ? ` · ${a.tier}` : "") + frequency + `</span>`;
    // Pick this reading: only meaningful when the token is genuinely stuck
    // on competing analyses (`hit.t.alternatives` non-empty — the ordinary
    // hover case has alternatives to *show*, not necessarily any left to
    // *resolve*, e.g. after a re-gloss that left the array stale until the
    // next save). `index` here is the raw index into `token.alternatives` —
    // this endpoint returns that array unfiltered, so it lines up exactly
    // with what `resolve-alternative` expects.
    if ((hit.t.alternatives || []).length) {
      row.classList.add("pickable");
      row.title = "Выбрать этот вариант";
      row.addEventListener("mousedown", (e) => {
        e.preventDefault();
        pickAlternative(hit.t.token_id, index, tokenEl);
      });
    }
    pop.append(row);
  });
  if (dictionary) {
    const d = document.createElement("div");
    d.className = "dict" + (dictionary.match === "surface" ? " surface" : "");
    const label = dictionary.match === "surface"
      ? `словарь (по форме), стр. ${dictionary.page}`
      : `словарь Арнольда, стр. ${dictionary.page}`;
    d.innerHTML = `<a href="${dictionary.url}" target="_blank" rel="noopener">${label} ↗</a>`;
    pop.append(d);
  }
  const addNote = document.createElement("div");
  addNote.className = "comment-add";
  addNote.textContent = "+ примечание";
  addNote.addEventListener("mousedown", (e) => {
    e.preventDefault();
    openCommentComposer(glossEl, hit.t.token_id, glossEl.dataset.morphId, null);
  });
  pop.append(addNote);
  const addDiscussion = document.createElement("div");
  addDiscussion.className = "discussion-add";
  addDiscussion.textContent = "+ обсуждение";
  addDiscussion.addEventListener("mousedown", (e) => {
    e.preventDefault();
    openDiscussionComposer(glossEl, hit.t.token_id);
  });
  pop.append(addDiscussion);
  placePopover(glossEl);
}

// An author resolving a token stuck on "several competing analyses"
// (invariant #7: the server would not guess between them) by picking one of
// its `alternatives` from the hover popover. Mirrors `openSegmentationEditor`'s
// `finish` for how a token mutation's response is applied: merge the
// returned token onto the live model, refresh undo/redo, and re-render.
async function pickAlternative(tokenId, index, tokenEl) {
  const hit = findToken(tokenId);
  if (!hit) return;
  try {
    const res = await api(`/tokens/${tokenId}/resolve-alternative`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ alternative_index: index }),
    });
    Object.assign(hit.t, res.token);
    setHistoryButtons(res.history);
    closePopover();
    if (tokenEl) {
      const fresh = renderToken(hit.t);
      tokenEl.replaceWith(fresh);
    }
    renderCounts();
    offerReadingAgain(tokenId).catch(error => reuseFeedback(error.message, true));
    const propagated = (res.propagation && res.propagation.tokens || 0) - 1;
    if (propagated > 0) {
      window.alert("Заодно исправлено ещё " + propagated
        + " такое же место в других документах.");
    }
  } catch (err) {
    closePopover();
    window.alert("Не удалось выбрать этот вариант: " + err.message);
  }
}

// ---------------------------------------------------------------------------
// line 1 editing + semi-automatic re-gloss (step 4)
// ---------------------------------------------------------------------------
async function reglossToken(word) {
  const newSurface = word.textContent.trim();
  const original = word.dataset.original;
  if (newSurface === original) return;
  const tokenId = word.dataset.tokenId;
  const res = await api(`/tokens/${tokenId}/regloss`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ surface: newSurface }),
  });
  const hit = findToken(tokenId);
  const previous = hit.t.morphemes;
  Object.assign(hit.t, res.token);
  // a hand-written line 2 is not re-glossed behind the author's back, so there
  // is nothing pending to accept or refuse
  if (res.reglossed) {
    hit.t.__prev = { morphemes: previous, surface: original };
    hit.t.morphemes.forEach((m) => { m.pending = true; });
  }
  word.dataset.original = newSurface;
  redrawToken(tokenId, hit.t);
  setHistoryButtons(res.history);
  offerReadingAgain(tokenId).catch(error => reuseFeedback(error.message, true));
}

function redrawToken(tokenId, token) {
  const el = blockOf(tokenId);
  if (el) el.replaceWith(renderToken(token));
  renderCounts();
}

// Alt+R: re-gloss a whole line from scratch, locked tokens and all — the
// escape hatch `reglossToken` above cannot offer, since it protects a hand-
// edited line 2 on purpose (invariant 3).
async function reglossLine(sentenceId) {
  if (!LOCK.you_are_holder) {
    reuseFeedback("Читателям доступны только обсуждения — правки требуют права редактирования.", true);
    return;
  }
  try {
    const res = await api(`/sentences/${sentenceId}/regloss`, { method: "POST" });
    const sentence = DOC.sentences.find((s) => s.sentence_id === sentenceId);
    if (sentence) sentence.tokens = res.sentence.tokens;
    render();
    setHistoryButtons(res.history);
  } catch (err) {
    reuseFeedback("Не удалось переглоссировать строку: " + err.message, true);
  }
}

// Del while a re-gloss is highlighted: keep line 1, restore lines 2-3.
async function revertRegloss(tokenId) {
  const hit = findToken(tokenId);
  if (!hit || !hit.t.__prev) return;
  const res = await api(`/tokens/${tokenId}/keep-morphology`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ morphemes: hit.t.__prev.morphemes }),
  });
  Object.assign(hit.t, res.token);
  delete hit.t.__prev;
  redrawToken(tokenId, hit.t);
}

// Create (existing=null) or edit (existing=the comment) a comment in the
// popover, in place of `window.prompt`: prompt loses multi-line text and is
// blocked outright in some browsers.
function openCommentComposer(anchorEl, tokenId, morphId, existing) {
  pop.innerHTML = "";
  const box = document.createElement("div");
  box.className = "comment-compose";
  const textarea = document.createElement("textarea");
  textarea.value = existing ? existing.text : "";
  textarea.placeholder = "Комментарий к выбранному месту…";
  box.append(textarea);
  const row = document.createElement("div");
  row.className = "row";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.textContent = "Отмена";
  cancel.addEventListener("click", closePopover);
  const save = document.createElement("button");
  save.type = "button";
  save.textContent = "Сохранить";
  save.addEventListener("click", submit);
  row.append(cancel, save);
  box.append(row);
  pop.append(box);
  placePopover(anchorEl);
  textarea.focus();

  async function submit() {
    const text = textarea.value.trim();
    if (!text) return;
    if (existing) {
      await api(`/comments/${existing.comment_id}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
    } else {
      await api("/comments", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token_id: tokenId, morph_id: morphId, text }),
      });
    }
    closePopover();
    DOC = await api("");
    render();
  }

  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
    else if (e.key === "Escape") { e.preventDefault(); closePopover(); }
    e.stopPropagation(); // Del/Backspace here must edit text, not the token
  });
}

// ---------------------------------------------------------------------------
// word context menu (right-click): the one entry point that needs neither a
// keyboard shortcut nor an already-hovered gloss to reach "Примечание"
// ---------------------------------------------------------------------------
let ctxMenu = null;
function closeContextMenu() {
  if (!ctxMenu) return;
  ctxMenu.remove();
  ctxMenu = null;
  document.removeEventListener("mousedown", onCtxMenuOutsideClick, true);
  document.removeEventListener("keydown", onCtxMenuEscape);
}
function onCtxMenuOutsideClick(e) {
  if (ctxMenu && !ctxMenu.contains(e.target)) closeContextMenu();
}
function onCtxMenuEscape(e) { if (e.key === "Escape") closeContextMenu(); }

document.addEventListener("contextmenu", (e) => {
  const owner = e.target.closest(".word, .token");
  if (!owner) return;
  e.preventDefault();
  closeContextMenu();
  const menu = document.createElement("div");
  menu.className = "ctxmenu";
  const item = document.createElement("div");
  item.className = "item";
  item.textContent = "Примечание";
  item.addEventListener("mousedown", (ev) => {
    ev.preventDefault();
    closeContextMenu();
    openCommentComposer(owner, owner.dataset.tokenId, null, null);
  });
  menu.append(item);
  const discussionItem = document.createElement("div");
  discussionItem.className = "item";
  discussionItem.textContent = "Обсуждение";
  discussionItem.addEventListener("mousedown", (ev) => {
    ev.preventDefault();
    closeContextMenu();
    openDiscussionComposer(owner, owner.dataset.tokenId);
  });
  menu.append(discussionItem);
  document.body.append(menu);
  menu.style.left = `${window.scrollX + e.clientX}px`;
  menu.style.top = `${window.scrollY + e.clientY}px`;
  ctxMenu = menu;
  document.addEventListener("mousedown", onCtxMenuOutsideClick, true);
  document.addEventListener("keydown", onCtxMenuEscape);
});

// ---------------------------------------------------------------------------
// event wiring
// ---------------------------------------------------------------------------
document.addEventListener("click", (e) => {
  const gloss = e.target.closest(".gloss");
  if (gloss && !editing) { openCellEditor(gloss); return; }
  const morph = e.target.closest(".morph");
  if (morph && !editing) { openSegmentationEditor(morph.closest(".token")); return; }
  if (!e.target.closest(".popover") && !e.target.closest(".gloss-input")) {
    if (!editing && !composerOpen()) closePopover();
  }
});

document.addEventListener("mouseover", (e) => {
  const gloss = e.target.closest(".gloss");
  if (!gloss || editing || composerOpen()) return;
  clearTimeout(hoverTimer);
  hoverTimer = setTimeout(() => showAlternatives(gloss), 260);
});
document.addEventListener("mouseout", (e) => {
  if (e.target.closest(".gloss") && !editing && !composerOpen()) {
    clearTimeout(hoverTimer);
    setTimeout(() => { if (!pop.matches(":hover")) closePopover(); }, 150);
  }
});

// line 1 ↔ its block: hovering either lights up the pair
document.addEventListener("mousemove", (e) => {
  const owner = e.target.closest(".word, .token");
  setLinked(owner ? owner.dataset.tokenId : null);
});

document.addEventListener("focusout", (e) => {
  if (e.target.classList?.contains("word")) reglossToken(e.target);
});

document.addEventListener("keydown", (e) => {
  // e.code, not e.key: on macOS Alt is a dead-key modifier and e.key for
  // Alt+N arrives as "˜", not "n" — e.code names the physical key either way.
  if (e.altKey && e.code === "KeyN") {
    e.preventDefault();
    setLineMode(false); // Alt alone already turned line-mode on; this isn't that
    const gloss = document.activeElement.closest?.(".gloss")
      || document.querySelector(".gloss:focus");
    const owner = (gloss || document.activeElement).closest?.(".token, .word");
    if (owner) {
      openCommentComposer(gloss || owner, owner.dataset.tokenId,
                          gloss ? gloss.dataset.morphId : null, null);
    }
    return;
  }
  if (e.altKey && e.code === "KeyR") {
    e.preventDefault();
    setLineMode(false);
    // Whichever line the caret or the mouse is currently over — a click
    // leaves focus on a gloss/word inside it, a hover just updates `linked`.
    const sentenceId = document.activeElement.closest?.(".sentence")?.dataset.sentenceId
      || (linked ? findToken(linked)?.s?.sentence_id : null);
    if (!sentenceId) {
      reuseFeedback("Наведите курсор на строку, которую нужно переглоссировать.", true);
      return;
    }
    reglossLine(sentenceId);
    return;
  }
  if (e.key === "Delete" || e.key === "Backspace") {
    // not while the caret is in line 1: there Del is an ordinary keystroke
    if (document.activeElement.classList?.contains("word")) return;
    const owner = document.activeElement.closest?.(".token");
    if (owner && findToken(owner.dataset.tokenId)?.t.__prev) {
      e.preventDefault();
      revertRegloss(owner.dataset.tokenId);
    }
  }
});

// Arriving from the inventory window with ?focus=<token_id>: scroll the token
// into view and mark it, so the form that was clicked is findable in a long text.
function focusRequestedToken() {
  const wanted = new URLSearchParams(window.location.search).get("focus");
  if (!wanted) return;
  const el = blockOf(wanted);
  if (!el) return;
  el.scrollIntoView({ block: "center" });
  el.classList.add("focused");
  // the highlight is a pointer, not a state: drop it once it has been seen
  setTimeout(() => el.classList.remove("focused"), 4000);
}


// ---------------------------------------------------------------------------
// undo / redo, and adding a line
// ---------------------------------------------------------------------------
function setHistoryButtons(state) {
  if (!state) return;
  HISTORY_STATE = state;
  document.getElementById("undo").disabled = !state.can_undo || !LOCK.you_are_holder;
  document.getElementById("redo").disabled = !state.can_redo || !LOCK.you_are_holder;
}

// The server owns the journal, so it also decides what changed. A whole-document
// reload would be wasteful here: an imported corpus text runs to megabytes.
async function step(direction) {
  const res = await api(`/${direction}`, { method: "POST" });
  setHistoryButtons(res);
  const applied = res.applied;
  if (!applied) return;
  if (applied.reload) { DOC = await api(""); render(); return; }

  if (applied.kind === "morpheme" && applied.morpheme) {
    const hit = findCell(applied.morpheme.morph_id);
    if (hit) {
      Object.assign(hit.m, applied.morpheme);
      const el = document.querySelector(
        `[data-morph-id="${CSS.escape(hit.m.morph_id)}"]`);
      if (el) el.replaceWith(renderCell(hit.m).querySelector(".gloss"));
      blockOf(hit.t.token_id)?.querySelector(".word-actions")?.replaceWith(renderWordActions(hit.t));
    }
  } else if (applied.kind === "token" && applied.token) {
    const hit = findToken(applied.token.token_id);
    if (hit) {
      Object.assign(hit.t, applied.token);
      const el = blockOf(hit.t.token_id);
      if (el) el.replaceWith(renderToken(hit.t));
      // a re-gloss can change the surface, so line 1 follows the token back
      const word = document.querySelector(
        `.word[data-token-id="${CSS.escape(hit.t.token_id)}"]`);
      if (word) {
        word.textContent = hit.t.surface;
        word.dataset.original = hit.t.surface;
      }
    }
  } else if (applied.kind === "cells") {
    (applied.morphemes || []).forEach((m) => {
      const hit = findCell(m.morph_id);
      if (hit) Object.assign(hit.m, m);
    });
    render();
  } else if (applied.kind === "tokens") {
    const sentence = DOC.sentences.find(
      (s) => s.sentence_id === applied.sentence_id);
    if (sentence) { sentence.tokens = applied.tokens; render(); }
  } else if (applied.kind === "translation") {
    const sentence = DOC.sentences.find(
      (s) => s.sentence_id === applied.sentence_id);
    if (sentence) {
      sentence.translation = applied.translation;
      render();
    }
  } else if (applied.kind === "comment") {
    const sentence = DOC.sentences.find(
      (s) => s.sentence_id === applied.sentence_id);
    if (sentence) { sentence.comments = applied.comments; render(); }
  }
  renderCounts();
}

document.getElementById("undo").addEventListener("click", () => step("undo"));
document.getElementById("redo").addEventListener("click", () => step("redo"));

document.addEventListener("keydown", (e) => {
  if (!(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== "z") return;
  // never steal the shortcut from a field the user is typing in
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || e.target.isContentEditable) return;
  e.preventDefault();
  step(e.shiftKey ? "redo" : "undo");
});

document.getElementById("new-line").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!LOCK.you_are_holder) {
    reuseFeedback("Читателям доступны только обсуждения — правки требуют права редактирования.", true);
    return;
  }
  const form = e.target;
  const res = await api("/lines", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: form.text.value,
                           translation: form.translation.value }),
  });
  DOC.sentences.splice(res.index, 0, ...(res.sentences || [res.sentence]));
  form.reset();
  render();
  setHistoryButtons(res.history);
});

// ---------------------------------------------------------------------------
// «Леммы…»: re-link the author's checked words to the lexicon
//
// The server computes the plan (GET /relink); the author unticks what should
// stay as it is and picks a lexeme where several fit; POST /relink recomputes
// and applies it as one undo step. Words the lexicon does not know become
// `lexnew_` entries and show up in the Словник queue ("в словарик").
// ---------------------------------------------------------------------------
const RELINK_ORDER = ["choose", "unbound", "rebound", "bound", "new", "renamed"];

async function openRelink() {
  if (!LOCK.you_are_holder) {
    reuseFeedback("Перепривязка лемм требует права редактирования.", true);
    return;
  }
  closeBulk();
  const panel = document.getElementById("bulk");
  panel.hidden = false;
  panel.classList.add("relink");
  panel.textContent = "Считаю…";
  let plan;
  try { plan = await api("/relink"); }
  catch (error) { panel.textContent = "Не удалось: " + error.message; return; }
  panel.innerHTML = "";
  const head = document.createElement("div");
  head.className = "bulk-head";
  if (!plan.changes.length) {
    head.textContent = "Все проверенные слова уже привязаны — менять нечего.";
    const close = document.createElement("button");
    close.type = "button"; close.textContent = "закрыть";
    close.addEventListener("click", closeBulk);
    panel.append(head, close);
    return;
  }
  head.textContent = "Леммы проверенных слов: " + RELINK_ORDER
    .filter(a => plan.counts[a]).map(a => `${plan.labels[a]} — ${plan.counts[a]}`).join("; ");
  panel.append(head);

  const skip = new Set();
  const choices = {};
  const lemmaText = (side) => side.lemma || (side.lemma_id ? side.lemma_id : "—");
  RELINK_ORDER.forEach(action => {
    const rows = plan.changes.filter(c => c.action === action);
    if (!rows.length) return;
    const group = document.createElement("details");
    group.open = action === "choose" || action === "unbound";
    const summary = document.createElement("summary");
    summary.textContent = `${plan.labels[action]} (${rows.length})`;
    group.append(summary);
    const list = document.createElement("ul");
    list.className = "bulk-list";
    rows.forEach(change => {
      const row = document.createElement("li");
      const label = document.createElement("label");
      const box = document.createElement("input");
      box.type = "checkbox"; box.checked = true;
      box.addEventListener("change", () => {
        if (box.checked) skip.delete(change.token_id); else skip.add(change.token_id);
      });
      label.append(box, document.createTextNode(
        ` стр. ${change.line}: ${change.surface} — ${change.gloss}`));
      row.append(label);
      const now = document.createElement("div");
      now.className = "bulk-now";
      if (change.candidates.length) {
        const select = document.createElement("select");
        select.add(new Option("— не выбирать —", ""));
        change.candidates.forEach(c => select.add(new Option(c.lemma || c.lexeme_id, c.lexeme_id)));
        select.addEventListener("change", () => {
          if (select.value) choices[change.token_id] = select.value;
          else delete choices[change.token_id];
        });
        now.append(document.createTextNode(`${lemmaText(change.before)} → `), select);
      } else {
        now.textContent = `${lemmaText(change.before)} → ${lemmaText(change.after)}`;
      }
      if (change.why) {
        const why = document.createElement("small");
        why.className = "muted";
        why.textContent = "  " + change.why;
        now.append(why);
      }
      row.append(now);
      list.append(row);
    });
    group.append(list);
    panel.append(group);
  });

  const bar = document.createElement("div");
  bar.className = "bulk-bar";
  const cancel = document.createElement("button");
  cancel.type = "button"; cancel.textContent = "не надо";
  cancel.addEventListener("click", closeBulk);
  const apply = document.createElement("button");
  apply.type = "button"; apply.textContent = "Применить";
  apply.addEventListener("click", async () => {
    apply.disabled = true;
    try {
      const res = await api("/relink", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ choices, skip: [...skip] }),
      });
      DOC = await api("");
      closeBulk(); render(); renderCounts(); setHistoryButtons(res.history);
      reuseFeedback(`Леммы: изменено ${res.applied}. Новые слова — в очереди Словника.`);
    } catch (error) {
      reuseFeedback("Не удалось перепривязать: " + error.message, true);
    } finally { apply.disabled = false; }
  });
  bar.append(cancel, apply);
  panel.append(bar);
}

document.getElementById("relink").addEventListener("click", openRelink);

// ---------------------------------------------------------------------------
(async function boot() {
  [DOC, PREFERENCES] = await Promise.all([api(""), projectRequest("/api/editor-preferences")]);
  await claimLockIfFree();
  const select = document.getElementById("default-reuse");
  select.value = String(PREFERENCES.reusable_default);
  select.onchange = async () => {
    try {
      PREFERENCES = await projectRequest("/api/editor-preferences", "PATCH", { reusable_default: select.value === "true" });
      reuseFeedback("Общая настройка сохранена для всех текстов проекта. Исключения «Только здесь» остаются в силе.");
    } catch (error) { select.value = String(PREFERENCES.reusable_default); reuseFeedback(error.message, true); }
  };
  const docReuse = document.getElementById("doc-reuse");
  docReuse.value = typeof DOC.reuse_default === "boolean" ? String(DOC.reuse_default) : "inherit";
  docReuse.onchange = async () => {
    const value = docReuse.value === "inherit" ? null : docReuse.value === "true";
    try {
      const res = await api("/reuse-default", {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reusable_default: value }),
      });
      DOC.reuse_default = res.reuse_default;
      reuseFeedback(value === null ? "Этот текст следует общей настройке."
        : `В этом тексте законченные разборы: ${value ? "запоминаются для всех текстов" : "сохраняются только здесь"}. Общая настройка проекта не изменилась.`);
    } catch (error) {
      docReuse.value = typeof DOC.reuse_default === "boolean" ? String(DOC.reuse_default) : "inherit";
      reuseFeedback(error.message, true);
    }
  };
  await loadMutes();
  render();
  focusRequestedToken();
  setHistoryButtons(await api("/history"));
})().catch((error) => reuseFeedback("Не удалось загрузить редактор: " + error.message, true));
