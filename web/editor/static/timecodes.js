"use strict";

// Timecodes are source metadata, not part of the IGT line. The parser already
// stores them in sentence.source_locator.timecode; this layer gives that field
// its own UI and optionally seeks a document-level audio source to the start
// of the range. The end time is intentionally informational only.

const MEDIA_TC_RE = /\b\d{1,4}:\d{2}(?::\d{2})?(?:\s*[-–—]\s*\d{1,4}:\d{2}(?::\d{2})?)?\b/;
const AUDIO_STORAGE = `turoyo:${DOC_ID}:audio-url`;
const TIMECODE_STORAGE = `turoyo:${DOC_ID}:timecodes`;

function storageGet(key, fallback = "") {
  try { return localStorage.getItem(key) ?? fallback; }
  catch (e) { return fallback; }
}
function storageSet(key, value) {
  try {
    if (value === null || value === undefined || value === "") localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch (e) { /* private mode etc. */ }
}

function localTimecodes() {
  try { return JSON.parse(storageGet(TIMECODE_STORAGE, "{}")) || {}; }
  catch (e) { return {}; }
}
function localTimecode(sentenceId) {
  const values = localTimecodes();
  return Object.prototype.hasOwnProperty.call(values, sentenceId) ? values[sentenceId] : null;
}
function saveLocalTimecode(sentenceId, value) {
  const values = localTimecodes();
  values[sentenceId] = value;
  try { localStorage.setItem(TIMECODE_STORAGE, JSON.stringify(values)); }
  catch (e) { /* private mode etc. */ }
}
function clearLocalTimecode(sentenceId) {
  const values = localTimecodes();
  if (!Object.prototype.hasOwnProperty.call(values, sentenceId)) return;
  delete values[sentenceId];
  try {
    if (Object.keys(values).length) localStorage.setItem(TIMECODE_STORAGE, JSON.stringify(values));
    else localStorage.removeItem(TIMECODE_STORAGE);
  } catch (e) { /* ignore */ }
}

function timecodeFor(sentence) {
  // A local override is used only by the offline/base app, whose API predates
  // writable timecode routes. Hosted sessions clear this as soon as PATCH wins.
  const local = localTimecode(sentence.sentence_id);
  if (local !== null) return local;
  const locator = sentence.source_locator || {};
  if (Object.prototype.hasOwnProperty.call(locator, "timecode")) {
    return locator.timecode || "";
  }
  const marker = locator.segment_marker || "";
  const match = marker.match(MEDIA_TC_RE);
  return match ? match[0].replace(/[–—]/g, "-").replace(/\s+/g, "") : "";
}

function audioUrl() {
  // Local fallback is an override only while the server has no persistent
  // media route. A successful PATCH clears it, so hosted sessions all converge
  // on document.audio_url while old/offline editors still remember changes.
  const local = storageGet(AUDIO_STORAGE, "");
  if (local) return local;
  if (DOC && Object.prototype.hasOwnProperty.call(DOC, "audio_url")) {
    return DOC.audio_url || "";
  }
  return "";
}

function clockSeconds(clock) {
  const values = clock.split(":").map(Number);
  if (values.some((n) => !Number.isFinite(n))) return null;
  if (values.length === 2) return values[0] * 60 + values[1];
  if (values.length === 3) return values[0] * 3600 + values[1] * 60 + values[2];
  return null;
}

function timecodeStartSeconds(value) {
  const match = (value || "").match(MEDIA_TC_RE);
  if (!match) return null;
  return clockSeconds(match[0].split(/\s*[-–—]\s*/)[0]);
}

function focusAudioInput() {
  const input = document.getElementById("audio-url");
  if (input) { input.focus(); input.select(); }
}

function ensureAudioPanel() {
  let panel = document.getElementById("audio-source-panel");
  if (panel) {
    const input = panel.querySelector("#audio-url");
    if (input && document.activeElement !== input) input.value = audioUrl();
    return panel;
  }

  const host = document.querySelector(".reuse-settings");
  if (!host) return null;
  panel = document.createElement("div");
  panel.id = "audio-source-panel";
  panel.className = "audio-source-panel";

  const label = document.createElement("label");
  label.textContent = "Аудио: ";
  const input = document.createElement("input");
  input.id = "audio-url";
  input.type = "url";
  input.placeholder = "прямая https://… ссылка на .m4a/.mp3";
  input.value = audioUrl();
  input.autocomplete = "off";
  label.append(input);

  const save = document.createElement("button");
  save.type = "button";
  save.textContent = "сохранить";

  const audio = document.createElement("audio");
  audio.id = "source-audio";
  audio.controls = true;
  audio.preload = "metadata";
  if (audioUrl()) {
    audio.src = audioUrl();
    audio.dataset.sourceUrl = audioUrl();
  }
  audio.addEventListener("error", () => {
    const code = audio.error?.code;
    const detail = code === 1 ? "загрузка отменена"
      : code === 2 ? "ошибка сети при загрузке"
      : code === 3 ? "ошибка декодирования аудио"
      : code === 4 ? "формат или источник не поддерживается браузером"
      : "неизвестная ошибка браузерного аудиоплеера";
    reuseFeedback(`Не удалось открыть аудиофайл: ${detail}.`, true);
  });

  const persist = async () => {
    const value = input.value.trim();
    if (value) {
      try {
        const parsed = new URL(value);
        if (!/^https?:$/.test(parsed.protocol)) throw new Error("scheme");
      } catch (e) {
        reuseFeedback("Ссылка на аудио должна начинаться с http:// или https://", true);
        return;
      }
    }
    save.disabled = true;
    try {
      const res = await api("/audio", {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: value }),
      });
      DOC.audio_url = res.audio_url;
      storageSet(AUDIO_STORAGE, "");
      input.value = res.audio_url;
      reuseFeedback(res.audio_url ? "Ссылка на аудио сохранена в документе." : "Аудио отвязано от документа.");
    } catch (error) {
      if (error.message !== "404") {
        reuseFeedback("Не удалось сохранить ссылку на аудио: " + error.message, true);
        save.disabled = false;
        return;
      }
      // Local/offline compatibility: old create_app has no metadata routes.
      storageSet(AUDIO_STORAGE, value);
      reuseFeedback("Ссылка сохранена в этом браузере (локальный режим редактора).", false);
    }
    const current = audioUrl();
    if (current) {
      audio.src = current; audio.dataset.sourceUrl = current; audio.load();
    } else {
      audio.removeAttribute("src"); delete audio.dataset.sourceUrl; audio.load();
    }
    save.disabled = false;
    render();
  };

  save.addEventListener("click", persist);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); persist(); }
  });

  panel.append(label, save, audio);
  host.append(panel);
  return panel;
}

async function playAtTimecode(value) {
  const url = audioUrl();
  if (!url) {
    reuseFeedback("Сначала укажите ссылку на аудиофайл.", true);
    focusAudioInput();
    return;
  }
  const seconds = timecodeStartSeconds(value);
  if (seconds === null) {
    reuseFeedback("Не удалось прочитать начало таймкода.", true);
    return;
  }
  const panel = ensureAudioPanel();
  const audio = panel?.querySelector("#source-audio");
  if (!audio) return;
  if (audio.dataset.sourceUrl !== url) {
    audio.src = url;
    audio.dataset.sourceUrl = url;
    audio.load();
  }
  const seek = async () => {
    try {
      audio.currentTime = seconds;
      await audio.play();
    } catch (error) {
      reuseFeedback("Не удалось перейти к таймкоду. Проверьте аудиофайл и его доступность.", true);
    }
  };
  if (audio.readyState >= 1) await seek();
  else audio.addEventListener("loadedmetadata", seek, { once: true });
}

async function saveTimecode(sentence, value) {
  value = value.trim().replace(/[–—]/g, "-").replace(/\s+/g, "");
  if (value && !new RegExp(`^${MEDIA_TC_RE.source}$`).test(value)) {
    throw new Error("формат: 05:53-06:13 или 05:53");
  }
  try {
    const res = await api(`/sentences/${sentence.sentence_id}/timecode`, {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: value }),
    });
    sentence.source_locator = sentence.source_locator || {};
    sentence.source_locator.timecode = res.timecode;
    clearLocalTimecode(sentence.sentence_id);
  } catch (error) {
    if (error.message !== "404") throw error;
    saveLocalTimecode(sentence.sentence_id, value);
  }
}

function openTimecodeEditor(sentence, controls) {
  if (!LOCK.you_are_holder) {
    reuseFeedback("Правка таймкода требует права редактирования.", true);
    return;
  }
  const input = document.createElement("input");
  input.className = "timecode-input";
  input.value = timecodeFor(sentence);
  input.placeholder = "05:53-06:13";
  controls.replaceChildren(input);
  input.focus(); input.select();

  let done = false;
  const finish = async (save) => {
    if (done) return;
    done = true;
    if (save) {
      try {
        await saveTimecode(sentence, input.value);
        reuseFeedback("Таймкод сохранён.");
      } catch (error) {
        reuseFeedback("Не удалось сохранить таймкод: " + error.message, true);
      }
    }
    render();
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); finish(true); }
    if (event.key === "Escape") { event.preventDefault(); finish(false); }
  });
  input.addEventListener("blur", () => finish(true));
}

function renderTimecodeControls(sentence) {
  const wrap = document.createElement("span");
  wrap.className = "timecode-controls";
  const value = timecodeFor(sentence);

  const button = document.createElement("button");
  button.type = "button";
  button.className = "timecode-link" + (value ? "" : " empty");
  button.textContent = value || "+ таймкод";
  button.title = value
    ? (audioUrl() ? "Проиграть аудио с начала этого интервала" : "Таймкод есть; добавьте ссылку на аудио для проигрывания")
    : "Добавить таймкод";
  button.addEventListener("click", () => {
    if (value) playAtTimecode(value);
    else openTimecodeEditor(sentence, wrap);
  });
  wrap.append(button);

  const edit = document.createElement("button");
  edit.type = "button";
  edit.className = "timecode-edit";
  edit.textContent = "✎";
  edit.title = "Изменить таймкод";
  edit.addEventListener("click", () => openTimecodeEditor(sentence, wrap));
  wrap.append(edit);
  return wrap;
}

function enhanceTimecodeLocators() {
  const blocks = [...document.querySelectorAll("#doc .sentence")];
  blocks.forEach((block, index) => {
    const sentence = DOC.sentences[index];
    const locator = block.querySelector(":scope > .sentence-locator");
    if (!sentence || !locator) return;
    const oldLabel = [...locator.children].find((child) => child.tagName === "SPAN");
    if (oldLabel) oldLabel.textContent = `Блок ${index + 1}`;
    const sep = document.createElement("span");
    sep.className = "locator-separator";
    sep.textContent = "·";
    locator.append(sep, renderTimecodeControls(sentence));
  });
}

// editor.js owns the main renderer. Wrap it rather than duplicating the editor
// so every ordinary gloss edit continues to use one source of truth.
const renderWithoutTimecodes = render;
render = function renderWithTimecodes() {
  renderWithoutTimecodes();
  ensureAudioPanel();
  enhanceTimecodeLocators();
};
