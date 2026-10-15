"use strict";

// Compact transport controls next to each timecode. Every row controls the
// same document-level <audio> element, so playback position survives ordinary
// editor re-renders and moving between transcript blocks.

function transportAudioElement() {
  return document.getElementById("source-audio");
}

// timecodes.js used to blame every HTMLMediaElement error on an indirect URL.
// That is misleading for real direct sources such as R2: the same event also
// covers network, decode, and unsupported-media failures. Translate only that
// legacy message; Yandex's more specific diagnostic remains untouched.
const reuseFeedbackWithoutAudioDiagnostic = reuseFeedback;
reuseFeedback = function reuseFeedbackWithAudioDiagnostic(message, isError = false) {
  if (message === "Аудио не открылось. Нужна прямая ссылка на сам файл, а не страница общего доступа.") {
    const code = transportAudioElement()?.error?.code;
    const reason = {
      1: "загрузка была прервана",
      2: "ошибка сети при загрузке",
      3: "браузер не смог декодировать аудио",
      4: "формат или источник не поддерживается браузером",
    }[code];
    message = reason
      ? `Не удалось открыть аудиофайл: ${reason}.`
      : "Не удалось открыть аудиофайл. Проверьте доступность файла и формат аудио.";
  }
  return reuseFeedbackWithoutAudioDiagnostic(message, isError);
};

function syncTransportButtons() {
  const audio = transportAudioElement();
  const playing = Boolean(audio && !audio.paused && !audio.ended);
  document.querySelectorAll(".timecode-play-toggle").forEach((button) => {
    button.textContent = playing ? "■" : "▶";
    button.title = playing ? "Остановить воспроизведение" : "Продолжить воспроизведение";
    button.setAttribute("aria-label", button.title);
  });
}

function sourceAudio() {
  const panel = ensureAudioPanel();
  const audio = panel?.querySelector("#source-audio") || null;
  if (audio && audio.dataset.transportBound !== "1") {
    audio.dataset.transportBound = "1";
    audio.addEventListener("play", syncTransportButtons);
    audio.addEventListener("pause", syncTransportButtons);
    audio.addEventListener("ended", syncTransportButtons);
  }
  return audio;
}

function currentAudioOrExplain() {
  const audio = sourceAudio();
  if (!audio || !(audio.currentSrc || audio.getAttribute("src"))) {
    reuseFeedback("Сначала нажмите таймкод, чтобы запустить аудио.", true);
    return null;
  }
  return audio;
}

async function toggleAudioPlayback(value) {
  const audio = sourceAudio();
  if (!audio) return;

  if (!(audio.currentSrc || audio.getAttribute("src"))) {
    await playAtTimecode(value);
    syncTransportButtons();
    return;
  }

  if (audio.paused || audio.ended) {
    try {
      await audio.play();
    } catch (error) {
      reuseFeedback("Не удалось продолжить воспроизведение.", true);
    }
  } else {
    audio.pause();
  }
  syncTransportButtons();
}

function skipAudio(seconds) {
  const audio = currentAudioOrExplain();
  if (!audio) return;
  const now = Number.isFinite(audio.currentTime) ? audio.currentTime : 0;
  let target = Math.max(0, now + seconds);
  if (Number.isFinite(audio.duration) && audio.duration > 0) {
    target = Math.min(audio.duration, target);
  }
  audio.currentTime = target;
}

const renderTimecodeControlsWithoutTransport = renderTimecodeControls;
renderTimecodeControls = function renderTimecodeControlsWithTransport(sentence) {
  const wrap = renderTimecodeControlsWithoutTransport(sentence);
  const value = timecodeFor(sentence);
  if (!value) return wrap;

  const transport = document.createElement("span");
  transport.className = "timecode-transport";

  const rewind = document.createElement("button");
  rewind.type = "button";
  rewind.className = "timecode-transport-button rewind";
  rewind.textContent = "−3с";
  rewind.title = "На 3 секунды назад";
  rewind.setAttribute("aria-label", rewind.title);
  rewind.addEventListener("click", (event) => {
    event.stopPropagation();
    skipAudio(-3);
  });

  const play = document.createElement("button");
  play.type = "button";
  play.className = "timecode-transport-button timecode-play-toggle";
  play.textContent = "▶";
  play.title = "Продолжить воспроизведение";
  play.setAttribute("aria-label", play.title);
  play.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleAudioPlayback(value);
  });

  const forward = document.createElement("button");
  forward.type = "button";
  forward.className = "timecode-transport-button forward";
  forward.textContent = "+3с";
  forward.title = "На 3 секунды вперёд";
  forward.setAttribute("aria-label", forward.title);
  forward.addEventListener("click", (event) => {
    event.stopPropagation();
    skipAudio(3);
  });

  transport.append(rewind, play, forward);
  wrap.append(transport);
  sourceAudio();
  syncTransportButtons();
  return wrap;
};

// ---------------------------------------------------------------------------
// Mini player: right of the −3с ▶ +3с the author used last, the current time
// and a slider over this block's interval (±10 s) to scrub. One element,
// moved to whichever block was touched last; the page re-renders often, so
// the active block is remembered by sentence id and re-attached after render.
// ---------------------------------------------------------------------------
let MINI_PLAYER_SENTENCE = null;

function clockText(seconds) {
  if (!Number.isFinite(seconds)) return "--:--";
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(r)}` : `${m}:${pad(r)}`;
}

function blockWindow(value) {
  const match = (value || "").match(MEDIA_TC_RE);
  if (!match) return null;
  const [from, to] = match[0].split(/\s*[-–—]\s*/).map(clockSeconds);
  if (from === null || from === undefined) return null;
  const end = Number.isFinite(to) && to > from ? to : from + 20;
  return { min: Math.max(0, from - 10), max: end + 10 };
}

function miniPlayer() {
  let box = document.getElementById("mini-player");
  if (box) return box;
  box = document.createElement("span");
  box.id = "mini-player";
  box.className = "mini-player";
  const slider = document.createElement("input");
  slider.type = "range"; slider.step = "0.1"; slider.min = "0"; slider.max = "1";
  slider.title = "Перемотка в пределах этого блока (±10 с)";
  const time = document.createElement("span");
  time.className = "mini-player-time";
  box.append(time, slider);
  let dragging = false;
  slider.addEventListener("pointerdown", () => { dragging = true; });
  slider.addEventListener("pointerup", () => { dragging = false; });
  slider.addEventListener("input", (event) => {
    event.stopPropagation();
    const audio = sourceAudio();
    if (audio) audio.currentTime = Number(slider.value);
    time.textContent = clockText(Number(slider.value));
  });
  slider.addEventListener("click", (event) => event.stopPropagation());
  box.update = () => {
    const audio = transportAudioElement();
    const now = audio && Number.isFinite(audio.currentTime) ? audio.currentTime : 0;
    time.textContent = clockText(now);
    if (!dragging) slider.value = String(Math.min(Number(slider.max), Math.max(Number(slider.min), now)));
  };
  box.setWindow = (win) => {
    slider.min = String(win.min); slider.max = String(win.max);
    box.update();
  };
  return box;
}

function bindMiniPlayerAudio() {
  const audio = sourceAudio();
  if (audio && audio.dataset.miniBound !== "1") {
    audio.dataset.miniBound = "1";
    ["timeupdate", "seeked", "loadedmetadata"].forEach((name) =>
      audio.addEventListener(name, () => document.getElementById("mini-player")?.update()));
  }
}

function attachMiniPlayer(transport, sentence) {
  const win = blockWindow(timecodeFor(sentence));
  if (!win) return;
  const box = miniPlayer();
  box.setWindow(win);
  transport.append(box);
  bindMiniPlayerAudio();
}

const renderTimecodeControlsWithoutMini = renderTimecodeControls;
renderTimecodeControls = function renderTimecodeControlsWithMini(sentence) {
  const wrap = renderTimecodeControlsWithoutMini(sentence);
  const transport = wrap.querySelector(".timecode-transport");
  if (!transport) return wrap;
  const activate = () => {
    MINI_PLAYER_SENTENCE = sentence.sentence_id;
    attachMiniPlayer(transport, sentence);
  };
  // the timecode link and the three buttons all make this block the active one
  wrap.addEventListener("click", activate, true);
  if (MINI_PLAYER_SENTENCE === sentence.sentence_id) attachMiniPlayer(transport, sentence);
  return wrap;
};
