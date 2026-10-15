// User-guide window: Markdown text, rendered server-side (see
// editor/guide.py) and edited whole through /api/guide. No external
// dependencies.

async function api(options) {
  const res = await fetch("/api/guide", {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await res.json();
  if (!res.ok) throw new Error(body.detail || res.status);
  return body;
}

const view = document.getElementById("guide-view");
const textarea = document.getElementById("guide-text");
const status = document.getElementById("guide-status");
const editButton = document.getElementById("guide-edit");
const saveButton = document.getElementById("guide-save");
const cancelButton = document.getElementById("guide-cancel");

let lastText = "";

function showView(data) {
  view.innerHTML = data.html;
  lastText = data.text;
  view.hidden = false;
  textarea.hidden = true;
  editButton.hidden = false;
  saveButton.hidden = true;
  cancelButton.hidden = true;
}

function showEditor() {
  textarea.value = lastText;
  view.hidden = true;
  textarea.hidden = false;
  editButton.hidden = true;
  saveButton.hidden = false;
  cancelButton.hidden = false;
  status.textContent = "";
}

async function load() {
  showView(await api({}));
}

async function save() {
  try {
    showView(await api({ method: "PATCH", body: JSON.stringify({ text: textarea.value }) }));
    status.textContent = "Сохранено.";
  } catch (err) {
    alert(`Не записано: ${err.message}`);
  }
}

editButton.addEventListener("click", showEditor);
cancelButton.addEventListener("click", () => showView({ text: lastText, html: view.innerHTML }));
saveButton.addEventListener("click", save);
load();
