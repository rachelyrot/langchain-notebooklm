// NotebookLM-style client. Talks to the FastAPI backend only over /api.

const api = {
  async get(url) {
    const r = await fetch(url);
    if (!r.ok) throw await err(r);
    return r.json();
  },
  async send(method, url, body) {
    const r = await fetch(url, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!r.ok) throw await err(r);
    return r.status === 204 ? null : r.json();
  },
};
async function err(r) {
  let detail = r.statusText;
  try {
    detail = (await r.json()).detail || detail;
  } catch {}
  const e = new Error(detail);
  e.status = r.status;
  return e;
}

// One conversation thread per page load. "New chat" starts a fresh one so the agent's
// short-term memory resets.
const state = { sources: [], threadId: crypto.randomUUID() };
const $ = (id) => document.getElementById(id);
// Deliberately corpus-agnostic: the notebook starts empty and holds whatever you add.
const SUGGESTIONS = [
  "Summarise the main points across my sources.",
  "Where do my sources disagree, and what does each one claim?",
  "What questions do my sources leave unanswered?",
];

function escapeHtml(s) {
  return String(s).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

/* ---- sources --------------------------------------------------------------- */

async function loadSources() {
  state.sources = await api.get("/api/sources");
  renderSources();
}

function renderSources() {
  const ul = $("sources");
  ul.innerHTML = "";
  const activeCount = state.sources.filter((s) => s.active).length;
  $("active-count").textContent = `${activeCount} active`;
  $("select-all").checked = activeCount === state.sources.length && state.sources.length > 0;

  for (const s of state.sources) {
    const li = document.createElement("li");
    if (!s.active) li.className = "inactive";
    const label = s.url ? `🌐 ${s.name}` : s.name;
    li.innerHTML = `
      <input type="checkbox" ${s.active ? "checked" : ""} />
      <span class="s-name" title="${escapeHtml(s.url || s.name)}">${escapeHtml(label)}</span>
      <span class="s-meta">${s.chars.toLocaleString()}</span>
      <button class="icon-btn" title="View">👁</button>
      <button class="icon-btn danger" title="Remove">✕</button>`;
    const [chk, , , viewBtn, delBtn] = li.children;
    chk.onchange = () => toggleSource(s.id, chk.checked);
    viewBtn.onclick = () => viewSource(s.id);
    delBtn.onclick = () => deleteSource(s.id);
    ul.appendChild(li);
  }
}

async function toggleSource(id, active) {
  await api.send("PATCH", `/api/sources/${id}`, { active });
  await loadSources();
}
async function deleteSource(id) {
  await api.send("DELETE", `/api/sources/${id}`);
  await loadSources();
}
async function viewSource(id) {
  const s = await api.get(`/api/sources/${id}`);
  openViewer(s.name, s.content);
}

async function addText(e) {
  e.preventDefault();
  const content = $("add-content").value.trim();
  if (!content) return;
  await api.send("POST", "/api/sources", { name: $("add-name").value, content });
  $("add-name").value = "";
  $("add-content").value = "";
  $("add-form").classList.add("hidden");
  await loadSources();
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch("/api/sources/upload", { method: "POST", body: fd });
  if (!r.ok) {
    addMessage("system", `⚠ ${(await err(r)).message}`);
    return;
  }
  $("add-form").classList.add("hidden");
  await loadSources();
}

/* ---- research (human in the loop) ------------------------------------------
   The agent searches from several angles and proposes pages; the run then pauses
   inside the graph until we send back which ones to keep. A run can pause more than
   once, so every leg goes through handleResearch. */

function researchBusy(busy, note = "") {
  $("research-btn").disabled = busy;
  $("research-status").textContent = note;
}

async function researchTopic() {
  const topic = $("research-topic").value.trim();
  if (!topic) return;

  researchBusy(true, "searching the web…");
  addMessage("system", `🔎 Researching “${topic}” — searching from several angles…`);
  try {
    await handleResearch(await api.send("POST", "/api/sources/research", { topic }));
    $("research-topic").value = "";
  } catch (e) {
    addMessage("system", `⚠ ${e.message}`);
    researchBusy(false);
  }
}

async function handleResearch(data) {
  await loadSources(); // anything already accepted shows up right away

  if (data.status === "awaiting_selection") {
    researchBusy(true, "waiting for your picks…");
    openPicker(data);
    return;
  }

  const names = data.sources.map((s) => s.name).join(", ") || "nothing new";
  addMessage("system", `✅ Added ${data.sources.length} source(s): ${names}\n\n${data.summary}`);
  researchBusy(false);
}

function openPicker(data) {
  state.runId = data.run_id;
  const ul = $("picker-list");
  ul.innerHTML = "";

  for (const c of data.candidates) {
    const li = document.createElement("li");
    const kind = c.action === "crawl_site" ? "🕸 whole site" : "📄 page";
    li.innerHTML = `
      <input type="checkbox" checked data-index="${c.index}" />
      <div class="p-body">
        <div class="p-reason">${escapeHtml(c.reason)}</div>
        <div class="p-url">${kind} · ${escapeHtml(c.url)}</div>
      </div>`;
    li.querySelector(".p-body").onclick = () => {
      const box = li.querySelector("input");
      box.checked = !box.checked;
      syncPickerAll();
    };
    li.querySelector("input").onchange = syncPickerAll;
    ul.appendChild(li);
  }

  $("picker-title").textContent = `Choose sources to add (${data.candidates.length} proposed)`;
  syncPickerAll();
  $("picker").classList.remove("hidden");
  $("overlay").classList.remove("hidden");
}

function pickerBoxes() {
  return [...$("picker-list").querySelectorAll("input[type=checkbox]")];
}

function syncPickerAll() {
  const boxes = pickerBoxes();
  $("picker-select-all").checked = boxes.length > 0 && boxes.every((b) => b.checked);
}

function closePicker() {
  $("picker").classList.add("hidden");
  $("overlay").classList.add("hidden");
}

async function sendPicks(approved) {
  closePicker();
  researchBusy(true, approved.length ? "fetching your picks…" : "skipping…");
  addMessage(
    "system",
    approved.length
      ? `📥 Fetching ${approved.length} page(s) you picked…`
      : "🚫 Skipped all proposals."
  );
  try {
    await handleResearch(
      await api.send("POST", `/api/sources/research/${state.runId}/decide`, { approved })
    );
  } catch (e) {
    addMessage("system", `⚠ ${e.message}`);
    researchBusy(false);
  }
}

async function selectAll(checked) {
  await Promise.all(
    state.sources
      .filter((s) => s.active !== checked)
      .map((s) => api.send("PATCH", `/api/sources/${s.id}`, { active: checked }))
  );
  await loadSources();
}

/* ---- chat ------------------------------------------------------------------ */

function renderEmptyState() {
  const m = $("messages");
  if (m.children.length) return;
  const div = document.createElement("div");
  div.className = "empty";
  div.innerHTML = `
    <div>Ask anything about your active sources — answers are grounded and cited.</div>
    <div class="suggestions">${SUGGESTIONS.map(
      (q) => `<button class="suggestion">${escapeHtml(q)}</button>`
    ).join("")}</div>`;
  div.querySelectorAll(".suggestion").forEach((b, i) => {
    b.onclick = () => {
      $("chat-input").value = SUGGESTIONS[i];
      $("chat-form").requestSubmit();
    };
  });
  m.appendChild(div);
}

function clearEmptyState() {
  const e = $("messages").querySelector(".empty");
  if (e) e.remove();
}

function addMessage(role, text, { pending = false } = {}) {
  clearEmptyState();
  const div = document.createElement("div");
  div.className = `msg ${role}${pending ? " pending" : ""}`;
  div.textContent = text; // text, never HTML — sources are scraped from the open web
  $("messages").appendChild(div);
  $("messages").scrollTop = $("messages").scrollHeight;
  return div;
}

function decorateAnswer(div, data) {
  div.className = "msg ai";
  div.textContent = data.answer;
  if (data.citations?.length) {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML = `<div><span class="label">sources:</span> ${data.citations
      .map((c) => `<span class="chip">${escapeHtml(c.source)}</span>`)
      .join("")}</div>`;
    div.appendChild(meta);
  }
  const actions = document.createElement("div");
  actions.className = "msg-actions";
  const save = document.createElement("button");
  save.className = "link-btn";
  save.textContent = "＋ Save to note";
  save.onclick = () => saveNote(data.answer);
  actions.appendChild(save);
  div.appendChild(actions);
}

// Reads a server-sent-event stream from a POST. EventSource would force the request
// into a GET, so the body is streamed back through fetch instead.
async function* sseEvents(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw await err(response);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop(); // the last one may still be arriving
    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data: "));
      if (line) yield JSON.parse(line.slice(6));
    }
  }
}

async function sendMessage(message) {
  addMessage("user", message);
  const bubble = addMessage("ai", "thinking…", { pending: true });
  $("send").disabled = true;

  let answer = "";
  const paint = () => {
    bubble.className = "msg ai";
    bubble.textContent = answer;
    $("messages").scrollTop = $("messages").scrollHeight;
  };

  try {
    for await (const event of sseEvents("/api/chat/stream", {
      message,
      thread_id: state.threadId,
    })) {
      if (event.type === "tool") {
        bubble.textContent = `searching your sources for “${event.args?.query ?? ""}”…`;
      } else if (event.type === "token") {
        answer += event.text;
        paint();
      } else if (event.type === "replace") {
        // a guardrail rewrote the answer after it had already been streamed
        answer = event.text;
        paint();
      } else if (event.type === "error") {
        throw new Error(event.detail);
      } else if (event.type === "done") {
        decorateAnswer(bubble, {
          answer,
          citations: event.sources.map((source) => ({ source })),
        });
      }
    }
  } catch (e) {
    bubble.remove();
    addMessage("system", `⚠ ${e.message}`);
  } finally {
    $("send").disabled = false;
    $("messages").scrollTop = $("messages").scrollHeight;
  }
}

/* ---- studio + notes -------------------------------------------------------- */

async function loadArtifacts() {
  const arts = await api.get("/api/studio/artifacts");
  const grid = $("artifacts");
  grid.innerHTML = "";
  for (const a of arts) {
    const btn = document.createElement("button");
    btn.className = "artifact";
    const badge = a.status === "planned" ? `<span class="a-badge">Soon</span>` : "";
    if (a.status === "planned") btn.classList.add("planned");
    btn.innerHTML = `<span class="a-icon">${a.icon}</span>
      <span class="a-title">${escapeHtml(a.title)}</span>${badge}`;
    btn.onclick = () => generateArtifact(a, btn);
    grid.appendChild(btn);
  }
}

// An HTML infographic is worth opening in place; a .pptx is only worth downloading.
function downloadLink(url, name, kind) {
  const wrapper = document.createElement("div");
  wrapper.className = "msg-actions";

  const link = document.createElement("a");
  link.className = "link-btn";
  link.href = url;
  if (kind === "infographic") {
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = "↗ Open the infographic";
  } else {
    link.download = name || "";
    link.textContent = `⬇ Download ${name || "the file"}`;
  }
  wrapper.appendChild(link);
  return wrapper;
}

// A generated artifact is saved as a note and opened straight away — the agent returns a
// structured object, so the note's markdown is rendered from fields, not from prose.
async function generateArtifact(a, btn) {
  if (a.status !== "ready") {
    addMessage("system", `⚠ ${a.title} generation is coming soon.`);
    return;
  }
  btn.disabled = true;
  btn.classList.add("busy");
  addMessage("system", `🛠 Building the ${a.title.toLowerCase()} from your active sources…`);
  try {
    const result = await api.send("POST", "/api/studio/generate", { kind: a.key });
    await loadNotes();

    if (result.download_url) {
      // an infographic or a deck: the note holds the text, the file is the artifact
      addMessage("system", `📦 ${result.note.title}`).appendChild(
        downloadLink(result.download_url, result.download_name, a.key)
      );
    } else {
      openViewer(result.note.title, result.note.content);
    }
  } catch (e) {
    addMessage("system", `⚠ ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.classList.remove("busy");
  }
}

async function loadNotes() {
  renderNotes(await api.get("/api/notes"));
}
function renderNotes(notes) {
  const ul = $("notes");
  ul.innerHTML = "";
  if (!notes.length) {
    ul.innerHTML = `<div class="note-empty">No notes yet — save a grounded answer.</div>`;
    return;
  }
  for (const n of notes) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="n-title" title="${escapeHtml(n.title)}">${escapeHtml(
      n.title
    )}</span><button class="icon-btn danger">✕</button>`;
    li.querySelector("button").onclick = async () => {
      await api.send("DELETE", `/api/notes/${n.id}`);
      loadNotes();
    };
    li.querySelector(".n-title").onclick = () => openViewer(n.title, n.content);
    ul.appendChild(li);
  }
}
async function saveNote(content) {
  const title = content.split("\n")[0].slice(0, 50);
  await api.send("POST", "/api/notes", { title, content });
  loadNotes();
}

/* ---- viewer modal ---------------------------------------------------------- */

function openViewer(title, content) {
  $("viewer-title").textContent = title;
  $("viewer-content").textContent = content;
  $("viewer").classList.remove("hidden");
  $("overlay").classList.remove("hidden");
}
function closeViewer() {
  $("viewer").classList.add("hidden");
  $("overlay").classList.add("hidden");
}

/* ---- wiring ---------------------------------------------------------------- */

$("add-toggle").onclick = () => $("add-form").classList.toggle("hidden");
$("add-cancel").onclick = () => $("add-form").classList.add("hidden");
$("add-form").addEventListener("submit", addText);
$("add-file").onchange = (e) => e.target.files[0] && uploadFile(e.target.files[0]);
$("research-btn").onclick = researchTopic;
$("research-topic").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault(); // the form's submit is "Add text", not research
    researchTopic();
  }
});
$("select-all").onchange = (e) => selectAll(e.target.checked);

$("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const msg = $("chat-input").value.trim();
  if (!msg) return;
  $("chat-input").value = "";
  sendMessage(msg);
});

$("new-chat").onclick = () => {
  state.threadId = crypto.randomUUID(); // fresh memory
  $("messages").innerHTML = "";
  renderEmptyState();
};

$("close-viewer").onclick = closeViewer;
// A paused run is waiting on an answer, so the picker only closes through a decision.
$("overlay").onclick = () => $("picker").classList.contains("hidden") && closeViewer();

$("picker-select-all").onchange = (e) =>
  pickerBoxes().forEach((b) => (b.checked = e.target.checked));
$("picker-add").onclick = () =>
  sendPicks(pickerBoxes().filter((b) => b.checked).map((b) => Number(b.dataset.index)));
$("picker-skip").onclick = () => sendPicks([]);
$("close-picker").onclick = () => sendPicks([]);

/* ---- init ------------------------------------------------------------------ */

(async function init() {
  await Promise.all([loadSources(), loadArtifacts(), loadNotes()]);
  renderEmptyState();
})();
