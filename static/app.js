/* Stripes RAG — Chat Frontend */

const API = "";  // same origin

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let conversations = JSON.parse(localStorage.getItem("stripes_conversations") || "[]");
let currentConvId = null;
let isStreaming = false;

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("new-chat-btn");
const historyList = document.getElementById("history-list");
const profileSelect = document.getElementById("profile-select");
const profileDesc = document.getElementById("profile-desc");
const modelSelect = document.getElementById("model-select");
const languageSelect = document.getElementById("language-select");
const temperatureSlider = document.getElementById("temperature");
const tempValue = document.getElementById("temp-value");
const kSlider = document.getElementById("k-slider");
const kValue = document.getElementById("k-value");
const maxStepsSlider = document.getElementById("max-steps");
const stepsValue = document.getElementById("steps-value");
const rerankerToggle = document.getElementById("reranker-toggle");

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
marked.setOptions({
  highlight: (code, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
});

async function init() {
  // Load profiles
  try {
    const profiles = await fetch(`${API}/profiles`).then(r => r.json());
    profiles.forEach((p, i) => {
      const opt = document.createElement("option");
      opt.value = p.name;
      opt.textContent = p.name;
      if (i === 1) opt.selected = true;  // default: Project Architect
      profileSelect.appendChild(opt);
    });
    updateProfileDesc();
  } catch (e) {
    console.error("Failed to load profiles:", e);
  }

  // Load models
  try {
    const models = await fetch(`${API}/models`).then(r => r.json());
    models.forEach((m, i) => {
      const opt = document.createElement("option");
      opt.value = JSON.stringify({ model_id: m.model_id, api_base: m.api_base, api_key: m.api_key });
      opt.textContent = m.name;
      if (i === 0) opt.selected = true;
      modelSelect.appendChild(opt);
    });
  } catch (e) {
    console.error("Failed to load models:", e);
  }

  // Slider labels
  temperatureSlider.addEventListener("input", () => { tempValue.textContent = temperatureSlider.value; });
  kSlider.addEventListener("input", () => { kValue.textContent = kSlider.value; });
  maxStepsSlider.addEventListener("input", () => { stepsValue.textContent = maxStepsSlider.value; });
  profileSelect.addEventListener("change", updateProfileDesc);

  // Load history
  renderHistory();

  // Start new chat or load last
  if (conversations.length > 0) {
    loadConversation(conversations[0].id);
  } else {
    newChat();
  }
}

function updateProfileDesc() {
  fetch(`${API}/profiles`).then(r => r.json()).then(profiles => {
    const p = profiles.find(p => p.name === profileSelect.value);
    profileDesc.textContent = p ? p.description : "";
  });
}

// ---------------------------------------------------------------------------
// Conversations (localStorage)
// ---------------------------------------------------------------------------
function saveConversations() {
  localStorage.setItem("stripes_conversations", JSON.stringify(conversations));
}

function newChat() {
  const conv = {
    id: Date.now().toString(),
    title: "New Chat",
    messages: [],
    created: new Date().toISOString(),
  };
  conversations.unshift(conv);
  saveConversations();
  loadConversation(conv.id);
  renderHistory();
}

function loadConversation(id) {
  currentConvId = id;
  const conv = conversations.find(c => c.id === id);
  if (!conv) return;

  messagesEl.innerHTML = "";
  conv.messages.forEach(msg => renderMessage(msg));
  renderHistory();
  scrollToBottom();
}

function deleteConversation(id, e) {
  e.stopPropagation();
  conversations = conversations.filter(c => c.id !== id);
  saveConversations();
  if (currentConvId === id) {
    if (conversations.length > 0) {
      loadConversation(conversations[0].id);
    } else {
      newChat();
    }
  }
  renderHistory();
}

function getCurrentConv() {
  return conversations.find(c => c.id === currentConvId);
}

function renderHistory() {
  historyList.innerHTML = "";
  conversations.forEach(conv => {
    const div = document.createElement("div");
    div.className = "history-item" + (conv.id === currentConvId ? " active" : "");
    div.innerHTML = `
      <span>${escapeHtml(conv.title)}</span>
      <span class="history-delete" onclick="deleteConversation('${conv.id}', event)">x</span>
    `;
    div.addEventListener("click", () => loadConversation(conv.id));
    historyList.appendChild(div);
  });
}

// ---------------------------------------------------------------------------
// Message rendering
// ---------------------------------------------------------------------------
function renderMessage(msg) {
  const div = document.createElement("div");
  div.className = `msg msg-${msg.role}`;

  if (msg.role === "status") {
    div.className = "msg msg-status";
    div.textContent = msg.content;
    messagesEl.appendChild(div);
    return div;
  }

  if (msg.role === "user") {
    div.textContent = msg.content;
  } else {
    div.innerHTML = marked.parse(msg.content || "");
  }

  // Sources
  if (msg.chunks && msg.chunks.length > 0) {
    const toggle = document.createElement("div");
    toggle.className = "sources-toggle";
    toggle.textContent = `Sources & References (${msg.chunks.length})`;
    const panel = document.createElement("div");
    panel.className = "sources-panel";
    msg.chunks.forEach(chunk => {
      const item = document.createElement("div");
      item.className = "source-item";
      let meta = `Similarity: ${chunk.similarity.toFixed(4)}`;
      if (chunk.reranker_score != null) {
        meta = `Reranker: ${chunk.reranker_score.toFixed(4)} | ${meta}`;
      }
      if (chunk.pages) meta += ` | Pages: ${chunk.pages}`;
      if (chunk.headings) meta += ` | ${chunk.headings}`;

      item.innerHTML = `
        <div class="source-header">${escapeHtml(chunk.source)}</div>
        <div class="source-meta">${meta}</div>
        <div class="source-content">${escapeHtml(chunk.content)}</div>
      `;
      item.querySelector(".source-header").addEventListener("click", () => {
        item.querySelector(".source-content").classList.toggle("open");
      });
      panel.appendChild(item);
    });
    toggle.addEventListener("click", () => panel.classList.toggle("open"));
    div.appendChild(toggle);
    div.appendChild(panel);
  }

  // Follow-ups
  if (msg.follow_ups && msg.follow_ups.length > 0) {
    const fuDiv = document.createElement("div");
    fuDiv.className = "follow-ups";
    msg.follow_ups.forEach(fq => {
      const btn = document.createElement("button");
      btn.className = "follow-up-btn";
      btn.textContent = fq;
      btn.addEventListener("click", () => sendMessage(fq));
      fuDiv.appendChild(btn);
    });
    div.appendChild(fuDiv);
  }

  // Stats
  if (msg.stats) {
    const statsDiv = document.createElement("div");
    statsDiv.className = "msg-stats";
    const s = msg.stats;
    statsDiv.textContent = `${s.elapsed}s | Retrieval: ${s.retrieval_time}s (${s.retrieval_calls} calls) | LLM: ${s.llm_time}s | ${s.input_tokens.toLocaleString()} in / ${s.output_tokens.toLocaleString()} out`;
    div.appendChild(statsDiv);
  }

  messagesEl.appendChild(div);
  return div;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Send message
// ---------------------------------------------------------------------------
async function sendMessage(text) {
  if (isStreaming || !text.trim()) return;

  const conv = getCurrentConv();
  if (!conv) return;

  // Update title from first message
  if (conv.messages.length === 0) {
    conv.title = text.slice(0, 50);
    saveConversations();
    renderHistory();
  }

  // Add user message
  const userMsg = { role: "user", content: text };
  conv.messages.push(userMsg);
  renderMessage(userMsg);
  saveConversations();
  scrollToBottom();

  // Clear input
  chatInput.value = "";
  isStreaming = true;
  sendBtn.disabled = true;
  sendBtn.textContent = "...";

  // Status indicator
  const statusEl = renderMessage({ role: "status", content: "Searching knowledge base..." });

  // Build request
  const modelData = JSON.parse(modelSelect.value);
  const request = {
    message: text,
    history: conv.messages.filter(m => m.role === "user" || m.role === "assistant").map(m => ({ role: m.role, content: m.content })),
    model: modelData.model_id,
    api_base: modelData.api_base,
    api_key: modelData.api_key,
    profile: profileSelect.value,
    language: languageSelect.value,
    temperature: parseFloat(temperatureSlider.value),
    k: parseInt(kSlider.value),
    max_steps: parseInt(maxStepsSlider.value),
    use_reranker: rerankerToggle.checked,
  };

  try {
    const response = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let assistantMsg = { role: "assistant", content: "", chunks: [], follow_ups: [], stats: null };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();  // keep incomplete line

      let eventType = null;
      for (const line of lines) {
        if (line.startsWith("event: ")) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith("data: ") && eventType) {
          const data = line.slice(6);
          handleSSEEvent(eventType, data, assistantMsg, statusEl);
          eventType = null;
        }
      }
    }

    // Remove status, add final message
    statusEl.remove();
    conv.messages.push(assistantMsg);
    renderMessage(assistantMsg);
    saveConversations();

  } catch (e) {
    statusEl.remove();
    renderMessage({ role: "assistant", content: `**Error:** ${e.message}` });
  }

  isStreaming = false;
  sendBtn.disabled = false;
  sendBtn.textContent = "Send";
  scrollToBottom();
}

function handleSSEEvent(type, data, msg, statusEl) {
  switch (type) {
    case "status":
      statusEl.textContent = data;
      break;
    case "chunks":
      msg.chunks = JSON.parse(data);
      break;
    case "answer":
      msg.content = JSON.parse(data);
      break;
    case "follow_ups":
      msg.follow_ups = JSON.parse(data);
      break;
    case "stats":
      msg.stats = JSON.parse(data);
      break;
    case "error":
      msg.content = `**Error:** ${JSON.parse(data)}`;
      break;
  }
  scrollToBottom();
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(chatInput.value);
});

newChatBtn.addEventListener("click", () => newChat());

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
init();
