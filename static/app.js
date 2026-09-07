"use strict";

/* ------------------------------------------------------------------ *
 * Minimal Markdown renderer.
 * No CDN is used on purpose: this machine's network cannot reach the
 * usual JS CDNs reliably. Input is HTML-escaped before any markup is
 * applied, and link targets are restricted to http(s).
 * ------------------------------------------------------------------ */

const ESCAPES = [["&", "&amp;"], ["<", "&lt;"], [">", "&gt;"], ['"', "&quot;"], ["'", "&#39;"]];

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (ch) => ESCAPES.find(([c]) => c === ch)[1]);
}

function safeLink(label, target) {
  return /^https?:\/\//i.test(target)
    ? `<a href="${target}" target="_blank" rel="noopener noreferrer">${label}</a>`
    : label;
}

function inlineMarkdown(text) {
  const stash = [];
  const keep = (html) => `\u0001${stash.push(html) - 1}\u0001`;

  let out = text.replace(/`([^`\n]+)`/g, (_, code) => keep(`<code>${code}</code>`));
  out = out.replace(/\[([^\]\n]*)\]\(([^)\s]+)\)/g, (_, label, url) => keep(safeLink(label, url)));
  out = out.replace(/(^|[\s(])(https?:\/\/[^\s<>()"']+)/g, (_, pre, url) =>
    pre + keep(safeLink(url, url.replace(/&amp;/g, "&")))
  );
  out = out
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
  return out.replace(/\u0001(\d+)\u0001/g, (_, i) => stash[Number(i)]);
}

const isTableRow = (line) => /^\s*\|.*\|\s*$/.test(line);
const isTableDivider = (line) => /^\s*\|[\s:|-]+\|\s*$/.test(line) && line.includes("-");
const splitRow = (line) =>
  line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());

function renderMarkdown(src) {
  const fences = [];
  let text = src.replace(/```([^\n`]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const cls = lang.trim() ? ` class="language-${escapeHtml(lang.trim())}"` : "";
    fences.push(`<pre><code${cls}>${escapeHtml(code.replace(/\n$/, ""))}</code></pre>`);
    return `\u0000${fences.length - 1}\u0000`;
  });

  text = escapeHtml(text);
  const lines = text.split("\n");

  let html = "";
  let paragraph = [];
  let listTag = null;
  let tableRows = [];

  const flushParagraph = () => {
    if (paragraph.length) html += `<p>${inlineMarkdown(paragraph.join("\n"))}</p>`;
    paragraph = [];
  };
  const flushList = () => {
    if (listTag) { html += `</${listTag}>`; listTag = null; }
  };
  const flushTable = () => {
    if (!tableRows.length) return;
    const rows = tableRows.filter((r) => !isTableDivider(r.raw));
    const hasHeader = tableRows.some((r) => isTableDivider(r.raw));
    let out = "<table>";
    rows.forEach((row, i) => {
      const tag = hasHeader && i === 0 ? "th" : "td";
      out += "<tr>" + splitRow(row.raw).map((c) => `<${tag}>${inlineMarkdown(c)}</${tag}>`).join("") + "</tr>";
    });
    html += out + "</table>";
    tableRows = [];
  };
  const flushAll = () => { flushParagraph(); flushList(); flushTable(); };

  for (const line of lines) {
    const fence = /^\u0000(\d+)\u0000$/.exec(line.trim());
    if (fence) { flushAll(); html += fences[Number(fence[1])]; continue; }

    if (isTableRow(line)) { flushParagraph(); flushList(); tableRows.push({ raw: line }); continue; }
    flushTable();

    if (!line.trim()) { flushParagraph(); flushList(); continue; }

    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      flushParagraph(); flushList();
      const level = heading[1].length;
      html += `<h${level}>${inlineMarkdown(heading[2])}</h${level}>`;
      continue;
    }
    if (/^\s*([-*_])\s*\1\s*\1[\s-*_]*$/.test(line)) { flushParagraph(); flushList(); html += "<hr>"; continue; }

    const quote = /^\s*&gt;\s?(.*)$/.exec(line);
    if (quote) {
      flushParagraph(); flushList();
      html += `<blockquote><p>${inlineMarkdown(quote[1])}</p></blockquote>`;
      continue;
    }

    const bullet = /^\s*[-*+]\s+(.*)$/.exec(line);
    const ordered = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    const item = bullet || ordered;
    if (item) {
      flushParagraph();
      const want = bullet ? "ul" : "ol";
      if (listTag !== want) { flushList(); html += `<${want}>`; listTag = want; }
      html += `<li>${inlineMarkdown(item[1])}</li>`;
      continue;
    }

    if (listTag) flushList();
    paragraph.push(line.trim());
  }
  flushAll();
  return html;
}

/* ------------------------------------------------------------------ *
 * State
 * ------------------------------------------------------------------ */

// Conversations live on the server now; this key is read once, by the migration.
const LEGACY_KEY = "llama-cpp-demo.history.v1";
// The two checkboxes stay in the browser and stay global: reopening an old
// conversation should not silently change what the next turn will do.
const PREFS_KEY = "llama-cpp-demo.prefs.v1";
const MAX_EDGE = 1568;

let messages = [];
let pendingImages = [];
// Parsed documents waiting to be sent: {name, kind, chars, text, warnings,
// truncated, status}. status is transient UI state ("uploading" | "error") and is
// stripped before anything is archived or put on the wire.
let pendingDocs = [];
// In-flight parses. send() refuses while this is non-zero so a half-uploaded file
// cannot leave without its body, and addDocs() serialises on it: two PDFs at once
// would make the server load a second ONNX layout session.
let docBusy = 0;
let streaming = false;
let abortController = null;
let liveBubble = null;
let liveStatus = null;
let switching = false;
let runtimeState = "stopped";
let vision = false;
// null until the first status arrives, so a restored web-search tick is dropped
// silently on page load and only a real capability *transition* warns the user.
let tools = null;
let canSwitch = false;
let currentModelId = "";

let sessions = [];
// null for a conversation that has not been saved yet — sessions are created
// lazily so the sidebar does not fill with rows somebody abandoned half-typed.
let currentSessionId = null;
let saveChain = Promise.resolve();
let saveFailed = false;

// Settings/About/Help dialog. modalOpen gates the drag handlers and the Escape
// chain, which both predate it and would otherwise act on the page underneath.
let modalOpen = false;
let activePanel = "settings";
let drawerWasOpen = false;
let modalReturnFocus = null;

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const emptyEl = $("empty");
const inputEl = $("input");
const sendBtn = $("send-btn");
const attachBtn = $("attach-btn");
const fileInput = $("file-input");
const docBtn = $("doc-btn");
const docInput = $("doc-input");
const previewsEl = $("previews");
const searchToggle = $("web-search");
const searchLabel = searchToggle.closest("label");
const SEARCH_TITLE = searchLabel.title;
const thinkingToggle = $("thinking");
const newChatBtn = $("new-chat-btn");
const copyBtn = $("copy-btn");
const modelSelect = $("model-select");
const sidebarEl = $("sidebar");
const sidebarToggle = $("sidebar-toggle");
const newBtn = $("new-btn");
const sessionListEl = $("session-list");
const sessionSearch = $("session-search");
const dropOverlay = $("drop-overlay");
const stateDot = $("state-dot");
const modelName = $("model-name");
const runtimeMeta = $("runtime-meta");
const stCpu = $("st-cpu");
const stRam = $("st-ram");
const stGpu = $("st-gpu");
const stVram = $("st-vram");
const stTemp = $("st-temp");
const gpuStatWrap = $("st-gpu-wrap");
const tempStatWrap = $("st-temp-wrap");
const GPU_STAT_TITLE = gpuStatWrap.title;

const topbarEl = document.querySelector(".topbar");
const workspaceEl = document.querySelector(".workspace");
const modalEl = $("modal");
const modalTitle = $("modal-title");
const modalBackdrop = $("modal-backdrop");
const modalCloseBtn = $("modal-close");
const railBtns = [...document.querySelectorAll(".rail-btn")];
const sideLinks = [...document.querySelectorAll(".side-link")];
const settingsFieldsEl = $("settings-fields");
const settingsMsgEl = $("settings-msg");
const settingsSaveBtn = $("settings-save");
const settingsResetBtn = $("settings-reset");
const aboutFieldsEl = $("about-fields");
const exportBtn = $("export-btn");
const exportScopeEl = $("export-scope");
const exportFormatsEl = $("export-formats");
const exportNoteEl = $("export-format-note");
const exportNameEl = $("export-name");
const exportMsgEl = $("export-msg");
const exportRunBtn = $("export-run");

function loadPrefs() {
  try {
    const saved = JSON.parse(localStorage.getItem(PREFS_KEY) || "{}");
    if (typeof saved.webSearch === "boolean") searchToggle.checked = saved.webSearch;
    if (typeof saved.thinking === "boolean") thinkingToggle.checked = saved.thinking;
  } catch {
    /* first run, or a value somebody hand-edited: keep the defaults */
  }
}

function savePrefs() {
  try {
    localStorage.setItem(
      PREFS_KEY,
      JSON.stringify({ webSearch: searchToggle.checked, thinking: thinkingToggle.checked })
    );
  } catch {
    /* private mode or quota: the checkboxes simply are not remembered */
  }
}

/* Keeps only what the server returned and drops the transient `status` a card in
   flight carries. warnings is copied, not aliased: an archived snapshot must not
   see a later mutation of the live card. */
function stripDocStatus(doc) {
  const d = doc || {};
  return {
    name: d.name || "",
    kind: d.kind || "",
    text: d.text || "",
    chars: d.chars || 0,
    pages: d.pages || 0,
    truncated: Boolean(d.truncated),
    warnings: [...(d.warnings || [])],
  };
}

/* An allowlist in both directions, never JSON.stringify(messages). renderMessage
   hangs live DOM nodes off each message as _body / _status / _reasoningEl; those
   serialise to {} and then throw inside updateLive's requestAnimationFrame when
   the conversation is reopened, which kills the stick-to-bottom scroll. */
function toStored(msg) {
  return {
    role: msg.role,
    content: msg.content || "",
    images: msg.images || [],
    had_images: (msg.images || []).length > 0 || Boolean(msg.hadImages),
    // name/kind/chars/text/warnings, not just text: the card has to re-render
    // identically when the conversation is reopened, including its warnings.
    documents: (msg.documents || []).map(stripDocStatus),
    reasoning: msg.reasoning || "",
    sources: msg.sources || [],
    usage: msg.usage || "",
    error: Boolean(msg.error),
  };
}

function fromStored(raw) {
  return {
    role: raw.role,
    content: raw.content || "",
    images: raw.images || [],
    hadImages: Boolean(raw.had_images),
    documents: (raw.documents || []).map(stripDocStatus),
    reasoning: raw.reasoning || "",
    sources: raw.sources || [],
    usage: raw.usage || "",
    error: Boolean(raw.error),
  };
}

/* `sid` and `snapshot` are passed in rather than read from the globals: send()'s
   finally runs after an abort, by which point 新建对话 may already have swapped in
   a different array. The finished turn belongs to the conversation it started in. */
async function saveConversation(sid, snapshot) {
  try {
    const resp = await fetch(sid ? `/api/sessions/${sid}` : "/api/sessions", {
      method: sid ? "PATCH" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: snapshot.map(toStored) }),
    });
    if (!resp.ok) {
      const detail = (await resp.json().catch(() => ({}))).detail;
      throw new Error(detail || `HTTP ${resp.status}`);
    }
    const record = await resp.json();
    saveFailed = false;
    // Adopt the new id only while the user is still looking at that conversation;
    // otherwise 新建对话 orphaned it mid-stream and only the list needs to know.
    if (!sid && currentSessionId === null && messages === snapshot) {
      currentSessionId = record.id;
    }
    await refreshSessions();
  } catch (err) {
    // currentSessionId is deliberately left alone so the next turn retries the
    // create, and the messages are never dropped. One alert per failure streak:
    // send()'s finally is not something the user waits on, so a retry queue would
    // be invisible, but believing a conversation was saved when it was not is worse.
    if (!saveFailed) {
      saveFailed = true;
      alert(`对话未能保存到本地历史：${err.message}`);
    }
  }
}

function queueSave(sid, snapshot) {
  const run = () => saveConversation(sid, snapshot);
  saveChain = saveChain.then(run);
  return saveChain;
}

/* ------------------------------------------------------------------ *
 * Clipboard
 * ------------------------------------------------------------------ */

const ICON_COPY = "M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z";
const ICON_CHECK = "M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z";
const ICON_EXPORT = "M10 3h4v8h4l-6 7-6-7h4V3zM5 20h14v2H5v-2z";

async function clipboardWrite(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  // HOST is configurable, so the page can be opened over a LAN address, which is
  // not a secure context and therefore has no navigator.clipboard at all.
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.readOnly = true;
  ta.style.cssText = "position:fixed;top:0;left:0;opacity:0";
  document.body.append(ta);
  ta.select();
  const ok = document.execCommand("copy");
  ta.remove();
  if (!ok) throw new Error("浏览器拒绝了复制");
}

function iconButton(label, size, path = ICON_COPY) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "msg-copy";
  btn.title = label;
  btn.setAttribute("aria-label", label);
  btn.innerHTML =
    `<svg viewBox="0 0 24 24" width="${size}" height="${size}" aria-hidden="true">` +
    `<path fill="currentColor" d="${path}"/></svg>`;
  return btn;
}

async function copyAndFlash(btn, text) {
  try {
    await clipboardWrite(text);
  } catch (err) {
    alert(`复制失败：${err.message}`);
    return;
  }
  const path = btn.querySelector("svg path");
  // Remembered once, not read back from an attribute: the topbar button is authored
  // in HTML and carries no data-label, and a rapid second click would otherwise
  // capture "已复制" as the thing to restore.
  btn._label ??= btn.title;
  path.setAttribute("d", ICON_CHECK);
  btn.classList.add("copied");
  btn.title = "已复制";
  clearTimeout(btn._timer);
  btn._timer = setTimeout(() => {
    path.setAttribute("d", ICON_COPY);
    btn.classList.remove("copied");
    btn.title = btn._label;
  }, 1200);
}

/* Copies raw Markdown rather than the rendered HTML, because that is what pastes
   usefully anywhere else. Reasoning and sources are left out on purpose: they are
   secondary to the answer, and one button has to mean one obvious thing. */
function conversationText() {
  return messages
    .filter((m) => m.content || (m.documents || []).length)
    .map((m) => {
      const who = m.role === "user" ? "你" : "助手";
      // Names only. Inlining the bodies would swamp the transcript — one PDF can
      // be 200k characters — but dropping them silently makes a turn that was just
      // a file plus a question copy as nothing at all.
      const docs = (m.documents || []).length
        ? `［附件：${m.documents.map((d) => d.name).join("、")}］\n`
        : "";
      return `${who}：\n${docs}${m.content}`;
    })
    .join("\n\n");
}

/* ------------------------------------------------------------------ *
 * Export
 * ------------------------------------------------------------------ */

/* The message a per-message export is pointed at, or null for the whole
   conversation. The object, never an index: newConversation() and
   deleteSession() reassign `messages`, so an index captured while the dialog is
   open can point at a different turn, or past the end. Re-checked with
   includes() at export time for the same reason. */
let exportTarget = null;

/* One source of truth for what gets exported. Whole-conversation export goes
   through conversationText() and therefore matches 「复制整个对话」 by
   construction — same text-only body, attachments named but not inlined, no
   reasoning and no sources — instead of by a second implementation that drifts. */
function exportSource() {
  if (!exportTarget) return conversationText();
  return messages.includes(exportTarget) ? exportTarget.content : "";
}

/* The top-level block list the server needs for PDF and DOCX.

   Recomputed from the markdown rather than read off msg._body.children, which
   would be the obvious thing: user turns have no _body (renderMessage hangs one
   only on assistant turns), so this is the form that works for both roles, for
   restored sessions and for a turn still streaming. Block boundaries come from
   the DOM because it is the authority — splitting the joined HTML server-side
   was tried both by regex and by html.parser.getpos() and both got the offsets
   wrong. */
function blocksFromMarkdown(src) {
  const holder = document.createElement("div");
  holder.innerHTML = renderMarkdown(src);
  return [...holder.children].map((el) => el.outerHTML);
}

/* A light re-authoring of the .md rules in style.css:373-400, for a file that
   will be read somewhere else and printed. The class name is kept so the two can
   be diffed rule by rule; only the variable values and the `pre` background
   differ.

   Two things are load-bearing. `pre` must be #f6f7f9 and NOT style.css's
   #10131a, which prints as a solid black block. And the body font has to be
   stated: .md inherits it from body (style.css:25-31) and a standalone file has
   nothing to inherit from, so omitting it silently falls back to Times.

   app/export.py's _PDF_CSS is the same rewrite aimed at MuPDF's HTML subset,
   which supports only a fraction of CSS. The three copies can drift; accepted,
   because sharing them would need a build step this app does not have. */
const EXPORT_CSS = `
:root {
  color-scheme: light;
  --accent: #0b57d0;
  --accent-soft: rgba(11, 87, 208, 0.08);
  --muted: #57606a;
  --panel-2: #f6f7f9;
  --border: #d0d7de;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 20px; background: #ffffff; color: #1f2328;
  font: 15px/1.65 "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", system-ui, sans-serif;
}
.md { max-width: 860px; margin: 0 auto; overflow-wrap: break-word; }
.md > :first-child { margin-top: 0; }
.md > :last-child { margin-bottom: 0; }
.md p { margin: 0 0 10px; }
.md h1, .md h2, .md h3, .md h4 { margin: 16px 0 8px; line-height: 1.35; font-weight: 600; }
.md h1 { font-size: 19px; } .md h2 { font-size: 17px; } .md h3 { font-size: 15.5px; } .md h4 { font-size: 15px; }
.md ul, .md ol { margin: 0 0 10px; padding-left: 22px; }
.md li { margin: 3px 0; }
.md blockquote {
  margin: 0 0 10px; padding: 4px 12px;
  border-left: 3px solid var(--accent); background: var(--accent-soft);
  border-radius: 0 8px 8px 0; color: var(--muted);
}
.md a { color: var(--accent); }
.md code {
  font-family: "Cascadia Code", Consolas, "Courier New", monospace;
  font-size: 13px; background: var(--panel-2);
  padding: 1.5px 5px; border-radius: 5px; border: 1px solid var(--border);
}
.md pre {
  margin: 0 0 10px; padding: 12px 14px; overflow-x: auto;
  background: #f6f7f9; border: 1px solid var(--border); border-radius: 10px;
}
.md pre code { background: none; border: none; padding: 0; font-size: 13px; line-height: 1.55; }
.md hr { border: none; border-top: 1px solid var(--border); margin: 14px 0; }
.md table { border-collapse: collapse; margin: 0 0 10px; font-size: 13.5px; width: 100%; }
.md th, .md td { border: 1px solid var(--border); padding: 5px 9px; text-align: left; }
.md th { background: var(--panel-2); }
.md del { color: var(--muted); }
`;

function exportHtmlDocument(src, title) {
  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(title)}</title>
<style>${EXPORT_CSS}</style>
</head>
<body>
<div class="md">
${renderMarkdown(src)}
</div>
</body>
</html>
`;
}

/* Inline markdown is noise inside a spreadsheet cell, so strip it. The mirror of
   inlineMarkdown's patterns, run on the raw source: table helpers see the text
   before any escaping, so this is not escapeHtml's inverse. */
function plainInline(text) {
  return text
    .replace(/`([^`\n]+)`/g, "$1")
    .replace(/\[([^\]\n]*)\]\([^)\s]+\)/g, "$1")
    .replace(/\*\*([^*\n]+)\*\*/g, "$1")
    .replace(/(^|[^*\w])\*([^*\n]+)\*/g, "$1$2")
    .replace(/~~([^~\n]+)~~/g, "$1");
}

const csvCell = (v) => (/[",\r\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);

/* Consecutive pipe rows grouped into tables. Reuses isTableRow / isTableDivider
   / splitRow rather than re-deriving them — they already handle the divider row,
   which is the easy thing to get wrong.

   Fenced code comes out first, exactly as renderMarkdown does: a | inside a code
   block is not a table row, and treating it as one would put source code in the
   CSV. */
function tablesFromMarkdown(src) {
  const tables = [];
  let rows = null;
  for (const line of src.replace(/```[^\n`]*\n?[\s\S]*?```/g, "\n").split("\n")) {
    if (!isTableRow(line)) {
      if (rows) { tables.push(rows); rows = null; }
      continue;
    }
    if (isTableDivider(line)) continue;
    (rows ??= []).push(splitRow(line).map(plainInline));
  }
  if (rows) tables.push(rows);
  return tables;
}

/* Empty string when there is no table, so the caller can refuse instead of
   writing a file with nothing in it. */
function toCsv(src) {
  return tablesFromMarkdown(src)
    .map((rows) => rows.map((r) => r.map(csvCell).join(",")).join("\r\n"))
    // CRLF is what RFC 4180 asks for. A blank line between tables keeps two of
    // them apart when one answer holds both.
    .join("\r\n\r\n");
}

/* Excel on this machine defaults to GBK, so a UTF-8 CSV without a byte-order
   mark opens as mojibake. Prepended at the call site rather than inside toCsv,
   so that function stays a plain CSV string. */
const CSV_BOM = "\uFEFF";

/* app/history.py:35 TITLE_CHARS. The same cap here and there is what lets a
   session title become a filename without a second rule. */
const TITLE_CHARS = 40;

const FS_UNSAFE = /[\\/:*?"<>|\u0000-\u001f]/g;
const FS_RESERVED = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i;

/* Not optional. The real archive at runtime/history/index.json holds a title
   starting '> **【角色设定】**：…' — both > and * are illegal in a Windows
   filename — and two sessions whose titles are byte-identical. */
function sanitiseFilename(name) {
  let s = String(name || "").replace(FS_UNSAFE, "_");
  // Windows silently strips leading and trailing dots and spaces, so "…" and "…."
  // are not the name the user typed once they land on disk.
  s = s.replace(/^[.\s]+/, "").replace(/[.\s]+$/, "").slice(0, TITLE_CHARS);
  // A device name is reserved regardless of extension, so "con.md" is still bad:
  // Windows resolves a path component to a device by the part before the first dot.
  // Hence the stem, not the whole string, against an anchored pattern.
  //
  // Deliberately no `if (!s)` here. Making the result non-empty is the caller's job,
  // not this function's: returning "_" for empty input is truthy, which silently
  // defeats the `|| title || "对话"` fallback chain in exportFilename and prefills
  // the 文件名 field with "_" when there is no session. Measured, not hypothetical.
  if (FS_RESERVED.test(s.split(".")[0])) s = `_${s}`;
  return s;
}

function exportFilename(fmt, suffix = "") {
  // sessions can legitimately be empty: /api/sessions 404s on a static server and
  // can fail on a real one, and an export still has to produce a file.
  const title = (sessions.find((s) => s.id === currentSessionId) || {}).title || "";
  const base = sanitiseFilename(exportNameEl.value) || sanitiseFilename(title) || "对话";
  // Minute-resolution local time. Two exports of the same conversation in the same
  // minute are deliberately the same name — same content, same name — and the
  // browser appends "(1)" for a real collision. The stamp is what separates two
  // different sessions that happen to share a title, which the archive does.
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  const stamp = `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
  return `${base}${suffix}-${stamp}.${fmt}`;
}

/* The only download path in the app. A real filename on a same-origin blob URL
   means no navigation and no "save as" dialog. */
function saveBlob(data, mime, filename) {
  const blob = data instanceof Blob ? data : new Blob([data], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.style.display = "none";
  document.body.append(a);
  a.click();
  // Not revoked synchronously: click() only queues the download, so revoking in
  // the same tick races it and can produce a 0-byte file. Four seconds covers any
  // save dialog; the cost is one object URL per export.
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 4000);
}

/* ------------------------------------------------------------------ *
 * Rendering
 * ------------------------------------------------------------------ */

function hostOf(url) {
  try { return new URL(url).host; } catch { return url; }
}

function buildReasoning(text, open) {
  const det = document.createElement("details");
  det.className = "reasoning";
  det.open = open;
  const sum = document.createElement("summary");
  sum.textContent = "思考过程";
  const body = document.createElement("div");
  body.className = "md";
  body.innerHTML = text ? renderMarkdown(text) : "";
  det.append(sum, body);
  return det;
}

function buildSources(sources) {
  if (!sources.length) return null;
  const wrap = document.createElement("div");
  wrap.className = "sources";
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "sources-toggle";
  toggle.textContent = `检索来源 ${sources.length} 条`;
  const list = document.createElement("div");
  list.className = "sources-list";
  list.hidden = true;
  for (const s of sources) {
    const row = document.createElement("div");
    row.className = "source";
    const idx = document.createElement("span");
    idx.className = "source-index";
    idx.textContent = `[${s.index}]`;
    const body = document.createElement("div");
    body.className = "source-body";
    const link = document.createElement("a");
    link.href = s.url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = s.title || s.url;
    const host = document.createElement("div");
    host.className = "source-host";
    host.textContent = hostOf(s.url);
    body.append(link, host);
    row.append(idx, body);
    list.append(row);
  }
  toggle.onclick = () => {
    list.hidden = !list.hidden;
    toggle.textContent = list.hidden ? `检索来源 ${sources.length} 条` : "收起来源";
  };
  wrap.append(toggle, list);
  return wrap;
}

/* One collapsible card per attachment. An extracted body can run to 200k
   characters, so it is folded away by default and scrollable when opened —
   pouring it into the bubble would bury the question it was attached to. */
function buildDocs(docs) {
  if (!docs || !docs.length) return null;
  const wrap = document.createElement("div");
  wrap.className = "msg-docs";
  for (const d of docs) {
    const card = document.createElement("div");
    card.className = "doc-card";
    if (d.truncated) card.classList.add("truncated");

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "doc-toggle";
    toggle.setAttribute("aria-expanded", "false");
    const name = document.createElement("span");
    name.className = "doc-name";
    name.textContent = d.name || "未命名文件";
    name.title = d.name || "";
    const badge = document.createElement("span");
    badge.className = "doc-badge";
    badge.textContent = d.kind || "文件";
    const size = document.createElement("span");
    size.className = "doc-size";
    size.textContent = d.pages
      ? `${(d.chars || 0).toLocaleString("zh-CN")} 字 · ${d.pages} 页`
      : `${(d.chars || 0).toLocaleString("zh-CN")} 字`;
    toggle.append(name, badge, size);

    const body = document.createElement("div");
    body.className = "doc-body";
    body.hidden = true;
    body.textContent = d.text || "";
    toggle.onclick = () => {
      body.hidden = !body.hidden;
      toggle.setAttribute("aria-expanded", String(!body.hidden));
    };

    card.append(toggle, body);
    for (const w of d.warnings || []) {
      const warn = document.createElement("div");
      warn.className = "doc-warn";
      warn.textContent = w;
      card.append(warn);
    }
    wrap.append(card);
  }
  return wrap;
}

function renderMessage(msg) {
  const wrap = document.createElement("div");
  wrap.className = `msg msg-${msg.role}`;

  const role = document.createElement("div");
  role.className = "msg-role";
  const label = document.createElement("span");
  label.textContent = msg.role === "user" ? "你" : "助手";
  role.append(label);
  // Read at click time, not here: the same msg object keeps mutating while the
  // answer streams in. Skipped while empty so the "正在思考…" placeholder does not
  // offer a button that would copy nothing; renderMessageInto rebuilds the node
  // once text has arrived.
  if (msg.content) {
    const copy = iconButton(msg.role === "user" ? "复制这条消息" : "复制这条回答", 14);
    copy.onclick = () => copyAndFlash(copy, msg.content);
    role.append(copy);

    const exp = iconButton(msg.role === "user" ? "导出这条消息" : "导出这条回答", 14, ICON_EXPORT);
    // Deliberately not copyAndFlash: it writes ICON_COPY back into the path after
    // 1.2 s, so a flashed export icon would end up wearing a copy icon. The dialog
    // appearing is the feedback here, so there is nothing to flash.
    exp.onclick = () => openExportFor(msg, exp);
    role.append(exp);
  }
  wrap.append(role);

  if (msg.images && msg.images.length) {
    const box = document.createElement("div");
    box.className = "msg-images";
    for (const src of msg.images) {
      const img = document.createElement("img");
      img.src = src;
      img.alt = "上传的图片";
      box.append(img);
    }
    wrap.append(box);
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (msg.error) bubble.classList.add("error");

  if (msg.role === "assistant") {
    const body = document.createElement("div");
    body.className = "md";
    body.innerHTML = msg.content ? renderMarkdown(msg.content) : "";
    bubble.append(body);
    msg._body = body;

    if (msg.reasoning) {
      const det = buildReasoning(msg.reasoning, false);
      msg._reasoningEl = det.querySelector(".md");
      bubble.insertBefore(det, body);
    }

    if (msg.status) {
      const status = document.createElement("div");
      status.className = "status-line";
      status.textContent = msg.status;
      bubble.prepend(status);
      msg._status = status;
    }
    const sources = buildSources(msg.sources || []);
    if (sources) bubble.append(sources);
    if (msg.usage) {
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = msg.usage;
      bubble.append(meta);
    }
  } else {
    const docs = buildDocs(msg.documents);
    if (docs) bubble.append(docs);
    // A text node, not bubble.textContent: that assignment would wipe the cards
    // just appended. Renders identically inside a block container.
    bubble.append(document.createTextNode(msg.content));
    // had_images is set for every message that carried pictures, including the ones
    // archived intact and rendered just above — the note is only for the conversations
    // migrated in from localStorage, which recorded the flag after dropping the pixels.
    if (msg.hadImages && !(msg.images && msg.images.length)) {
      const note = document.createElement("div");
      note.className = "meta";
      note.textContent = "（这条消息的图片没有保存下来）";
      bubble.append(note);
    }
  }

  wrap.append(bubble);
  return wrap;
}

function renderAll() {
  messagesEl.replaceChildren();
  emptyEl.hidden = messages.length > 0;
  for (const msg of messages) messagesEl.append(renderMessage(msg));
  liveBubble = null;
  liveStatus = null;
  scrollToBottom(true);
  // Every path that changes the message list comes through here, so this is what
  // stops the copy button from staying enabled after 清空对话 until the next poll.
  syncControls();
}

let atBottom = true;
const chatEl = $("chat");
chatEl.addEventListener("scroll", () => {
  atBottom = chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight < 90;
});

function scrollToBottom(force = false) {
  if (force || atBottom) chatEl.scrollTop = chatEl.scrollHeight;
}

let renderQueued = false;
function updateLive(force = false) {
  if (renderQueued && !force) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    const msg = messages[messages.length - 1];
    if (!msg || msg.role !== "assistant" || !msg._body) return;
    msg._body.innerHTML = msg.content ? renderMarkdown(msg.content) : "";
    msg._body.classList.toggle("cursor", streaming);
    if (msg._reasoningEl) msg._reasoningEl.innerHTML = renderMarkdown(msg.reasoning || "");
    scrollToBottom();
  });
}

/* ------------------------------------------------------------------ *
 * Images
 * ------------------------------------------------------------------ */

function downscale(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("读取失败"));
    reader.onload = () => {
      const img = new Image();
      img.onerror = () => reject(new Error("不是有效图片"));
      img.onload = () => {
        const scale = Math.min(1, MAX_EDGE / Math.max(img.width, img.height));
        const w = Math.max(1, Math.round(img.width * scale));
        const h = Math.max(1, Math.round(img.height * scale));
        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        canvas.getContext("2d").drawImage(img, 0, 0, w, h);
        const isPng = file.type === "image/png";
        resolve(canvas.toDataURL(isPng ? "image/png" : "image/jpeg", isPng ? undefined : 0.88));
      };
      img.src = reader.result;
    };
    reader.readAsDataURL(file);
  });
}

async function addFiles(fileList) {
  if (!vision) return;
  const files = [...fileList].filter((f) => f.type.startsWith("image/"));
  for (const file of files) {
    if (pendingImages.length >= 4) {
      alert("每条消息最多 4 张图片");
      break;
    }
    try {
      pendingImages.push(await downscale(file));
    } catch (err) {
      alert(`图片处理失败：${err.message}`);
    }
  }
  renderPreviews();
}

/* ------------------------------------------------------------------ *
 * Documents
 * ------------------------------------------------------------------ */

// Must match MAX_DOCS_PER_MESSAGE in app/main.py; past it the request 422s.
const MAX_DOCS = 3;
const DOC_EXT = new Set(["md", "markdown", "txt", "pdf", "docx", "xlsx", "pptx"]);
// Named separately rather than left to fall through as "unknown": these look like
// they should work, so silence would read as a bug rather than a format limit.
const LEGACY_EXT = { doc: "docx", xls: "xlsx", ppt: "pptx" };

function extOf(name) {
  const s = String(name || "");
  const i = s.lastIndexOf(".");
  return i < 0 ? "" : s.slice(i + 1).toLowerCase();
}

/* Extension first, MIME second: a dropped .txt arrives as text/plain and a .docx
   as application/vnd…, but browsers disagree enough that the name is the more
   reliable signal for everything except pictures. */
function classify(file) {
  const ext = extOf(file.name);
  if (DOC_EXT.has(ext) || ext in LEGACY_EXT) return "doc";
  if (String(file.type || "").startsWith("image/")) return "image";
  return "other";
}

/* No vision gate here, unlike addFiles: text extraction has nothing to do with the
   mmproj projector, so greying this out on a text-only model would be the most
   annoying possible way to get the capability matrix wrong. */
async function addDocs(fileList) {
  // Snapshotted before the first await — the caller resets input.value straight
  // after, which empties the live FileList out from under an async loop.
  const files = [...fileList];
  for (const file of files) {
    const ext = extOf(file.name);
    if (ext in LEGACY_EXT) {
      alert(`不支持 2007 以前的 .${ext}，请在 Office 或 WPS 里「另存为」.${LEGACY_EXT[ext]} 后再上传`);
      continue;
    }
    if (!DOC_EXT.has(ext)) continue; // dispatchFiles already reported it
    if (pendingDocs.filter((d) => d.status !== "error").length >= MAX_DOCS) {
      alert(`每条消息最多 ${MAX_DOCS} 个文件`);
      break;
    }
    const card = {
      name: file.name, kind: "", text: "", chars: 0, pages: 0,
      truncated: false, warnings: [], status: "uploading",
    };
    pendingDocs.push(card);
    renderPreviews();
    docBusy++;
    syncControls();
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      const resp = await fetch("/api/documents", { method: "POST", body: form });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
      // Replaced in place, not pushed: a file added while this one was in flight
      // must not end up queued behind it. -1 means the user removed the card.
      const at = pendingDocs.indexOf(card);
      if (at >= 0) pendingDocs[at] = { ...stripDocStatus(data), status: "" };
    } catch (err) {
      // The card stays, red, with the server's own reason. No alert: the chip is
      // already in front of the user and an alert would cover it.
      card.status = "error";
      card.error = err.message || "解析失败";
    } finally {
      docBusy--;
      renderPreviews();
      syncControls();
    }
  }
}

/* One entry point for drag and paste, which can carry both kinds at once.
   Unknown types are named rather than ignored, because "nothing happened" gives
   the user nothing to act on. */
function dispatchFiles(fileList) {
  const images = [];
  const docs = [];
  const other = [];
  for (const f of fileList) {
    const kind = classify(f);
    if (kind === "image") images.push(f);
    else if (kind === "doc") docs.push(f);
    else other.push(f.name);
  }
  if (other.length) {
    alert(`不支持的文件类型：${other.slice(0, 3).join("、")}${other.length > 3 ? " 等" : ""}`);
  }
  // addFiles is still vision-gated, so on a text-only model the pictures fall away
  // there exactly as they did before documents existed.
  if (images.length) addFiles(images);
  if (docs.length) addDocs(docs);
}

function renderPreviews() {
  previewsEl.replaceChildren();
  pendingImages.forEach((src, i) => {
    const box = document.createElement("div");
    box.className = "preview";
    const img = document.createElement("img");
    img.src = src;
    img.alt = `待发送图片 ${i + 1}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "\u00d7";
    remove.title = "移除";
    remove.onclick = () => {
      pendingImages.splice(i, 1);
      renderPreviews();
    };
    box.append(img, remove);
    previewsEl.append(box);
  });
  pendingDocs.forEach((doc, i) => {
    const box = document.createElement("div");
    box.className = "preview-doc";
    if (doc.status === "uploading") box.classList.add("uploading");
    if (doc.status === "error") box.classList.add("failed");
    const name = document.createElement("div");
    name.className = "pd-name";
    name.textContent = doc.name;
    name.title = doc.name;
    const meta = document.createElement("div");
    meta.className = "pd-meta";
    if (doc.status === "uploading") {
      meta.textContent = "上传解析中…";
    } else if (doc.status === "error") {
      meta.textContent = doc.error || "解析失败";
    } else {
      const bits = [doc.kind || "文件", `${(doc.chars || 0).toLocaleString("zh-CN")} 字`];
      if (doc.pages) bits.push(`${doc.pages} 页`);
      if (doc.warnings && doc.warnings.length) bits.push("有提示");
      meta.textContent = bits.join(" · ");
      if (doc.warnings && doc.warnings.length) meta.title = doc.warnings.join("\n");
    }
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "\u00d7";
    remove.title = "移除";
    remove.onclick = () => {
      pendingDocs.splice(i, 1);
      renderPreviews();
      syncControls();
    };
    box.append(name, meta, remove);
    previewsEl.append(box);
  });
}

attachBtn.onclick = () => fileInput.click();
fileInput.onchange = () => { addFiles(fileInput.files); fileInput.value = ""; };
docBtn.onclick = () => docInput.click();
docInput.onchange = () => { addDocs(docInput.files); docInput.value = ""; };

let dragDepth = 0;
/* Widened from plain `vision`: the banner is the only hint that dropping does
   anything at all, and a .pdf must still land on a text-only model. The file list
   is readable during dragenter, so "is any of these a document" is answerable. */
function dropWanted(e) {
  if (modalOpen || ![...e.dataTransfer.types].includes("Files")) return false;
  return vision || [...e.dataTransfer.files].some((f) => classify(f) === "doc");
}
window.addEventListener("dragenter", (e) => {
  if (!dropWanted(e)) return;
  dragDepth++;
  dropOverlay.classList.add("active");
});
// Deliberately not vision-gated: without an unconditional preventDefault, dropping
// a file on a text-only model navigates the tab away from the app.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("dragleave", () => {
  if (--dragDepth <= 0) { dragDepth = 0; dropOverlay.classList.remove("active"); }
});
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  dropOverlay.classList.remove("active");
  // Gated as well as dragenter: the banner cannot appear while the dialog is open,
  // but the drop still lands, and attaching to an invisible composer is worse than
  // refusing. Same reason the preventDefault above stays unconditional. The vision
  // check moved into dispatchFiles, which still gates images but not documents.
  if (modalOpen) return;
  if (e.dataTransfer.files.length) dispatchFiles(e.dataTransfer.files);
});

inputEl.addEventListener("paste", (e) => {
  const files = [...(e.clipboardData?.files || [])];
  if (files.length) { e.preventDefault(); dispatchFiles(files); }
});

/* ------------------------------------------------------------------ *
 * Sending
 * ------------------------------------------------------------------ */

// Must match MAX_CHAT_MESSAGES in app/main.py. A resumed conversation can be much
// longer than one sitting, so send the tail: past the cap the request 422s and the
// conversation would otherwise become permanently unsendable.
const MAX_CHAT_MESSAGES = 200;

/* A turn that produced no words must not go back on the wire. Five paths leave an
   assistant bubble empty: an abort via 停止 / Enter-mid-stream / 新建对话 (send()'s
   catch skips AbortError, so neither content nor error is set), a stream that ends
   with no delta, a thinking-only turn, a sources-only turn, a done-only turn — plus
   content that is whitespace, which nothing here ever trims. Replaying one hands the
   model […, user X, assistant "", user X], and it answers with an immediate EOS.
   Measured on the session that reported 罢工: GENERATED 1 tokens, ctx=5972,
   truncated=0 — 4.5% of the window used, so context was never the constraint.

   error turns go too: their content is our own "请求失败：…", and the model learns to
   say it. The bubble keeps its red styling, which renderMessage drives off msg.error.

   Filtered HERE and nowhere else. The bubble and the archive both keep the record, so
   an already-broken session self-heals on its next send with no data migration.

   A user turn survives on attachments alone: send() refuses a text-less turn, but one
   restored from the archive may be pixels-only. */
function isSendable(msg) {
  const text = (msg.content || "").trim();
  if (msg.role === "user") {
    return Boolean(text) || (msg.documents || []).length > 0 || (msg.images || []).length > 0;
  }
  return !msg.error && Boolean(text);
}

function wireMessages() {
  // Slice first, then filter: MAX_CHAT_MESSAGES guards the array the server validates,
  // and filtering can only ever shorten it. Both `out` and the index `i` below must come
  // from the same array or they drift apart.
  const out = messages.slice(-MAX_CHAT_MESSAGES).filter(isSendable);
  return out.map((msg, i) => {
    // out.length - 1, NOT - 2. send() pushes the empty "正在思考…" placeholder before
    // this runs, and isSendable removes it, so the newest user turn is the LAST element
    // here. Leaving -2 hands the pixels to the turn before it — an assistant turn, whose
    // isLastUser is false, so the images go nowhere at all and vision silently stops
    // working. That -1 is safe because it is an invariant, not a hope: send() returns
    // unless inputEl.value.trim() is non-empty, so the user turn it pushes always passes
    // isSendable, and this function has exactly one call site. Hence out.length >= 1 and
    // out[out.length - 1].role === "user".
    const isLastUser = msg.role === "user" && i === out.length - 1;
    // Only the newest user turn keeps its pixels; re-sending older images
    // would re-run the vision encoder over the whole history every turn.
    // The vision check matters after switching to a text-only model, whose
    // server was started without --mmproj and rejects any image part.
    //
    // Documents are the OPPOSITE and must not be narrowed to the newest turn:
    // "刚才那份文档的第二章讲了什么" only works if the body attached three turns
    // ago is still there. Every turn keeps its bodies and the server's fit_budget
    // decides what to drop, because only the server knows the real n_ctx. Trimming
    // here instead would be invisible to the user and impossible to recover from.
    return {
      role: msg.role,
      content: msg.content,
      images: isLastUser && vision ? msg.images || [] : [],
      documents: (msg.documents || []).map((d) => ({ name: d.name, text: d.text })),
    };
  });
}

async function send() {
  if (streaming) return stop();
  // A parse in flight has no body yet, so this would attach a filename with nothing
  // behind it. syncControls disables the button for exactly this, but Enter does not
  // go through the button.
  if (docBusy) return;

  const text = inputEl.value.trim();
  if (!text && !pendingImages.length && !pendingDocs.length) return;
  if (!text) { alert("请输入文字：图片和文件需要配合一个问题一起发送"); return; }

  // Captured for the finally below. `turn` is the array itself, not a copy: the
  // pushes that follow are what it needs to see, and 新建对话 reassigns the module
  // binding rather than emptying this array in place.
  const sid = currentSessionId;
  const turn = messages;

  // Cards still showing red are left out: their chip names the reason, so the user
  // can see what is missing, and sending an empty body would look like success.
  const docs = pendingDocs.filter((d) => !d.status).map(stripDocStatus);
  messages.push({
    role: "user",
    content: text,
    images: [...pendingImages],
    documents: docs,
  });
  pendingImages = [];
  pendingDocs = [];
  renderPreviews();
  inputEl.value = "";
  autoGrow();

  const assistant = { role: "assistant", content: "", sources: [], status: "正在思考…" };
  messages.push(assistant);
  emptyEl.hidden = true;
  renderAll();

  const node = messagesEl.lastElementChild;
  liveBubble = node.querySelector(".bubble");
  assistant._body = node.querySelector(".md");
  assistant._status = node.querySelector(".status-line");
  liveStatus = assistant._status;

  streaming = true;
  atBottom = true;
  sendBtn.textContent = "停止";
  abortController = new AbortController();
  syncControls();

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: wireMessages(),
        web_search: searchToggle.checked,
        thinking: thinkingToggle.checked,
      }),
      signal: abortController.signal,
    });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let sep;
      while ((sep = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const dataLines = [];
        let eventName = "message";
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) eventName = line.slice(6).trim();
          else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
        }
        if (!dataLines.length) continue;
        handleEvent(eventName, JSON.parse(dataLines.join("\n")), assistant);
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      assistant.error = true;
      assistant.content = assistant.content || `请求失败：${err.message}`;
    }
  } finally {
    streaming = false;
    abortController = null;
    sendBtn.textContent = "发送";
    assistant.status = "";
    syncControls();
    renderMessageInto(node, assistant);
    queueSave(sid, turn);
  }
}

function renderMessageInto(node, msg) {
  const fresh = renderMessage(msg);
  node.replaceWith(fresh);
}

function handleEvent(name, data, assistant) {
  if (name === "reasoning") {
    assistant.reasoning = (assistant.reasoning || "") + (data.text || "");
    if (!assistant._reasoningEl && liveBubble) {
      const det = buildReasoning("", true);
      assistant._reasoningEl = det.querySelector(".md");
      liveBubble.insertBefore(det, assistant._body);
    }
    updateLive();
  } else if (name === "delta") {
    assistant.status = "";
    if (liveStatus) { liveStatus.remove(); liveStatus = null; }
    assistant.content += data.text || "";
    updateLive();
  } else if (name === "status") {
    assistant.status = data.text || "";
    if (liveStatus) liveStatus.textContent = assistant.status;
    else if (liveBubble) {
      liveStatus = document.createElement("div");
      liveStatus.className = "status-line";
      liveStatus.textContent = assistant.status;
      liveBubble.prepend(liveStatus);
    }
  } else if (name === "sources") {
    assistant.sources.push(...(data.items || []));
    assistant.status = "已获取检索结果，正在整理答案…";
    if (liveStatus) liveStatus.textContent = assistant.status;
  } else if (name === "done") {
    const u = data.usage || {};
    if (u.prompt_tokens || u.completion_tokens) {
      assistant.usage = `${u.prompt_tokens ?? "?"} 输入 / ${u.completion_tokens ?? "?"} 输出 tokens`;
    }
    if (data.sources?.length) assistant.sources = data.sources.map((s) => ({ ...s, url: s.url }));
    assistant.status = "";
  } else if (name === "error") {
    assistant.error = true;
    assistant.content = assistant.content
      ? `${assistant.content}\n\n**出错：** ${data.text}`
      : `出错：${data.text}`;
    assistant.status = "";
    if (liveStatus) { liveStatus.remove(); liveStatus = null; }
    updateLive(true);
  }
}

function stop() {
  if (abortController) abortController.abort();
}

function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 190)}px`;
}

inputEl.addEventListener("input", autoGrow);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send();
  }
});
sendBtn.onclick = send;

copyBtn.onclick = () => copyAndFlash(copyBtn, conversationText());

searchToggle.onchange = savePrefs;
thinkingToggle.onchange = savePrefs;

/* ------------------------------------------------------------------ *
 * Session sidebar
 * ------------------------------------------------------------------ */

const ICON_RENAME =
  "M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04a1 1 0 0 0 0-1.41l-2.34-2.34a1 1 0 0 0-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z";
const ICON_DELETE = "M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z";

function relTime(iso) {
  const then = new Date(iso).getTime();
  if (!then) return "";
  const minutes = Math.floor((Date.now() - then) / 60000);
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  if (hours < 48) return "昨天";
  const d = new Date(then);
  const now = new Date();
  return d.getFullYear() === now.getFullYear()
    ? `${d.getMonth() + 1}月${d.getDate()}日`
    : `${d.getFullYear()}/${d.getMonth() + 1}/${d.getDate()}`;
}

async function refreshSessions() {
  try {
    const resp = await fetch("/api/sessions");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    sessions = (await resp.json()).sessions || [];
  } catch {
    // The conversation on screen is unaffected, so leave the list as it was
    // rather than emptying it; a dead backend is already reported by the dot.
    return;
  }
  renderSessions();
}

function sessionRow(s) {
  const row = document.createElement("div");
  row.className = "session";
  row.dataset.id = s.id;
  row.tabIndex = 0;
  row.setAttribute("role", "button");
  if (s.id === currentSessionId) row.classList.add("active");
  row.title = `${s.title}\n${s.message_count} 条消息${s.model ? ` · ${s.model}` : ""}`;

  const main = document.createElement("div");
  main.className = "session-main";
  const title = document.createElement("div");
  title.className = "session-title";
  title.textContent = s.title;
  const meta = document.createElement("div");
  meta.className = "session-meta";
  meta.textContent = `${relTime(s.updated)} · ${s.message_count} 条`;
  main.append(title, meta);

  // Separate buttons rather than a double-click on the row: dblclick fires after
  // two clicks, and the first one re-renders the list, destroying the very node
  // the pending dblclick was aimed at.
  const rename = iconButton("重命名", 14, ICON_RENAME);
  rename.onclick = (e) => { e.stopPropagation(); renameSession(s); };
  const del = iconButton("删除", 14, ICON_DELETE);
  del.classList.add("danger");
  del.onclick = (e) => { e.stopPropagation(); deleteSession(s); };

  row.append(main, rename, del);
  row.onclick = () => openSession(s.id);
  row.onkeydown = (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openSession(s.id); }
  };
  return row;
}

function renderSessions() {
  // Filtered here rather than on the server: the index already carries a digest of
  // each conversation's text, so matching title and body is a local string search.
  const needle = sessionSearch.value.trim().toLowerCase();
  const shown = needle
    ? sessions.filter((s) =>
        `${s.title}\n${s.digest || ""}`.toLowerCase().includes(needle))
    : sessions;

  sessionListEl.replaceChildren();
  if (!shown.length) {
    const empty = document.createElement("div");
    empty.className = "session-empty";
    empty.textContent = needle ? "没有匹配的对话" : "还没有历史对话";
    sessionListEl.append(empty);
  }
  for (const s of shown) sessionListEl.append(sessionRow(s));
  // This just replaced every node syncControls annotated, so re-apply — the same
  // reason renderAll ends the same way.
  syncControls();
}

// Opening a conversation changes the selection, not the data, so renderSessions is
// the wrong tool: rebuilding would also drop focus from a row opened with the
// keyboard. The empty-state placeholder has no dataset.id and simply loses.
function markActiveSession() {
  for (const row of sessionListEl.children) {
    row.classList.toggle("active", row.dataset.id === currentSessionId);
  }
}

async function openSession(id) {
  if (streaming || id === currentSessionId) return;
  try {
    const resp = await fetch(`/api/sessions/${id}`);
    if (!resp.ok) {
      const detail = (await resp.json().catch(() => ({}))).detail;
      throw new Error(detail || `HTTP ${resp.status}`);
    }
    const body = await resp.json();
    messages = (body.messages || []).map(fromStored);
    currentSessionId = body.id;
    // Set rather than left to the scroll listener: .chat is scroll-behavior:smooth,
    // so restoring a long conversation animates and the listener would sample
    // atBottom mid-flight and leave the view stuck at the top.
    atBottom = true;
    renderAll();
    markActiveSession();
    setSidebar(false);
  } catch (err) {
    alert(`打开对话失败：${err.message}`);
    // Most likely a row whose file is gone; refetch so it drops out of the list.
    await refreshSessions();
  }
}

function newConversation() {
  // stop() is fire-and-forget: send()'s finally still runs afterwards, but it
  // saves the captured snapshot to the captured id, so this cannot corrupt it.
  if (streaming) stop();
  // Only an unsaved conversation needs a warning — a persisted one is still in the
  // list, so there is nothing to lose and nothing to confirm.
  if (!currentSessionId && messages.length && !confirm("当前对话还没有保存，确定放弃？")) return;
  messages = [];
  currentSessionId = null;
  // The checkboxes are deliberately left alone: they are global preferences, so a
  // new conversation starts the way the last one was left.
  atBottom = true;
  renderAll();
  markActiveSession();
}

async function renameSession(s) {
  if (streaming) return;
  const answer = prompt("重命名对话", s.title);
  if (answer === null) return;
  const title = answer.trim();
  if (!title || title === s.title) return;
  try {
    // Title only — a rename must not ship the conversation's images back.
    const resp = await fetch(`/api/sessions/${s.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    if (!resp.ok) {
      const detail = (await resp.json().catch(() => ({}))).detail;
      throw new Error(detail || `HTTP ${resp.status}`);
    }
    await refreshSessions();
  } catch (err) {
    alert(`重命名失败：${err.message}`);
  }
}

async function deleteSession(s) {
  if (streaming) return;
  if (!confirm(`删除对话「${s.title}」？删除后无法恢复。`)) return;
  try {
    const resp = await fetch(`/api/sessions/${s.id}`, { method: "DELETE" });
    if (!resp.ok) {
      const detail = (await resp.json().catch(() => ({}))).detail;
      throw new Error(detail || `HTTP ${resp.status}`);
    }
    if (currentSessionId === s.id) {
      messages = [];
      currentSessionId = null;
      atBottom = true;
      renderAll();
    }
    await refreshSessions();
  } catch (err) {
    alert(`删除失败：${err.message}`);
  }
}

function setSidebar(open) {
  sidebarEl.classList.toggle("open", open);
  sidebarToggle.setAttribute("aria-expanded", String(open));
}

newChatBtn.onclick = newConversation;
newBtn.onclick = newConversation;
sidebarToggle.onclick = () => setSidebar(!sidebarEl.classList.contains("open"));
sessionSearch.oninput = renderSessions;
window.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  // First claim, and an early return: closing the dialog must not also clear the
  // session filter or shut the drawer that is sitting behind it.
  if (modalOpen) { closePanel(); return; }
  if (sessionSearch.value) { sessionSearch.value = ""; renderSessions(); }
  else setSidebar(false);
});

/* One-time import of the single conversation the browser used to hold. The old key
   also carried the two checkboxes, so those move to their own key on the way past. */
async function migrateLegacyHistory() {
  let saved;
  try {
    saved = JSON.parse(localStorage.getItem(LEGACY_KEY) || "null");
  } catch {
    saved = null;
  }
  if (!saved || typeof saved !== "object") return;

  if (typeof saved.webSearch === "boolean" || typeof saved.thinking === "boolean") {
    searchToggle.checked = Boolean(saved.webSearch);
    thinkingToggle.checked = Boolean(saved.thinking);
    savePrefs();
  }

  const legacy = Array.isArray(saved.messages) ? saved.messages : [];
  if (!legacy.length) {
    localStorage.removeItem(LEGACY_KEY);
    return;
  }
  try {
    const resp = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // The old store dropped images to stay inside the localStorage quota, so
      // these arrive with the flag and no pixels; toStored preserves that.
      body: JSON.stringify({
        title: "导入的历史对话",
        messages: legacy.map((m) => toStored({
          role: m.role,
          content: m.content || "",
          images: [],
          hadImages: Boolean(m.hadImages),
          reasoning: m.reasoning || "",
          sources: m.sources || [],
          usage: m.usage || "",
          error: Boolean(m.error),
        })),
      }),
    });
    if (!resp.ok) {
      const detail = (await resp.json().catch(() => ({}))).detail;
      throw new Error(detail || `HTTP ${resp.status}`);
    }
    // Removed only once the server has it: a failed import is retried on the next
    // load rather than throwing away the only copy.
    localStorage.removeItem(LEGACY_KEY);
  } catch (err) {
    alert(`导入浏览器里保存的旧对话失败，原数据已保留：${err.message}`);
  }
}

async function bootSessions() {
  await migrateLegacyHistory();
  await refreshSessions();
  // Reopen the most recent conversation, which is what a reload used to do.
  if (sessions.length) await openSession(sessions[0].id);
}

/* ------------------------------------------------------------------ *
 * Settings / About / Help / Export dialog
 * ------------------------------------------------------------------ */

const PANEL_TITLE = { settings: "设置", about: "关于", help: "帮助", export: "导出" };

/* Chinese label for every field Settings has. A field the backend adds without
   this table being updated falls through to its raw name and still renders, so a
   gap is visible rather than silently hiding a setting. */
const FIELD_LABEL = {
  runtime_backend: "推理后端",
  llama_server_url: "外部 llama-server",
  llama_port: "llama-server 端口",
  n_gpu_layers: "GPU 层数",
  // "默认" because models listed in n_ctx_overrides ignore this value; the number
  // that is actually in force is on the 关于 panel.
  n_ctx: "默认上下文长度",
  n_ctx_overrides: "按模型上下文",
  server_extra_args: "llama-server 附加参数",
  server_startup_timeout: "启动超时（秒）",
  model_path: "启动默认模型",
  mmproj_path: "视觉投影文件",
  system_prompt: "系统提示词",
  temperature: "温度",
  top_p: "核采样 top_p",
  max_tokens: "回答 token 上限",
  max_tool_rounds: "工具调用轮数上限",
  enable_thinking: "默认开启深度思考",
  thinking_max_tokens: "思考 token 上限",
  search_provider: "检索源",
  search_max_results: "检索结果条数",
  bocha_api_key: "BOCHA 密钥",
  tavily_api_key: "Tavily 密钥",
  host: "监听地址",
  port: "监听端口",
};

/* Only these eight get a control; everything else renders as text. The bounds
   mirror SettingsPatch in app/settings_store.py — that model is what the server
   enforces, so a wider range here would only buy a round trip that ends in 422. */
const FIELD_WIDGET = {
  system_prompt: { kind: "textarea" },
  temperature: { kind: "number", step: 0.1, min: 0, max: 2 },
  top_p: { kind: "number", step: 0.05, min: 0.01, max: 1 },
  max_tokens: { kind: "number", step: 128, min: 64, max: 32768 },
  thinking_max_tokens: { kind: "number", step: 128, min: 64, max: 32768 },
  max_tool_rounds: { kind: "number", step: 1, min: 0, max: 8 },
  search_provider: { kind: "select", options: ["auto", "bing", "bocha", "tavily"] },
  search_max_results: { kind: "number", step: 1, min: 1, max: 10 },
};

/* Defaults to the settings slot so the eight existing call sites stay as they
   are; the export pane has its own. */
function say(text, kind, el = settingsMsgEl) {
  el.textContent = text;
  el.className = kind ? `panel-msg ${kind}` : "panel-msg";
}

function fieldRow(f) {
  const row = document.createElement("div");
  row.className = "field";
  row.dataset.name = f.name;
  row.classList.toggle("overridden", Boolean(f.overridden));

  const name = document.createElement("div");
  name.className = "field-name";
  const label = document.createElement("div");
  label.textContent = FIELD_LABEL[f.name] || f.name;
  const key = document.createElement("code");
  key.textContent = f.name;
  name.append(label, key);

  const body = document.createElement("div");
  body.className = "field-body";
  const widget = f.editable ? FIELD_WIDGET[f.name] : null;

  if (!widget) {
    const value = document.createElement("div");
    value.className = "field-value";
    value.textContent = String(f.value);
    body.append(value);
  } else if (widget.kind === "textarea") {
    const ta = document.createElement("textarea");
    ta.className = "field-textarea";
    ta.value = f.value;
    body.append(ta);
  } else if (widget.kind === "select") {
    const sel = document.createElement("select");
    sel.className = "field-select";
    for (const opt of widget.options) sel.append(option(opt, opt));
    sel.value = f.value;
    body.append(sel);
  } else {
    const inp = document.createElement("input");
    inp.className = "field-input";
    inp.type = "number";
    inp.step = widget.step;
    inp.min = widget.min;
    inp.max = widget.max;
    inp.value = f.value;
    body.append(inp);
  }

  // Baseline for the diff in collectPatch, so saving one change does not write all
  // eight and stripe the whole panel as overridden.
  if (widget) row.dataset.base = String(f.value);

  const note = document.createElement("div");
  note.className = "field-reason";
  note.textContent = f.editable
    ? (f.overridden ? "已覆盖 .env 中的值" : "")
    : f.reason;
  if (note.textContent) body.append(note);

  row.append(name, body);
  return row;
}

function renderSettings(data) {
  settingsFieldsEl.replaceChildren(...(data.fields || []).map(fieldRow));
  settingsResetBtn.disabled = !(data.overridden || []).length;
  settingsResetBtn.title = (data.overridden || []).length
    ? `删除 runtime/settings_override.json，丢弃已覆盖的 ${data.overridden.length} 项`
    : "当前没有覆盖任何 .env 中的值";
}

async function loadSettings() {
  say("");
  try {
    const resp = await fetch("/api/settings");
    if (!resp.ok) {
      const detail = (await resp.json().catch(() => ({}))).detail;
      throw new Error(detail || `HTTP ${resp.status}`);
    }
    renderSettings(await resp.json());
  } catch (err) {
    settingsFieldsEl.replaceChildren();
    say(`读取设置失败：${err.message}`, "err");
  }
}

/* Returns only the fields whose control differs from what the server sent, plus
   the labels of any control holding an impossible value. */
function collectPatch() {
  const patch = {};
  const bad = [];
  for (const row of settingsFieldsEl.children) {
    const name = row.dataset.name;
    const widget = FIELD_WIDGET[name];
    if (!widget) continue;
    const control = row.querySelector(".field-input, .field-textarea, .field-select");
    if (!control) continue;
    row.classList.remove("invalid");

    let value;
    if (widget.kind === "number") {
      // valueAsNumber, never Number(control.value): a cleared number input reports
      // "", which Number() turns into 0 — and 0 is a perfectly legal temperature,
      // so the server would accept it. This is the only place that can catch it.
      value = control.valueAsNumber;
      if (!Number.isFinite(value) || value < widget.min || value > widget.max) {
        row.classList.add("invalid");
        bad.push(FIELD_LABEL[name] || name);
        continue;
      }
    } else {
      value = control.value;
      if (widget.kind === "textarea" && !value.trim()) {
        row.classList.add("invalid");
        bad.push(FIELD_LABEL[name] || name);
        continue;
      }
    }
    if (String(value) !== row.dataset.base) patch[name] = value;
  }
  return { patch, bad };
}

async function saveSettings() {
  const { patch, bad } = collectPatch();
  if (bad.length) { say(`取值不合法：${bad.join("、")}`, "err"); return; }
  if (!Object.keys(patch).length) { say("没有改动", "warn"); return; }
  settingsSaveBtn.disabled = true;
  try {
    const resp = await fetch("/api/settings", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(body.detail || `HTTP ${resp.status}`);
    renderSettings(body);
    say(
      body.persisted
        ? "已保存，对下一条消息生效"
        : "已生效，但没能写入磁盘，重启后会丢失",
      body.persisted ? "ok" : "warn"
    );
    // search_provider shows in the topbar's meta line, so read it back rather than
    // waiting up to 15s for the next poll.
    refreshStatus();
  } catch (err) {
    say(`保存失败：${err.message}`, "err");
  } finally {
    syncControls();
  }
}

async function resetSettings() {
  if (!confirm("删除 runtime/settings_override.json，全部恢复为 .env 中的值？")) return;
  settingsResetBtn.disabled = true;
  try {
    const resp = await fetch("/api/settings", { method: "DELETE" });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(body.detail || `HTTP ${resp.status}`);
    renderSettings(body);
    say(
      body.persisted ? "已恢复 .env 中的值" : "内存中已恢复，但磁盘上的文件删不掉",
      body.persisted ? "ok" : "warn"
    );
    refreshStatus();
  } catch (err) {
    say(`恢复失败：${err.message}`, "err");
  } finally {
    syncControls();
  }
}

function aboutRow(label, text) {
  const row = document.createElement("div");
  row.className = "field";
  const name = document.createElement("div");
  name.className = "field-name";
  const span = document.createElement("div");
  span.textContent = label;
  name.append(span);
  const body = document.createElement("div");
  body.className = "field-body";
  const value = document.createElement("div");
  value.className = "field-value";
  value.textContent = text;
  body.append(value);
  row.append(name, body);
  return row;
}

/* Refetched on every open rather than cached: session count, model size and the
   runtime state all move while the app runs, and a stale "关于" is worse than
   one extra request. */
async function loadAbout() {
  aboutFieldsEl.replaceChildren();
  let a;
  try {
    const resp = await fetch("/api/about");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    a = await resp.json();
  } catch (err) {
    aboutFieldsEl.replaceChildren(aboutRow("错误", `读取失败：${err.message}`));
    return;
  }
  const r = a.runtime || {};
  const s = a.sessions || {};
  const caps = [];
  if (r.vision) caps.push("图片理解");
  if (r.tools !== false) caps.push("工具调用 / 联网检索");
  const rows = [
    ["运行状态", `${STATE_TEXT[r.state] || r.state || "–"}${r.detail ? ` — ${String(r.detail).split("\n")[0]}` : ""}`],
    ["推理后端", `${r.backend || "–"}${a.external ? "（外部 llama-server，不由本应用启动）" : ""}`],
    ["服务地址", r.base_url || "–"],
    ["llama.cpp 版本", a.build_info || "未知（外部服务未提供）"],
    ["当前模型", r.model || "–"],
    ["模型文件", a.model_size_gb ? `${a.model_path} · ${a.model_size_gb} GB` : a.model_path || "–"],
    ["视觉投影", a.mmproj_path || (r.vision ? "–" : "无（不支持图片理解）")],
    ["上下文长度", `${r.n_ctx ?? "–"} tokens`],
    ["GPU 层数", String(r.n_gpu_layers ?? "–")],
    ["模型能力", caps.join(" · ") || "纯文本对话"],
    ["检索源", r.search_provider || "–"],
    ["历史对话", s.readable === false
      ? "目录不可读"
      : `${s.count ?? 0} 个 · ${fmtSize(s.bytes || 0)} · runtime/history/`],
  ];
  aboutFieldsEl.replaceChildren(...rows.map(([k, v]) => aboutRow(k, v)));
}

function setPanel(name) {
  activePanel = PANEL_TITLE[name] ? name : "settings";
  modalTitle.textContent = PANEL_TITLE[activePanel];
  for (const btn of railBtns) btn.classList.toggle("active", btn.dataset.panel === activePanel);
  for (const id of Object.keys(PANEL_TITLE)) {
    $(`panel-${id}`).classList.toggle("active", id === activePanel);
  }
  if (activePanel === "settings") loadSettings();
  else if (activePanel === "about") loadAbout();
  else if (activePanel === "export") renderExportPane();
}

/* inert on the two containers is the whole focus trap — no hand-written Tab cycle
   — and it also blocks the topbar and the conversation behind the dialog. It does
   not stop the 2s telemetry poll, the 15s status poll or a running stream, which
   is the point: opening the panel costs nothing that is in flight. The trade is
   that 停止 becomes unreachable until the panel is closed, which is why Escape and
   a backdrop click both close it. The dialog is the last thing in <body> for the
   same reason. */
function openPanel(name, trigger) {
  if (modalOpen) { setPanel(name); return; }
  modalOpen = true;
  modalReturnFocus = trigger || null;
  topbarEl.toggleAttribute("inert", true);
  workspaceEl.toggleAttribute("inert", true);
  modalEl.classList.add("active");
  // An open drawer would sit under the backdrop with no way back to it.
  drawerWasOpen = sidebarEl.classList.contains("open");
  if (drawerWasOpen) setSidebar(false);
  setPanel(name);
  modalCloseBtn.focus();
}

function closePanel() {
  if (!modalOpen) return;
  modalOpen = false;
  modalEl.classList.remove("active");
  // inert comes off first: focus() inside an inert subtree is a no-op, so the
  // opposite order drops focus to <body> and the next Tab starts over at the top.
  topbarEl.removeAttribute("inert");
  workspaceEl.removeAttribute("inert");
  if (drawerWasOpen) setSidebar(true);
  (modalReturnFocus || sidebarToggle).focus();
  modalReturnFocus = null;
}

/* ------------------------------------------------------------------ *
 * Export pane
 * ------------------------------------------------------------------ */

/* Surfaced through #export-format-note. This is where renderMarkdown's fidelity
   limits become something the user reads instead of something they hit: no images
   ever reach the markdown output, and a paragraph's line breaks are already
   collapsed to spaces by the time there is HTML to export. */
const FORMAT_NOTE = {
  md: "逐字保存 Markdown 原文。换行要紧时选它——段落里的换行在渲染成 HTML 时就已经合并成空格了。",
  html: "白底网页，可以直接用浏览器打开或打印。图片不会被导出。",
  csv: "只导出正文里的 Markdown 表格，行内语法会剥掉；没有表格时不会产生文件。多张表之间空一行。",
  pdf: "按 A4 分页排好版，适合打印，链接可以点击。由服务端生成，仍然不调用模型。",
  docx: "Word 文档，链接可以点击，可以在 Word 里继续编辑。由服务端生成，仍然不调用模型。",
};

function exportFormat() {
  const checked = exportFormatsEl.querySelector("input[name=export-format]:checked");
  return checked ? checked.value : "md";
}

function renderExportPane() {
  // Rebuilt on every open rather than kept in sync: `messages` grows while an
  // answer streams, and a stale scope list would offer a turn that has moved.
  if (exportTarget && !messages.includes(exportTarget)) exportTarget = null;

  const scopes = [];
  if (messages.length) scopes.push(option("all", `整段对话（${messages.length} 条）`));
  messages.forEach((m, i) => {
    if (!m.content) return;
    // A preview, because "第 7 条 · 助手" is not enough to pick by in a long
    // conversation. textContent via option(), so it cannot inject markup.
    const head = m.content.trim().split("\n")[0].slice(0, 20);
    const who = m.role === "user" ? "你" : "助手";
    scopes.push(option(String(i), `第 ${i + 1} 条 · ${who}${head ? ` · ${head}` : ""}`));
  });
  exportScopeEl.replaceChildren(...scopes);
  exportScopeEl.disabled = !scopes.length;
  exportRunBtn.disabled = !scopes.length;
  exportScopeEl.value = exportTarget ? String(messages.indexOf(exportTarget)) : "all";

  // Reset on every open, the way loadSettings re-reads from the server. Nothing
  // re-renders between editing this and clicking 导出, so an edit is never lost.
  const title = (sessions.find((s) => s.id === currentSessionId) || {}).title || "";
  exportNameEl.value = sanitiseFilename(title) || "对话";
  exportNoteEl.textContent = FORMAT_NOTE[exportFormat()];
  say("", "", exportMsgEl);
  hintIfPromptLacksExport();
}

/* The sentence added to DEFAULT_SYSTEM_PROMPT never reaches a user who saved a
   prompt of their own: precedence is runtime/settings_override.json, then .env,
   then the default. Appending to a file the user hand-edited is exactly the kind
   of change that destroys someone's work, so this stays a read-only hint — and it
   has to say the button still works, because approach A is click-driven and the
   sentence only affects whether the model mentions it. Losing it costs
   discoverability, not function. */
async function hintIfPromptLacksExport() {
  let value = "";
  let overridden = false;
  try {
    const resp = await fetch("/api/settings");
    if (!resp.ok) return;
    const field = ((await resp.json()).fields || []).find((f) => f.name === "system_prompt");
    if (!field) return;
    value = String(field.value || "");
    overridden = Boolean(field.overridden);
  } catch {
    return; // no server behind this page: nothing to warn about
  }
  if (value.includes("导出")) return;
  // The fetch is async, so it can land after the user has already exported.
  if (exportMsgEl.textContent) return;
  const where = overridden ? "你在设置面板里保存过的版本" : ".env 里的 SYSTEM_PROMPT";
  say(`当前生效的系统提示词是${where}，不含「输出 Markdown 并提示导出」这一句，` +
      "模型可能不会主动提到导出。导出按钮本身照常可用。", "warn", exportMsgEl);
}

function openExportFor(msg, trigger) {
  exportTarget = msg;
  openPanel("export", trigger);
}

async function runExport() {
  if (exportTarget && !messages.includes(exportTarget)) {
    say("这条消息已不在当前对话里", "warn", exportMsgEl);
    renderExportPane();
    return;
  }
  const fmt = exportFormat();
  const src = exportSource();
  if (!src.trim()) {
    say("没有可导出的正文", "warn", exportMsgEl);
    return;
  }

  const title = (sessions.find((s) => s.id === currentSessionId) || {}).title || "";
  const suffix = exportTarget ? `-msg${messages.indexOf(exportTarget) + 1}` : "";
  const filename = exportFilename(fmt, suffix);

  exportRunBtn.disabled = true;
  try {
    if (fmt === "md") {
      saveBlob(src, "text/markdown;charset=utf-8", filename);
    } else if (fmt === "html") {
      saveBlob(exportHtmlDocument(src, title || filename), "text/html;charset=utf-8", filename);
    } else if (fmt === "csv") {
      const csv = toCsv(src);
      if (!csv) {
        say("这段内容里没有 Markdown 表格，CSV 没有可导出的内容", "warn", exportMsgEl);
        return;
      }
      saveBlob(CSV_BOM + csv, "text/csv;charset=utf-8", filename);
    } else {
      const resp = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ format: fmt, blocks: blocksFromMarkdown(src), title }),
      });
      if (!resp.ok) {
        const detail = (await resp.json().catch(() => ({}))).detail;
        // typeof guard, unlike the `detail || HTTP n` idiom elsewhere: ExportIn's
        // Field(min_length/max_length) makes Pydantic answer 422 with a LIST, and
        // a list is truthy, so that idiom would show "[object Object]".
        throw new Error(typeof detail === "string" ? detail : `HTTP ${resp.status}`);
      }
      const pages = Number(resp.headers.get("X-Export-Pages") || 0);
      const truncated = resp.headers.get("X-Export-Truncated") === "1";
      saveBlob(await resp.blob(), resp.headers.get("Content-Type") || "", filename);
      // The header is ASCII and the flag is what carries the meaning, because
      // Starlette encodes headers as latin-1 and the warning is Chinese.
      say(truncated
        ? `已导出 ${filename}，内容超过 ${pages} 页上限，只保留了前面部分`
        : `已导出 ${filename}${pages ? `（${pages} 页）` : ""}`, truncated ? "warn" : "ok", exportMsgEl);
      return;
    }
    say(`已导出 ${filename}`, "ok", exportMsgEl);
  } catch (err) {
    say(`导出失败：${err.message}`, "err", exportMsgEl);
  } finally {
    // Not syncControls' business: it runs on a 15s poll and would fight the
    // transient disable. Export is deliberately usable mid-stream.
    exportRunBtn.disabled = !exportScopeEl.options.length;
  }
}

for (const btn of sideLinks) btn.onclick = () => openPanel(btn.dataset.panel, btn);
for (const btn of railBtns) btn.onclick = () => setPanel(btn.dataset.panel);
modalCloseBtn.onclick = closePanel;
modalBackdrop.onclick = closePanel;
settingsSaveBtn.onclick = saveSettings;
settingsResetBtn.onclick = resetSettings;
exportBtn.onclick = () => openExportFor(null, exportBtn);
exportRunBtn.onclick = runExport;
// The <select> is authoritative and exportTarget is derived from it, not the
// other way round: renderExportPane() rebuilds the options on every open, and a
// bare exportTarget would go stale as soon as 新建对话 or deleteSession reassigns
// `messages`. Number(v) because <select> values are strings and messages is an
// array; `|| null` covers a scope that names a turn which has since vanished.
exportScopeEl.onchange = () => {
  const v = exportScopeEl.value;
  exportTarget = v === "all" ? null : (messages[Number(v)] || null);
};
exportFormatsEl.onchange = () => { exportNoteEl.textContent = FORMAT_NOTE[exportFormat()]; };

/* ------------------------------------------------------------------ *
 * Runtime status
 * ------------------------------------------------------------------ */

const STATE_CLASS = { ready: "dot-ready", external: "dot-ready", loading: "dot-loading",
  starting: "dot-loading", switching: "dot-loading", error: "dot-error", stopped: "dot-error" };
const STATE_TEXT = { ready: "运行中", external: "外部服务", loading: "加载模型中",
  starting: "启动中", switching: "切换模型中", error: "不可用", stopped: "已停止" };

/* Every disabled state is derived here and nowhere else — the 15s poll below would
   otherwise revert whatever a one-off handler had set. */
function syncControls() {
  // `!streaming` on the docBusy term: during a stream this button reads 停止, and
  // a parse finishing in the background must not make the answer unstoppable.
  sendBtn.disabled = switching || runtimeState === "error" || (!streaming && docBusy > 0);
  copyBtn.disabled = !messages.length;
  // Only the entry button is derived here. The dialog's own 导出 button belongs to
  // runExport's finally instead: this poll would otherwise revert the transient
  // disable, and export is deliberately usable while a stream is still running.
  exportBtn.disabled = !messages.length;
  // Disabling the select while streaming is what avoids racing with stop(), which
  // is fire-and-forget and leaves `streaming` set until send()'s finally runs.
  modelSelect.disabled = streaming || switching || !canSwitch;
  attachBtn.disabled = !vision;
  attachBtn.title = vision ? "上传图片" : "当前模型不支持图片理解";
  fileInput.disabled = !vision;
  searchToggle.disabled = !tools;
  searchLabel.title = tools ? SEARCH_TITLE : "当前模型不支持工具调用，联网检索不可用";
  // inert, not disabled: disabled on a container still lets its children's click
  // handlers fire, whereas inert blocks both pointer and keyboard. The search box
  // sits outside the list precisely so filtering survives a stream.
  // newChatBtn is left enabled on purpose — it calls stop() first.
  newBtn.disabled = streaming;
  sessionListEl.toggleAttribute("inert", streaming);
  // Only the save button, not the sidebar entries that open the dialog: reading
  // the help text mid-stream costs nothing, whereas saving would really change the
  // turn in flight — llm.py re-reads get_settings() on every tool round.
  settingsSaveBtn.disabled = streaming;
}

function applyVision(next) {
  const had = vision;
  vision = !!next;
  if (had && !vision && pendingImages.length) {
    pendingImages = [];
    renderPreviews();
    alert("当前模型不支持图片理解，已清空待发送的图片");
  }
  syncControls();
}

function applyTools(next) {
  const had = tools;
  tools = next !== false; // a missing field means "unknown" -> allow
  if (!tools && searchToggle.checked) {
    searchToggle.checked = false;
    savePrefs(); // otherwise the next reload restores it ticked
    // A tick restored from localStorage is not something the user just did, so
    // only a real capability change speaks up.
    if (had) alert("当前模型不支持工具调用，已关闭联网检索");
  }
  syncControls();
}

function applyStatus(s) {
  runtimeState = s.state;
  // The optimistic flag set on click outlives this line; a second tab learns
  // about an in-flight switch from here instead.
  switching = switching || s.state === "switching";
  canSwitch = !!s.can_switch;
  stateDot.className = `dot ${STATE_CLASS[s.state] || ""}`;
  modelName.textContent = s.model || "未知模型";
  const bits = [STATE_TEXT[s.state] || s.state, s.backend, `ctx ${s.n_ctx}`];
  if (s.vision) bits.push("图片理解");
  bits.push(s.tools === false ? "检索 不可用" : `检索 ${s.search_provider}`);
  runtimeMeta.textContent = s.state === "error" ? s.detail.split("\n")[0] : bits.join(" · ");
  runtimeMeta.title = s.detail || "";
  applyVision(s.vision);
  applyTools(s.tools);
}

async function refreshStatus() {
  try {
    const resp = await fetch("/api/status");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    applyStatus(await resp.json());
    return runtimeState;
  } catch {
    runtimeState = "error";
    stateDot.className = "dot dot-error";
    runtimeMeta.textContent = "无法连接后端服务";
    syncControls();
    return "error";
  }
}

async function pollUntilReady() {
  const state = await refreshStatus();
  if (state === "loading" || state === "starting" || state === "switching") {
    setTimeout(pollUntilReady, 2000);
  }
}

/* ------------------------------------------------------------------ *
 * System telemetry
 * ------------------------------------------------------------------ */

const STATS_MS = 2000;
// A 24 GB card running the 27B model sits near its ceiling by design, so these
// mark "nothing else can coexist" and "about to fail", not merely "busy".
const VRAM_WARN = 0.9;
const VRAM_ERR = 0.96;

function setStat(el, text, level) {
  el.textContent = text;
  el.className = level || "";
}

function applyStats(s) {
  setStat(stCpu, s.cpu_percent == null ? "–" : `${s.cpu_percent}%`);
  setStat(stRam, s.ram_total_gb == null ? "–" : `${s.ram_used_gb}/${s.ram_total_gb}G`);
  if (!s.gpu_available) {
    setStat(stGpu, "–");
    setStat(stVram, "–");
    setStat(stTemp, "–");
    gpuStatWrap.title = tempStatWrap.title = "未检测到 NVIDIA 驱动，GPU 遥测不可用";
    return;
  }
  gpuStatWrap.title = GPU_STAT_TITLE;
  setStat(stGpu, s.gpu_percent == null ? "–" : `${s.gpu_percent}%`);
  const fill = s.vram_total_gb ? s.vram_used_gb / s.vram_total_gb : 0;
  setStat(
    stVram,
    s.vram_total_gb == null ? "–" : `${s.vram_used_gb}/${s.vram_total_gb}G`,
    fill >= VRAM_ERR ? "err" : fill >= VRAM_WARN ? "warn" : ""
  );
  setStat(stTemp, s.gpu_temp_c == null ? "–" : `${s.gpu_temp_c}°C`);
  tempStatWrap.title = s.gpu_watts == null
    ? "GPU 芯片温度"
    : `GPU 芯片温度 · 功耗 ${s.gpu_watts} W`;
}

async function refreshStats() {
  try {
    const resp = await fetch("/api/stats");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    applyStats(await resp.json());
  } catch {
    // Keep the last good numbers: one missed poll says nothing, and a dead
    // backend is already reported by the status dot.
  }
}

function pollStats() {
  // A background tab would otherwise poll ~43k times a day for numbers nobody
  // is looking at.
  if (document.visibilityState === "visible") refreshStats();
}

/* ------------------------------------------------------------------ *
 * Model picker
 * ------------------------------------------------------------------ */

function fmtSize(bytes) {
  if (!bytes) return "?";
  const units = ["B", "KB", "MB", "GB"];
  let n = bytes;
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i >= 3 ? 1 : 0)} ${units[i]}`;
}

function option(value, text, title) {
  const el = document.createElement("option");
  el.value = value;
  el.textContent = text;
  if (title) el.title = title;
  return el;
}

async function loadModels() {
  try {
    const resp = await fetch("/api/models");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    canSwitch = !!data.can_switch;
    currentModelId = data.active || "";
    const known = (data.models || []).some((m) => m.id === data.active);

    const opts = [];
    // .env may point outside the scanned directory; name that model rather than
    // leaving the dropdown silently empty.
    if (!known && data.active) opts.push(option("", `${data.active}（不在模型目录）`));
    for (const m of data.models || []) {
      opts.push(option(
        m.id,
        `${m.name} · ${fmtSize(m.size)}${m.vision ? " · 图片" : ""}`,
        m.vision ? "支持图片理解" : "纯文本模型",
      ));
    }
    modelSelect.replaceChildren(...opts);
    modelSelect.value = known ? data.active : "";
    modelSelect.title = canSwitch
      ? `切换本地模型（${data.dir}），会重启 llama-server，当前对话保留`
      : "外部 llama-server 模式下无法切换模型";
  } catch {
    canSwitch = false;
    modelSelect.replaceChildren(option("", "模型列表不可用"));
  }
  syncControls();
}

async function switchModel(id) {
  const prev = currentModelId;
  if (!id || id === prev) { modelSelect.value = prev || ""; return; }
  if (!confirm(`切换到 ${id}？\nllama-server 会重启，加载大模型可能需要一两分钟。\n当前对话会保留。`)) {
    modelSelect.value = prev || "";
    return;
  }

  switching = true;
  syncControls();
  stateDot.className = "dot dot-loading";
  runtimeMeta.textContent = "切换模型中…";
  // Started before the await so it runs concurrently with the pending POST: the
  // topbar shows live llama-server progress (hover the meta line) while it loads.
  pollUntilReady();

  try {
    const resp = await fetch("/api/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    const body = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(body.detail || `HTTP ${resp.status}`);
    currentModelId = id;
    applyStatus(body);
  } catch (err) {
    alert(`切换失败：${err.message}`);
    modelSelect.value = currentModelId;
  } finally {
    switching = false;
    syncControls();
    refreshStatus();
    // Re-reads `active`, so the select snaps back by itself after a rollback.
    loadModels();
  }
}

modelSelect.onchange = () => switchModel(modelSelect.value);

loadPrefs();
renderAll();
autoGrow();
pollUntilReady();
loadModels();
bootSessions();
setInterval(refreshStatus, 15000);
refreshStats();
setInterval(pollStats, STATS_MS);
