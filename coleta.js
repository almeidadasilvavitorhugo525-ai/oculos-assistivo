// =====================================================================
// MODO COLETA: guarda fotos da câmera para treinar a IA depois.
// - Foto automática a cada AUTO_MS (sem rótulo: marcar depois).
// - Botões / voz ("degrau", "buraco"...): guarda a foto daquele instante com o rótulo.
// - Tudo fica no próprio aparelho (IndexedDB). "Exportar" baixa um .zip.
// Usa video, running, sonar, log e $ do app.js.
// =====================================================================
const COLLECT_LABELS = [
  { id: "degrau", text: "Degrau / escada", words: ["degrau", "escada", "degraus"] },
  { id: "buraco", text: "Buraco", words: ["buraco"] },
  { id: "meio-fio", text: "Meio-fio", words: ["meio-fio", "meio fio", "guia", "calçada"] },
  { id: "poste", text: "Poste", words: ["poste", "árvore", "pilar"] },
  { id: "porta", text: "Porta", words: ["porta"] },
  { id: "pessoa", text: "Pessoa", words: ["pessoa", "gente"] },
  { id: "carro", text: "Carro", words: ["carro", "ônibus", "caminhão"] },
  { id: "moto", text: "Moto", words: ["moto", "motocicleta", "bicicleta"] },
  { id: "livre", text: "Caminho livre", words: ["livre", "nada"] },
];
const AUTO_MS = 2000;
const COLLECT_SIDE = 960; // lado maior da foto guardada
const JSZIP_URL = "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js";

const collect = {
  on: false,
  auto: true,
  timer: null,
  recog: null,
  listening: false,
  db: null,
  count: 0,
  counts: {},
};

// ---------- banco de fotos (IndexedDB) ----------
function collectDb() {
  if (collect.db) return Promise.resolve(collect.db);
  return new Promise((resolve, reject) => {
    const req = indexedDB.open("oculos-coleta", 1);
    req.onupgradeneeded = () => req.result.createObjectStore("fotos", { keyPath: "id", autoIncrement: true });
    req.onsuccess = () => { collect.db = req.result; resolve(req.result); };
    req.onerror = () => reject(req.error);
  });
}

function dbAll(mode, fn) {
  return collectDb().then((db) => new Promise((resolve, reject) => {
    const tx = db.transaction("fotos", mode);
    const result = fn(tx.objectStore("fotos"));
    tx.oncomplete = () => resolve(result && "result" in result ? result.result : undefined);
    tx.onerror = () => reject(tx.error);
  }));
}

async function refreshCounts() {
  try {
    const items = await dbAll("readonly", (s) => s.getAll());
    collect.count = items.length;
    collect.counts = {};
    for (const it of items) collect.counts[it.label] = (collect.counts[it.label] || 0) + 1;
  } catch (err) {
    log("Coleta: não consegui ler as fotos (" + err.message + ")");
  }
  renderCounts();
}

function renderCounts() {
  $("collectCount").textContent = `${collect.count} fotos guardadas`;
  for (const l of [...COLLECT_LABELS, { id: "sem-rotulo" }]) {
    const el = document.querySelector(`[data-count="${l.id}"]`);
    if (el) el.textContent = collect.counts[l.id] || 0;
  }
}

// ---------- captura ----------
function grabFrame() {
  return new Promise((resolve) => {
    if (!running || !video.videoWidth) return resolve(null);
    const scale = Math.min(1, COLLECT_SIDE / Math.max(video.videoWidth, video.videoHeight));
    const c = document.createElement("canvas");
    c.width = Math.round(video.videoWidth * scale);
    c.height = Math.round(video.videoHeight * scale);
    c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
    c.toBlob(resolve, "image/jpeg", 0.85);
  });
}

async function savePhoto(label, source) {
  const blob = await grabFrame();
  if (!blob) return;
  try {
    await dbAll("readwrite", (s) => s.add({ time: Date.now(), label, source, blob }));
  } catch (err) {
    log("Coleta: erro ao guardar (" + err.message + "). Memória cheia?");
    return;
  }
  collect.count++;
  collect.counts[label] = (collect.counts[label] || 0) + 1;
  renderCounts();
  if (label !== "sem-rotulo") {
    // confirmação curta, sem fala: um bipe e uma vibração
    sonar.tone(1200, 0, 0.2, 0.06);
    if (navigator.vibrate) navigator.vibrate(40);
    log(`Coleta: foto "${label}" guardada (${source}).`);
  }
}

// ---------- voz ----------
function labelFromSpeech(text) {
  const t = text.toLowerCase();
  for (const l of COLLECT_LABELS) if (l.words.some((w) => t.includes(w))) return l.id;
  return null;
}

function startVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    $("collectVoice").textContent = "Voz: não disponível neste navegador";
    return;
  }
  const r = new SR();
  r.lang = "pt-BR";
  r.continuous = true;
  r.interimResults = false;
  r.onresult = (e) => {
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (!e.results[i].isFinal) continue;
      const said = e.results[i][0].transcript;
      const label = labelFromSpeech(said);
      if (label) savePhoto(label, "voz");
      else log(`Coleta: ouvi "${said.trim()}" (não é um rótulo).`);
    }
  };
  r.onerror = (e) => {
    if (e.error === "not-allowed" || e.error === "service-not-allowed") {
      collect.listening = false;
      $("collectVoice").textContent = "Voz: microfone não permitido";
    }
  };
  r.onend = () => { if (collect.on && collect.listening) try { r.start(); } catch {} }; // o Chrome para sozinho de tempos em tempos
  collect.recog = r;
  collect.listening = true;
  try { r.start(); } catch {}
  $("collectVoice").textContent = "Voz: ouvindo (diga \"degrau\", \"buraco\", \"poste\"...)";
}

function stopVoice() {
  collect.listening = false;
  if (collect.recog) try { collect.recog.stop(); } catch {}
  collect.recog = null;
}

// ---------- liga / desliga ----------
function setAuto(on) {
  collect.auto = on;
  clearInterval(collect.timer);
  collect.timer = null;
  if (on && collect.on) collect.timer = setInterval(() => savePhoto("sem-rotulo", "automática"), AUTO_MS);
  $("collectAutoBtn").textContent = on ? "Foto automática: LIGADA" : "Foto automática: DESLIGADA";
  $("collectAutoBtn").setAttribute("aria-pressed", String(on));
}

function setCollect(on) {
  if (on && !running) {
    speaker.say("Toque em Iniciar primeiro.", { priority: PRI_SYSTEM });
    return;
  }
  collect.on = on;
  $("collectPanel").hidden = !on;
  $("collectToggle").textContent = on ? "Sair do modo coleta" : "Modo coleta (fotos para treinar)";
  $("collectToggle").setAttribute("aria-pressed", String(on));
  if (on) {
    if (navigator.storage && navigator.storage.persist) navigator.storage.persist().catch(() => {});
    setAuto(collect.auto);
    startVoice();
    speaker.say("Modo coleta ligado.", { priority: PRI_SYSTEM });
  } else {
    setAuto(collect.auto); // só para o timer
    stopVoice();
  }
}

// ---------- exportar / apagar ----------
async function exportZip() {
  const btn = $("collectExportBtn");
  btn.disabled = true;
  btn.textContent = "Preparando...";
  try {
    if (!window.JSZip) await loadScript(JSZIP_URL);
    const items = await dbAll("readonly", (s) => s.getAll());
    if (!items.length) { log("Coleta: nenhuma foto para exportar."); return; }
    const zip = new JSZip();
    const rows = ["arquivo,rotulo,origem,data"];
    for (const it of items) {
      const stamp = new Date(it.time).toISOString().replace(/[:.]/g, "-");
      const name = `${it.label}/${stamp}_${it.id}.jpg`;
      zip.file(name, it.blob);
      rows.push(`${name},${it.label},${it.source},${new Date(it.time).toISOString()}`);
    }
    zip.file("rotulos.csv", rows.join("\n"));
    const blob = await zip.generateAsync({ type: "blob" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `oculos-coleta-${new Date().toISOString().slice(0, 10)}.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
    log(`Coleta: exportadas ${items.length} fotos (${(blob.size / 1e6).toFixed(1)} MB).`);
  } catch (err) {
    log("Coleta: erro ao exportar (" + err.message + ")");
  } finally {
    btn.disabled = false;
    btn.textContent = "Exportar fotos (.zip)";
  }
}

let clearArmed = null;
async function clearPhotos() {
  const btn = $("collectClearBtn");
  if (!clearArmed) { // pede um segundo toque em vez de janela de confirmação
    btn.textContent = "Toque de novo para APAGAR tudo";
    clearArmed = setTimeout(() => { clearArmed = null; btn.textContent = "Apagar fotos"; }, 4000);
    return;
  }
  clearTimeout(clearArmed);
  clearArmed = null;
  btn.textContent = "Apagar fotos";
  await dbAll("readwrite", (s) => s.clear());
  await refreshCounts();
  log("Coleta: fotos apagadas.");
}

// ---------- montagem dos botões ----------
(function setupCollect() {
  const grid = $("collectLabels");
  for (const l of COLLECT_LABELS) {
    const b = document.createElement("button");
    b.className = "label-btn";
    b.innerHTML = `${l.text} <span data-count="${l.id}">0</span>`;
    b.addEventListener("click", () => savePhoto(l.id, "botão"));
    grid.appendChild(b);
  }
  $("collectToggle").addEventListener("click", () => setCollect(!collect.on));
  $("collectAutoBtn").addEventListener("click", () => setAuto(!collect.auto));
  $("collectExportBtn").addEventListener("click", exportZip);
  $("collectClearBtn").addEventListener("click", clearPhotos);
  stopBtn.addEventListener("click", () => { if (collect.on) setCollect(false); });
  refreshCounts();
})();
