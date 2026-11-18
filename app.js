// Óculos Assistivo — protótipo no navegador.
//
// Como funciona:
//  1. SENSOR DE RÉ: o mapa de profundidade (Depth Anything) mede o quanto
//     o que está à frente está perto. Isso vira bipes: mais rápidos e agudos
//     quanto mais perto, puxados para o lado (fone estéreo) onde está o obstáculo.
//     Vale para qualquer coisa: parede, poste, porta, móvel.
//  2. VOZ SÓ PARA O CRÍTICO: veículo no caminho, pessoa colada à frente e
//     obstáculo na altura da cabeça. Cada aviso é falado UMA vez; só repete se
//     aquilo sumir por um tempo e voltar.
//  3. DESCRIÇÃO SOB DEMANDA: botão "O que tem na minha frente?" (ou toque no vídeo)
//     manda uma foto para o Claude, que responde com uma frase curta.

// ---------- Elementos da página ----------
const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");
const depthView = $("depthView");
const startBtn = $("startBtn");
const stopBtn = $("stopBtn");
const muteBtn = $("muteBtn");
const describeBtn = $("describeBtn");
const descriptionBox = $("description");
const sonarBox = $("sonar");
const logBox = $("log");
const statusBox = $("status");
const depthStatus = $("depthStatus");
const apiKeyInput = $("apiKeyInput");

// ---------- Estado ----------
let detector = null;   // { name, gpu, kind, detect(): Promise<[{cls, score, bbox:[x,y,w,h]}]> }
let running = false;
let muted = false;
let wakeLock = null;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function log(text) {
  const time = new Date().toLocaleTimeString();
  logBox.textContent = `[${time}] ${text}\n` + logBox.textContent;
}

// ---------- Bibliotecas ----------
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
const ANTHROPIC_SDK_URL = "https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@1.3.0/+esm";

let transformersPromise = null;
function loadTransformers() {
  transformersPromise ??= import(TRANSFORMERS_URL).then((tf) => {
    // Com a página isolada (coi-sw.js), a IA no processador pode usar vários núcleos.
    const wasm = tf.env.backends?.onnx?.wasm;
    if (wasm && self.crossOriginIsolated) {
      wasm.numThreads = Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));
    }
    return tf;
  });
  return transformersPromise;
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error("não carregou " + src));
    document.head.appendChild(s);
  });
}

// =====================================================================
// FILA DE FALA — uma frase por vez; urgente interrompe; nada repetido.
// =====================================================================
const PRI_INFO = 1;
const PRI_ALERT = 2;
const PRI_URGENT = 3;  // interrompe a fala atual
const PRI_SYSTEM = 4;  // mensagens do app (fala mesmo silenciado)

const MAX_QUEUE_AGE = 2500; // ms: frase de alerta esperando mais que isso é descartada
const MAX_QUEUE = 2;

const speaker = {
  queue: [],
  current: null,
  safetyTimer: null,

  get busy() { return !!this.current; },

  say(text, { priority = PRI_INFO, key = text, keepFresh = false } = {}) {
    if (!("speechSynthesis" in window)) return false;
    if (muted && priority < PRI_SYSTEM) return false;
    const item = { text, priority, key, keepFresh, time: Date.now() };

    if (this.current && this.current.key === key) return false;

    if (this.current && priority >= PRI_URGENT && priority > this.current.priority) {
      this.queue = this.queue.filter((q) => q.priority >= priority && q.key !== key);
      this.queue.unshift(item);
      this.interrupt();
      return true;
    }

    this.queue = this.queue.filter((q) => q.key !== key);
    this.queue.push(item);
    this.queue.sort((a, b) => b.priority - a.priority || a.time - b.time);
    if (this.queue.length > MAX_QUEUE) this.queue.length = MAX_QUEUE;
    if (!this.current) this.next();
    return true;
  },

  next() {
    const now = Date.now();
    this.queue = this.queue.filter(
      (q) => q.keepFresh || q.priority >= PRI_SYSTEM || now - q.time <= MAX_QUEUE_AGE
    );
    const item = this.queue.shift();
    if (!item) return;

    const u = new SpeechSynthesisUtterance(item.text);
    u.lang = "pt-BR";
    u.rate = 1.1;
    item.utterance = u; // manter referência evita que o Chrome perca o onend
    this.current = item;

    const done = () => {
      if (this.current !== item) return;
      clearTimeout(this.safetyTimer);
      this.current = null;
      setTimeout(() => this.next(), 150);
    };
    u.onend = done;
    u.onerror = done;
    // Se o navegador nunca avisar o fim, libera a fila mesmo assim.
    this.safetyTimer = setTimeout(done, 1500 + item.text.length * 95);

    if (item.priority >= PRI_URGENT && navigator.vibrate) navigator.vibrate([200, 80, 200]);
    window.speechSynthesis.speak(u);
  },

  interrupt() {
    clearTimeout(this.safetyTimer);
    this.current = null;
    window.speechSynthesis.cancel();
    setTimeout(() => { if (!this.current) this.next(); }, 80);
  },

  clear() {
    this.queue = [];
    clearTimeout(this.safetyTimer);
    this.current = null;
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  },
};

// =====================================================================
// AVISOS FALADOS UMA VEZ SÓ
// Cada aviso tem uma chave (ex.: "car"). Depois de falado, fica "anunciado"
// enquanto continuar sendo visto. Só pode ser falado de novo se sumir por
// FORGET_MS e depois voltar.
// =====================================================================
const FORGET_MS = 10000;
const announced = {}; // chave -> último momento em que foi visto

function announceOnce(key, text, priority, now) {
  if (announced[key] !== undefined) {
    announced[key] = now; // continua presente: só renova, não fala
    return;
  }
  announced[key] = now;
  const spoken = speaker.say(text, { priority, key });
  log(muted ? `${text} (silenciado)` : spoken ? text : `${text} (não falado: já falando)`);
}

function forgetOld(now) {
  for (const key of Object.keys(announced)) {
    if (now - announced[key] > FORGET_MS) delete announced[key];
  }
}

// =====================================================================
// SENSOR DE RÉ (bipes)
// =====================================================================
const SONAR_LEVEL_AT = [0.65, 0.8, 0.95];   // proximidade relativa para os níveis 1, 2, 3
const SONAR_INTERVAL = [0, 750, 380, 160];  // ms entre bipes por nível
const SONAR_FREQ = [0, 440, 620, 880];      // Hz por nível
const SONAR_STALE_MS = 2500;                // sem profundidade recente: silencia
const SONAR_SMOOTH = 0.6;                   // suavização (0 = sem, 1 = congelado)

const sonar = {
  level: 0, side: 1, time: 0,     // side: 0 esquerda, 1 frente, 2 direita
  cols: [0, 0, 0],                // proximidade suavizada por coluna
  audio: null, timer: null, nextBeep: 0, lastVibe: 0,

  start() {
    try {
      this.audio ??= new (window.AudioContext || window.webkitAudioContext)();
      this.audio.resume();
    } catch (err) {
      log("Áudio indisponível: " + err.message);
    }
    clearInterval(this.timer);
    this.timer = setInterval(() => this.tick(), 40);
  },

  stop() {
    clearInterval(this.timer);
    this.timer = null;
    this.level = 0;
    this.cols = [0, 0, 0];
    sonarBox.textContent = "";
  },

  tone(freq, pan, volume, duration = 0.07, delay = 0) {
    const a = this.audio;
    if (!a) return;
    const t = a.currentTime + delay;
    const osc = a.createOscillator();
    const gain = a.createGain();
    osc.type = "sine";
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(volume, t + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + duration);
    let node = osc.connect(gain);
    if (a.createStereoPanner) {
      const p = a.createStereoPanner();
      p.pan.value = pan;
      node = node.connect(p);
    }
    node.connect(a.destination);
    osc.start(t);
    osc.stop(t + duration + 0.02);
  },

  tick() {
    if (!running || muted) return;
    const level = Date.now() - this.time > SONAR_STALE_MS ? 0 : this.level;
    if (!level) return;
    const now = performance.now();
    if (now < this.nextBeep) return;
    const pan = [-0.8, 0, 0.8][this.side];
    // Mais baixo enquanto a voz fala, para não atrapalhar.
    this.tone(SONAR_FREQ[level], pan, speaker.busy ? 0.08 : 0.3);
    this.nextBeep = now + SONAR_INTERVAL[level];
    if (level === 3 && navigator.vibrate && now - this.lastVibe > 700) {
      navigator.vibrate(120);
      this.lastVibe = now;
    }
  },

  update(closeness) {
    this.cols = this.cols.map((old, i) => SONAR_SMOOTH * old + (1 - SONAR_SMOOTH) * closeness[i]);
    const lv = this.cols.map((c) => SONAR_LEVEL_AT.filter((t) => c >= t).length);
    // Laterais contam um nível a menos (corredor estreito não deve apitar sem parar).
    const eff = [Math.max(0, lv[0] - 1), lv[1], Math.max(0, lv[2] - 1)];
    const best = Math.max(...eff);
    this.level = best;
    this.side = eff[1] === best ? 1 : eff[0] >= eff[2] ? 0 : 2;
    this.time = Date.now();
    const bar = (n) => "▮".repeat(n) + "▯".repeat(3 - n);
    sonarBox.textContent = `Esq ${bar(eff[0])}   Frente ${bar(eff[1])}   Dir ${bar(eff[2])}`;
  },
};

// =====================================================================
// PROFUNDIDADE
// =====================================================================
const DEPTH_MODEL = "onnx-community/depth-anything-v2-small";
const DEPTH_INPUT_WIDTH = 320;
const DEPTH_SIZE = 196;
const HEAD_THRESHOLD = 0.9;   // topo da imagem tão perto quanto o chão aos pés = altura da cabeça
const HEAD_CONFIRM = 2;

let depthPipeline = null;
let depthOnGpu = false;
let RawImage = null;
let lastDepth = null;   // valores 0–255; 255 = ponto mais perto da cena
let depthTime = 0;
let depthMs = 0;
let headStreak = 0;
const depthCanvas = document.createElement("canvas");

const DEVICES = navigator.gpu
  ? [{ device: "webgpu", dtype: "fp16", name: "GPU" }, { device: "wasm", dtype: "q8", name: "CPU" }]
  : [{ device: "wasm", dtype: "q8", name: "CPU" }];

// Percentil dos valores de profundidade numa região (frações da imagem).
function regionPercentile(depth, x0, x1, y0, y1, pct) {
  const { data, width, height } = depth;
  const values = [];
  const xa = Math.max(0, Math.floor(x0 * width)), xb = Math.min(width, Math.floor(x1 * width));
  const ya = Math.max(0, Math.floor(y0 * height)), yb = Math.min(height, Math.floor(y1 * height));
  for (let y = ya; y < yb; y += 2)
    for (let x = xa; x < xb; x += 2) values.push(data[y * width + x]);
  if (!values.length) return 0;
  values.sort((a, b) => a - b);
  return values[Math.min(values.length - 1, Math.floor(pct * values.length))];
}

// Referência: o chão logo à frente dos pés (parte de baixo, no meio).
// Algo tão perto quanto esse chão, na altura do corpo, é obstáculo.
function analyzeDepth(depth, now) {
  const ref = Math.max(regionPercentile(depth, 0.3, 0.7, 0.88, 1, 0.5), 30);
  const closeness = [[0, 0.33], [0.33, 0.67], [0.67, 1]].map(
    ([x0, x1]) => regionPercentile(depth, x0, x1, 0.1, 0.75, 0.85) / ref
  );
  sonar.update(closeness);

  const top = regionPercentile(depth, 0.35, 0.65, 0, 0.3, 0.7);
  headStreak = top >= HEAD_THRESHOLD * ref ? headStreak + 1 : 0;
  if (headStreak >= HEAD_CONFIRM) {
    announceOnce("head", "Cuidado! Obstáculo na altura da cabeça.", PRI_URGENT, now);
  }

  lastDepth = depth;
  depthTime = now;
}

function drawDepth(depth) {
  const { data, width, height } = depth;
  depthView.width = width;
  depthView.height = height;
  const dctx = depthView.getContext("2d");
  const img = dctx.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    const v = data[i];
    img.data[4 * i] = v;
    img.data[4 * i + 1] = 0.6 * v;
    img.data[4 * i + 2] = 255 - v;
    img.data[4 * i + 3] = 255;
  }
  dctx.putImageData(img, 0, 0);
}

async function runDepth() {
  const started = performance.now();
  depthCanvas.width = DEPTH_INPUT_WIDTH;
  depthCanvas.height = Math.round((video.videoHeight * DEPTH_INPUT_WIDTH) / video.videoWidth);
  depthCanvas.getContext("2d").drawImage(video, 0, 0, depthCanvas.width, depthCanvas.height);
  const { depth } = await depthPipeline(RawImage.fromCanvas(depthCanvas));
  if (!running) return;
  analyzeDepth(depth, Date.now());
  drawDepth(depth);
  depthMs = performance.now() - started;
}

// Quão perto está o centro da caixa de um objeto (0 longe … 1 ponto mais perto). null = sem dado.
function boxCloseness(bbox, now) {
  if (!lastDepth || now - depthTime > SONAR_STALE_MS) return null;
  const [x, y, w, h] = bbox;
  const W = video.videoWidth, H = video.videoHeight;
  return regionPercentile(lastDepth, (x + 0.2 * w) / W, (x + 0.8 * w) / W, (y + 0.2 * h) / H, (y + 0.8 * h) / H, 0.6) / 255;
}

// =====================================================================
// OBJETOS (YOLO) — usados só para os avisos críticos
// =====================================================================
const YOLO_MODEL = "onnx-community/yolov10n";
const YOLO_SIZE = 640;              // o modelo só aceita 640x640
const YOLO_INTERVAL_CPU = 1500;     // sem placa de vídeo, o YOLO roda a cada 1,5 s

const SCORES = {
  yolo: { normal: 0.5, vehicle: 0.6, vehicleIndoor: 0.85 },
  coco: { normal: 0.6, vehicle: 0.75, vehicleIndoor: 0.9 },
};
const NEAR_RATIO = 0.35;      // altura do objeto / altura da imagem
const NEAR_BOTTOM = 0.9;      // ou pé do objeto quase na base da imagem
const VERY_NEAR_RATIO = 0.6;
const FAR_DEPTH = 0.35;       // profundidade baixa = longe, ignora
const CONFIRM_HITS = 3;
const CONFIRM_HITS_VEHICLE = 4;
const MISS_TOLERANCE = 2;
const INDOOR_HOLD = 15000;

const LABELS = {
  person: "pessoa", car: "carro", bicycle: "bicicleta", motorcycle: "moto",
  bus: "ônibus", truck: "caminhão", chair: "cadeira", bench: "banco",
  dog: "cachorro", cat: "gato", "fire hydrant": "hidrante",
  "stop sign": "placa de pare", "potted plant": "vaso de planta",
  "traffic light": "semáforo", "dining table": "mesa", couch: "sofá",
  bed: "cama", refrigerator: "geladeira", suitcase: "mala", toilet: "vaso sanitário",
};
const VEHICLES = new Set(["car", "motorcycle", "bus", "truck", "bicycle"]);
const BOXY_VEHICLES = new Set(["car", "bus", "truck"]);
const INDOOR_CLUES = new Set([
  "couch", "bed", "tv", "refrigerator", "dining table", "toilet", "sink", "oven",
  "microwave", "laptop", "keyboard", "mouse", "remote", "book", "clock", "vase",
  "toaster", "hair drier", "toothbrush",
]);

const tracks = {};   // chave -> { hits, misses }
let indoorUntil = 0;
let yoloMs = 0;

const YOLO_DEVICES = navigator.gpu
  ? [{ device: "webgpu", dtype: "fp32", name: "GPU" }, { device: "wasm", dtype: "uint8", name: "CPU" }]
  : [{ device: "wasm", dtype: "uint8", name: "CPU" }];

async function createYoloDetector() {
  const tf = await loadTransformers();
  RawImage ??= tf.RawImage;

  // Preparo manual da imagem: reduz para caber em 640 e completa com cinza.
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = YOLO_SIZE;
  const cctx = canvas.getContext("2d", { willReadFrequently: true });
  const N = YOLO_SIZE * YOLO_SIZE;
  const input = new Float32Array(3 * N);
  const toTensor = (source, w, h) => {
    const scale = YOLO_SIZE / Math.max(w, h);
    cctx.fillStyle = "#727272";
    cctx.fillRect(0, 0, YOLO_SIZE, YOLO_SIZE);
    cctx.drawImage(source, 0, 0, Math.round(w * scale), Math.round(h * scale));
    const px = cctx.getImageData(0, 0, YOLO_SIZE, YOLO_SIZE).data;
    for (let i = 0; i < N; i++) {
      input[i] = px[4 * i] / 255;
      input[N + i] = px[4 * i + 1] / 255;
      input[2 * N + i] = px[4 * i + 2] / 255;
    }
    return { tensor: new tf.Tensor("float32", input, [1, 3, YOLO_SIZE, YOLO_SIZE]), scale };
  };

  let lastErr = null;
  for (const opt of YOLO_DEVICES) {
    try {
      statusBox.textContent = `Baixando modelo de objetos (${opt.name}, só na primeira vez)...`;
      const model = await tf.AutoModel.from_pretrained(YOLO_MODEL, { device: opt.device, dtype: opt.dtype });
      const labels = model.config.id2label;
      const detect = async () => {
        const { tensor, scale } = toTensor(video, video.videoWidth, video.videoHeight);
        const { output0 } = await model({ images: tensor });
        const out = [];
        for (const [x0, y0, x1, y1, score, id] of output0.tolist()[0]) {
          if (score < 0.3) continue;
          out.push({ cls: labels[id], score, bbox: [x0 / scale, y0 / scale, (x1 - x0) / scale, (y1 - y0) / scale] });
        }
        return out;
      };
      // Primeira execução (na GPU prepara os programas; alguns aparelhos falham aqui).
      statusBox.textContent = `Preparando modelo de objetos (${opt.name})...`;
      const blank = document.createElement("canvas");
      blank.width = blank.height = 64;
      await model({ images: toTensor(blank, 64, 64).tensor });
      return { name: `YOLOv10n (${opt.name})`, kind: "yolo", gpu: opt.device === "webgpu", detect };
    } catch (err) {
      lastErr = err;
      console.warn("YOLO falhou em", opt.device, err);
    }
  }
  throw lastErr || new Error("YOLO indisponível");
}

async function createCocoDetector() {
  await loadScript("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js");
  await loadScript("https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js");
  const model = await cocoSsd.load({ base: "mobilenet_v2" });
  return {
    name: "COCO-SSD (reserva)", kind: "coco", gpu: false,
    detect: async () =>
      (await model.detect(video, 20, 0.3)).map((p) => ({ cls: p.class, score: p.score, bbox: p.bbox })),
  };
}

function corridorFor(w, h) {
  return { xLeftTop: 0.35 * w, xRightTop: 0.65 * w, yTop: 0.4 * h, xLeftBottom: 0.05 * w, xRightBottom: 0.95 * w, yBottom: h };
}

function insideCorridor(x, y, c) {
  if (y < c.yTop || y > c.yBottom) return false;
  const t = (y - c.yTop) / (c.yBottom - c.yTop);
  const left = c.xLeftTop + t * (c.xLeftBottom - c.xLeftTop);
  const right = c.xRightTop + t * (c.xRightBottom - c.xRightTop);
  return x >= left && x <= right;
}

function drawCorridor(c) {
  ctx.beginPath();
  ctx.moveTo(c.xLeftTop, c.yTop);
  ctx.lineTo(c.xRightTop, c.yTop);
  ctx.lineTo(c.xRightBottom, c.yBottom);
  ctx.lineTo(c.xLeftBottom, c.yBottom);
  ctx.closePath();
  ctx.strokeStyle = "rgba(0, 255, 150, 0.7)";
  ctx.lineWidth = 2;
  ctx.stroke();
}

function directionText(d) {
  return d === "frente" ? "à frente" : "à " + d;
}

// Decide se um objeto confirmado merece aviso falado. Retorna { text, priority } ou null.
function spokenAlertFor(c) {
  const name = c.label.charAt(0).toUpperCase() + c.label.slice(1);
  const dist = c.veryNear ? "muito perto" : "perto";
  if (VEHICLES.has(c.cls)) {
    const urgent = c.direction === "frente" || c.veryNear;
    return { text: `${urgent ? "Cuidado! " : ""}${name} ${directionText(c.direction)}, ${dist}.`, priority: urgent ? PRI_URGENT : PRI_ALERT };
  }
  if (c.cls === "person") {
    return c.veryNear && c.direction === "frente" ? { text: "Pessoa à frente, muito perto.", priority: PRI_ALERT } : null;
  }
  // Sem profundidade não há sensor de ré: aí os demais objetos também são falados.
  if (!depthPipeline) return { text: `${name} ${directionText(c.direction)}, ${dist}.`, priority: PRI_INFO };
  return null; // o sensor de ré já cobre móveis e afins
}

function processDetections(detections, now) {
  const W = video.videoWidth, H = video.videoHeight;
  if (detections.some((d) => INDOOR_CLUES.has(d.cls) && d.score >= 0.5)) indoorUntil = now + INDOOR_HOLD;
  const indoor = now < indoorUntil;
  const s = SCORES[detector.kind];

  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const corridor = corridorFor(W, H);
  drawCorridor(corridor);

  const seen = new Set();
  const confirmed = [];

  for (const d of detections) {
    if (!LABELS[d.cls]) continue;
    const vehicle = VEHICLES.has(d.cls);
    if (d.score < (vehicle ? (indoor ? s.vehicleIndoor : s.vehicle) : s.normal)) continue;

    const [x, y, w, h] = d.bbox;
    const cx = x + w / 2, bottom = y + h, ratio = h / H;
    const closeness = boxCloseness(d.bbox, now);
    const inPath = insideCorridor(cx, bottom, corridor);
    const near = (ratio > NEAR_RATIO || bottom / H > NEAR_BOTTOM) && !(closeness !== null && closeness < FAR_DEPTH);
    // Contra "carro" falso: alto e estreito não é carro; dentro de casa sem profundidade, não arrisca.
    const plausible = !vehicle || !((BOXY_VEHICLES.has(d.cls) && w / h < 0.6) || (indoor && closeness === null));

    const color = !plausible ? "#888888" : inPath && near ? "#ff3b3b" : inPath ? "#ffd23b" : "#3bd6ff";
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = color;
    ctx.font = "16px sans-serif";
    ctx.fillText(`${LABELS[d.cls]} ${(100 * d.score).toFixed(0)}%${closeness === null ? "" : " · " + (100 * closeness).toFixed(0)}`, x, y > 20 ? y - 6 : y + 16);

    if (!(inPath && near && plausible) || seen.has(d.cls)) continue;
    seen.add(d.cls);
    const t = (tracks[d.cls] ??= { hits: 0, misses: 0 });
    t.hits++;
    t.misses = 0;
    if (t.hits >= (vehicle ? CONFIRM_HITS_VEHICLE : CONFIRM_HITS)) {
      const direction = cx < 0.4 * W ? "esquerda" : cx > 0.6 * W ? "direita" : "frente";
      const veryNear = ratio >= VERY_NEAR_RATIO || (closeness !== null && closeness >= 0.9);
      confirmed.push({ cls: d.cls, label: LABELS[d.cls], direction, veryNear });
    }
  }
  for (const key of Object.keys(tracks)) {
    if (!seen.has(key) && ++tracks[key].misses > MISS_TOLERANCE) delete tracks[key];
  }

  // Mais importante primeiro (veículos), e cada um falado uma vez só.
  confirmed.sort((a, b) => VEHICLES.has(b.cls) - VEHICLES.has(a.cls));
  for (const c of confirmed) {
    const alert = spokenAlertFor(c);
    if (alert) announceOnce(c.cls, alert.text, alert.priority, now);
    else if (announced[c.cls] !== undefined) announced[c.cls] = now;
  }
}

// =====================================================================
// LAÇO PRINCIPAL — profundidade e objetos em sequência (sem disputar a máquina)
// =====================================================================
async function mainLoop() {
  let lastYolo = 0;
  let lastStatus = 0;
  while (running) {
    if (!video.videoWidth) {
      await sleep(100);
      continue;
    }
    try {
      if (depthPipeline) await runDepth();
      const yoloDue = detector && (detector.gpu || !depthPipeline || performance.now() - lastYolo >= YOLO_INTERVAL_CPU);
      if (running && yoloDue) {
        const t0 = performance.now();
        lastYolo = t0;
        const detections = await detector.detect();
        yoloMs = performance.now() - t0;
        if (running) processDetections(detections, Date.now());
      }
    } catch (err) {
      statusBox.textContent = "Erro na análise: " + err.message;
      await sleep(500);
    }
    forgetOld(Date.now());

    if (performance.now() - lastStatus > 1500) {
      lastStatus = performance.now();
      const cores = self.crossOriginIsolated ? "vários núcleos" : "1 núcleo";
      const indoor = Date.now() < indoorUntil ? "interno" : "externo";
      statusBox.textContent = `${detector.name}: ${(yoloMs / 1000).toFixed(2)} s · ${cores} · ambiente ${indoor}`;
      if (depthPipeline) depthStatus.textContent = `Sensor de ré (${depthOnGpu ? "GPU" : "CPU"}): ${(depthMs / 1000).toFixed(2)} s por leitura`;
    }
    await sleep(15);
  }
}

// =====================================================================
// DESCRIÇÃO DA CENA (Claude, sob demanda)
// =====================================================================
const KEY_STORAGE = "oculos.anthropicKey";
const DESCRIBE_MODEL = "claude-opus-5-5";
const DESCRIBE_SYSTEM =
  "Você é os olhos de uma pessoa cega que está andando com a câmera do celular apontada para a frente, " +
  "na altura do peito. Descreva a cena em português do Brasil em no máximo duas frases curtas, que serão lidas em voz alta. " +
  "Comece pelo que importa para andar com segurança: obstáculos, degraus, escadas, buracos, portas, pessoas e veículos, " +
  "dizendo a posição (à esquerda, à frente, à direita) e a distância aproximada em passos. " +
  "Se houver texto útil visível (placa, aviso, número, nome de loja), leia. " +
  "Não use markdown, listas nem frases de introdução. Se a imagem estiver escura ou borrada, diga isso em uma frase.";

let anthropicSdk = null;
let describing = false;

function getKey() {
  try { return localStorage.getItem(KEY_STORAGE) || ""; } catch { return ""; }
}

function captureJpegBase64(maxSide = 1024) {
  const scale = Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight));
  const c = document.createElement("canvas");
  c.width = Math.round(video.videoWidth * scale);
  c.height = Math.round(video.videoHeight * scale);
  c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.75).split(",")[1];
}

async function describeScene() {
  if (describing) return;
  if (!running || !video.videoWidth) {
    speaker.say("Toque em Iniciar primeiro.", { priority: PRI_SYSTEM });
    return;
  }
  const apiKey = getKey();
  if (!apiKey) {
    speaker.say("Configure a chave da IA primeiro.", { priority: PRI_SYSTEM });
    $("aiSettings").open = true;
    return;
  }

  describing = true;
  describeBtn.disabled = true;
  descriptionBox.textContent = "Analisando...";
  sonar.tone(660, 0, 0.25, 0.09);
  sonar.tone(990, 0, 0.25, 0.09, 0.12); // sinal sonoro de "pedido recebido"
  const started = performance.now();

  try {
    anthropicSdk ??= (await import(ANTHROPIC_SDK_URL)).default;
    const Anthropic = anthropicSdk;
    const client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true });
    const response = await client.beta.messages.create({
      model: DESCRIBE_MODEL,
      max_tokens: 2000,
      output_config: { effort: "low" },
      betas: ["server-side-fallback-2026-07-01"],
      fallbacks: "default",
      system: DESCRIBE_SYSTEM,
      messages: [{
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: captureJpegBase64() } },
          { type: "text", text: "O que tem na minha frente?" },
        ],
      }],
    });

    let text;
    if (response.stop_reason === "refusal") {
      text = "Não consegui descrever esta cena.";
    } else {
      text = response.content.filter((b) => b.type === "text").map((b) => b.text).join(" ").trim()
        || "Não consegui descrever esta cena.";
    }
    descriptionBox.textContent = text;
    log(`Descrição (${((performance.now() - started) / 1000).toFixed(1)} s): ${text}`);
    speaker.say(text, { priority: PRI_ALERT, key: "describe", keepFresh: true });
  } catch (err) {
    const A = anthropicSdk;
    let msg = "Erro ao descrever: " + err.message;
    if (A && err instanceof A.AuthenticationError) msg = "A chave da IA é inválida.";
    else if (A && err instanceof A.RateLimitError) msg = "Muitos pedidos seguidos. Tente de novo em instantes.";
    else if (A && err instanceof A.APIConnectionError) msg = "Sem conexão com a internet.";
    else if (A && err instanceof A.APIError) msg = `Erro da IA (${err.status}).`;
    descriptionBox.textContent = msg;
    log(msg + (err.message ? ` [${err.message}]` : ""));
    speaker.say(msg, { priority: PRI_SYSTEM });
  } finally {
    describing = false;
    describeBtn.disabled = !running;
  }
}

$("saveKeyBtn").addEventListener("click", () => {
  const key = apiKeyInput.value.trim();
  if (!key) return;
  try { localStorage.setItem(KEY_STORAGE, key); } catch {}
  apiKeyInput.value = "";
  $("aiSettings").open = false;
  log("Chave da IA salva neste aparelho.");
});
$("clearKeyBtn").addEventListener("click", () => {
  try { localStorage.removeItem(KEY_STORAGE); } catch {}
  log("Chave da IA apagada.");
});

// =====================================================================
// CÂMERA E CONTROLES
// =====================================================================
async function requestWakeLock() {
  try {
    if ("wakeLock" in navigator) wakeLock = await navigator.wakeLock.request("screen");
  } catch (err) {
    log("Não foi possível manter a tela acesa: " + err.message);
  }
}

document.addEventListener("visibilitychange", () => {
  if (running && document.visibilityState === "visible") requestWakeLock();
});

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  running = true;
  sonar.start();
  mainLoop();
}

function stopAll() {
  running = false;
  speaker.clear();
  sonar.stop();
  const stream = video.srcObject;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (wakeLock) wakeLock.release().catch(() => {});
  wakeLock = null;
  for (const k of Object.keys(tracks)) delete tracks[k];
  for (const k of Object.keys(announced)) delete announced[k];
  lastDepth = null;
  headStreak = 0;
  indoorUntil = 0;
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  sonar.start(); // o áudio precisa ser liberado dentro do toque do usuário
  speaker.say("Iniciando.", { priority: PRI_SYSTEM });
  requestWakeLock();
  try {
    await startCamera();
    statusBox.textContent = `Detectando com ${detector.name}...`;
    stopBtn.disabled = false;
    describeBtn.disabled = false;
  } catch (err) {
    statusBox.textContent = "Erro ao acessar a câmera: " + err.message;
    startBtn.disabled = false;
  }
});

stopBtn.addEventListener("click", () => {
  stopAll();
  stopBtn.disabled = true;
  describeBtn.disabled = true;
  startBtn.disabled = false;
  statusBox.textContent = "Parado. Toque em Iniciar para voltar.";
});

muteBtn.addEventListener("click", () => {
  muted = !muted;
  if (muted) speaker.clear();
  muteBtn.textContent = muted ? "Ativar som" : "Silenciar";
  muteBtn.setAttribute("aria-pressed", String(muted));
});

describeBtn.addEventListener("click", describeScene);
$("video").addEventListener("click", describeScene); // tocar no vídeo também descreve
document.addEventListener("keydown", (e) => {
  if (e.key === "d" && document.activeElement !== apiKeyInput) describeScene();
});

// =====================================================================
// CARREGAMENTO DOS MODELOS
// =====================================================================
async function loadDepth() {
  let tf;
  try {
    tf = await loadTransformers();
  } catch (err) {
    depthStatus.textContent = "Sensor de ré: não carregou (" + err.message + ")";
    return;
  }
  RawImage ??= tf.RawImage;
  for (const opt of DEVICES) {
    try {
      depthStatus.textContent = `Sensor de ré: baixando modelo (${opt.name}, só na primeira vez)...`;
      const pipe = await tf.pipeline("depth-estimation", DEPTH_MODEL, { device: opt.device, dtype: opt.dtype });
      pipe.processor.image_processor.size = { height: DEPTH_SIZE, width: DEPTH_SIZE };
      depthOnGpu = opt.device === "webgpu";
      depthPipeline = pipe;
      depthStatus.textContent = `Sensor de ré: pronto (${opt.name})`;
      return;
    } catch (err) {
      console.warn("Profundidade falhou em", opt.device, err);
    }
  }
  depthStatus.textContent = "Sensor de ré indisponível neste aparelho. Objetos serão falados.";
}

(async function init() {
  statusBox.textContent = "Carregando modelo de detecção...";
  try {
    detector = await createYoloDetector();
  } catch (err) {
    log("YOLO não carregou (" + err.message + "). Usando COCO-SSD.");
    try {
      detector = await createCocoDetector();
    } catch (err2) {
      statusBox.textContent = "Não foi possível carregar a detecção: " + err2.message;
      return;
    }
  }
  loadDepth(); // depois do detector, para não disputarem o download
  statusBox.textContent = `Modelo carregado (${detector.name}). Toque em Iniciar.`;
  startBtn.disabled = false;
})();
