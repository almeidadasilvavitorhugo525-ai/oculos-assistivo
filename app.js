// Óculos Assistivo — protótipo no navegador.
// Câmera + YOLOv10n (objetos) + Depth Anything (profundidade) + alertas por voz.
// Se o YOLO não carregar no aparelho, cai para o COCO-SSD (mais fraco).

// ---------- Elementos da página ----------
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const depthView = document.getElementById("depthView");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const muteBtn = document.getElementById("muteBtn");
const logBox = document.getElementById("log");
const statusBox = document.getElementById("status");
const depthStatus = document.getElementById("depthStatus");

// ---------- Estado ----------
let detector = null;   // { name, detect(): Promise<[{cls, score, bbox:[x,y,w,h]}]> } (bbox em px do vídeo)
let running = false;
let muted = false;
let wakeLock = null;

// ---------- Bibliotecas ----------
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
let transformersPromise = null;
function loadTransformers() {
  transformersPromise ??= import(TRANSFORMERS_URL);
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

// Ordem de preferência para rodar os modelos: placa de vídeo (WebGPU), senão processador.
const DEVICES = navigator.gpu
  ? [{ device: "webgpu", dtype: "fp16", name: "GPU" }, { device: "wasm", dtype: "fp32", name: "CPU" }]
  : [{ device: "wasm", dtype: "fp32", name: "CPU" }];

// ---------- Ajustes da detecção de objetos ----------
const YOLO_MODEL = "onnx-community/yolov10n";
const YOLO_SIZE = 640;  // o modelo só aceita 640x640

// Confiança mínima por detector (o YOLO dá notas mais "honestas" que o COCO-SSD).
const SCORES = {
  yolo: { normal: 0.5, vehicle: 0.6, vehicleIndoor: 0.85 },
  coco: { normal: 0.6, vehicle: 0.75, vehicleIndoor: 0.9 },
};

const NEAR_RATIO = 0.35;         // altura do objeto / altura da imagem para contar como "perto"
const NEAR_BOTTOM = 0.9;         // ou: pé do objeto quase na base da imagem (está junto de você)
const VERY_NEAR_RATIO = 0.6;     // acima disso: "muito perto"
const FAR_DEPTH = 0.35;          // profundidade no objeto < 35% do ponto mais perto = longe, não alerta
const VERY_NEAR_DEPTH = 0.9;     // profundidade quase igual ao ponto mais perto = muito perto
const CONFIRM_FRAMES = 3;        // quadros em que precisa aparecer antes de alertar
const CONFIRM_FRAMES_VEHICLE = 4;
const MISS_TOLERANCE = 2;        // quadros que pode "sumir" sem perder a contagem
const GLOBAL_COOLDOWN = 4000;    // intervalo mínimo entre alertas comuns (ms)
const REPEAT_COOLDOWN = 12000;   // intervalo para repetir o mesmo alerta (ms)
const GROWTH_RATIO = 0.15;       // se o objeto cresceu isso, avisa de novo (está se aproximando)
const INDOOR_HOLD = 15000;       // por quanto tempo vale a última pista de "ambiente interno" (ms)

const tracks = {};       // chave -> { hits, misses }
const lastAlerts = {};   // chave -> { time, ratio }
let lastAlertTime = 0;
let indoorUntil = 0;

const LABELS = {
  person: "pessoa", car: "carro", bicycle: "bicicleta", motorcycle: "moto",
  bus: "ônibus", truck: "caminhão", chair: "cadeira", bench: "banco",
  dog: "cachorro", cat: "gato", "fire hydrant": "hidrante",
  "stop sign": "placa de pare", "potted plant": "vaso de planta",
  "traffic light": "semáforo", "dining table": "mesa", couch: "sofá",
  bed: "cama", refrigerator: "geladeira", suitcase: "mala", toilet: "vaso sanitário",
};
const VEHICLES = new Set(["car", "motorcycle", "bus", "truck", "bicycle"]);
const BOXY_VEHICLES = new Set(["car", "bus", "truck"]); // nunca são bem mais altos que largos
const PRIORITY = { car: 3, motorcycle: 3, bus: 3, truck: 3, bicycle: 3, obstacle: 2, head: 2 };

// Objetos que indicam ambiente interno (casa, escritório). Lá, "carro" quase sempre é engano.
const INDOOR_CLUES = new Set([
  "couch", "bed", "tv", "refrigerator", "dining table", "toilet", "sink", "oven",
  "microwave", "laptop", "keyboard", "mouse", "remote", "book", "clock", "vase",
  "toaster", "hair drier", "toothbrush",
]);

// ---------- Ajustes da profundidade ----------
const DEPTH_MODEL = "onnx-community/depth-anything-v2-small";
const DEPTH_INPUT_WIDTH = 320;
const DEPTH_SIZE = 196;
const OBSTACLE_THRESHOLD = 0.95; // meio da imagem tão perto quanto o chão = obstáculo
const HEAD_THRESHOLD = 0.9;      // topo da imagem tão perto quanto o chão = altura da cabeça
const DEPTH_CONFIRM = 2;         // análises seguidas antes de alertar
const DEPTH_FRESH_MS = 1500;     // validade de uma análise de profundidade

let depthPipeline = null;
let RawImage = null;
const depthStreak = { obstacle: 0, head: 0 };
let depthHazards = [];
let depthTime = 0;
let lastDepth = null;    // último mapa de profundidade (valores 0–255, 255 = ponto mais perto)
const depthCanvas = document.createElement("canvas");

// =====================================================================
// FILA DE FALA
// Uma frase por vez. Alerta urgente interrompe o que estiver falando.
// Frases antigas na fila são descartadas (a situação já mudou).
// Não confia em speechSynthesis.speaking (no Chrome do Android ela erra).
// =====================================================================
const PRI_INFO = 1;    // aviso comum
const PRI_ALERT = 2;   // alerta importante
const PRI_URGENT = 3;  // perigo: interrompe a fala atual
const PRI_SYSTEM = 4;  // mensagens do app (fala mesmo silenciado)

const MAX_QUEUE_AGE = 2500; // ms: frase esperando mais que isso é descartada
const MAX_QUEUE = 2;

const speaker = {
  queue: [],
  current: null,     // { text, priority, key, utterance }
  safetyTimer: null,

  say(text, { priority = PRI_INFO, key = text } = {}) {
    if (!("speechSynthesis" in window)) return false;
    if (muted && priority < PRI_SYSTEM) return false;

    const item = { text, priority, key, time: Date.now() };

    // Já está falando exatamente isso: não repete.
    if (this.current && this.current.key === key) return false;

    // Urgente interrompe a fala atual se ela for menos importante.
    if (this.current && priority >= PRI_URGENT && priority > this.current.priority) {
      this.queue = this.queue.filter((q) => q.priority >= priority && q.key !== key);
      this.queue.unshift(item);
      this.interrupt();
      return true;
    }

    // Substitui frase da mesma chave que ainda esperava na fila.
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
      (q) => q.priority >= PRI_SYSTEM || now - q.time <= MAX_QUEUE_AGE
    );
    const item = this.queue.shift();
    if (!item) return;

    const u = new SpeechSynthesisUtterance(item.text);
    u.lang = "pt-BR";
    u.rate = 1.15;
    item.utterance = u; // manter referência evita que o Chrome perca o onend
    this.current = item;

    const done = () => {
      if (this.current !== item) return;
      clearTimeout(this.safetyTimer);
      this.current = null;
      setTimeout(() => this.next(), 150); // pequena pausa entre frases
    };
    u.onend = done;
    u.onerror = done;

    // Segurança: se o navegador nunca avisar o fim, libera a fila mesmo assim.
    const estimate = 1200 + item.text.length * 90;
    this.safetyTimer = setTimeout(done, estimate);

    if (item.priority >= PRI_URGENT && navigator.vibrate) navigator.vibrate([200, 80, 200]);
    window.speechSynthesis.speak(u);
  },

  interrupt() {
    clearTimeout(this.safetyTimer);
    this.current = null;
    window.speechSynthesis.cancel();
    // O Chrome às vezes ignora speak() logo depois de cancel(); espera um pouco.
    setTimeout(() => { if (!this.current) this.next(); }, 80);
  },

  clear() {
    this.queue = [];
    clearTimeout(this.safetyTimer);
    this.current = null;
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  },
};

// ---------- Frases ----------
function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function directionText(direction) {
  return direction === "frente" ? "à frente" : "à " + direction;
}

// Formato fixo e curto: "Objeto, direção, distância."
function buildAlert(c) {
  if (c.cls === "head") {
    return { text: "Cuidado! Obstáculo na altura da cabeça.", priority: PRI_URGENT };
  }
  if (c.cls === "obstacle") {
    return { text: "Obstáculo à frente, perto.", priority: PRI_ALERT };
  }
  const distance = c.veryNear ? "muito perto" : "perto";
  if (VEHICLES.has(c.cls)) {
    const urgent = c.direction === "frente" || c.veryNear;
    return {
      text: `${urgent ? "Cuidado! " : ""}${capitalize(c.label)} ${directionText(c.direction)}, ${distance}.`,
      priority: urgent ? PRI_URGENT : PRI_ALERT,
    };
  }
  return {
    text: `${capitalize(c.label)} ${directionText(c.direction)}, ${distance}.`,
    priority: c.veryNear ? PRI_ALERT : PRI_INFO,
  };
}

function log(text) {
  const time = new Date().toLocaleTimeString();
  logBox.textContent = `[${time}] ${text}\n` + logBox.textContent;
}

// ---------- Profundidade ----------
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

// No Depth Anything, valor maior = mais perto.
function analyzeDepth(depth) {
  const floor = regionPercentile(depth, 0.35, 0.65, 0.85, 1, 0.9);
  const middle = regionPercentile(depth, 0.35, 0.65, 0.3, 0.7, 0.7);
  const top = regionPercentile(depth, 0.35, 0.65, 0, 0.3, 0.7);
  const ref = Math.max(floor, 1);
  const hazards = [];

  depthStreak.obstacle = middle >= OBSTACLE_THRESHOLD * ref ? depthStreak.obstacle + 1 : 0;
  depthStreak.head = top >= HEAD_THRESHOLD * ref ? depthStreak.head + 1 : 0;

  if (depthStreak.obstacle >= DEPTH_CONFIRM)
    hazards.push({ key: "obstacle-frente", cls: "obstacle", label: "obstáculo", direction: "frente", ratio: middle / ref, confirmed: true });
  if (depthStreak.head >= DEPTH_CONFIRM)
    hazards.push({ key: "head-frente", cls: "head", label: "obstáculo na altura da cabeça", direction: "frente", ratio: top / ref, confirmed: true });

  depthHazards = hazards;
  lastDepth = depth;
  depthTime = Date.now();
}

// Quão perto está a parte central da caixa do objeto, de 0 (longe) a 1 (ponto mais perto da cena).
// Retorna null se não houver profundidade recente.
function boxCloseness(bbox, now) {
  if (!lastDepth || now - depthTime > DEPTH_FRESH_MS) return null;
  const [x, y, w, h] = bbox;
  const W = video.videoWidth, H = video.videoHeight;
  const x0 = (x + 0.2 * w) / W, x1 = (x + 0.8 * w) / W;
  const y0 = (y + 0.2 * h) / H, y1 = (y + 0.8 * h) / H;
  return regionPercentile(lastDepth, x0, x1, y0, y1, 0.6) / 255;
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

async function depthLoop() {
  while (running && depthPipeline) {
    if (!video.videoWidth) {
      await new Promise((r) => setTimeout(r, 200));
      continue;
    }
    const started = performance.now();
    depthCanvas.width = DEPTH_INPUT_WIDTH;
    depthCanvas.height = Math.round((video.videoHeight * DEPTH_INPUT_WIDTH) / video.videoWidth);
    depthCanvas.getContext("2d").drawImage(video, 0, 0, depthCanvas.width, depthCanvas.height);
    try {
      const { depth } = await depthPipeline(RawImage.fromCanvas(depthCanvas));
      if (!running) break;
      analyzeDepth(depth);
      drawDepth(depth);
      depthStatus.textContent = `Profundidade: ${((performance.now() - started) / 1000).toFixed(1)} s por análise`;
    } catch (err) {
      depthStatus.textContent = "Profundidade: erro (" + err.message + ")";
      break;
    }
    await new Promise((r) => setTimeout(r, 50));
  }
}

// ---------- Detectores de objetos ----------
// Medido num PC: na GPU, fp32 foi ~3x mais rápido que fp16. Na CPU, uint8 é o mais rápido.
const YOLO_DEVICES = navigator.gpu
  ? [{ device: "webgpu", dtype: "fp32", name: "GPU" }, { device: "wasm", dtype: "uint8", name: "CPU" }]
  : [{ device: "wasm", dtype: "uint8", name: "CPU" }];

async function createYoloDetector() {
  const tf = await loadTransformers();
  RawImage ??= tf.RawImage;

  // Preparo da imagem feito à mão (bem mais rápido que o AutoProcessor):
  // reduz para caber em 640, encosta no canto superior esquerdo e completa com cinza.
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
          out.push({
            cls: labels[id],
            score,
            bbox: [x0 / scale, y0 / scale, (x1 - x0) / scale, (y1 - y0) / scale],
          });
        }
        return out;
      };

      // Primeira execução: na GPU ela demora (prepara os programas da placa) e
      // alguns aparelhos aceitam WebGPU mas falham aqui. Melhor descobrir agora.
      statusBox.textContent = `Preparando modelo de objetos (${opt.name})...`;
      const blank = document.createElement("canvas");
      blank.width = blank.height = 64;
      await model({ images: toTensor(blank, 64, 64).tensor });

      return { name: `YOLOv10n (${opt.name})`, kind: "yolo", detect };
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
  // "mobilenet_v2" é mais lento que o "lite" padrão, mas erra bem menos.
  const model = await cocoSsd.load({ base: "mobilenet_v2" });
  return {
    name: "COCO-SSD (reserva)",
    kind: "coco",
    detect: async () =>
      (await model.detect(video, 20, 0.3)).map((p) => ({ cls: p.class, score: p.score, bbox: p.bbox })),
  };
}

// ---------- Corredor de caminhada ----------
// Trapézio na parte de baixo da imagem.
function corridorFor(w, h) {
  return {
    xLeftTop: 0.35 * w, xRightTop: 0.65 * w, yTop: 0.4 * h,
    xLeftBottom: 0.05 * w, xRightBottom: 0.95 * w, yBottom: h,
  };
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

// ---------- Decisão do alerta ----------
function minScore(cls, indoor) {
  const s = SCORES[detector.kind];
  if (!VEHICLES.has(cls)) return s.normal;
  return indoor ? s.vehicleIndoor : s.vehicle;
}

// Filtros contra alarme falso de veículo.
function vehicleLooksReal(cls, bbox, indoor, closeness) {
  const [, , w, h] = bbox;
  // Carro/ônibus/caminhão bem mais alto que largo: quase sempre é porta, armário etc.
  if (BOXY_VEHICLES.has(cls) && w / h < 0.6) return false;
  // Dentro de casa sem mapa de profundidade para conferir a distância: não arrisca alertar veículo.
  if (indoor && closeness === null) return false;
  return true;
}

// Escolhe o alerta mais importante que ainda não foi dito recentemente.
function pickAlert(candidates, now) {
  candidates.sort(
    (a, b) => (PRIORITY[b.cls] || 1) - (PRIORITY[a.cls] || 1) || b.ratio - a.ratio
  );
  for (const c of candidates) {
    if (!c.confirmed) {
      const need = VEHICLES.has(c.cls) ? CONFIRM_FRAMES_VEHICLE : CONFIRM_FRAMES;
      if ((tracks[c.key]?.hits || 0) < need) continue;
    }
    const last = lastAlerts[c.key];
    const grew = last && c.ratio - last.ratio >= GROWTH_RATIO;
    if (!last || now - last.time >= REPEAT_COOLDOWN || grew) return c;
  }
  return null;
}

let frameCount = 0;
let fpsStart = performance.now();

async function detectLoop() {
  while (running) {
    if (!video.videoWidth) {
      await new Promise((r) => setTimeout(r, 100));
      continue;
    }
    let detections;
    try {
      detections = await detector.detect();
    } catch (err) {
      statusBox.textContent = "Erro na detecção: " + err.message;
      await new Promise((r) => setTimeout(r, 500));
      continue;
    }
    if (!running) return;
    processDetections(detections);

    frameCount++;
    const elapsed = performance.now() - fpsStart;
    if (elapsed >= 2000) {
      const indoor = Date.now() < indoorUntil;
      statusBox.textContent = `${detector.name} · ${(frameCount * 1000 / elapsed).toFixed(1)} quadros/s · ambiente ${indoor ? "interno" : "externo"}`;
      frameCount = 0;
      fpsStart = performance.now();
    }
    // Pausa curta (não espera a tela redesenhar: em algumas janelas isso trava em ~1 s).
    await new Promise((r) => setTimeout(r, 15));
  }
}

function processDetections(detections) {
  const now = Date.now();
  const W = video.videoWidth, H = video.videoHeight;

  // Pista de ambiente interno.
  if (detections.some((d) => INDOOR_CLUES.has(d.cls) && d.score >= 0.5)) indoorUntil = now + INDOOR_HOLD;
  const indoor = now < indoorUntil;

  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const corridor = corridorFor(W, H);
  drawCorridor(corridor);

  const candidates = [];
  const seen = new Set();

  for (const d of detections) {
    if (!LABELS[d.cls]) continue;
    if (d.score < minScore(d.cls, indoor)) continue;

    const [x, y, w, h] = d.bbox;
    const cx = x + w / 2;
    const bottom = y + h;
    const ratio = h / H;
    const closeness = boxCloseness(d.bbox, now);
    const inPath = insideCorridor(cx, bottom, corridor);
    const depthSaysFar = closeness !== null && closeness < FAR_DEPTH;
    const near = (ratio > NEAR_RATIO || bottom / H > NEAR_BOTTOM) && !depthSaysFar;
    const plausible = !VEHICLES.has(d.cls) || vehicleLooksReal(d.cls, d.bbox, indoor, closeness);

    const color = !plausible ? "#888888" : inPath && near ? "#ff3b3b" : inPath ? "#ffd23b" : "#3bd6ff";
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = color;
    ctx.font = "16px sans-serif";
    const dist = closeness === null ? "" : ` · ${(100 * closeness).toFixed(0)}`;
    ctx.fillText(`${LABELS[d.cls]} ${(100 * d.score).toFixed(0)}%${dist}`, x, y > 20 ? y - 6 : y + 16);

    if (inPath && near && plausible) {
      const direction = cx < 0.4 * W ? "esquerda" : cx > 0.6 * W ? "direita" : "frente";
      const key = `${d.cls}-${direction}`;
      if (!seen.has(key)) {
        seen.add(key);
        const t = (tracks[key] ??= { hits: 0, misses: 0 });
        t.hits++;
        t.misses = 0;
        const veryNear = ratio >= VERY_NEAR_RATIO || (closeness !== null && closeness >= VERY_NEAR_DEPTH);
        candidates.push({ key, cls: d.cls, label: LABELS[d.cls], direction, ratio, veryNear });
      }
    }
  }
  // Objeto que sumiu: tolera alguns quadros antes de zerar (a detecção pisca).
  for (const key of Object.keys(tracks)) {
    if (seen.has(key)) continue;
    if (++tracks[key].misses > MISS_TOLERANCE) delete tracks[key];
  }

  // Junta os perigos vindos da profundidade (se a análise for recente).
  if (now - depthTime <= DEPTH_FRESH_MS) {
    const objectInFront = candidates.some((c) => c.direction === "frente");
    for (const hz of depthHazards) {
      // Se já há um objeto identificado à frente, não fala "obstáculo" genérico.
      if (hz.cls === "obstacle" && objectInFront) continue;
      candidates.push(hz);
    }
    if (depthHazards.length) {
      ctx.fillStyle = "rgba(255, 59, 59, 0.18)";
      ctx.fillRect(0.35 * W, 0, 0.3 * W, 0.7 * H);
    }
  }

  const chosen = pickAlert(candidates, now);
  if (chosen) {
    const alert = buildAlert(chosen);
    // Alertas urgentes não esperam o intervalo global.
    if (alert.priority >= PRI_URGENT || now - lastAlertTime >= GLOBAL_COOLDOWN) {
      lastAlertTime = now;
      lastAlerts[chosen.key] = { time: now, ratio: chosen.ratio };
      const spoken = speaker.say(alert.text, { priority: alert.priority, key: chosen.key });
      log(muted ? `${alert.text} (silenciado)` : spoken ? alert.text : `${alert.text} (já falando)`);
    }
  }
}

// ---------- Câmera e controles ----------
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
    video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;
  running = true;
  frameCount = 0;
  fpsStart = performance.now();
  detectLoop();
  if (depthPipeline) depthLoop();
}

function stopAll() {
  running = false;
  speaker.clear();
  const stream = video.srcObject;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (wakeLock) wakeLock.release().catch(() => {});
  wakeLock = null;
  for (const key of Object.keys(tracks)) delete tracks[key];
  depthHazards = [];
  lastDepth = null;
  depthStreak.obstacle = 0;
  depthStreak.head = 0;
  indoorUntil = 0;
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  speaker.say("Iniciando detecção", { priority: PRI_SYSTEM });
  requestWakeLock();
  try {
    await startCamera();
    statusBox.textContent = `Detectando com ${detector.name}...`;
    stopBtn.disabled = false;
  } catch (err) {
    statusBox.textContent = "Erro ao acessar a câmera: " + err.message;
    startBtn.disabled = false;
  }
});

stopBtn.addEventListener("click", () => {
  stopAll();
  stopBtn.disabled = true;
  startBtn.disabled = false;
  statusBox.textContent = "Parado. Toque em Iniciar para voltar.";
});

muteBtn.addEventListener("click", () => {
  muted = !muted;
  if (muted) speaker.clear();
  muteBtn.textContent = muted ? "Ativar som" : "Silenciar";
  muteBtn.setAttribute("aria-pressed", String(muted));
});

// ---------- Carregamento dos modelos ----------
async function loadDepth() {
  let tf;
  try {
    tf = await loadTransformers();
  } catch (err) {
    depthStatus.textContent = "Profundidade: não carregou (" + err.message + ")";
    return;
  }
  RawImage ??= tf.RawImage;
  for (const opt of DEVICES) {
    try {
      depthStatus.textContent = `Profundidade: baixando modelo (${opt.name}, só na primeira vez)...`;
      depthPipeline = await tf.pipeline("depth-estimation", DEPTH_MODEL, {
        device: opt.device,
        dtype: opt.device === "wasm" ? "q8" : opt.dtype,
      });
      depthPipeline.processor.image_processor.size = { height: DEPTH_SIZE, width: DEPTH_SIZE };
      depthStatus.textContent = `Profundidade: pronta (${opt.name})`;
      if (running) depthLoop();
      return;
    } catch (err) {
      console.warn("Profundidade falhou em", opt.device, err);
    }
  }
  depthStatus.textContent = "Profundidade: indisponível neste aparelho. Seguindo só com a detecção de objetos.";
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
  // Profundidade depois do detector, para não disputarem o download.
  loadDepth();
  statusBox.textContent = `Modelo carregado (${detector.name}). Toque em Iniciar.`;
  startBtn.disabled = false;
})();
