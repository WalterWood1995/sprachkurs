// 照片数字人：一张正脸照片 → 面部 478 个点（MediaPipe FaceLandmarker，Apache-2.0）→ 三角网格贴图（WebGL），
// 每帧按「说话音量 / 眨眼 / 微笑 / 抬眉 / 呼吸」挪动嘴、眼、眉附近的点；嘴张开的地方露出画在底层的口腔。
//
//   const f = await createPhotoFace(容器, 照片地址);       // 照片旁边有同名 .json（预先算好的面部点）就直接用，
//   f.level(0~1);   // 每帧喂老师说话的音量                  没有就在浏览器里现算一次（要下载 ~10 MB 的模型）
//   f.mood("happy" | "love" | "sad" | "neutral" …);  f.dispose();
//   const pts = await detect(照片地址);                    // 只算面部点（后台上传照片时用，结果存成 .json）
//
// 只适合正脸、五官没遮挡的照片；找不到脸会抛错，调用方退回静态照片或卡通头像。

const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
  150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const IN_UP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308];     // 上唇内沿（左嘴角 → 右嘴角）
const IN_LO = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308];     // 下唇内沿
const EYES = [
  { up: [246, 161, 160, 159, 158, 157, 173], lo: [7, 163, 144, 145, 153, 154, 155], c: [33, 133] },
  { up: [466, 388, 387, 386, 385, 384, 398], lo: [249, 390, 373, 374, 380, 381, 382], c: [263, 362] },
];
const BROWS = [[70, 63, 105, 66, 107, 55, 65, 52, 53, 46], [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]];
const MOODS = [
  [/开心|高兴|兴奋|好笑|轻松|俏皮|得意|欣慰|愉快|调皮|乐|笑|happy/, "happy"],
  [/温柔|喜欢|爱|甜|暖|疼惜|宠|感动|love/, "love"],
  [/心疼|难过|伤心|难受|失落|沉重|遗憾|委屈|低落|sad/, "sad"],
  [/担心|紧张|害怕|着急|不安|慌|fear/, "fear"],
];
const TARGET = { neutral: [0.15, 0], happy: [0.65, 0.25], love: [0.45, 0.15], sad: [0, 0.35], fear: [0, 0.45], angry: [0, 0] };

export function moodOf(text) {
  for (const [re, m] of MOODS) if (re.test(text || "")) return m;
  return "neutral";
}

let landmarkerP = null;
function landmarker() {
  if (!landmarkerP) landmarkerP = (async () => {
    const { FaceLandmarker, FilesetResolver } = await import(MP + "/vision_bundle.mjs");
    const fs = await FilesetResolver.forVisionTasks(MP + "/wasm");
    return FaceLandmarker.createFromOptions(fs, { baseOptions: { modelAssetPath: MODEL, delegate: "CPU" },
      runningMode: "IMAGE", numFaces: 1 });
  })();
  return landmarkerP;
}

function loadImage(url) {
  return new Promise((ok, bad) => {
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => ok(im);
    im.onerror = () => bad(new Error("照片加载失败"));
    im.src = url;
  });
}

// 面部点：[[x, y], …]，按照片像素算（不是 0~1）
export async function detect(urlOrImage) {
  const im = typeof urlOrImage === "string" ? await loadImage(urlOrImage) : urlOrImage;
  const lm = await landmarker();
  const r = lm.detect(im);
  const f = r && r.faceLandmarks && r.faceLandmarks[0];
  if (!f || f.length < 468) throw new Error("照片里没找到正脸");
  return { w: im.naturalWidth, h: im.naturalHeight, pts: f.map(p => [+(p.x * im.naturalWidth).toFixed(1), +(p.y * im.naturalHeight).toFixed(1)]) };
}

async function landmarksFor(url, im) {
  const j = url.replace(/\.(jpe?g|png|webp)(\?.*)?$/i, ".json$2");
  if (j !== url) {
    try {
      const r = await fetch(j, { cache: "no-cache" });
      if (r.ok) {
        const d = await r.json();
        if (d && d.pts && d.pts.length >= 468) {
          const sx = im.naturalWidth / (d.w || im.naturalWidth), sy = im.naturalHeight / (d.h || im.naturalHeight);
          return d.pts.map(([x, y]) => [x * sx, y * sy]);
        }
      }
    } catch (e) { /* 没有预先算好的，就现算 */ }
  }
  return (await detect(im)).pts;
}

// Bowyer–Watson 三角剖分（五六百个点，一次几十毫秒）
function delaunay(P) {
  const n = P.length;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of P) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  const d = Math.max(x1 - x0, y1 - y0) * 30, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const Q = P.concat([[cx - d, cy - d], [cx + d, cy - d], [cx, cy + d]]);
  const mk = (a, b, c) => {
    const [ax, ay] = Q[a], [bx, by] = Q[b], [qx, qy] = Q[c];
    const D = 2 * (ax * (by - qy) + bx * (qy - ay) + qx * (ay - by)) || 1e-9;
    const ux = ((ax * ax + ay * ay) * (by - qy) + (bx * bx + by * by) * (qy - ay) + (qx * qx + qy * qy) * (ay - by)) / D;
    const uy = ((ax * ax + ay * ay) * (qx - bx) + (bx * bx + by * by) * (ax - qx) + (qx * qx + qy * qy) * (bx - ax)) / D;
    return { a, b, c, x: ux, y: uy, r: (ax - ux) ** 2 + (ay - uy) ** 2 };
  };
  let T = [mk(n, n + 1, n + 2)];
  for (let i = 0; i < n; i++) {
    const [px, py] = Q[i], keep = [], edges = new Map();
    for (const t of T) {
      if ((px - t.x) ** 2 + (py - t.y) ** 2 < t.r) {
        for (const [u, v] of [[t.a, t.b], [t.b, t.c], [t.c, t.a]]) {
          const k = u < v ? u + "," + v : v + "," + u;
          edges.set(k, edges.has(k) ? null : [u, v]);
        }
      } else keep.push(t);
    }
    for (const e of edges.values()) if (e) keep.push(mk(e[0], e[1], i));
    T = keep;
  }
  return T.filter(t => t.a < n && t.b < n && t.c < n).map(t => [t.a, t.b, t.c]);
}

const VS = `attribute vec2 p; attribute vec2 uv; uniform vec2 res; varying vec2 v;
void main(){ vec2 c = p / res * 2.0 - 1.0; gl_Position = vec4(c.x, -c.y, 0.0, 1.0); v = uv; }`;
const FS = `precision mediump float; varying vec2 v; uniform sampler2D t; void main(){ gl_FragColor = texture2D(t, v); }`;

export async function createPhotoFace(container, url) {
  const im = await loadImage(url);
  const L = await landmarksFor(url, im);
  const iw = im.naturalWidth, ih = im.naturalHeight;

  // ---- 点：面部 478 个 + 脸外一圈（固定，用来吸收下巴的拉伸）+ 照片边框 ----
  const ov = OVAL.map(i => L[i]);
  const fx0 = Math.min(...ov.map(p => p[0])), fx1 = Math.max(...ov.map(p => p[0]));
  const fy0 = Math.min(...ov.map(p => p[1])), fy1 = Math.max(...ov.map(p => p[1]));
  const fcx = (fx0 + fx1) / 2, fcy = (fy0 + fy1) / 2;
  const P = L.slice(0, 478).map(([x, y], i) => [x + (i % 7) * 1e-3, y + (i % 5) * 1e-3]);   // 微扰：避免点完全重合
  const nFace = P.length;
  const clampX = x => Math.max(0, Math.min(iw, x)), clampY = y => Math.max(0, Math.min(ih, y));
  for (const k of [1.35, 1.75]) for (const [x, y] of ov) P.push([clampX(fcx + (x - fcx) * k), clampY(fcy + (y - fcy) * k)]);
  for (let i = 0; i <= 6; i++) { const t = i / 6; P.push([iw * t, 0], [iw * t, ih], [0, ih * t], [iw, ih * t]); }
  // 去掉重合的点（脸靠边时外圈会被夹到边框上）
  const seen = new Set(), U = [], remap = [];
  P.forEach((p, i) => {
    const k = Math.round(p[0] * 2) + "," + Math.round(p[1] * 2);
    if (i >= nFace && seen.has(k)) { remap.push(-1); return; }
    seen.add(k); remap.push(U.length); U.push(p);
  });
  const N = U.length;

  // ---- 三角形：去掉跨过上下唇之间的那些（嘴张开时这里是空的，露出底层的口腔） ----
  const up = new Set(IN_UP.slice(1, -1)), lo = new Set(IN_LO.slice(1, -1)), inner = new Set([...IN_UP, ...IN_LO]);
  const tris = delaunay(U).filter(([a, b, c]) => {
    const vs = [a, b, c];
    if (!vs.every(v => v < nFace && inner.has(v))) return true;
    return !(vs.some(v => up.has(v)) && vs.some(v => lo.has(v)));
  });

  // ---- 量一量这张脸 ----
  const dist = (a, b) => Math.hypot(L[a][0] - L[b][0], L[a][1] - L[b][1]);
  const W = dist(61, 291), Mx = (L[13][0] + L[14][0]) / 2, My = (L[13][1] + L[14][1]) / 2, chinY = L[152][1];
  // 上下嘴唇的分界线：上下唇内沿的中线（笑起来嘴角上翘时，比「两个嘴角连线」准得多）
  const mid = IN_UP.map((u, k) => [(L[u][0] + L[IN_LO[k]][0]) / 2, (L[u][1] + L[IN_LO[k]][1]) / 2]).sort((a, b) => a[0] - b[0]);
  const lineY = x => {
    if (x <= mid[0][0]) return mid[0][1];
    if (x >= mid[mid.length - 1][0]) return mid[mid.length - 1][1];
    for (let k = 1; k < mid.length; k++) if (x <= mid[k][0]) {
      const [x0, y0] = mid[k - 1], [x1, y1] = mid[k];
      return y0 + (y1 - y0) * ((x - x0) / ((x1 - x0) || 1));
    }
    return My;
  };
  const UPPER = new Set([...IN_UP, 0, 37, 39, 40, 185, 267, 269, 270, 409, 11, 12, 72, 38, 41, 42, 302, 268, 271, 272]);
  const CORNER = new Set([61, 291, 78, 308]);
  const eyes = EYES.map(e => {
    const cx = (L[e.c[0]][0] + L[e.c[1]][0]) / 2, ew = Math.abs(L[e.c[0]][0] - L[e.c[1]][0]);
    const top = e.up.reduce((s, i) => s + L[i][1], 0) / e.up.length, bot = e.lo.reduce((s, i) => s + L[i][1], 0) / e.lo.length;
    return { cx, ew, top, bot, gap: Math.max(0, bot - top), up: new Set(e.up) };
  });
  const brows = BROWS.map(b => [b.reduce((s, i) => s + L[i][0], 0) / b.length, b.reduce((s, i) => s + L[i][1], 0) / b.length]);
  const eyeTop = Math.min(...eyes.map(e => e.top));

  // 每个点对各个动作的权重（只跟原始位置有关，先算好）
  const wJaw = new Float32Array(N), wLift = new Float32Array(N), wBlink = new Float32Array(N), wBrow = new Float32Array(N);
  const wSmX = new Float32Array(N), wSmY = new Float32Array(N), blinkGap = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const [x, y] = U[i], face = i < nFace;
    const rx = (x - Mx) / (W * 0.8);
    const lower = face && CORNER.has(i) ? false : face && UPPER.has(i) ? false : y > lineY(x);
    if (face && CORNER.has(i)) {
      wJaw[i] = 0.35;   // 嘴角跟着下巴动一点点
    } else if (lower) {
      const fy = y <= chinY ? 1 : Math.max(0, 1 - (y - chinY) / (W * 0.55));
      wJaw[i] = Math.exp(-rx * rx * 1.6) * fy;
    } else {
      const u = (My - y) / (W * 0.28);
      if (u < 1) wLift[i] = Math.exp(-rx * rx * 2) * (1 - Math.max(0, u));
    }
    for (const [k, s] of [[61, -1], [291, 1]]) {
      const r = Math.hypot(x - L[k][0], y - L[k][1]) / (W * 0.4);
      if (r < 1.8) { const w = Math.exp(-r * r * 1.3); wSmX[i] += s * w; wSmY[i] += w; }
    }
    for (const e of eyes) {
      if (!face && y > e.top) continue;
      const lid = face && e.up.has(i);
      if (lid) {
        const r = (x - e.cx) / (e.ew * 0.5);
        wBlink[i] = Math.exp(-r * r * 1.2); blinkGap[i] = e.gap;
      } else if (y < e.top) {
        const r = (x - e.cx) / (e.ew * 0.62), t = (e.top - y) / (e.ew * 0.7);
        if (Math.abs(r) < 1.5 && t < 1) { const w = Math.exp(-r * r * 1.5) * (1 - t); if (w > wBlink[i]) { wBlink[i] = w; blinkGap[i] = e.gap; } }
      }
    }
    if (y < eyeTop) for (const [bx, by] of brows) {
      const r = Math.hypot(x - bx, y - by) / (W * 0.45);
      if (r < 1.8) wBrow[i] = Math.max(wBrow[i], Math.exp(-r * r * 1.2));
    }
  }

  // ---- 画布：底层 2D（口腔）+ 上层 WebGL（贴图网格） ----
  const wrap = document.createElement("div");
  wrap.style.cssText = "position:absolute;inset:0;transform-origin:50% 70%";
  const base = document.createElement("canvas"), cv = document.createElement("canvas");
  for (const c of [base, cv]) { c.style.cssText = "position:absolute;inset:0;width:100%;height:100%"; wrap.append(c); }
  container.innerHTML = "";
  container.append(wrap);
  const gl = cv.getContext("webgl", { premultipliedAlpha: false, alpha: true, antialias: true });
  if (!gl) { container.innerHTML = ""; throw new Error("浏览器不支持 WebGL"); }
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog);
  gl.useProgram(prog);
  const aP = gl.getAttribLocation(prog, "p"), aUV = gl.getAttribLocation(prog, "uv"), uRes = gl.getUniformLocation(prog, "res");
  const uvs = new Float32Array(N * 2);
  U.forEach(([x, y], i) => { uvs[2 * i] = x / iw; uvs[2 * i + 1] = y / ih; });
  const bUV = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, bUV); gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(aUV); gl.vertexAttribPointer(aUV, 2, gl.FLOAT, false, 0, 0);
  const pos = new Float32Array(N * 2);
  const bP = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, bP); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(aP); gl.vertexAttribPointer(aP, 2, gl.FLOAT, false, 0, 0);
  const idx = new Uint16Array(tris.flat());
  const bI = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, bI); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
  const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, im);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const bx = base.getContext("2d");

  // 取景：脸放在画面中上部，脸高约占画面六成
  let cw = 0, ch = 0, s = 1, ox = 0, oy = 0;
  function resize() {
    const r = container.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
    cw = Math.max(2, Math.round(r.width * dpr)); ch = Math.max(2, Math.round(r.height * dpr));
    for (const c of [base, cv]) { c.width = cw; c.height = ch; }
    s = Math.max(Math.min(ch * 0.6 / (fy1 - fy0), cw * 0.82 / (fx1 - fx0)), cw / iw, ch / ih);   // 照片要铺满小窗
    ox = Math.min(0, Math.max(cw - iw * s, cw / 2 - fcx * s));
    oy = Math.min(0, Math.max(ch - ih * s, ch * 0.47 - fcy * s));
    gl.viewport(0, 0, cw, ch); gl.uniform2f(uRes, cw, ch);
  }
  resize();

  // ---- 状态 ----
  let lv = 0, open = 0, smile = TARGET.neutral[0], brow = 0, tgt = TARGET.neutral, blink = 0, blinkAt = performance.now() + 1500,
    raf = 0, dead = false;
  const D = new Float32Array(N * 2), UF = new Float32Array(U.flat());
  const MOUTH = [...IN_UP, ...IN_LO.slice().reverse()].map(k => remap[k]);
  function poly(src) {   // 嘴的内轮廓（src：按照片像素的点坐标，平铺数组）
    bx.beginPath();
    MOUTH.forEach((k, j) => { const X = src[2 * k] * s + ox, Y = src[2 * k + 1] * s + oy; if (j) bx.lineTo(X, Y); else bx.moveTo(X, Y); });
    bx.closePath();
  }
  function frame(now) {
    if (dead) return;
    raf = requestAnimationFrame(frame);
    const r = container.getBoundingClientRect();
    if (Math.abs(r.width * Math.min(2, window.devicePixelRatio || 1) - cw) > 2) resize();
    open += ((lv > 0.04 ? lv : 0) - open) * 0.45;
    smile += (tgt[0] - smile) * 0.06; brow += (tgt[1] + lv * 0.25 - brow) * 0.08;
    if (now > blinkAt) { blink = 1; if (now > blinkAt + 120) { blink = 0; blinkAt = now + 2200 + Math.random() * 3800; } }
    const jaw = open * W * 0.24, lift = open * W * 0.035, sx = smile * W * 0.07, sy = smile * W * 0.05, br = brow * W * 0.05;
    for (let i = 0; i < N; i++) {
      const dx = wSmX[i] * sx, dy = wJaw[i] * jaw - wLift[i] * lift - wSmY[i] * sy + wBlink[i] * blink * blinkGap[i] * 0.95
        - wBrow[i] * br;
      D[2 * i] = U[i][0] + dx; D[2 * i + 1] = U[i][1] + dy;
      pos[2 * i] = D[2 * i] * s + ox; pos[2 * i + 1] = D[2 * i + 1] * s + oy;
    }
    // 底层：嘴里（深色）+ 照片上原本露出的牙（原样；照片是闭着嘴的话这块面积是零）
    bx.clearRect(0, 0, cw, ch);
    if (open > 0.01) {
      poly(D);
      const g = bx.createLinearGradient(0, My * s + oy - W * s * 0.1, 0, My * s + oy + jaw * s + W * s * 0.1);
      g.addColorStop(0, "#2a0f0f"); g.addColorStop(0.7, "#4a1a1c"); g.addColorStop(1, "#7a3a3a");
      bx.fillStyle = g; bx.fill();
    }
    bx.save(); poly(UF); bx.clip(); bx.drawImage(im, ox, oy, iw * s, ih * s); bx.restore();
    gl.bindBuffer(gl.ARRAY_BUFFER, bP); gl.bufferSubData(gl.ARRAY_BUFFER, 0, pos);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawElements(gl.TRIANGLES, idx.length, gl.UNSIGNED_SHORT, 0);
    // 呼吸、轻轻晃头、说话时微微点头
    const t = now / 1000;
    wrap.style.transform = `translateY(${(Math.sin(t * 1.1) * 0.5 + lv * 1.2).toFixed(2)}px) rotate(${(Math.sin(t * 0.45) * 0.9).toFixed(2)}deg) scale(${(1 + Math.sin(t * 1.1) * 0.004).toFixed(4)})`;
  }
  raf = requestAnimationFrame(frame);

  return {
    level(v) { lv = Math.max(0, Math.min(1, +v || 0)); },
    mood(m) { tgt = TARGET[m] || TARGET.neutral; },
    frame,   // 页面在后台时 requestAnimationFrame 不跑；测试时可以手动推一帧
    dispose() {
      dead = true; cancelAnimationFrame(raf);
      try { gl.getExtension("WEBGL_lose_context")?.loseContext(); } catch (e) { /* 没关系 */ }
      container.innerHTML = "";
    },
  };
}
