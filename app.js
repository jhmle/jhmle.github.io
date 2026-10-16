const $ = (id) => document.getElementById(id);
const setStatus = (text, kind = "idle") => { const s = $("status"); s.textContent = text; s.className = "pill pill-" + kind; };

if (!navigator.gpu) {
  setStatus("This needs a WebGPU browser (recent desktop Chrome or Edge).", "error");
  $("load").disabled = true;
  throw new Error("no WebGPU");
}

const gemma = new Worker("gemma.worker.js", { type: "module" });
const kokoro = new Worker("kokoro.worker.js", { type: "module" });

const TARGET = 16000;      // Gemma wants 16 kHz mono
const MAX_SECONDS = 30;    // Gemma's audio limit
const SILENCE_END = 1.2;   // seconds of quiet that end your turn
let ready = 0, busy = false, starting = false, S = null;   // S = live conversation session
let state = "off", speakNow = true, genDone = true, skipped = false, pending = 0, captureInfo = "";

// ---------------------------------------------------------------- turn states (what the stage shows)
const STATES = {
  off:       ["Not started", "Ready when you are", "Press Start conversation and just talk."],
  listening: ["Your turn", "Go ahead, I'm listening", "Speak whenever you like. I'll answer when you pause."],
  hearing:   ["Your turn", "Hearing you…", "Keep going. Pause when you're done."],
  thinking:  ["Gemma's turn", "Thinking…", "Working out a reply."],
  speaking:  ["Gemma's turn", "Speaking", "Wait for the reply, or press Skip reply to cut in."],
};
function setState(s) {
  state = s;
  const [turn, title, hint] = STATES[s];
  $("stage").dataset.state = s;
  $("turnText").textContent = turn; $("title").textContent = title; $("hint").textContent = hint;
  if (s === "listening" || s === "hearing") setStatus("Live · your turn", "ready");
  else if (s === "thinking" || s === "speaking") setStatus("Live · Gemma's turn", "busy");
  else if (ready === 2) setStatus("Ready", "ready");
  refresh();
}
function refresh() {
  $("mic").disabled = ready !== 2 || starting;
  $("skip").disabled = !S || !(state === "thinking" || state === "speaking");
  for (const id of ["up", "txt", "sample"]) $(id).disabled = ready !== 2 || busy || !!S || starting;
}
const setBusy = (b) => { busy = b; refresh(); };

// ---------------------------------------------------------------- loading
$("load").onclick = () => {
  $("load").disabled = true;
  setStatus("Downloading models…", "loading");
  gemma.postMessage({ type: "load", dtype: new URLSearchParams(location.search).get("dtype") || "q4f16" });
  kokoro.postMessage({ type: "load" });
};
const onReady = () => {
  if (++ready < 2) return;
  $("setup").classList.add("done");
  setState("off");
};
const fail = (e) => setStatus("Error: " + e.message, "error");

// ---------------------------------------------------------------- speech playback (24 kHz chunks, back to back)
let actx, nextT = 0, sources = [];
function play(samples, rate) {
  actx ??= new AudioContext();
  const b = actx.createBuffer(1, samples.length, rate); b.copyToChannel(samples, 0);
  const s = actx.createBufferSource(); s.buffer = b; s.connect(actx.destination);
  nextT = Math.max(nextT, actx.currentTime + 0.05); s.start(nextT); nextT += b.duration;
  sources.push(s);
  s.onended = () => { sources = sources.filter((x) => x !== s); settle(); };
}
function stopSpeech() {
  sources.forEach((s) => { s.onended = null; try { s.stop(); } catch {} });
  sources = []; nextT = 0;
}
// When Gemma has finished and every spoken chunk has played, hand the turn back to the user.
const canResume = () => genDone && pending === 0 && sources.length === 0 && (state === "thinking" || state === "speaking");
function settle() {
  setTimeout(() => { if (canResume()) listen(); }, 450);   // short gap so the tail of our own voice isn't picked up
}
function skip() {
  skipped = true; speakNow = false; stopSpeech();
  if (genDone) settle(); else setState("thinking");
}
$("skip").onclick = skip;

// ---------------------------------------------------------------- sentence chunker -> TTS
let buf = "";
const cleanForSpeech = (t) => t.replace(/[*_#`>~]+/g, "").replace(/\s+/g, " ").trim();
function speak(text) {
  const c = cleanForSpeech(text);
  if (c.length > 1 && /[a-z0-9]/i.test(c)) { pending++; kokoro.postMessage({ type: "speak", text: c, voice: $("voice").value }); }
}
function feed(t, final) {
  buf += t; $("out").textContent += t;
  if (speakNow) {
    let m;
    while ((m = buf.match(/^([\s\S]*?[.!?\n])\s/))) { speak(m[1]); buf = buf.slice(m[0].length); }
    if (final && buf.trim()) speak(buf);
  }
  if (final) buf = "";
}

gemma.onmessage = ({ data: d }) => {
  if (d.type === "progress") $("pg").value = d.pct;
  else if (d.type === "ready") onReady();
  else if (d.type === "debug") { $("dbg").textContent = (captureInfo ? captureInfo + "\n" : "") + d.text; console.log(d.text); }
  else if (d.type === "token") feed(d.text, false);
  else if (d.type === "done") { genDone = true; feed("", true); setBusy(false); settle(); }
  else if (d.type === "error") { genDone = true; setBusy(false); S ? listen() : setState("off"); fail(d); }
};
kokoro.onmessage = ({ data: d }) => {
  if (d.type === "progress") $("pk").value = d.pct;
  else if (d.type === "ready") {
    $("voice").innerHTML = d.voices.map((v) => `<option>${v}</option>`).join("");
    $("voice").value = d.voices.includes("af_heart") ? "af_heart" : d.voices[0];
    onReady();
  }
  else if (d.type === "audio") {
    pending = Math.max(0, pending - 1);
    if (!skipped && state !== "off") { play(d.samples, d.rate); if (state === "thinking") setState("speaking"); }
    else settle();
  }
  else if (d.type === "error") { pending = Math.max(0, pending - 1); fail(d); settle(); }
};

// ---------------------------------------------------------------- audio helpers
function resample(input, from, to = TARGET) {
  if (from === to) return input;
  const ratio = from / to, n = Math.floor(input.length / ratio), out = new Float32Array(n);
  for (let i = 0; i < n; i++) {           // box filter: average the source samples under each output sample
    const a = Math.floor(i * ratio), b = Math.min(Math.ceil((i + 1) * ratio), input.length);
    let s = 0; for (let j = a; j < b; j++) s += input[j];
    out[i] = b > a ? s / (b - a) : 0;
  }
  return out;
}
function concat(chunks, total) {
  const out = new Float32Array(total); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}
function prepare(x) {                      // cap length, reject silence, lift quiet mics (gain capped)
  if (x.length > TARGET * MAX_SECONDS) x = x.slice(0, TARGET * MAX_SECONDS);
  let peak = 0; for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); if (a > peak) peak = a; }
  if (peak < 0.01) return { samples: null, peak, gain: 1 };
  let gain = 1;
  if (peak < 0.3) { gain = Math.min(0.7 / peak, 20); x = x.map((v) => v * gain); }
  return { samples: x, peak, gain };
}
async function decodeBlob(blob) {          // for uploaded files / sample clip
  const ctx = new AudioContext({ sampleRate: TARGET });
  const ab = await ctx.decodeAudioData(await blob.arrayBuffer()); await ctx.close();
  const out = new Float32Array(ab.length);
  for (let c = 0; c < ab.numberOfChannels; c++) {
    const ch = ab.getChannelData(c);
    for (let i = 0; i < out.length; i++) out[i] += ch[i] / ab.numberOfChannels;
  }
  return out;
}
function grab(video) {
  const c = document.createElement("canvas"), k = 448 / Math.max(video.videoWidth, video.videoHeight);
  c.width = Math.round(video.videoWidth * k); c.height = Math.round(video.videoHeight * k);
  c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
  return new Promise((r) => c.toBlob(r, "image/jpeg", 0.85));
}

// ---------------------------------------------------------------- send to Gemma
async function run(audio, frames) {
  if (busy) return;
  const text = S ? "" : $("prompt").value.trim();
  if (!audio && !frames.length && !text) { setStatus("Nothing to send. Type a message or upload a file.", "error"); return; }
  setBusy(true); stopSpeech();
  skipped = false; genDone = false; buf = ""; $("out").textContent = "";
  speakNow = $("speakOn").checked;
  setState("thinking");
  gemma.postMessage({ type: "run", audio, frames, text, task: "chat" });
}
async function sendClip(mono, label, frames = []) {
  const p = prepare(mono);
  captureInfo = `${label}: ${(Math.min(mono.length, TARGET * MAX_SECONDS) / TARGET).toFixed(1)}s | peak ${p.peak.toFixed(3)} | gain x${p.gain.toFixed(1)}`;
  $("dbg").textContent = captureInfo;
  if (!p.samples) {
    if (S) { listen(); return; }
    setStatus(`That clip is silent (peak ${p.peak.toFixed(4)}). Check your input device and mic permission.`, "error");
    return;
  }
  await run(p.samples, frames);
}

// ---------------------------------------------------------------- live conversation (voice-activity loop)
function listen() {                        // start (or restart) waiting for the user to speak
  if (!S) { setState("off"); return; }
  Object.assign(S, { pre: [], chunks: [], total: 0, len: 0, voice: 0, silent: 0, run: 0 });
  setState("listening");
}
function onChunk(e) {
  if (!S) return;
  const d = new Float32Array(e.inputBuffer.getChannelData(0));
  let sum = 0; for (let i = 0; i < d.length; i++) sum += d[i] * d[i];
  const rms = Math.sqrt(sum / d.length); S.lvl = rms;
  if (state !== "listening" && state !== "hearing") return;     // ignore the mic while Gemma is thinking or talking
  const dur = d.length / S.ctx.sampleRate, loud = rms > Math.max(0.012, S.floor * 3.5);
  if (state === "listening") {
    if (!loud) { S.run = 0; S.floor = S.floor * 0.95 + rms * 0.05; S.pre.push(d); if (S.pre.length > 4) S.pre.shift(); return; }
    S.pre.push(d);
    if (++S.run < 2) return;               // two loud blocks in a row = real speech, not a click
    S.chunks = S.pre; S.pre = [];
    S.total = S.chunks.reduce((n, c) => n + c.length, 0); S.len = S.total / S.ctx.sampleRate;
    S.voice = dur * 2; S.silent = 0; setState("hearing");
    return;
  }
  S.chunks.push(d); S.total += d.length; S.len += dur;
  if (loud) { S.silent = 0; S.voice += dur; } else S.silent += dur;
  if (S.silent >= SILENCE_END || S.len >= MAX_SECONDS) finish();
}
async function finish() {
  const s = S; setState("thinking");        // stops further capture immediately
  const chunks = s.chunks, total = s.total;
  if (s.voice < 0.35) { listen(); return; } // a cough or bump, not a sentence
  let frames = [];
  if (s.useCam && $("cam").videoWidth) { try { frames = [await grab($("cam"))]; } catch {} }
  const rate = s.ctx.sampleRate;
  await sendClip(resample(concat(chunks, total), rate, TARGET), `mic (${rate} Hz to 16 kHz)`, frames);
}
function viz() {                            // the orb grows with your voice
  if (!S) return;
  S.raf = requestAnimationFrame(viz);
  const on = state === "listening" || state === "hearing";
  S.sm = (S.sm || 0) * 0.7 + (on ? Math.min(S.lvl * 6, 1) : 0) * 0.3;
  $("stage").style.setProperty("--lvl", S.sm.toFixed(3));
}
async function begin() {
  if (S || starting || ready < 2) return;
  starting = true; refresh(); stopSpeech();
  const useCam = $("useCam").checked;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: useCam ? { facingMode: "user" } : false,
    });
  } catch (e) {
    starting = false; refresh();
    setStatus("Mic or camera access failed: " + e.message, "error");
    return;
  }
  const ctx = new AudioContext(); await ctx.resume();
  const src = ctx.createMediaStreamSource(stream), sp = ctx.createScriptProcessor(4096, 1, 1), mute = ctx.createGain();
  mute.gain.value = 0;                      // keeps the processor running without playing the mic back
  src.connect(sp); sp.connect(mute); mute.connect(ctx.destination);
  S = { stream, ctx, src, sp, mute, useCam, floor: 0.004, lvl: 0, sm: 0, raf: 0 };
  sp.onaudioprocess = onChunk;
  if (useCam) { $("cam").srcObject = stream; try { await $("cam").play(); } catch {} $("stage").classList.add("cam"); }
  starting = false; skipped = false;
  $("mic").classList.add("on"); $("micLabel").textContent = "End conversation";
  listen(); viz();
}
function end() {
  const s = S; if (!s) return; S = null;
  cancelAnimationFrame(s.raf); s.sp.onaudioprocess = null;
  try { s.src.disconnect(); s.sp.disconnect(); s.mute.disconnect(); } catch {}
  s.stream.getTracks().forEach((t) => t.stop()); s.ctx.close();
  $("cam").srcObject = null; $("stage").classList.remove("cam"); $("stage").style.setProperty("--lvl", 0);
  stopSpeech(); speakNow = false; skipped = true;
  $("mic").classList.remove("on"); $("micLabel").textContent = "Start conversation";
  setState(busy ? "thinking" : "off");      // if Gemma is mid-reply it finishes silently, then we go idle
}
$("mic").onclick = () => (S ? end() : begin());

// ---------------------------------------------------------------- one-off inputs (only when no conversation is running)
async function frameFromFile(f) {
  const v = $("cam"), url = URL.createObjectURL(f);
  v.src = url;
  await new Promise((res, rej) => { v.onloadeddata = res; v.onerror = rej; });
  v.currentTime = isFinite(v.duration) ? v.duration / 2 : 0;
  await new Promise((res) => (v.onseeked = res));
  const frame = await grab(v);
  URL.revokeObjectURL(url); v.removeAttribute("src"); v.load();
  return frame;
}
$("up").onclick = () => $("file").click();
$("file").onchange = async (e) => {
  const f = e.target.files[0]; e.target.value = ""; if (!f) return;
  setStatus("Reading file…", "busy");
  let mono = null, frames = [];
  try { mono = await decodeBlob(f); } catch { /* no audio track */ }
  if (f.type.startsWith("video/")) { try { frames = [await frameFromFile(f)]; } catch {} }
  if (mono) await sendClip(mono, f.name, frames);
  else if (frames.length) await run(null, frames);
  else setStatus("Couldn't read audio or video from that file.", "error");
};
$("txt").onclick = () => run(null, []);

// Known-good clip: if this works but your mic does not, the problem is the recording, not the model.
$("sample").onclick = async () => {
  try {
    setStatus("Fetching test clip…", "busy");
    const r = await fetch("https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav");
    await sendClip(await decodeBlob(await r.blob()), "test clip");
  } catch (e) { setStatus("Test clip failed: " + e.message, "error"); }
};

setState("off");
