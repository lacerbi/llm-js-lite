// llm-js-lite MVP — Browser-only WebGPU chat with Transformers.js
// Notes:
// - WebGPU-first: If WebGPU is unavailable, Load remains disabled.
// - Uses @xenova/transformers via CDN at runtime (no build step).
// - Persists settings + transcript in localStorage.

const el = (id) => document.getElementById(id);
const $chat = el('chat');
const $composer = el('composer');
const $input = el('input');
const $send = el('btn-send');
const $stop = el('btn-stop');
const $load = el('btn-load');
const $unload = el('btn-unload');
const $progress = el('progress');
const $modelStatus = el('model-status');
const $capBanner = el('capability-banner');
const $statusBackend = el('backend');
const $statusDevice = el('device');
const $statusState = el('state');
const $diag = el('diag');

// Settings elements
const $temperature = el('temperature');
const $top_p = el('top_p');
const $top_k = el('top_k');
const $repetition_penalty = el('repetition_penalty');
const $max_new_tokens = el('max_new_tokens');
const $systemPrompt = el('system-prompt');
const $saveSettings = el('btn-save-settings');
const $resetConvo = el('btn-reset-convo');
const $clearCache = el('btn-clear-cache');

// Runtime state
let Transformers = null; // module
let pipe = null; // text-generation pipeline
let tokenizer = null; // tokenizer for chat template, if available
let abortController = null;
let busy = false;
let modelLoaded = false;
let activeModelId = null;

const MODEL_IDS = {
  primary: 'onnx-community/Qwen3-1.7B-ONNX',
  // Known-good, smaller fallback supported by Transformers.js
  fallback: 'Xenova/Qwen2.5-0.5B-Instruct',
};

// App state
const STORAGE = {
  settings: 'llm-js-lite:settings',
  transcript: 'llm-js-lite:transcript',
};

const DEFAULT_SETTINGS = {
  temperature: 0.7,
  top_p: 0.95,
  top_k: 40,
  repetition_penalty: 1.05,
  max_new_tokens: 256,
  system_prompt: `You are a concise, helpful AI assistant. Provide clear, accurate answers.`,
};

let settings = loadSettings();
let messages = loadTranscript();

// Utility: simple renderer
function addMessage(role, content) {
  const item = document.createElement('div');
  item.className = `msg ${role}`;
  const roleEl = document.createElement('div');
  roleEl.className = 'role';
  roleEl.textContent = role;
  const contentEl = document.createElement('div');
  contentEl.className = 'content';
  contentEl.textContent = content;
  item.appendChild(roleEl);
  item.appendChild(contentEl);
  $chat.appendChild(item);
  $chat.scrollTop = $chat.scrollHeight;
  return { item, contentEl };
}

function updateLastAssistantContent(text) {
  const items = $chat.querySelectorAll('.msg.assistant .content');
  const last = items[items.length - 1];
  if (last) last.textContent = text;
}

function setBusy(nextBusy) {
  busy = nextBusy;
  $statusState.textContent = `State: ${busy ? 'generating' : 'idle'}`;
  $send.disabled = busy || !modelLoaded;
  $stop.disabled = !busy;
  $input.disabled = busy || !modelLoaded;
}

function setModelStatus(text) {
  $modelStatus.textContent = text;
}

function setProgress(text) {
  $progress.textContent = text ?? '';
}

function saveSettings(next) {
  settings = { ...settings, ...next };
  localStorage.setItem(STORAGE.settings, JSON.stringify(settings));
}

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE.settings);
    return raw ? { ...DEFAULT_SETTINGS, ...JSON.parse(raw) } : { ...DEFAULT_SETTINGS };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

function saveTranscript() {
  localStorage.setItem(STORAGE.transcript, JSON.stringify(messages));
}

function loadTranscript() {
  try {
    const raw = localStorage.getItem(STORAGE.transcript);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function renderTranscript() {
  $chat.innerHTML = '';
  for (const m of messages) addMessage(m.role, m.content);
}

function hydrateSettingsUI() {
  $temperature.value = String(settings.temperature);
  $top_p.value = String(settings.top_p);
  $top_k.value = String(settings.top_k);
  $repetition_penalty.value = String(settings.repetition_penalty);
  $max_new_tokens.value = String(settings.max_new_tokens);
  $systemPrompt.value = settings.system_prompt;
}

function readSettingsFromUI() {
  const coerce = (n, def) => {
    const v = Number(n);
    return Number.isFinite(v) ? v : def;
  };
  saveSettings({
    temperature: coerce($temperature.value, DEFAULT_SETTINGS.temperature),
    top_p: coerce($top_p.value, DEFAULT_SETTINGS.top_p),
    top_k: Math.max(0, Math.floor(coerce($top_k.value, DEFAULT_SETTINGS.top_k))),
    repetition_penalty: coerce($repetition_penalty.value, DEFAULT_SETTINGS.repetition_penalty),
    max_new_tokens: Math.max(16, Math.floor(coerce($max_new_tokens.value, DEFAULT_SETTINGS.max_new_tokens))),
    system_prompt: $systemPrompt.value || DEFAULT_SETTINGS.system_prompt,
  });
  hydrateSettingsUI();
}

async function detectWebGPU() {
  if (!('gpu' in navigator)) {
    $capBanner.textContent = 'WebGPU not available. Enable it or use a supported browser.';
    $capBanner.style.color = '#ff8a8a';
    $load.disabled = true;
    $send.disabled = true;
    $statusBackend.textContent = 'Backend: — (WebGPU unavailable)';
    return false;
  }
  $capBanner.textContent = 'WebGPU available';
  $capBanner.style.color = '#8affc7';
  $load.disabled = false; // allow load attempt

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter) {
      $statusDevice.textContent = `Device: ${adapter.name || 'WebGPU adapter'}`;
    }
  } catch {
    // ignore
  }
  return true;
}

function clearTransformersCache() {
  // Best-effort clear of the known IndexedDB used by Transformers.js
  try { indexedDB.deleteDatabase('transformers-cache'); } catch {}
  // Optional: clear Cache Storage keys that may exist
  if (typeof caches !== 'undefined') {
    caches.keys().then(keys => keys.forEach(k => {
      if (k.includes('transformers') || k.includes('onnx')) caches.delete(k);
    }));
  }
}

async function ensureTransformers() {
  if (Transformers) return Transformers;
  // Lazy-load from CDN when needed with fallbacks
  const CANDIDATES = [
    'https://cdn.jsdelivr.net/npm/@xenova/transformers@3.2.1',
    'https://cdn.jsdelivr.net/npm/@xenova/transformers',
    'https://cdn.jsdelivr.net/npm/@xenova/transformers@3',
    'https://unpkg.com/@xenova/transformers',
  ];
  let lastErr = null;
  for (const url of CANDIDATES) {
    try {
      Transformers = await import(url);
      break;
    } catch (e) {
      lastErr = e;
    }
  }
  if (!Transformers) {
    throw lastErr || new Error('Failed to load @xenova/transformers from CDN');
  }
  // Prefer WebGPU; do not silently fall back in MVP flow
  try {
    const { env } = Transformers;
    env.useBrowserCache = true;
    env.allowRemoteModels = true;
    env.allowLocalModels = false; // avoid /models/* lookups on our own host
    env.localModelPath = null;    // explicit: no local model directory
    // Prefer fetching from HF; keep default remote host
    // Optional wasm tuning when wasm is used elsewhere in stack
    if (env.backends?.onnx?.wasm) {
      env.backends.onnx.wasm.numThreads = Math.max(2, navigator.hardwareConcurrency ? Math.min(4, navigator.hardwareConcurrency) : 2);
    }
  } catch (e) {
    console.warn('Env setup warning:', e);
  }
  return Transformers;
}

async function loadModel() {
  if (modelLoaded) return;
  setModelStatus('Loading…');
  setProgress('');
  $load.disabled = true; $unload.disabled = true; $send.disabled = true;
  $statusBackend.textContent = 'Backend: WebGPU (requested)';

  const tryLoad = async (modelId) => {
    const TF = await ensureTransformers();
    const { pipeline, AutoTokenizer } = TF;

    const progress_callback = (data) => {
      try {
        if (data?.status === 'progress') {
          const { file, loaded, total } = data;
          const pct = total ? Math.round((loaded / total) * 100) : 0;
          setProgress(`${file || 'artifact'} — ${loaded}/${total || '?'} bytes (${pct}%)`);
        } else if (data?.status) {
          setProgress(`${data.status}…`);
        }
      } catch {}
    };

    // Initialize tokenizer if possible (chat template)
    tokenizer = null;
    try {
      tokenizer = await AutoTokenizer.from_pretrained(modelId, {
        progress_callback,
        revision: 'main',
      });
    } catch (e) {
      console.warn('Tokenizer load failed, continuing with naive prompt formatting:', e);
    }

    // Initialize text-generation pipeline
    pipe = await pipeline('text-generation', modelId, {
      device: 'webgpu',
      progress_callback,
      revision: 'main',
    });
    activeModelId = modelId;
  };

  try {
    await tryLoad(MODEL_IDS.primary);
    modelLoaded = true;
    setModelStatus('Ready');
    setProgress('');
    $send.disabled = false; $unload.disabled = false; $input.disabled = false;
  } catch (err) {
    console.error(err);
    const msg = String(err?.message || err);
    const looksUnsupported = /Unsupported model type: qwen3/i.test(msg) || /split is not a function/i.test(msg) || /qwen3/i.test(msg);

    if (looksUnsupported) {
      setModelStatus('Qwen3 not yet supported by Transformers.js');
      const proceed = window.confirm('Qwen3-1.7B appears unsupported in Transformers.js at the moment. Load a compatible fallback (Xenova/Qwen2.5-0.5B-Instruct) instead?');
      if (proceed) {
        try {
          setModelStatus('Loading fallback model…');
          await tryLoad(MODEL_IDS.fallback);
          modelLoaded = true;
          setModelStatus('Ready (fallback: Qwen2.5-0.5B)');
          setProgress('');
          $send.disabled = false; $unload.disabled = false; $input.disabled = false;
          return;
        } catch (e2) {
          console.error(e2);
          setModelStatus('Fallback load failed');
          setProgress(String(e2?.message || e2));
        }
      }
    }

    setModelStatus('Load failed');
    setProgress(msg);
    $load.disabled = false;
  }
}

function unloadModel() {
  // Best-effort release
  pipe = null;
  tokenizer = null;
  modelLoaded = false;
  setModelStatus('Unloaded');
  $send.disabled = true; $unload.disabled = true; $input.disabled = true; $load.disabled = false;
}

// Build prompt with optional chat template
function buildPrompt(userText) {
  const convo = [];
  if (settings.system_prompt) {
    convo.push({ role: 'system', content: settings.system_prompt });
  }

  // Sliding window by last N messages to keep context bounded (coarse)
  const MAX_TURNS = 6;
  const history = messages.slice(-MAX_TURNS * 2); // user+assistant pairs
  for (const m of history) convo.push({ role: m.role, content: m.content });
  convo.push({ role: 'user', content: userText });

  // Try tokenizer.apply_chat_template if available
  try {
    if (tokenizer && typeof tokenizer.apply_chat_template === 'function') {
      return tokenizer.apply_chat_template(convo, { add_generation_prompt: true, tokenize: false });
    }
  } catch (e) {
    console.warn('Chat template apply failed; using fallback', e);
  }

  // Fallback: simple role-tagged formatting
  const lines = [];
  for (const turn of convo) {
    const role = turn.role === 'assistant' ? 'Assistant' : turn.role === 'system' ? 'System' : 'User';
    lines.push(`${role}: ${turn.content}`);
  }
  lines.push('Assistant:');
  return lines.join('\n');
}

async function generate(userText) {
  if (!pipe) throw new Error('Model not loaded');

  const prompt = buildPrompt(userText);
  const assistantMsg = { role: 'assistant', content: '' };
  messages.push({ role: 'user', content: userText });
  saveTranscript();
  addMessage('user', userText);
  addMessage('assistant', '');

  // Streaming generation
  setBusy(true);
  abortController = new AbortController();

  const startTime = performance.now();
  let ttfToken = null; // ms
  let tokenCount = 0;

  try {
    const result = await pipe(prompt, {
      max_new_tokens: settings.max_new_tokens,
      temperature: settings.temperature,
      top_p: settings.top_p,
      top_k: settings.top_k,
      repetition_penalty: settings.repetition_penalty,
      // Stream tokens as they are generated
      callback_function: (data) => {
        // Expected shape: { token: { text, id, logprob, special }, generated_text? }
        const text = data?.token?.text ?? '';
        if (text) {
          if (ttfToken === null) ttfToken = performance.now() - startTime;
          tokenCount += 1;
          assistantMsg.content += text;
          updateLastAssistantContent(assistantMsg.content);
        }
      },
      abort_signal: abortController.signal,
    });

    // Final text may also be provided; ensure sync
    const generated = Array.isArray(result) ? result[0]?.generated_text : result?.generated_text;
    if (generated && generated.length >= assistantMsg.content.length) {
      assistantMsg.content = generated;
      updateLastAssistantContent(assistantMsg.content);
    }

    messages.push(assistantMsg);
    saveTranscript();

    const dt = performance.now() - startTime;
    const tokPerSec = tokenCount > 1 ? (tokenCount - 1) / ((dt - (ttfToken || 0)) / 1000) : 0;
    $diag.textContent = `TTFT: ${ttfToken ? ttfToken.toFixed(0) + ' ms' : '—'}, tok/s: ${tokPerSec ? tokPerSec.toFixed(1) : '—'}`;
  } catch (err) {
    if (err?.name === 'AbortError') {
      setModelStatus('Generation aborted');
    } else {
      console.error(err);
      setModelStatus('Generation error');
      addMessage('assistant', `[Error] ${err?.message || String(err)}`);
    }
  } finally {
    setBusy(false);
    abortController = null;
  }
}

// Event wiring
$saveSettings.addEventListener('click', () => {
  readSettingsFromUI();
  setModelStatus('Settings saved');
});

$resetConvo.addEventListener('click', () => {
  messages = [];
  saveTranscript();
  renderTranscript();
  setModelStatus('Conversation reset');
});

$clearCache.addEventListener('click', async () => {
  clearTransformersCache();
  setModelStatus('Cache cleared');
});

if ($composer) {
  $composer.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = ($input.value || '').trim();
    if (!text || busy || !modelLoaded) return;
    $input.value = '';
    await generate(text);
  });
}

$send.addEventListener('click', async () => {
  const text = ($input.value || '').trim();
  if (!text || busy || !modelLoaded) return;
  $input.value = '';
  await generate(text);
});

$stop.addEventListener('click', () => {
  if (abortController) abortController.abort();
});

$load.addEventListener('click', async () => {
  await loadModel();
});

$unload.addEventListener('click', () => {
  unloadModel();
});

// Initialize UI
(async function init() {
  hydrateSettingsUI();
  renderTranscript();
  $input.disabled = true; $send.disabled = true; $stop.disabled = true; $unload.disabled = true;
  await detectWebGPU();
})();
