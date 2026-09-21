const $ = (id) => document.getElementById(id);
const frame = $('frame');
let state = null;
let pollTimer = null;
let frameTimer = null;
let camera = { azimuth: 135, elevation: -18, distance: 1.15 };
let drag = null;
// A fast slider drag can generate more input events than the local API needs.
// Serialize motion requests and keep only the newest one.  This removes stale
// command latency without changing the simulator/control-loop rate.
let motionInFlight = false;
let pendingMotion = null;
// Use a time-based origin so refreshing the page cannot restart the sequence
// below a value already accepted by the running replay process.
let motionSequence = Date.now() * 1000 + Math.floor(Math.random() * 1000);

async function post(payload) {
  await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
}

function fmt(value, digits = 2) { return Number(value || 0).toFixed(digits); }

function setText(id, value) { $(id).textContent = value; }

async function flushMotion() {
  if (motionInFlight || pendingMotion === null) return;
  const payload = pendingMotion;
  pendingMotion = null;
  motionInFlight = true;
  try {
    await post(payload);
  } finally {
    motionInFlight = false;
    if (pendingMotion !== null) flushMotion();
  }
}

function queueMotion(payload) {
  pendingMotion = { ...payload, command: 'motion', motion_seq: ++motionSequence };
  flushMotion();
}

function updateBar(id, value, max) {
  const ratio = Math.max(0, Math.min(1, Math.abs(Number(value || 0)) / max));
  $(id).style.width = `${Math.max(3, ratio * 100)}%`;
}

function drawTimeline(history) {
  const canvas = $('timeline');
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(74 * dpr));
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = rect.width, h = 74;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = '#2b3542'; ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) { const y = (h * i) / 4; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  if (!history || history.length < 2) return;
  const max = Math.max(5, ...history.map(p => Math.abs(p.drift_cm || 0)));
  ctx.strokeStyle = '#55d8b7'; ctx.lineWidth = 1.8; ctx.beginPath();
  history.forEach((p, i) => {
    const x = (i / (history.length - 1)) * w;
    const y = h - (Math.abs(p.drift_cm || 0) / max) * (h - 6) - 3;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  // pitch 叠加曲线（橙色）：1.62 Hz 点头在旧策略下会明显振荡，
  // 协同控制器下几乎是一条平线 —— 这是"高频点头"最直观的判据。
  const pitchMax = Math.max(1.0, ...history.map(p => Math.abs(p.pitch_deg || 0)));
  ctx.strokeStyle = '#f0a24b'; ctx.lineWidth = 1.4; ctx.beginPath();
  history.forEach((p, i) => {
    const x = (i / (history.length - 1)) * w;
    const y = h / 2 - ((p.pitch_deg || 0) / pitchMax) * (h / 2 - 6);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = '#718092'; ctx.font = '10px system-ui';
  ctx.fillText(`${fmt(max, 1)} cm drift`, 5, 13);
  ctx.fillStyle = '#f0a24b';
  ctx.fillText(`pitch ±${fmt(pitchMax, 1)}°`, 5, 25);
}

function refreshState(next) {
  state = next;
  const live = next.playing && !next.done;
  const driftFail = Number(next.peak_drift_cm || 0) > 5.0;
  const unstable = Boolean(next.failed) || !next.upright_ok || !next.height_ok || Boolean(next.body_contact);
  const failed = unstable || driftFail;
  const levelBadge = $('levelBadge');
  levelBadge.textContent = failed ? (driftFail ? 'DRIFT > 5 CM' : 'UNSTABLE / FAILED') : next.done ? 'EPISODE COMPLETE' : 'S2 EVALUATION';
  levelBadge.classList.toggle('bad', failed);
  $('statusDot').className = `dot ${failed ? 'bad' : next.done ? 'done' : live ? 'live' : ''}`;
  const status = failed ? (driftFail ? 'DRIFT FAIL' : 'UNSTABLE / FALLEN') : next.done ? 'DONE' : live ? 'RUNNING' : 'PAUSED';
  setText('statusText', status);
  setText('simState', status);
  setText('frameHint', failed ? (driftFail ? 'DRIFT > 5 CM' : `FALLEN: ${String(next.failure_reason || 'PHYSICAL LIMIT').toUpperCase()}`) : next.done ? next.termination.toUpperCase() : live ? 'PLAYING' : 'PAUSED');
  setText('playButton', live ? 'Ⅱ Pause' : '▶ Play');
  $('playButton').classList.toggle('primary', !live);
  setText('episodeValue', next.episode);
  setText('stepValue', next.step);
  setText('timeValue', `${fmt(next.sim_time_s, 2)}s`);
  setText('checkpointName', next.checkpoint.split('/').pop());
  setText('driftValue', `${fmt(next.drift_cm)} cm`);
  setText('peakDriftValue', `${fmt(next.peak_drift_cm)} cm`);
  setText('tailValue', `${fmt(next.tail_drift_cm)} cm`);
  setText('timelineValue', `${fmt(next.drift_cm)} cm`);
  setText('pitchValue', `${fmt(next.pitch_deg)} deg`);
  setText('heightValue', `${fmt(next.height_m, 3)} m`);
  setText('vxValue', `${fmt(next.body_vx, 3)} m/s`);
  setText('rewardValue', fmt(next.reward, 3));
  setText('terminationValue', next.termination || 'ready');
  // ---- 腿 + 轮协同分工 ----
  setText('modeBadge', next.controller_only ? '协同控制器（无策略）' : 'PPO 策略 + 协同');
  setText('coordMixValue', `mix ${fmt(next.coord_mix, 2)}`);
  setText('pitchRmsValue', `${fmt(next.pitch_rms_deg, 3)} deg`);
  setText('pitchPeakValue', `${fmt(next.pitch_peak_deg, 3)} deg`);
  setText('pitchRateValue', `${fmt(next.pitch_rate_rms, 4)} rad/s`);
  setText('driftPeakValue', `${fmt(next.drift_peak_cm)} cm`);
  setText('currentRmsValue', `${fmt(next.current_rms_a, 3)} A`);
  setText('currentDeltaValue', `${fmt(next.current_delta_rms_a, 4)} A`);
  // ---- 腿长（腿负责机体高度）----
  const legRange = next.leg_length_range_m || [];
  if (legRange.length === 2) {
    const slider = $('legLength');
    slider.min = legRange[0].toFixed(3);
    slider.max = legRange[1].toFixed(3);
    setText('legRangeLabel', `${legRange[0].toFixed(3)} – ${legRange[1].toFixed(3)} m`);
    setText('legMinLabel', `${legRange[0].toFixed(3)} m`);
    setText('legMaxLabel', `${legRange[1].toFixed(3)} m`);
    updateBar('legCmdBar', (next.leg_length_cmd_m - legRange[0]), Math.max(1e-6, legRange[1] - legRange[0]));
  }
  const legMeas = Number(next.leg_length_meas_m || 0);
  const legCmd = Number(next.leg_length_cmd_m || 0);
  const legControl = Number(next.leg_length_control_m || legCmd);
  const requestedError = Number(next.leg_length_error_requested_mm || ((legMeas - legCmd) * 1000));
  const controlError = Number(next.leg_length_error_control_mm || ((legMeas - legControl) * 1000));
  setText('legMeasValue', `${fmt(legMeas, 4)} m`);
  setText('legErrValue', `${controlError.toFixed(1)} mm`);
  setText('legRequestedErrValue', `${requestedError.toFixed(1)} mm`);
  setText('legControlValue', `${fmt(legControl, 4)} m`);
  setText('legCmdValue', `${fmt(legCmd, 4)} m`);
  const legOffset = Number(next.coord_leg_offset_rad || 0);
  const coordCurrent = Number(next.coord_current_a || 0);
  setText('legOffsetValue', `${fmt(legOffset, 4)} rad`);
  setText('coordCurrentValue', `${fmt(coordCurrent)} A`);
  updateBar('legOffsetBar', legOffset, 0.35);
  updateBar('coordCurrentBar', coordCurrent, 8);
  setText('speedValue', `${fmt(next.speed, 2)}x`);
  setText('rtfValue', `${fmt(next.real_time_factor, 2)}x`);
  $('speed').value = next.speed;
  const motion = next.motion || {};
  if (document.activeElement !== $('legLength')) $('legLength').value = motion.leg_length_m ?? $('legLength').value;
  if (document.activeElement !== $('wheelSpeed')) $('wheelSpeed').value = motion.wheel_speed_rad_s ?? $('wheelSpeed').value;
  if (document.activeElement !== $('yawRate')) $('yawRate').value = motion.yaw_rate_rad_s ?? $('yawRate').value;
  setText('legLengthValue', `${fmt(motion.leg_length_m, 3)} m`);
  setText('wheelSpeedValue', `${fmt(motion.wheel_speed_rad_s, 1)} rad/s`);
  setText('yawRateValue', `${fmt(motion.yaw_rate_rad_s)} rad/s`);
  const currents = next.wheel_current_a || [0, 0];
  const legs = next.leg_lengths_m || [0, 0];
  setText('wheelLValue', `${fmt(currents[0])} A`); setText('wheelRValue', `${fmt(currents[1])} A`);
  setText('legLValue', `${fmt(legs[0], 3)} m`); setText('legRValue', `${fmt(legs[1], 3)} m`);
  updateBar('wheelLBar', currents[0], 10); updateBar('wheelRBar', currents[1], 10);
  updateBar('legLBar', legs[0], .34); updateBar('legRBar', legs[1], .34);
  $('connectionText').textContent = 'Local API connected';
  drawTimeline(next.history || []);
}

async function poll() {
  try { refreshState(await (await fetch('/api/state', { cache: 'no-store' })).json()); }
  catch (error) { $('connectionText').textContent = 'API disconnected'; $('statusDot').className = 'dot'; }
  pollTimer = setTimeout(poll, 120);
}

function refreshFrame() {
  frame.src = `/api/frame.jpg?t=${Date.now()}`;
  frameTimer = setTimeout(refreshFrame, 160);
}

$('playButton').addEventListener('click', () => post({ command: state && state.playing ? 'pause' : 'play' }));
$('stepButton').addEventListener('click', () => post({ command: 'step', count: 1 }));
$('resetButton').addEventListener('click', () => post({ command: 'reset', seed: Number($('seed').value) || 2000 }));
$('newSeedButton').addEventListener('click', () => { $('seed').value = Math.floor(1000 + Math.random() * 9000); });
$('speed').addEventListener('input', (event) => { setText('speedValue', `${fmt(event.target.value, 2)}x`); post({ command: 'speed', value: Number(event.target.value) }); });
function sendMotion() {
  const leg = Number($('legLength').value);
  const wheel = Number($('wheelSpeed').value);
  const yaw = Number($('yawRate').value);
  setText('legLengthValue', `${fmt(leg, 3)} m`);
  setText('wheelSpeedValue', `${fmt(wheel, 1)} rad/s`);
  setText('yawRateValue', `${fmt(yaw)} rad/s`);
  queueMotion({ leg_length_m: leg, wheel_speed_rad_s: wheel, yaw_rate_rad_s: yaw });
}
['legLength', 'wheelSpeed', 'yawRate'].forEach((id) => $(id).addEventListener('input', sendMotion));
document.querySelectorAll('.quick-leg button').forEach((btn) => {
  btn.addEventListener('click', () => {
    $('legLength').value = btn.dataset.leg;
    sendMotion();
  });
});
$('motionResetButton').addEventListener('click', () => {
  $('legLength').value = '0.184'; $('wheelSpeed').value = '0'; $('yawRate').value = '0';
  sendMotion(); post({ command: 'motion_reset' });
});
$('followButton').addEventListener('click', () => {
  const enabled = $('followButton').textContent.endsWith('OFF');
  $('followButton').textContent = enabled ? 'FOLLOW ON' : 'FOLLOW OFF';
  post({ command: 'follow', value: enabled });
});
document.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('[data-preset]').forEach(item => item.classList.remove('active'));
  button.classList.add('active'); post({ command: 'camera', preset: button.dataset.preset });
}));
$('fitButton').addEventListener('click', () => post({ command: 'camera', preset: 'three-quarter' }));
$('fullscreenButton').addEventListener('click', () => document.documentElement.requestFullscreen?.());

const viewport = $('viewport');
viewport.addEventListener('pointerdown', (event) => { drag = { x: event.clientX, y: event.clientY, az: camera.azimuth, el: camera.elevation }; viewport.setPointerCapture(event.pointerId); });
viewport.addEventListener('pointermove', (event) => {
  if (!drag) return;
  camera.azimuth = drag.az - (event.clientX - drag.x) * .45;
  camera.elevation = Math.max(-80, Math.min(25, drag.el + (event.clientY - drag.y) * .3));
  post({ command: 'camera', azimuth: camera.azimuth, elevation: camera.elevation });
});
viewport.addEventListener('pointerup', () => { drag = null; });
viewport.addEventListener('wheel', (event) => { event.preventDefault(); camera.distance = Math.max(.45, Math.min(3, camera.distance + event.deltaY * .0015)); post({ command: 'camera', distance: camera.distance }); }, { passive: false });

poll(); refreshFrame();
