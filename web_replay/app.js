const $ = (id) => document.getElementById(id);
const fmt = (v, n = 2) => Number(v || 0).toFixed(n);
async function post(payload) {
  await fetch('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
}
function refresh(s) {
  $('mode').textContent = s.mode || 'PPO DIRECT';
  $('stepValue').textContent = s.step || 0;
  $('rewardValue').textContent = fmt(s.reward, 3);
  $('termination').textContent = s.termination || 'ready';
  $('height').textContent = `${fmt(s.height_m, 3)} m`;
  $('pitch').textContent = `${fmt(Number(s.pitch_deg), 2)}°`;
  $('vx').textContent = `${fmt(s.body_vx, 3)} m/s`;
  const wheel = s.wheel_target_rad_s || [0, 0];
  const current = s.wheel_current_a || [0, 0];
  const legs = s.leg_lengths_m || [0, 0];
  $('wheelL').textContent = `${fmt(wheel[0], 2)} rad/s`;
  $('wheelR').textContent = `${fmt(wheel[1], 2)} rad/s`;
  $('currentL').textContent = `${fmt(current[0], 2)} A`;
  $('currentR').textContent = `${fmt(current[1], 2)} A`;
  $('legL').textContent = `${fmt(legs[0], 3)} m`;
  $('legR').textContent = `${fmt(legs[1], 3)} m`;
  $('legTarget').textContent = fmt(s.leg_length_target_m, 4);
  $('legError').textContent = fmt(s.leg_length_error_mm, 1);
  $('play').textContent = s.playing ? 'Ⅱ 暂停' : '▶ 播放';
  $('speed').value = s.speed || $('speed').value;
  const motion = s.motion || {};
  $('wheelSpeed').value = motion.wheel_speed_rad_s || 0;
  $('yawRate').value = motion.yaw_rate_rad_s || 0;
  $('wheelSpeedValue').textContent = fmt(motion.wheel_speed_rad_s, 1);
  $('yawValue').textContent = fmt(motion.yaw_rate_rad_s, 1);
}
async function poll() {
  try { refresh(await (await fetch('/api/state', { cache: 'no-store' })).json()); } catch (_) {}
  setTimeout(poll, 150);
}
function sendMotion() {
  post({ command: 'motion', wheel_speed_rad_s: Number($('wheelSpeed').value), yaw_rate_rad_s: Number($('yawRate').value) });
}
$('play').onclick = () => post({ command: 'play' });
$('step').onclick = () => post({ command: 'step' });
$('reset').onclick = () => post({ command: 'reset', seed: Number($('seed').value) || 2000 });
$('speed').oninput = (e) => post({ command: 'speed', value: Number(e.target.value) });
$('wheelSpeed').oninput = sendMotion;
$('yawRate').oninput = sendMotion;
setInterval(() => { $('frame').src = `/api/frame.jpg?t=${Date.now()}`; }, 180);
poll();
