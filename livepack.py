"""Live Pack: client-side animated cross-section of the solved design.

Self-diagnosing: any JS error paints itself onto the canvas and the status
line. Badges are pre-filled server-side so the numbers show even if a
browser blocks the script. Canvas uses CSS height + devicePixelRatio buffer
scaling (retina-safe).
"""
import json


def live_pack_html(state: dict) -> str:
    s = json.dumps(state)
    S = state
    b_q = f"<b>{S['q_kw']:.2f} kW</b> at {S['c_rms']:.2f}C rms"
    b_t = (f"can <b>{S['T_b']:.1f}°C</b> · core <b>{S['T_core']:.1f}°C</b>"
           f"<br>limit {S['T_limit']:.0f}°C")
    b_w = (f"water {S['T_w_in']:.0f}→{S['T_w_in']+S['dT_water']:.1f}°C · "
           f"{S['flow_lpm']:.0f} L/min")
    b_u = (f"{'guided' if S['mode']=='serpentine' else 'oil'} "
           f"<b>{S['u_mm_s']:.1f} mm/s</b> · spread {S['spread']:.1f} °C")
    tpl = """
<div id="lp-root" style="font-family:Inter,-apple-system,'Segoe UI',sans-serif">
<style>
 #lp-root{position:relative;background:#0B1220;border-radius:16px;
   padding:0;overflow:hidden;box-shadow:0 1px 3px rgba(16,24,40,.12)}
 #lp-cv{width:100%;height:388px;display:block}
 #lp-bar{position:absolute;left:12px;top:10px;display:flex;gap:8px;
   align-items:center;background:rgba(15,23,42,.55);backdrop-filter:blur(8px);
   border:1px solid rgba(255,255,255,.14);border-radius:12px;padding:6px 10px}
 .lpb{border:0;background:transparent;color:#CBD5E1;font-weight:600;
   font-size:12px;padding:4px 10px;border-radius:8px;cursor:pointer;
   transition:all .15s}
 .lpb:hover{background:rgba(255,255,255,.10);color:#fff}
 .lpb.on{background:rgba(255,255,255,.16);color:#fff}
 #lp-spd{width:88px;accent-color:#8B9CF9}
 #lp-status{position:absolute;right:14px;top:14px;font-size:11px;
   color:#94A3B8}
 .lp-badge{position:absolute;background:rgba(15,23,42,.60);
   backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.14);
   border-radius:10px;padding:5px 11px;font-size:11.5px;color:#CBD5E1;
   pointer-events:none}
 .lp-badge b{font-size:13px;color:#fff}
</style>
<canvas id="lp-cv"></canvas>
<div id="lp-bar">
  <button class="lpb on" id="lp-play">&#10074;&#10074; Pause</button>
  <input type="range" id="lp-spd" min="0.2" max="4" step="0.1" value="1">
  <button class="lpb on" id="lp-oil">Oil</button>
  <button class="lpb on" id="lp-wat">Water</button>
  <button class="lpb on" id="lp-heat">Glow</button>
  <button class="lpb" id="lp-exag">Exaggerate</button>
</div>
<div id="lp-status">live</div>
<div class="lp-badge" id="bq" style="left:12px;bottom:12px">__BQ__</div>
<div class="lp-badge" id="bt" style="right:12px;top:52px;text-align:right">__BT__</div>
<div class="lp-badge" id="bw" style="right:12px;bottom:12px;text-align:right">__BW__</div>
<div class="lp-badge" id="bu" style="left:12px;bottom:52px">__BU__</div>
</div>
<script>
(function(){
"use strict";
const status = document.getElementById('lp-status');
function fail(msg){ status.textContent = 'error: ' + msg;
  status.style.color = '#FCA5A5';
  try { const c = document.getElementById('lp-cv').getContext('2d');
    c.setTransform(1,0,0,1,0,0); c.fillStyle='#FCA5A5';
    c.font='13px Inter,sans-serif'; c.fillText('Live Pack error: '+msg,14,26);
  } catch(e){} }
window.addEventListener('error', e => fail(e.message));
try {
const S = __STATE__;
const cv = document.getElementById('lp-cv');
const cx = cv.getContext('2d');
let W = 800, H = 388;
function fit(){
  W = cv.clientWidth || cv.parentElement.clientWidth || 800;
  H = cv.clientHeight || 388;
  const d = window.devicePixelRatio || 1;
  cv.width = Math.round(W * d); cv.height = Math.round(H * d);
  cx.setTransform(d, 0, 0, d, 0, 0);
}
fit();
if (window.ResizeObserver) new ResizeObserver(fit).observe(cv);

let play=true, spd=1, showOil=true, showWat=true, showHeat=true, exag=false;
function tog(id, fn, isPlay){
  const b = document.getElementById(id);
  b.addEventListener('click', function(){
    b.classList.toggle('on');
    const on = b.classList.contains('on');
    fn(on);
    if (isPlay) b.innerHTML = on ? '&#10074;&#10074; Pause' : '&#9654; Play';
  });
}
tog('lp-play', v => play = v, true);
tog('lp-oil',  v => showOil = v);
tog('lp-wat',  v => showWat = v);
tog('lp-heat', v => showHeat = v);
tog('lp-exag', v => exag = v);
document.getElementById('lp-spd').addEventListener('input',
  e => spd = parseFloat(e.target.value));

function tcol(T, a){
  const lo = exag ? S.T_w_in : 15,
        hi = exag ? Math.max(S.T_core, S.T_b + 2) : 60;
  let f = Math.min(Math.max((T - lo) / (hi - lo), 0), 1);
  const r = Math.round(80 + f * (239 - 80)),
        g = Math.round(140 + f * (68 - 140)),
        b = Math.round(235 + f * (68 - 235));
  return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
}
function rrect(x, y, w, h, r, fill, stroke){
  cx.beginPath();
  cx.moveTo(x + r, y);
  cx.arcTo(x + w, y, x + w, y + h, r);
  cx.arcTo(x + w, y + h, x, y + h, r);
  cx.arcTo(x, y + h, x, y, r);
  cx.arcTo(x, y, x + w, y, r);
  cx.closePath();
  if (fill) cx.fill();
  if (stroke) cx.stroke();
}

const NP = 150, PW = 70;
const oilP = [], watP = [];
for (let i = 0; i < NP; i++)
  oilP.push({ x: Math.random(), y: Math.random() });
for (let i = 0; i < PW; i++)
  watP.push({ x: Math.random(), lane: i % S.n_tubes_draw });

function field(px, py){
  const u = S.u_mm_s * 3.2 * spd;
  if (S.mode === 'serpentine'){
    const lane = Math.floor(py * S.n_rows_draw);
    return { vx: (lane % 2 ? -1 : 1) * u * 2.2,
             vy: Math.sin(px * 12 + lane) * 2 };
  }
  if (py < 0.16) return { vx: (px < 0.5 ? 1 : -1) * u * 1.6, vy: -1 };
  if (px < 0.14 || px > 0.86) return { vx: 0, vy: u * 1.5 };
  return { vx: Math.sin(py * 9) * 3, vy: -u * 1.4 };
}

let last = performance.now();
function frame(now){
  try {
    const dt = Math.min((now - last) / 1000, 0.05); last = now;
    const m = 22, bx = m, by = 14, bw = W - 2 * m, bh = H - 28;
    const oilTop = by + bh * (1 - S.fill_frac);
    const cTop = by + bh * (1 - S.cell_top_frac);
    const cBot = by + bh * (1 - S.cell_bot_frac);
    // background + enclosure
    cx.fillStyle = '#0B1220'; cx.fillRect(0, 0, W, H);
    cx.strokeStyle = 'rgba(255,255,255,.55)'; cx.lineWidth = 2;
    cx.strokeRect(bx - 7, by - 7, bw + 14, bh + 14);
    cx.fillStyle = '#101A2E'; cx.fillRect(bx, by, bw, bh);
    cx.fillStyle = 'rgba(245,158,11,0.14)';
    cx.fillRect(bx, oilTop, bw, by + bh - oilTop);
    cx.strokeStyle = 'rgba(245,158,11,.55)'; cx.lineWidth = 1;
    cx.beginPath(); cx.moveTo(bx, oilTop); cx.lineTo(bx + bw, oilTop);
    cx.stroke();
    // cells
    const n = S.n_rows_draw, pitch = bw / (n + 0.6),
          cw = pitch * S.d_over_p;
    for (let i = 0; i < n; i++){
      const x = bx + pitch * (0.4 + i) + (pitch - cw) / 2;
      if (showHeat){
        const gcx = x + cw / 2, gcy = (cTop + cBot) / 2;
        const gl = cx.createRadialGradient(gcx, gcy, 2, gcx, gcy, cw * 1.6);
        const a = Math.min(0.10 + S.q_kw * 0.05, 0.35);
        gl.addColorStop(0, 'rgba(239,68,68,' + a + ')');
        gl.addColorStop(1, 'rgba(239,68,68,0)');
        cx.fillStyle = gl;
        cx.fillRect(x - cw, cTop - cw, cw * 3, (cBot - cTop) + cw * 2);
      }
      cx.fillStyle = tcol(S.T_b, 0.95);
      cx.strokeStyle = 'rgba(255,255,255,.28)'; cx.lineWidth = 1;
      rrect(x, cTop, cw, cBot - cTop, 4, true, true);
      cx.fillStyle = tcol(S.T_core, 0.9);
      rrect(x + cw * 0.32, cTop + 3, cw * 0.36, (cBot - cTop) - 6, 3,
            true, false);
      if (S.mode === 'serpentine' && i < n - 1){
        cx.fillStyle = 'rgba(139,156,249,.9)';
        cx.fillRect(x + cw + (pitch - cw) / 2 - 1.2, cTop, 2.4, cBot - cTop);
      }
    }
    // tubes + water
    const tubeY = S.interstitial ? (cTop + cBot) / 2 : (by + oilTop) / 2 + 8;
    const nt = S.n_tubes_draw, tp = bw / (nt + 1);
    for (let j = 0; j < nt; j++){
      const tx = bx + tp * (j + 1);
      cx.fillStyle = 'rgba(165,180,204,.35)';
      cx.beginPath(); cx.arc(tx, tubeY, 10, 0, 6.283); cx.fill();
      cx.fillStyle = '#D97706';
      cx.beginPath(); cx.arc(tx, tubeY, 5.2, 0, 6.283); cx.fill();
    }
    if (showWat){
      for (const p of watP){
        if (play) { p.x += dt * S.flow_norm * 0.25 * spd;
                    if (p.x > 1) p.x -= 1; }
        const tx = bx + tp * (p.lane + 1);
        const ang = p.x * 6.283;
        cx.fillStyle = tcol(S.T_w_in + S.dT_water * p.x, 0.95);
        cx.beginPath();
        cx.arc(tx + Math.cos(ang) * 3.2, tubeY + Math.sin(ang) * 3.2,
               1.7, 0, 6.283);
        cx.fill();
      }
    }
    // oil particles
    if (showOil){
      const topFrac = (oilTop - by) / bh;
      for (const p of oilP){
        if (play){
          const v = field(p.x, p.y);
          p.x += v.vx * dt / bw; p.y += v.vy * dt / bh;
          if (p.x < 0) p.x += 1; if (p.x > 1) p.x -= 1;
          if (p.y < topFrac + 0.005) p.y = 0.985;
          if (p.y > 0.995) p.y = topFrac + 0.01;
        }
        const px = bx + p.x * bw, py = by + p.y * bh;
        if (py < oilTop) continue;
        cx.fillStyle = 'rgba(251,191,36,.75)';
        cx.beginPath(); cx.arc(px, py, 1.6, 0, 6.283); cx.fill();
      }
    }
    // weakest link tag
    const wy = S.weak === 'Water film' ? tubeY :
               S.weak.indexOf('tube') >= 0 ? tubeY + 18 : (cTop + cBot) / 2;
    cx.fillStyle = '#FCA5A5'; cx.font = '600 11px Inter,sans-serif';
    cx.fillText('weakest link: ' + S.weak, bx + bw * 0.36,
                Math.max(wy - 16, 30));
    status.textContent = play ? 'live' : 'paused';
  } catch (e){ fail(e.message); return; }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
} catch (e){ fail(e.message); }
})();
</script>
"""
    return (tpl.replace("__STATE__", s).replace("__BQ__", b_q)
               .replace("__BT__", b_t).replace("__BW__", b_w)
               .replace("__BU__", b_u))
