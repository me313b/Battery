"""Live Pack: a client-side animated cross-section of the solved design.

Pure HTML5 canvas + vanilla JS, seeded from the solver state. All animation
controls (play, speed, layers, exaggerate) run in the browser - no Streamlit
rerun, so it stays silky. Colours encode the real solved temperatures.
"""
import json


def live_pack_html(state: dict) -> str:
    s = json.dumps(state)
    return """
<div id="lp-root" style="font-family:Inter,-apple-system,'Segoe UI',sans-serif;">
<style>
  #lp-root{position:relative;background:#fff;border:1px solid #E7EAF0;
    border-radius:16px;box-shadow:0 1px 3px rgba(16,24,40,.05);padding:10px}
  #lp-bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;
    padding:2px 6px 8px 6px;font-size:12.5px;color:#475569}
  #lp-bar b{color:#0F172A}
  .lp-btn{border:1px solid #E7EAF0;border-radius:8px;background:#fff;
    padding:3px 12px;cursor:pointer;font-weight:600;font-size:12px;
    transition:all .15s}
  .lp-btn:hover{background:#F1F5F9}
  .lp-btn.on{background:linear-gradient(120deg,#6366F1,#06B6D4);color:#fff;
    border-color:transparent}
  #lp-cv{width:100%;display:block;border-radius:10px}
  .lp-badge{position:absolute;background:rgba(255,255,255,.92);
    border:1px solid #E7EAF0;border-radius:10px;padding:4px 10px;
    font-size:11.5px;color:#334155;box-shadow:0 1px 4px rgba(16,24,40,.08);
    pointer-events:none}
  .lp-badge b{font-size:13px;color:#0F172A}
  input[type=range]{accent-color:#6366F1;vertical-align:middle}
</style>
<div id="lp-bar">
  <button class="lp-btn on" id="lp-play">Pause</button>
  <span>speed <input type="range" id="lp-spd" min="0.2" max="4" step="0.1" value="1"></span>
  <button class="lp-btn on" id="lp-oil">Oil flow</button>
  <button class="lp-btn on" id="lp-wat">Water</button>
  <button class="lp-btn on" id="lp-heat">Heat glow</button>
  <button class="lp-btn" id="lp-exag">Exaggerate ΔT</button>
  <span style="margin-left:auto;color:#64748B">live from the solver - drag nothing, it moves itself</span>
</div>
<div style="position:relative">
  <canvas id="lp-cv" height="360"></canvas>
  <div class="lp-badge" id="bq"  style="left:12px;top:10px"></div>
  <div class="lp-badge" id="bt"  style="right:12px;top:10px;text-align:right"></div>
  <div class="lp-badge" id="bw"  style="right:12px;bottom:12px;text-align:right"></div>
  <div class="lp-badge" id="bu"  style="left:12px;bottom:12px"></div>
</div>
</div>
<script>
const S = __STATE__;
const cv = document.getElementById('lp-cv'), cx = cv.getContext('2d');
function fit(){ cv.width = cv.parentElement.clientWidth * devicePixelRatio;
  cv.height = 360 * devicePixelRatio; cx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);}
fit(); addEventListener('resize', fit);

let play=true, spd=1, showOil=true, showWat=true, showHeat=true, exag=false;
const tog=(id,fn)=>{const b=document.getElementById(id);
  b.onclick=()=>{b.classList.toggle('on'); fn(b.classList.contains('on'));
    if(id==='lp-play') b.textContent=play?'Pause':'Play';};};
tog('lp-play', v=>play=v); tog('lp-oil', v=>showOil=v);
tog('lp-wat', v=>showWat=v); tog('lp-heat', v=>showHeat=v);
tog('lp-exag', v=>exag=v);
document.getElementById('lp-spd').oninput=e=>spd=+e.target.value;

// temperature -> colour (blue 15C .. red 60C, exaggerate narrows the band)
function tcol(T,a=1){
  const lo = exag? S.T_w_in : 15, hi = exag? Math.max(S.T_core,S.T_b+2) : 60;
  let f=Math.min(Math.max((T-lo)/(hi-lo),0),1);
  const r=Math.round(59+f*(239-59)), g=Math.round(130+f*(68-130)),
        b=Math.round(246+f*(68-246));
  return `rgba(${r},${g},${b},${a})`;
}

// geometry in canvas space (side section, y-z)
function G(){
  const W=cv.clientWidth, H=340, m=26;
  const box={x:m,y:16,w:W-2*m,h:H-16};
  const oilTop=box.y+box.h*(1-S.fill_frac);
  const cellTop=box.y+box.h*(1-S.cell_top_frac),
        cellBot=box.y+box.h*(1-S.cell_bot_frac);
  return {W,H,box,oilTop,cellTop,cellBot};
}

// particles
const NP=150, PW=70;
let oilP=[...Array(NP)].map(()=>({x:Math.random(),y:Math.random(),ph:Math.random()*6.28}));
let watP=[...Array(PW)].map((_,i)=>({x:Math.random(),lane:i%S.n_tubes_draw}));

function oilField(px,py,g){ // px,py in [0,1] within box; returns vx,vy (px/s)
  const u=S.u_mm_s*3.2*spd;               // visual gain
  if(S.mode==='serpentine'){
    const lane=Math.floor(py*S.n_rows_draw);
    const dir=(lane%2? -1:1);
    return {vx:dir*u*2.2, vy:Math.sin(px*12+lane)*2};
  }
  // thermosiphon / stirred loop: up in middle band, across top, down at walls
  const cx0=0.5, r=Math.hypot(px-cx0, py-0.55);
  const upzone = px>0.16 && px<0.84 && py>0.18;
  if(py<0.16) return {vx:(px<0.5?-1:1)*-u*1.6, vy:-1};
  if(px<0.14||px>0.86) return {vx:0, vy:u*1.5};
  return {vx:Math.sin(py*9)*3, vy:-u*(upzone?1.6:0.6)};
}

let last=performance.now();
function frame(now){
  const dt=Math.min((now-last)/1000,0.05); last=now;
  const g=G(); const {box,oilTop,cellTop,cellBot}=g;
  cx.clearRect(0,0,g.W,360);

  // enclosure + headspace + oil
  cx.fillStyle='#F8FAFC'; cx.fillRect(box.x-8,box.y-8,box.w+16,box.h+16);
  cx.strokeStyle='#0F172A'; cx.lineWidth=2;
  cx.strokeRect(box.x-8,box.y-8,box.w+16,box.h+16);
  cx.fillStyle='#fff'; cx.fillRect(box.x,box.y,box.w,box.h);
  cx.fillStyle='rgba(245,158,11,0.13)';
  cx.fillRect(box.x,oilTop,box.w,box.y+box.h-oilTop);
  cx.strokeStyle='rgba(180,83,9,.5)'; cx.lineWidth=1;
  cx.beginPath(); cx.moveTo(box.x,oilTop); cx.lineTo(box.x+box.w,oilTop); cx.stroke();

  // cells (rows across width) with heat glow + true colour
  const n=S.n_rows_draw, pitch=box.w/(n+0.6), cw=pitch*(S.d_over_p);
  for(let i=0;i<n;i++){
    const x=box.x+pitch*(0.4+i)+ (pitch-cw)/2;
    if(showHeat){
      const gl=cx.createRadialGradient(x+cw/2,(cellTop+cellBot)/2,2,
                x+cw/2,(cellTop+cellBot)/2,cw*1.5);
      const a=Math.min(0.10+S.q_kw*0.05,0.35);
      gl.addColorStop(0,`rgba(239,68,68,${a})`); gl.addColorStop(1,'rgba(239,68,68,0)');
      cx.fillStyle=gl; cx.fillRect(x-cw,cellTop-cw,cw*3,(cellBot-cellTop)+cw*2);
    }
    cx.fillStyle=tcol(S.T_b,0.95);
    cx.strokeStyle='rgba(15,23,42,.25)';
    roundRect(x,cellTop,cw,cellBot-cellTop,4,true,true);
    cx.fillStyle=tcol(S.T_core,0.9);
    roundRect(x+cw*0.32,cellTop+3,cw*0.36,(cellBot-cellTop)-6,3,true,false);
    // serpentine plates
    if(S.mode==='serpentine' && i<n-1){
      cx.fillStyle='rgba(99,102,241,.85)';
      cx.fillRect(x+cw+ (pitch-cw)/2-1.2, cellTop, 2.4, cellBot-cellTop);
    }
  }

  // tube band with fins + water particles
  const tubeY = S.interstitial ? (cellTop+cellBot)/2 : (box.y+oilTop)/2 + 8;
  const nt=S.n_tubes_draw, tp=box.w/(nt+1);
  for(let j=0;j<nt;j++){
    const tx=box.x+tp*(j+1);
    cx.fillStyle='rgba(165,180,204,.45)';
    cx.beginPath(); cx.arc(tx,tubeY,10,0,6.28); cx.fill();
    cx.fillStyle='#D97706';
    cx.beginPath(); cx.arc(tx,tubeY,5.2,0,6.28); cx.fill();
  }
  if(showWat){
    watP.forEach(p=>{
      p.x+= dt*S.flow_norm*0.25*spd; if(p.x>1)p.x-=1;
      const tx=box.x+tp*(p.lane+1);
      const ang=p.x*6.283, rr=3.2;
      const Tw=S.T_w_in + S.dT_water*p.x;
      cx.fillStyle=tcol(Tw,0.95);
      cx.beginPath();
      cx.arc(tx+Math.cos(ang)*rr, tubeY+Math.sin(ang)*rr,1.6,0,6.28); cx.fill();
    });
  }

  // oil particles
  if(showOil && play){
    oilP.forEach(p=>{
      const v=oilField(p.x,p.y,g);
      p.x+= v.vx*dt/box.w; p.y+= v.vy*dt/box.h;
      if(p.x<0)p.x+=1; if(p.x>1)p.x-=1;
      if(p.y<(box.y? (oilTop-box.y)/box.h :0)) p.y=0.98;
      if(p.y>0.99)p.y=(oilTop-box.y)/box.h+0.01;
    });
  }
  if(showOil){
    oilP.forEach(p=>{
      const px=box.x+p.x*box.w, py=box.y+p.y*box.h;
      if(py<oilTop) return;
      cx.fillStyle='rgba(180,83,9,.55)';
      cx.beginPath(); cx.arc(px,py,1.5,0,6.28); cx.fill();
    });
  }

  // weakest-link tag
  const wy = S.weak==='Water film' ? tubeY : S.weak==='Oil to tube+fins' ?
             tubeY+18 : (cellTop+cellBot)/2;
  cx.fillStyle='#B91C1C'; cx.font='600 11px Inter';
  cx.fillText('weakest link: '+S.weak, box.x+box.w*0.36, Math.max(wy-16,26));

  // badges
  bq.innerHTML=`<b>${S.q_kw.toFixed(2)} kW</b> at ${S.c_rms.toFixed(2)}C rms`;
  bt.innerHTML=`can <b>${S.T_b.toFixed(1)}°C</b> · core <b>${S.T_core.toFixed(1)}°C</b><br>limit ${S.T_limit.toFixed(0)}°C`;
  bw.innerHTML=`water ${S.T_w_in.toFixed(0)}→${(S.T_w_in+S.dT_water).toFixed(1)}°C · ${S.flow_lpm.toFixed(0)} L/min`;
  bu.innerHTML=`${S.mode==='serpentine'?'guided':'oil'} <b>${S.u_mm_s.toFixed(1)} mm/s</b> · spread ${S.spread.toFixed(1)} K`;

  if(play) requestAnimationFrame(frame); else setTimeout(()=>requestAnimationFrame(frame),120);
}
function roundRect(x,y,w,h,r,f,s){cx.beginPath();
  cx.moveTo(x+r,y);cx.arcTo(x+w,y,x+w,y+h,r);cx.arcTo(x+w,y+h,x,y+h,r);
  cx.arcTo(x,y+h,x,y,r);cx.arcTo(x,y,x+w,y,r);cx.closePath();
  if(f)cx.fill(); if(s)cx.stroke();}
const bq=document.getElementById('bq'), bt=document.getElementById('bt'),
      bw=document.getElementById('bw'), bu=document.getElementById('bu');
requestAnimationFrame(frame);
</script>
""".replace("__STATE__", s)
