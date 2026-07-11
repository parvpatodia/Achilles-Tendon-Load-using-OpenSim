"""Self-contained HTML dashboard for the real-time streaming demo.

Bakes the per-stride results into one standalone .html (no server, no network,
no build step) so it opens on any laptop or projector and cannot fail on stage.
The animation replays the strides: the lab reference, the uncalibrated
surrogate, and the calibrated surrogate draw across each gait cycle, with live
peak-load, asymmetry, calibration-status, and inference-latency readouts.

Kept separate from the matplotlib figures in plots.py: this module only renders
HTML from an already-computed result list.
"""
from __future__ import annotations

import json
from pathlib import Path

_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mirai | Live Achilles tendon load</title>
<style>
  :root{--bg:#0c2019;--panel:#12312a;--ink:#eef7f2;--muted:#9fb8ad;--teal:#2ee0a4;
        --blue:#38b6f0;--coral:#ff7a5c;--line:#22463b;}
  *{box-sizing:border-box}
  body{margin:0;background:radial-gradient(1100px 560px at 18% -12%,#14352b 0%,var(--bg) 62%);
       color:var(--ink);font:15px/1.5 -apple-system,Segoe UI,Roboto,Inter,sans-serif}
  .wrap{max-width:1040px;margin:0 auto;padding:22px 20px 40px}
  header{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
  .brand{font-weight:800;letter-spacing:1px;color:var(--teal)}
  h1{font-size:20px;margin:0;font-weight:700}
  .sub{color:var(--muted);font-size:13px;margin:6px 0 18px}
  .grid{display:grid;grid-template-columns:1fr 296px;gap:18px}
  @media(max-width:820px){.grid{grid-template-columns:1fr}}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}
  .ct{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px}
  svg{width:100%;height:auto;display:block}
  .metric{margin-bottom:15px}
  .lab{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px}
  .big{font-size:40px;font-weight:800;color:var(--teal);line-height:1.05}
  .unit{font-size:14px;color:var(--muted);font-weight:600}
  .asi-bar{height:10px;border-radius:6px;background:#0e261f;position:relative;margin-top:7px;overflow:hidden}
  .asi-mid{position:absolute;left:50%;top:0;bottom:0;width:1px;background:#3c5f54}
  .asi-fill{position:absolute;top:0;bottom:0;background:var(--blue)}
  .badge{display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:700;margin-top:3px}
  .cal{background:rgba(46,224,164,.16);color:var(--teal);border:1px solid rgba(46,224,164,.4)}
  .warm{background:rgba(255,122,92,.16);color:var(--coral);border:1px solid rgba(255,122,92,.4)}
  .legend{display:flex;gap:16px;font-size:12px;color:var(--muted);margin-top:10px;flex-wrap:wrap}
  .legend i{display:inline-block;width:16px;border-top:3px solid;vertical-align:middle;margin-right:6px}
  .row{display:flex;justify-content:space-between;font-size:13px;color:var(--muted);margin:4px 0}
  .row b{color:var(--ink)}
  .foot{color:var(--muted);font-size:12px;margin-top:16px;border-top:1px solid var(--line);padding-top:12px}
  button{background:var(--teal);color:#04140e;border:0;border-radius:8px;padding:8px 15px;font-weight:700;cursor:pointer;margin-top:12px}
</style></head>
<body><div class="wrap">
  <header><span class="brand">MIRAI</span><h1>Live Achilles tendon load</h1></header>
  <div class="sub" id="sub"></div>
  <div class="grid">
    <div class="card">
      <div class="ct">Achilles tendon force &middot; this stride</div>
      <svg id="chart" viewBox="0 0 700 360" preserveAspectRatio="xMidYMid meet"></svg>
      <div class="legend">
        <span><i style="border-color:#c9d9d2"></i>lab reference</span>
        <span><i style="border-color:var(--coral);border-top-style:dashed"></i>surrogate, uncalibrated</span>
        <span><i style="border-color:var(--teal)"></i>surrogate, calibrated</span>
      </div>
    </div>
    <div class="card">
      <div class="metric"><div class="lab">Peak tendon load</div>
        <div><span class="big" id="peak">--</span> <span class="unit">body weights</span></div></div>
      <div class="metric"><div class="lab">Left / right asymmetry</div>
        <div id="asi" style="font-weight:700">--</div>
        <div class="asi-bar"><div class="asi-mid"></div><div class="asi-fill" id="asifill"></div></div></div>
      <div class="metric"><div class="lab">Calibration</div><div id="cal"></div></div>
      <div class="metric"><div class="lab">Inference speed</div><div id="lat" style="font-weight:700"></div></div>
      <div class="row"><span>Stride</span><b id="scount"></b></div>
      <div class="row"><span>Running speed</span><b id="spd"></b></div>
      <button id="btn">Replay</button>
    </div>
  </div>
  <div class="foot" id="disc"></div>
</div>
<script>
const DATA = __DATA__;
const S=DATA.strides, PH=DATA.phase, N=PH.length;
document.getElementById('sub').textContent=DATA.subtitle;
document.getElementById('disc').textContent=DATA.disclaimer;
const svg=document.getElementById('chart'), NS='http://www.w3.org/2000/svg';
const W=700,H=360,PL=54,PR=16,PT=14,PB=42;
let YMAX=2; S.forEach(s=>[s.t,s.raw,s.p].forEach(c=>c.forEach(v=>{if(v>YMAX)YMAX=v})));
YMAX=Math.ceil(YMAX*1.1);
const X=p=>PL+p/100*(W-PL-PR), Yf=f=>H-PB-f/YMAX*(H-PT-PB);
const mk=(n,a,t)=>{const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);if(t!=null)e.textContent=t;return e;};
function poly(arr,n,attr){let d='';for(let i=0;i<n;i++){d+=(i?'L':'M')+X(PH[i]).toFixed(1)+' '+Yf(arr[i]).toFixed(1)+' ';}return mk('path',Object.assign({d,fill:'none','stroke-linejoin':'round'},attr));}
// static grid
for(let f=0;f<=YMAX;f++){const y=Yf(f);svg.appendChild(mk('line',{x1:PL,y1:y,x2:W-PR,y2:y,stroke:'#1c3a31','stroke-width':f?0.6:1.3}));svg.appendChild(mk('text',{x:PL-8,y:y+4,fill:'#7f978c','font-size':11,'text-anchor':'end'},f));}
[0,25,50,75,100].forEach(p=>svg.appendChild(mk('text',{x:X(p),y:H-20,fill:'#7f978c','font-size':11,'text-anchor':'middle'},p)));
svg.appendChild(mk('text',{x:(PL+W-PR)/2,y:H-4,fill:'#7f978c','font-size':11,'text-anchor':'middle'},'gait cycle (%)'));
svg.appendChild(mk('text',{x:16,y:H/2,fill:'#7f978c','font-size':11,'text-anchor':'middle',transform:'rotate(-90 16 '+(H/2)+')'},'tendon force (BW)'));
const dyn=mk('g',{}); svg.appendChild(dyn);
const $=id=>document.getElementById(id);
let si=0,k=0,hold=0,playing=true;
function panel(s){
  $('scount').textContent=(s.i+1)+' / '+S.length+'  ('+s.side+')';
  $('spd').textContent=s.speed.toFixed(1)+' m/s';
  $('lat').textContent=s.lat.toFixed(2)+' ms  (~'+Math.round(1000/s.lat).toLocaleString()+' strides/s)';
  if(s.is_cal){$('cal').innerHTML='<span class="badge warm">calibrating &middot; onboarding step '+(s.i+1)+'/'+DATA.calib_k+'</span>';}
  else{$('cal').innerHTML='<span class="badge cal">calibrated to this athlete</span>';}
  if(isFinite(s.asi)){const a=Math.max(-30,Math.min(30,s.asi));$('asi').textContent=(s.asi>=0?'+':'')+s.asi.toFixed(1)+'%  ('+(s.asi>=0?'right':'left')+' higher)';
    const f=$('asifill');const w=Math.abs(a)/30*50;f.style.width=w+'%';f.style.left=(a>=0?50:50-w)+'%';}
  else{$('asi').textContent='--';$('asifill').style.width='0%';}
}
function draw(s,upto){
  dyn.textContent='';
  dyn.appendChild(poly(s.t,N,{stroke:'#c9d9d2','stroke-width':2,opacity:.45}));
  if(s.cal)dyn.appendChild(poly(s.raw,N,{stroke:'#ff7a5c','stroke-width':1.6,opacity:.75,'stroke-dasharray':'5 4'}));
  dyn.appendChild(poly(s.p,upto,{stroke:'#2ee0a4','stroke-width':3}));
  const j=Math.min(upto,N-1),cx=X(PH[j]);
  dyn.appendChild(mk('line',{x1:cx,y1:PT,x2:cx,y2:H-PB,stroke:'#2ee0a4',opacity:.28}));
  dyn.appendChild(mk('circle',{cx,cy:Yf(s.p[j]),r:4.5,fill:'#2ee0a4'}));
  let mx=0;for(let i=0;i<=j;i++)if(s.p[i]>mx)mx=s.p[i];
  $('peak').textContent=mx.toFixed(2);
}
function frame(){
  if(!playing)return;
  const s=S[si];
  if(k===0)panel(s);
  k+=2; const done=k>=N; draw(s,Math.min(k,N));
  if(done){ if(hold++<26){requestAnimationFrame(frame);return;} hold=0;k=0;si=(si+1)%S.length; }
  requestAnimationFrame(frame);
}
$('btn').onclick=()=>{si=0;k=0;hold=0;if(!playing){playing=true;requestAnimationFrame(frame);}};
requestAnimationFrame(frame);
</script>
</body></html>
"""


def _round(arr, n=3):
    return [round(float(v), n) for v in arr]


def render_realtime_dashboard(results, summary, out_path: Path, subtitle: str | None = None) -> Path:
    """Write a standalone HTML dashboard of the streamed strides to out_path."""
    strides = [{
        "i": r.index, "side": r.side, "speed": float(r.speed_ms),
        "is_cal": bool(r.is_calibration_stride), "cal": bool(r.calibrated),
        "lat": float(r.latency_ms), "asi": (float(r.asymmetry_pct) if r.asymmetry_pct == r.asymmetry_pct else float("nan")),
        "t": _round(r.true_curve), "raw": _round(r.raw_pred_curve), "p": _round(r.pred_curve),
    } for r in results]

    sub = subtitle or (
        f"Athlete {summary.subject_id} held out of training · recommended linear surrogate "
        f"· calibrated on the first {summary.calib_k} strides · "
        f"{summary.mean_latency_ms:.2f} ms/stride (~{round(summary.strides_per_sec):,} strides/s). "
        f"Peak-load error {summary.peak_mape_uncal:.0f}% → {summary.peak_mape_cal:.0f}% after calibration."
    )
    data = {
        "phase": _round(results[0].phase, 2),
        "strides": strides,
        "calib_k": summary.calib_k,
        "subtitle": sub,
        "disclaimer": (
            "Replay of real recorded running cycles (BMClab Fukuchi 2017). The surrogate output is "
            "genuine per stride; the arrival of strides over time is simulated. The athlete is held out "
            "of training and the first strides calibrate to them (a one-time onboarding step, from the "
            "lab reference). On the Mirai insole the same engine consumes the live signal instead of a "
            "replay. Tendon load is a musculoskeletal-model estimate, not a direct measurement."
        ),
    }
    html = _TEMPLATE.replace("__DATA__", json.dumps(data))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
