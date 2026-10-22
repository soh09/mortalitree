#!/usr/bin/env python3
"""
Interactive viewer for checking the NeonTreeEvaluation benchmark by eye: human crown
labels vs our LiDAR watershed crowns, on NEON's co-registered 10 cm RGB and 1 m CHM.

  RGB                | RGB + labels + watershed
  CHM hillshade      | CHM hillshade + labels + watershed

Every crown carries the outcome assigned by neontree_eval.evaluate (match, merged,
over-segmentation, poor delineation, miss / false positive), so the colours are the
benchmark's own verdicts. Click a crown to flag it; flags, a per-plot verdict and a
note are kept in the browser and exported as JSON.

    python baselines/watershed_eval/make_neontree_viewer.py OUT_DIR NAME=PRED_DIR [NAME=PRED_DIR ...] [--min-top 5]
    open OUT_DIR/index.html

Each PRED_DIR is in neontree_watershed.py's format (plots.json + predictions_{tag}.csv),
e.g. current=ws5 tuned=ws_tuned. The viewer switches between the sets; human labels
are shared, but their outcomes depend on which set they are scored against.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import neontree_eval as E

ND = np.array([0.42, 0.62, 0.86])        # CHM no-data


def hillshade(z, az=315, alt=40):
    gy, gx = np.gradient(z)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
    a, al = np.radians(az), np.radians(alt)
    return np.clip(np.sin(al) * np.sin(slope) + np.cos(al) * np.cos(slope) * np.cos(a - asp), 0, 1)


def height_colour(h):
    stops = np.array([[0.953, 0.945, 0.918], [0.624, 0.780, 0.624], [0.247, 0.498, 0.310], [0.071, 0.239, 0.141]])
    t = np.clip(h / 40, 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2); f = (t - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def tag_of(min_top):
    return f"{min_top:g}m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("sets", nargs="+", help="NAME=PRED_DIR")
    ap.add_argument("--min-top", type=float, default=5.0)
    a = ap.parse_args()
    out = Path(a.out); img = out / "img"; img.mkdir(parents=True, exist_ok=True)

    sets = {}
    for spec in a.sets:
        name, d = spec.split("=", 1)
        meta, preds, extra = E.load(Path(d), tag_of(a.min_top))
        sets[name] = (meta, preds, E.evaluate(meta, preds, extra))
    meta = next(iter(sets.values()))[0]
    recs = []
    for plot in meta:
        m = meta[plot]
        with rasterio.open(m["rgb"]) as src:
            rgb = src.read([1, 2, 3]); rtr = src.transform; H, W = src.height, src.width
        Image.fromarray(np.transpose(rgb, (1, 2, 0)).astype(np.uint8)).save(img / f"{plot}_rgb.jpg", quality=90)

        with rasterio.open(m["chm"]) as src:
            chm = src.read(1).astype(np.float32); ctr = src.transform; nod_v = src.nodata
        nod = ~np.isfinite(chm) | (chm < -1) | ((chm == nod_v) if nod_v is not None else False)
        chm = np.where(nod, 0, np.clip(chm, 0, None))
        hs = height_colour(chm) * (0.35 + 0.65 * hillshade(chm * 1.5))[..., None]; hs[nod] = ND
        scale = max(1, round(abs(ctr.a) / abs(rtr.a)))
        Image.fromarray((hs * 255).astype(np.uint8)).resize((chm.shape[1] * scale, chm.shape[0] * scale),
                                                           Image.NEAREST).save(img / f"{plot}_chm.png")
        # where the CHM sits in RGB pixel space (the grids are offset by up to 0.4 m)
        (cx0, cy0), (cx1, cy1) = ~rtr * (ctr * (0, 0)), ~rtr * (ctr * (chm.shape[1], chm.shape[0]))
        chm_frame = [round(100 * cx0 / W, 3), round(100 * cy0 / H, 3),
                     round(100 * (cx1 - cx0) / W, 3), round(100 * (cy1 - cy0) / H, 3)]

        g = E.read_gt(m["xml"])
        rnd = lambda b: [round(float(v), 1) for v in b]
        nan = lambda v: None if np.isnan(v) else round(float(v), 3)
        per = {}
        for name, (_, preds, res) in sets.items():
            r = res[plot]; pb = np.array(preds.get(plot, []), float).reshape(-1, 4)
            per[name] = {"P": nan(r["precision"]), "R": nan(r["recall"]), "gt_out": [str(o) for o in r["gl"]],
                         "pred": [rnd(b) + [str(o), round(float(h), 1)] for b, o, h in zip(pb, r["pl"], r["p_height"])]}
        recs.append({"plot": plot, "site": m["site"], "W": W, "H": H, "px": m["rgb_px_m"], "chm_frame": chm_frame,
                     "gt": [rnd(b) for b in g], "sets": per})
    recs.sort(key=lambda x: (x["site"], x["plot"]))
    (out / "data.js").write_text("const D=" + json.dumps(recs, separators=(",", ":")) + ";"
                                 + f"const MIN_TOP={a.min_top:g};const SETS={json.dumps(list(sets))};")
    (out / "index.html").write_text(HTML)
    print(f"{len(recs)} plots, {sum(len(r['gt']) for r in recs)} labels; crowns per set: "
          + ", ".join(f"{n} {sum(len(r['sets'][n]['pred']) for r in recs)}" for n in sets) + f" -> {out/'index.html'}")


HTML = r"""<!doctype html>
<meta charset="utf-8"><title>NEON crowns check</title>
<style>
:root{color-scheme:light dark;--bg:#fcfcfb;--card:#fff;--ink:#0b0b0b;--dim:#52514e;--line:#e3e2de;--acc:#2a78d6;
  --c-match:#2a78d6;--c-merged:#eb6834;--c-poor:#1baf7a;--c-split:#eda100;--c-miss:#e87ba4;--c-gt:#ffffff;--c-ws:#e87ba4}
@media(prefers-color-scheme:dark){:root{--bg:#1a1a19;--card:#242422;--ink:#fff;--dim:#c3c2b7;--line:#3a3a37;--acc:#3987e5}}
*{box-sizing:border-box}
body{margin:0;padding:20px 24px;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--dim);font-size:13px;margin-bottom:8px}
.bar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;padding:10px 0 14px;border-bottom:1px solid var(--line);margin-bottom:14px}
.bar label{color:var(--dim);font-size:13px;display:flex;gap:6px;align-items:center}
select,button,textarea{font:inherit;font-size:13px;padding:4px 9px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink)}
button{cursor:pointer}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;color:var(--dim);margin:0 0 14px}
.legend i{display:inline-block;width:13px;height:13px;border:2px solid;margin-right:5px;vertical-align:-2px;border-radius:2px}
.legend i.d{border-style:dashed}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;cursor:zoom-in}
.thumb{position:relative;aspect-ratio:1;overflow:hidden;background:#000}
.thumb img,.thumb svg{position:absolute;inset:0;width:100%;height:100%}
.meta{padding:7px 10px;font-size:11.5px;color:var(--dim);display:flex;justify-content:space-between;gap:6px}
.meta b{color:var(--ink);font-weight:600}
.v{font-weight:600;color:var(--acc)}
#lb{position:fixed;inset:0;background:#0d0d0cf2;display:none;z-index:50;color:#eee;grid-template-rows:auto 1fr auto;padding:12px 16px;gap:10px}
#lb.on{display:grid}
#lbbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:center;font-size:12.5px}
#lbbar button{background:#ffffff14;color:#eee;border-color:#ffffff33}
#lbbar button.on{background:#fff;color:#111}
#lbbar .k{color:#a9a9a4}
#main{display:flex;gap:14px;justify-content:center;align-items:center;min-height:0}
#quad{display:grid;grid-template-columns:repeat(2,var(--S));grid-template-rows:repeat(2,var(--S));gap:8px}
.pane{position:relative;overflow:hidden;background:#000;border-radius:6px;cursor:crosshair}
.pane .cap{position:absolute;left:8px;top:6px;z-index:5;font-size:12px;color:#fff;text-shadow:0 1px 3px #000;pointer-events:none}
.world{position:absolute;inset:0;transform-origin:0 0}
.world img{position:absolute;image-rendering:pixelated;user-select:none;-webkit-user-drag:none}
.world svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
svg rect{fill:none}
.world svg rect{fill:transparent;vector-effect:non-scaling-stroke;cursor:pointer}
.world svg rect.flag{fill:#ffffff40}
.xh{position:absolute;pointer-events:none;z-index:4;background:#fff;opacity:.85;mix-blend-mode:difference}
#side{width:250px;font-size:12.5px;align-self:stretch;display:flex;flex-direction:column;gap:8px;overflow:auto}
#side h3{font-size:13px;margin:4px 0 0;color:#fff}
#side table{border-collapse:collapse;width:100%}
#side td{padding:1px 4px;color:#cfcfca}#side td:last-child{text-align:right;color:#fff}
#side .vb{display:flex;flex-wrap:wrap;gap:6px}
#side .vb button{background:#ffffff14;color:#eee;border-color:#ffffff33;font-size:12px;padding:3px 8px}
#side .vb button.on{background:#fff;color:#111}
#side textarea{width:100%;height:70px;background:#ffffff10;color:#eee;border-color:#ffffff33}
#lbcap{text-align:center;font-size:12.5px;color:#a9a9a4}
#tip{position:fixed;z-index:60;pointer-events:none;background:#111;color:#eee;font-size:12px;padding:4px 7px;border-radius:5px;display:none}
</style>
<h1>NeonTreeEvaluation — human crowns vs LiDAR watershed</h1>
<div class="sub" id="hdr"></div>
<div class="bar">
  <label>watershed <select id="set"></select></label>
  <label>site <select id="site"></select></label>
  <label>sort <select id="srt"><option value="plot">plot<option value="R">recall, lowest first<option value="Rd">recall, highest first<option value="n">most labels first</select></label>
  <label>show <select id="flt"><option value="all">all<option value="todo">not yet reviewed<option value="done">reviewed<option value="flagged">with flagged crowns</select></label>
  <button id="exp">Export review (JSON)</button>
</div>
<div class="legend" id="leg"></div>
<div id="grid"></div>
<div id="lb">
  <div id="lbbar">
    <button id="bPrev">‹ prev (J)</button><button id="bNext">next (K) ›</button>
    <button id="bGT" class="on">human labels (H)</button><button id="bWS" class="on">watershed (W)</button>
    <button id="bCol" class="on">colour by outcome (C)</button>
    <span class="k">watershed set:</span><span id="setbtns"></span>
    <button id="bFit">fit view (F)</button><button id="bClose">close (Esc)</button>
  </div>
  <div id="main"><div id="quad"></div><div id="side"></div></div>
  <div id="lbcap"></div>
</div>
<div id="tip"></div>
<script src="data.js"></script>
<script>
const $=id=>document.getElementById(id);
const SKEY='neontree_review_v2';
const store={get(){try{return JSON.parse(localStorage.getItem(SKEY)||'{}')}catch(e){return{}}},set(v){try{localStorage.setItem(SKEY,JSON.stringify(v))}catch(e){}}};
let REV=store.get(), CUR=[], SET=SETS[0];
$('set').innerHTML=SETS.map(n=>`<option>${n}</option>`).join('');
const GRP={match:'match',merged:'merged',merge:'merged',poor:'poor',over_seg:'split',fragment:'split',
           'miss_<3m':'miss','miss_3-5m':'miss','miss_>5m':'miss',false_pos:'miss'};
const NAMEGT={match:'matched',merged:'merged into a bigger watershed crown',over_seg:'split into several watershed crowns',poor:'poorly delineated (IoU 0.1–0.4)',
  'miss_<3m':'missed (under 3 m tall)','miss_3-5m':'missed (3–5 m tall)','miss_>5m':'missed (over 5 m tall)'};
const NAMEWS={match:'matched',merge:'merges several labels',fragment:'fragment of one label',poor:'poorly delineated',false_pos:'false positive'};
const col=(o,src)=>SHOW.col?`var(--c-${GRP[o]})`:(src=='gt'?'var(--c-gt)':'var(--c-ws)');
let SHOW={gt:true,ws:true,col:true};
// legend
$('leg').innerHTML=[['match','matched'],['merged','merged (under-segmented)'],['split','split (over-segmented)'],['poor','poorly delineated'],['miss','missed / false positive']]
  .map(([k,t])=>`<span><i style="border-color:var(--c-${k})"></i>${t}</span>`).join('')+
  `<span><i style="border-color:var(--dim)"></i>human label (solid)</span><span><i class="d" style="border-color:var(--dim)"></i>watershed crown (dashed)</span>`;
// sites
const sites=[...new Set(D.map(d=>d.site))].sort();
$('site').innerHTML=`<option value="CA">California (SJER, SOAP, TEAK)</option><option value="all">all sites</option>`+sites.map(s=>`<option>${s}</option>`).join('');
const rev=p=>REV[p]||{verdict:'',note:'',flags:{}};
const nflags=p=>Object.values(rev(p).flags).filter(Boolean).length;
function boxes(d,big){
  const W=d.W,H=d.H,sw=big?2:1.4; let s=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
  const fl=rev(d.plot).flags, st=d.sets[SET];
  if(SHOW.ws) st.pred.forEach((b,i)=>{s+=`<rect data-s="ws" data-i="${i}" class="${fl['ws:'+SET+':'+i]?'flag':''}" x="${b[0]}" y="${b[1]}" width="${b[2]-b[0]}" height="${b[3]-b[1]}" stroke="${col(b[4],'ws')}" stroke-width="${sw}" stroke-dasharray="5 3"/>`});
  if(SHOW.gt) d.gt.forEach((b,i)=>{s+=`<rect data-s="gt" data-i="${i}" class="${fl['gt'+i]?'flag':''}" x="${b[0]}" y="${b[1]}" width="${b[2]-b[0]}" height="${b[3]-b[1]}" stroke="${col(st.gt_out[i],'gt')}" stroke-width="${sw}"/>`});
  return s+'</svg>';
}
const f3=v=>v==null?'–':v.toFixed(2);
function draw(){
  const s=$('site').value,o=$('srt').value,f=$('flt').value;
  CUR=D.filter(d=>(s=='all'||(s=='CA'?['SJER','SOAP','TEAK'].includes(d.site):d.site==s))&&
    (f=='all'||(f=='todo'?!rev(d.plot).verdict:f=='done'?!!rev(d.plot).verdict:nflags(d.plot)>0)));
  const R=d=>d.sets[SET].R;
  CUR.sort((a,b)=>o=='R'?(R(a)??2)-(R(b)??2):o=='Rd'?(R(b)??-1)-(R(a)??-1):o=='n'?b.gt.length-a.gt.length:a.plot.localeCompare(b.plot));
  const nr=D.filter(d=>rev(d.plot).verdict).length;
  $('hdr').textContent=`${CUR.length} of ${D.length} plots · 40 m × 40 m each · showing watershed set "${SET}" · ${nr} reviewed · click a plot to inspect`;
  $('grid').innerHTML=CUR.map(d=>`<div class="card" data-p="${d.plot}"><div class="thumb"><img src="img/${d.plot}_rgb.jpg" loading="lazy">${boxes(d,false)}</div>
    <div class="meta"><span><b>${d.plot}</b><br>${d.gt.length} labels · ${d.sets[SET].pred.length} watershed · P ${f3(d.sets[SET].P)} / R ${f3(d.sets[SET].R)}</span>
    <span>${rev(d.plot).verdict?`<span class="v">${rev(d.plot).verdict}</span>`:''}${nflags(d.plot)?`<br>${nflags(d.plot)} flagged`:''}</span></div></div>`).join('');
}
['site','srt','flt'].forEach(i=>$(i).addEventListener('input',draw));
$('set').addEventListener('input',e=>{SET=e.target.value;draw()});
$('grid').addEventListener('click',e=>{const c=e.target.closest('.card');if(c)open(CUR.findIndex(d=>d.plot==c.dataset.p))});

// ---- lightbox ----
let I=-1,V={s:1,tx:0,ty:0},S=0;
function layout(){S=Math.floor(Math.min((innerHeight-130)/2,(innerWidth-330)/2));$('quad').style.setProperty('--S',S+'px')}
function chmImg(d){const f=d.chm_frame;return `<img src="img/${d.plot}_chm.png" style="left:${f[0]}%;top:${f[1]}%;width:${f[2]}%;height:${f[3]}%">`}
function rgbImg(d){return `<img src="img/${d.plot}_rgb.jpg" style="left:0;top:0;width:100%;height:100%">`}
function open(i){if(i<0)return;I=i;V={s:1,tx:0,ty:0};layout();build();$('lb').classList.add('on')}
function counts(list,names){const c={};list.forEach(o=>c[o]=(c[o]||0)+1);
  return Object.keys(names).filter(k=>c[k]).map(k=>`<tr><td><i style="display:inline-block;width:9px;height:9px;background:var(--c-${GRP[k]});margin-right:5px"></i>${names[k]}</td><td>${c[k]}</td></tr>`).join('')}
function build(){
  const d=CUR[I],r=rev(d.plot),st=d.sets[SET];
  $('setbtns').innerHTML=SETS.map((n,k)=>`<button data-set="${n}" class="${n==SET?'on':''}">${n} (${k+1})</button>`).join(' ');
  $('setbtns').querySelectorAll('[data-set]').forEach(b=>b.onclick=()=>{SET=b.dataset.set;$('set').value=SET;build()});
  const P=[['RGB (10 cm)',rgbImg(d),false],['RGB + crowns',rgbImg(d),true],['CHM hillshade (1 m)',chmImg(d),false],['CHM hillshade + crowns',chmImg(d),true]];
  $('quad').innerHTML=P.map(([cap,im,lab])=>`<div class="pane"><div class="cap">${cap}</div><div class="world">${im}${lab?boxes(d,true):''}</div><div class="xh h"></div><div class="xh v"></div></div>`).join('');
  const pr=SETS.map(n=>`${n}: P ${f3(d.sets[n].P)} / R ${f3(d.sets[n].R)} · ${d.sets[n].pred.length} crowns`).join('<br>');
  $('side').innerHTML=`<h3>${d.plot}</h3><div>${d.site}</div><div>${pr}</div>
    <h3>Human labels (${d.gt.length}) vs "${SET}"</h3><table>${counts(st.gt_out,NAMEGT)}</table>
    <h3>Watershed "${SET}" (${st.pred.length})</h3><table>${counts(st.pred.map(b=>b[4]),NAMEWS)}</table>
    <h3>Your verdict</h3><div class="vb">${['benchmark looks right','labels look wrong','watershed worse than shown','unsure'].map(v=>`<button data-v="${v}" class="${r.verdict==v?'on':''}">${v}</button>`).join('')}</div>
    <h3>Note</h3><textarea id="note" placeholder="optional">${r.note||''}</textarea>
    <div style="color:#a9a9a4">Click a crown to flag it (${nflags(d.plot)} flagged). Hover for its outcome.</div>`;
  $('side').querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>{const x=rev(d.plot);x.verdict=x.verdict==b.dataset.v?'':b.dataset.v;REV[d.plot]=x;store.set(REV);build()});
  $('note').oninput=e=>{const x=rev(d.plot);x.note=e.target.value;REV[d.plot]=x;store.set(REV)};
  apply();
  $('lbcap').textContent=`${I+1}/${CUR.length} · scroll = zoom · drag = pan · hover = linked crosshair · click crown = flag · 1/2/3 = switch watershed set · 1 RGB px = ${(+d.px).toFixed(2)} m`;
}
function apply(){document.querySelectorAll('.world').forEach(w=>w.style.transform=`translate(${V.tx}px,${V.ty}px) scale(${V.s})`)}
function toFrame(p,e){const r=p.getBoundingClientRect();return[((e.clientX-r.left)-V.tx)/(S*V.s),((e.clientY-r.top)-V.ty)/(S*V.s)]}
function crosshair(f){document.querySelectorAll('.pane').forEach(p=>{const x=V.tx+f[0]*S*V.s,y=V.ty+f[1]*S*V.s;
  p.querySelector('.h').style.cssText=`left:0;right:0;top:${y}px;height:1px`;p.querySelector('.v').style.cssText=`top:0;bottom:0;left:${x}px;width:1px`})}
function clamp(){const m=S*V.s;V.tx=Math.min(0,Math.max(S-m,V.tx));V.ty=Math.min(0,Math.max(S-m,V.ty))}
let drag=null,moved=false;
$('quad').addEventListener('wheel',e=>{const p=e.target.closest('.pane');if(!p)return;e.preventDefault();
  const r=p.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,k=Math.exp(-e.deltaY*0.0015);
  const ns=Math.min(16,Math.max(1,V.s*k)),kk=ns/V.s;V.tx=mx-(mx-V.tx)*kk;V.ty=my-(my-V.ty)*kk;V.s=ns;clamp();apply();crosshair(toFrame(p,e))},{passive:false});
$('quad').addEventListener('mousedown',e=>{if(e.target.closest('.pane')){drag=[e.clientX,e.clientY,V.tx,V.ty];moved=false;e.preventDefault()}});
addEventListener('mouseup',e=>{
  if(drag&&!moved){const rc=e.target.closest&&e.target.closest('rect');
    if(rc){const d=CUR[I],x=rev(d.plot),k=rc.dataset.s=='gt'?'gt'+rc.dataset.i:'ws:'+SET+':'+rc.dataset.i;x.flags[k]=!x.flags[k];REV[d.plot]=x;store.set(REV);build()}}
  drag=null});
$('quad').addEventListener('mousemove',e=>{const p=e.target.closest('.pane');if(!p)return;
  if(drag){const dx=e.clientX-drag[0],dy=e.clientY-drag[1];if(Math.abs(dx)+Math.abs(dy)>3)moved=true;V.tx=drag[2]+dx;V.ty=drag[3]+dy;clamp();apply()}
  crosshair(toFrame(p,e));
  const rc=e.target.closest('rect'),tip=$('tip');
  if(rc&&!drag){const d=CUR[I],st=d.sets[SET],i=+rc.dataset.i,b=rc.dataset.s=='gt'?d.gt[i]:st.pred[i];
    const name=rc.dataset.s=='gt'?`human label: ${NAMEGT[st.gt_out[i]]} (vs "${SET}")`:`watershed "${SET}" crown: ${NAMEWS[b[4]]} · top ${b[5]} m`;
    const w=((b[2]-b[0])*d.px).toFixed(1),h=((b[3]-b[1])*d.px).toFixed(1);
    tip.textContent=`${name} · ${w} × ${h} m`;tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+12)+'px'}
  else tip.style.display='none'});
$('quad').addEventListener('mouseleave',()=>$('tip').style.display='none');
function step(k){if(CUR.length)open((I+k+CUR.length)%CUR.length)}
function toggle(w){SHOW[w]=!SHOW[w];$({gt:'bGT',ws:'bWS',col:'bCol'}[w]).classList.toggle('on',SHOW[w]);build()}
function close(){$('lb').classList.remove('on');$('tip').style.display='none';draw()}
$('bPrev').onclick=()=>step(-1);$('bNext').onclick=()=>step(1);$('bClose').onclick=close;
$('bGT').onclick=()=>toggle('gt');$('bWS').onclick=()=>toggle('ws');$('bCol').onclick=()=>toggle('col');
$('bFit').onclick=()=>{V={s:1,tx:0,ty:0};apply()};
addEventListener('keydown',e=>{if(!$('lb').classList.contains('on')||e.target.tagName=='TEXTAREA')return;const l=e.key.toLowerCase();
  if(e.key=='Escape')close();if(l=='j')step(-1);if(l=='k')step(1);if(l=='h')toggle('gt');if(l=='w')toggle('ws');if(l=='c')toggle('col');
  if(l=='f'){V={s:1,tx:0,ty:0};apply()}
  const n=parseInt(e.key);if(n>=1&&n<=SETS.length){SET=SETS[n-1];$('set').value=SET;build()}});
addEventListener('resize',()=>{if($('lb').classList.contains('on')){layout();clamp();apply()}});
$('exp').onclick=()=>{const rows=Object.entries(REV).filter(([p,v])=>v.verdict||v.note||Object.values(v.flags).some(Boolean)).map(([p,v])=>{
    const d=D.find(x=>x.plot==p)||{gt:[],pred:[]};
    const flagged=Object.entries(v.flags).filter(([,on])=>on).map(([k])=>{
      if(k.startsWith('gt')){const i=+k.slice(2);return{source:'label',index:i,box_px:(d.gt[i]||[]),
        outcome_by_set:Object.fromEntries(SETS.map(n=>[n,d.sets?.[n]?.gt_out[i]]))}}
      const [,set,ix]=k.split(':'),b=d.sets?.[set]?.pred[+ix]||[];return{source:'watershed',set,index:+ix,box_px:b.slice(0,4),outcome:b[4]}});
    return{plot:p,site:d.site,verdict:v.verdict,note:v.note,flagged}});
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:'application/json'}));a.download='neontree_review.json';a.click()};
draw();
</script>
"""

if __name__ == "__main__":
    main()
