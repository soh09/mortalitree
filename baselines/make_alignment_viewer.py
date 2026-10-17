#!/usr/bin/env python3
"""
Alignment viewer for CHM-NAIP pairs: NAIP vs LiDAR crowns.

  pre-fire NAIP     | pre-fire NAIP + LiDAR crowns
  post-fire NAIP    | post-fire NAIP + LiDAR crowns     (only if POST_DIR given)
  LiDAR hillshade   | LiDAR hillshade + crowns

Everything is placed in one map frame (the 256 m CHM chip), so all panels share
zoom, pan and a crosshair. Each NAIP flight has its own positional error, so the
crown labels on the pre- and post-fire panels are nudged independently (keys 1/2
choose which), and both offsets are saved per chip and exported.

Requires scipy<1.12 on this macOS.

    python baselines/make_alignment_viewer.py CHM_TAR NAIP_TAR CROWNS_JSON OUT_DIR [POST_DIR]
    open OUT_DIR/index.html

POST_DIR is the output of fetch_postfire_naip.py ({key}_post.npy + postfire.json),
already resampled onto each chip's pre-fire NAIP grid.
"""
import json, sys, tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.io import MemoryFile
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).parent))
from chm_crowns import crowns_from_chm, read_chm

UP = 2                                  # crown-outline raster at 0.5 m for thinner lines
PINK = (232, 123, 164)
ND = np.array([0.42, 0.62, 0.86])       # no LiDAR returns


def hillshade(z, az=315, alt=40):
    gy, gx = np.gradient(z)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
    a, al = np.radians(az), np.radians(alt)
    return np.clip(np.sin(al) * np.sin(slope) + np.cos(al) * np.cos(slope) * np.cos(a - asp), 0, 1)


def height_colour(h):
    stops = np.array([[0.953, 0.945, 0.918], [0.624, 0.780, 0.624], [0.247, 0.498, 0.310], [0.071, 0.239, 0.141]])
    t = np.clip(h / 50, 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2); f = (t - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def stretch(b, valid=None):
    v = b[valid] if valid is not None and valid.any() else b
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    return np.clip((b - lo) / max(hi - lo, 1), 0, 1)


def to_jpg(bands, path):
    valid = bands.any(0)
    rgb = np.stack([stretch(bands[j].astype(np.float32), valid) for j in range(3)], -1)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path, quality=88)


def main(chm_tar, naip_tar, crowns_json, out, post_dir=None):
    out = Path(out); img = out / "img"; img.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(crowns_json))
    post_meta = json.load(open(Path(post_dir) / "postfire.json")) if post_dir else {}
    ctf, ntf = tarfile.open(chm_tar), tarfile.open(naip_tar)
    cn = {Path(m.name).stem: m.name for m in ctf.getmembers() if m.name.endswith(".tif")}
    nn = {Path(m.name).stem.rsplit("_", 1)[0]: m.name for m in ntf.getmembers() if m.name.endswith(".tif")}
    recs = []
    for i, r in enumerate(meta):
        k = r["key"]
        chm, nod, ct, ccrs = read_chm(ctf, cn[k])
        with MemoryFile(ntf.extractfile(nn[k]).read()).open() as n:
            naip = n.read()[:4]; nt = n.transform; ncrs = n.crs
        if ccrs != ncrs:
            print("  skip (CRS differs)", k); continue
        sm, seg, cr = crowns_from_chm(chm)

        # hillshade, height-coloured, no-data in blue
        hs = height_colour(sm) * (0.35 + 0.65 * hillshade(sm * 1.5))[..., None]
        hs[nod] = ND
        Image.fromarray((hs * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST).save(img / f"{k}_hs.jpg", quality=88)

        to_jpg(naip, img / f"{k}_naip.jpg")
        post = None
        pf = Path(post_dir) / f"{k}_post.npy" if post_dir else None
        if pf is not None and pf.exists():
            to_jpg(np.load(pf), img / f"{k}_post.jpg")
            pm = post_meta.get(k, {})
            post = {"date": (pm.get("post_dates") or [""])[0], "cover": pm.get("post_cover")}

        # crown outlines, transparent PNG in the CHM frame
        keep = np.isin(seg, [g.label for g in cr])
        lab = np.where(keep, seg, 0).repeat(UP, 0).repeat(UP, 1)
        b = find_boundaries(lab, mode="inner")
        rgba = np.zeros(b.shape + (4,), np.uint8); rgba[b] = (*PINK, 255)
        Image.fromarray(rgba).save(img / f"{k}_crowns.png", optimize=True)

        # treetops in frame metres (x east, y down from chip top)
        tops = [g.coords[np.argmax(sm[g.coords[:, 0], g.coords[:, 1]])] for g in cr]
        tops = [[round(float(c) + 0.5, 1), round(float(r_) + 0.5, 1)] for r_, c in tops]

        # where the NAIP chip sits in the CHM frame (percent of 256 m); post uses the same grid
        X0, Y0 = ct * (0, 0); W = chm.shape[1] * abs(ct.a)
        nX0, nY0 = nt * (0, 0); nW = naip.shape[2] * abs(nt.a); nH = naip.shape[1] * abs(nt.e)
        frame = [round((nX0 - X0) / W * 100, 3), round((Y0 - nY0) / W * 100, 3),
                 round(nW / W * 100, 3), round(nH / W * 100, 3)]

        dy_m, dx_m = r["shift_m"]          # auto-estimate (pre-fire NAIP): +dx right, +dy down, metres
        recs.append({"key": k, "lidar": r["lidar_date"], "naip": r["naip_date"], "n": len(cr),
                     "forest": r["forest_frac"], "frame": frame, "auto": [dx_m, dy_m],
                     "auto_corr": r["corr_best"], "tops": tops, "inside": r.get("inside"), "post": post})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(meta)}")

    recs.sort(key=lambda x: x["n"])
    (out / "data.js").write_text("const D=" + json.dumps(recs, separators=(",", ":")) + ";")
    (out / "index.html").write_text(HTML)
    print(f"{len(recs)} pairs ({sum(r['post'] is not None for r in recs)} with post-fire NAIP) -> {out/'index.html'}")


HTML = r"""<!doctype html>
<meta charset="utf-8"><title>LiDAR–NAIP alignment</title>
<style>
:root{color-scheme:light dark;--bg:#fcfcfb;--card:#fff;--ink:#0b0b0b;--dim:#52514e;--line:#e3e2de;--acc:#2a78d6}
@media(prefers-color-scheme:dark){:root{--bg:#1a1a19;--card:#242422;--ink:#fff;--dim:#c3c2b7;--line:#3a3a37;--acc:#3987e5}}
*{box-sizing:border-box}
body{margin:0;padding:20px 24px;background:var(--bg);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--dim);font-size:13px;margin-bottom:14px}
.bar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;padding:10px 0 14px;border-bottom:1px solid var(--line);margin-bottom:18px}
.bar label{color:var(--dim);font-size:13px;display:flex;gap:6px;align-items:center}
select,button{font:inherit;font-size:13px;padding:4px 9px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink);cursor:pointer}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;cursor:zoom-in}
.thumb{position:relative;aspect-ratio:1;overflow:hidden;background:#000}
.thumb img{position:absolute;display:block}
.meta{padding:7px 10px;font-size:11.5px;color:var(--dim);display:flex;justify-content:space-between;gap:6px}
.meta b{color:var(--ink);font-weight:600}
.saved{color:var(--acc);font-weight:600}
#lb{position:fixed;inset:0;background:#0d0d0cf2;display:none;z-index:50;color:#eee;
    grid-template-rows:auto 1fr auto;padding:12px 16px;gap:10px}
#lb.on{display:grid}
#lbbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:center;font-size:12.5px}
#lbbar button{background:#ffffff14;color:#eee;border-color:#ffffff33}
#lbbar button.on{background:#fff;color:#111}
#lbbar .k{color:#a9a9a4}
#quad{display:grid;grid-template-columns:repeat(2,var(--S));grid-auto-rows:var(--S);gap:8px;justify-content:center;align-content:center}
.pane{position:relative;overflow:hidden;background:#000;border-radius:6px;cursor:crosshair;outline:2px solid transparent;outline-offset:-2px}
.pane.active{outline-color:#7fb3f0}
.pane .cap{position:absolute;left:8px;top:6px;z-index:5;font-size:12px;color:#fff;text-shadow:0 1px 3px #000;pointer-events:none}
.world{position:absolute;inset:0;transform-origin:0 0}
.world img{position:absolute;image-rendering:pixelated;user-select:none;-webkit-user-drag:none}
.world svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.lbl{position:absolute;inset:0}
.xh{position:absolute;pointer-events:none;z-index:4;background:#fff;opacity:.85;mix-blend-mode:difference}
#lbcap{text-align:center;font-size:13px;line-height:1.6}
#lbcap .s{color:#a9a9a4}
</style>
<h1>LiDAR crowns vs NAIP — alignment check</h1>
<div class="sub" id="hdr"></div>
<div class="bar">
  <label>thumbnail <select id="thm"><option value="pre">pre-fire NAIP<option value="post">post-fire NAIP<option value="hs">LiDAR</select></label>
  <label>sort <select id="srt"><option value="n">crown count<option value="nd">crown count, most first<option value="auto">auto-estimated offset<option value="key">location</select></label>
  <label>LiDAR year <select id="yr"><option value="all">all</select></label>
  <label>show <select id="flt"><option value="all">all<option value="saved">with saved offset<option value="unsaved">without<option value="inside">fully inside perimeter<option value="edge">on perimeter edge</select></label>
  <button id="exp">Export saved offsets (JSON)</button>
</div>
<div id="grid"></div>
<div id="lb">
  <div id="lbbar">
    <button id="bPrev">‹ prev (J)</button><button id="bNext">next (K) ›</button>
    <button id="bTops" class="on">treetops (T)</button><button id="bOut" class="on">outlines (O)</button>
    <span class="k">arrows move:</span><button id="bSelPre" class="on">pre-fire (1)</button><button id="bSelPost">post-fire (2)</button>
    <span class="k">pre</span><b id="offPre"></b><span class="k">post</span><b id="offPost"></b>
    <button id="bAuto">auto-estimate, pre (A)</button><button id="bReset">reset (R)</button><button id="bSave">save offsets (S)</button>
    <button id="bFit">fit view (F)</button><button id="bClose">close (Esc)</button>
  </div>
  <div id="quad"></div>
  <div id="lbcap"></div>
</div>
<script src="data.js"></script>
<script>
const $=id=>document.getElementById(id);
const SKEY='align_offsets_v2';
const store={get(){try{return JSON.parse(localStorage.getItem(SKEY)||'{}')}catch(e){return{}}},
             set(v){try{localStorage.setItem(SKEY,JSON.stringify(v))}catch(e){}}};
let SAVED=store.get(), CUR=[];
const HASPOST=D.some(d=>d.post);
if(!HASPOST){$('thm').querySelector('[value=post]').remove();$('bSelPost').style.display='none';$('offPost').previousElementSibling.style.display='none';$('offPost').style.display='none'}
[...new Set(D.map(d=>d.lidar.slice(0,4)))].sort().forEach(y=>{const o=document.createElement('option');o.value=o.textContent=y;$('yr').appendChild(o)});
const pct=v=>v+'%';
function framed(d,src){return `<img src="${src}" style="left:${pct(d.frame[0])};top:${pct(d.frame[1])};width:${pct(d.frame[2])};height:${pct(d.frame[3])}">`}
const SRC={pre:d=>framed(d,`img/${d.key}_naip.jpg`), post:d=>framed(d,`img/${d.key}_post.jpg`),
           hs:d=>`<img src="img/${d.key}_hs.jpg" style="left:0;top:0;width:100%;height:100%">`};
function draw(){
  const s=$('srt').value,y=$('yr').value,f=$('flt').value,th=$('thm').value;
  CUR=D.filter(d=>(y=='all'||d.lidar.startsWith(y))&&(f=='all'||(f=='inside'?d.inside===true:f=='edge'?d.inside===false:(f=='saved')==(d.key in SAVED))));
  CUR.sort((a,b)=>s=='n'?a.n-b.n:s=='nd'?b.n-a.n:s=='auto'?Math.hypot(...a.auto)-Math.hypot(...b.auto):a.key.localeCompare(b.key));
  $('hdr').textContent=`${CUR.length} of ${D.length} chip pairs · 256 m × 256 m each · ${Object.keys(SAVED).length} saved · click a pair to inspect`;
  $('grid').innerHTML=CUR.map(d=>`<div class="card" data-k="${d.key}"><div class="thumb">${(th=='post'&&!d.post?SRC.pre:SRC[th])(d)}
     <img src="img/${d.key}_crowns.png" style="left:0;top:0;width:100%;height:100%"></div>
     <div class="meta"><span><b>${d.n}</b> crowns · LiDAR ${d.lidar} · NAIP ${d.naip}${d.post?' / '+d.post.date:''}</span>
     ${d.inside===false?'<span>edge</span>':''}${d.key in SAVED?'<span class="saved">saved</span>':''}</div></div>`).join('');
}
['thm','srt','yr','flt'].forEach(i=>$(i).addEventListener('input',draw));
$('grid').addEventListener('click',e=>{const c=e.target.closest('.card');if(c)open(CUR.findIndex(d=>d.key==c.dataset.k))});

// ---- lightbox ----
let I=-1, V={s:1,tx:0,ty:0}, OFF={pre:[0,0],post:[0,0]}, ACT='pre', SHOW={tops:true,out:true}, S=0, ROWS=2;
function panes(d){
  const P=[[`pre-fire NAIP · ${d.naip}`,'pre',false],[`pre-fire NAIP + LiDAR crowns`,'pre',true]];
  if(d.post) P.push([`post-fire NAIP · ${d.post.date}`,'post',false],[`post-fire NAIP + LiDAR crowns`,'post',true]);
  P.push([`LiDAR hillshade · ${d.lidar}`,'hs',false],[`LiDAR hillshade + crowns`,'hs',true]);
  return P;
}
function layout(){S=Math.floor(Math.min((innerHeight-150-(ROWS-1)*8)/ROWS,(innerWidth-60)/2));$('quad').style.setProperty('--S',S+'px')}
function asOff(v){if(!v)return{pre:[0,0],post:[0,0]};if(Array.isArray(v))return{pre:v.slice(),post:[0,0]};return{pre:(v.pre||[0,0]).slice(),post:(v.post||[0,0]).slice()}}
function open(i){if(i<0)return;I=i;const d=CUR[I];OFF=asOff(SAVED[d.key]);if(!d.post)ACT='pre';V={s:1,tx:0,ty:0};ROWS=d.post?3:2;layout();build();$('lb').classList.add('on')}
function tops(d){return `<svg viewBox="0 0 256 256" style="display:${SHOW.tops?'block':'none'}">`+
  d.tops.map(p=>`<circle cx="${p[0]}" cy="${p[1]}" r="0.9" fill="#0b0b0b" stroke="#fff" stroke-width="0.35"/>`).join('')+`</svg>`}
function build(){
  const d=CUR[I];
  $('quad').innerHTML=panes(d).map(([cap,base,lab])=>`<div class="pane${lab&&base==ACT?' active':''}"><div class="cap">${cap}</div><div class="world">
     ${SRC[base](d)}
     ${lab?`<div class="lbl ${base=='hs'?'':'nudge-'+base}"><img src="img/${d.key}_crowns.png" style="left:0;top:0;width:100%;height:100%;display:${SHOW.out?'block':'none'}">${tops(d)}</div>`:''}
     </div><div class="xh h"></div><div class="xh v"></div></div>`).join('');
  apply();
  const sv=SAVED[d.key];
  $('lbcap').innerHTML=`<b>${d.key}</b> <span class="s">· LiDAR ${d.lidar} · NAIP ${d.naip}${d.post?' → '+d.post.date:''} · ${d.n} crowns · ${(d.forest*100).toFixed(0)}% of chip >5 m${d.inside===false?' · on perimeter edge':d.inside?' · inside perimeter':''} · ${I+1}/${CUR.length}</span><br>
    <span class="s">auto-estimate (pre-fire only) ${fmt(d.auto)}, corr ${d.auto_corr} — failed a random-shift control, treat as a guess</span>
    ${sv?` · <span style="color:#7fb3f0">saved</span>`:''}<br>
    <span class="s">scroll = zoom · drag = pan · 1/2 = pick which NAIP the arrows move · arrows = nudge 0.6 m (shift: 3 m) · hover = linked crosshair</span>`;
}
function fmt(o){const e=o[0],s=o[1];return `${e>=0?'+':''}${e.toFixed(1)} m E, ${-s>=0?'+':''}${(-s).toFixed(1)} m N`}
function apply(){
  document.querySelectorAll('.world').forEach(w=>w.style.transform=`translate(${V.tx}px,${V.ty}px) scale(${V.s})`);
  for(const k of ['pre','post']) document.querySelectorAll('.nudge-'+k).forEach(n=>n.style.transform=`translate(${OFF[k][0]/256*100}%,${OFF[k][1]/256*100}%)`);
  $('offPre').textContent=fmt(OFF.pre); $('offPost').textContent=fmt(OFF.post);
  $('bSelPre').classList.toggle('on',ACT=='pre'); $('bSelPost').classList.toggle('on',ACT=='post');
}
function select(k){if(k=='post'&&!CUR[I].post)return;ACT=k;build()}
function toFrame(pane,e){const r=pane.getBoundingClientRect();return[((e.clientX-r.left)-V.tx)/(S*V.s),((e.clientY-r.top)-V.ty)/(S*V.s)]}
function crosshair(f){
  document.querySelectorAll('.pane').forEach(p=>{const x=V.tx+f[0]*S*V.s,y=V.ty+f[1]*S*V.s;
    p.querySelector('.h').style.cssText=`left:0;right:0;top:${y}px;height:1px`;p.querySelector('.v').style.cssText=`top:0;bottom:0;left:${x}px;width:1px`});
}
let drag=null;
$('quad').addEventListener('wheel',e=>{const p=e.target.closest('.pane');if(!p)return;e.preventDefault();
  const r=p.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,k=Math.exp(-e.deltaY*0.0015);
  const ns=Math.min(24,Math.max(1,V.s*k)),kk=ns/V.s;V.tx=mx-(mx-V.tx)*kk;V.ty=my-(my-V.ty)*kk;V.s=ns;clamp();apply();crosshair(toFrame(p,e))},{passive:false});
$('quad').addEventListener('mousedown',e=>{if(e.target.closest('.pane')){drag=[e.clientX,e.clientY,V.tx,V.ty];e.preventDefault()}});
addEventListener('mouseup',()=>drag=null);
$('quad').addEventListener('mousemove',e=>{const p=e.target.closest('.pane');if(!p)return;
  if(drag){V.tx=drag[2]+e.clientX-drag[0];V.ty=drag[3]+e.clientY-drag[1];clamp();apply()}crosshair(toFrame(p,e))});
function clamp(){const m=S*V.s;V.tx=Math.min(0,Math.max(S-m,V.tx));V.ty=Math.min(0,Math.max(S-m,V.ty))}
function step(d){if(CUR.length)open((I+d+CUR.length)%CUR.length)}
function save(){SAVED[CUR[I].key]={pre:OFF.pre.slice(),post:OFF.post.slice()};store.set(SAVED);build()}
function toggle(w){SHOW[w]=!SHOW[w];$(w=='tops'?'bTops':'bOut').classList.toggle('on',SHOW[w]);build()}
function close(){$('lb').classList.remove('on');draw()}
$('bPrev').onclick=()=>step(-1);$('bNext').onclick=()=>step(1);$('bClose').onclick=close;
$('bTops').onclick=()=>toggle('tops');$('bOut').onclick=()=>toggle('out');
$('bSelPre').onclick=()=>select('pre');$('bSelPost').onclick=()=>select('post');
$('bAuto').onclick=()=>{OFF.pre=CUR[I].auto.slice();apply()};$('bReset').onclick=()=>{OFF[ACT]=[0,0];apply()};
$('bSave').onclick=save;$('bFit').onclick=()=>{V={s:1,tx:0,ty:0};apply()};
addEventListener('keydown',e=>{if(!$('lb').classList.contains('on'))return;
  const st=e.shiftKey?3:0.6,k=e.key,o=OFF[ACT];
  if(k=='ArrowLeft'){o[0]-=st;apply();e.preventDefault()} if(k=='ArrowRight'){o[0]+=st;apply();e.preventDefault()}
  if(k=='ArrowUp'){o[1]-=st;apply();e.preventDefault()} if(k=='ArrowDown'){o[1]+=st;apply();e.preventDefault()}
  const l=k.toLowerCase();
  if(k=='Escape')close(); if(l=='j')step(-1); if(l=='k')step(1); if(k=='1')select('pre'); if(k=='2')select('post');
  if(l=='t')toggle('tops'); if(l=='o')toggle('out'); if(l=='a'){OFF.pre=CUR[I].auto.slice();apply()}
  if(l=='r'){OFF[ACT]=[0,0];apply()} if(l=='s')save(); if(l=='f'){V={s:1,tx:0,ty:0};apply()}
});
addEventListener('resize',()=>{if($('lb').classList.contains('on')){layout();clamp();apply()}});
$('exp').onclick=()=>{const rows=Object.entries(SAVED).map(([k,v])=>{const o=asOff(v);
    return{key:k,pre_east_m:+o.pre[0].toFixed(2),pre_north_m:+(-o.pre[1]).toFixed(2),post_east_m:+o.post[0].toFixed(2),post_north_m:+(-o.post[1]).toFixed(2)}});
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:'application/json'}));a.download='naip_lidar_offsets.json';a.click()};
draw();
</script>
"""

if __name__ == "__main__":
    main(*sys.argv[1:6])
