"""
tilekit — cut, label and package image tiles from the MCS 3950 project image sets.

The project image sets are folders of pixel-aligned PNGs, one per year:

    <site>/1935.png  1958.png  1968.png  2000.png  2020.png  info.txt

Pixel (row, col) is the same ground spot in every year, top = north, black = no data.
This module does, for YOUR site, what was done to build the Assignment 2 dataset:

    1. choose tile positions          sample_positions()
    2. cut the tiles                  cut_tiles()
    3. label them in a browser page   make_labeller()     (pixel-aligned context from other years)
    4. turn labels into a dataset     build_dataset()     (split by SITE, not at random)
    5. check label quality            agreement()
    6. map a whole site with a model  sliding_map()

Only numpy and Pillow are required (torch only for sliding_map). It contains no network or
training code on purpose: use the network you build in Assignment 2, or a pretrained one (L14, AL8).

Command line (run `python tilekit.py -h` for details):

    python tilekit.py cut      --sites data/urban/charlottetown --year 1958 --size 64 --n 400 --out tiles/
    python tilekit.py labeller --tiles tiles/ --classes urban,agriculture,vegetated,water --context 1958,2020
    python tilekit.py dataset  --tiles tiles/ --labels labels_*.json --test-sites stratford --out mydata.npz
"""
from __future__ import annotations

import argparse, base64, csv, glob, io, json, os, re, sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None          # the Charlottetown block is 8000 x 6000
YEAR_RE = re.compile(r"^(\d{4})\.png$")


# ============================================================================ loading
def site_years(site_dir):
    """Years available in a site folder, e.g. [1935, 1958, 1968, 2000, 2020]."""
    return sorted(int(m.group(1)) for f in os.listdir(site_dir) if (m := YEAR_RE.match(f)))


def load_year(site_dir, year, grey=True):
    """One year of a site as uint8: (H, W) if grey else (H, W, 3); plus a no-data mask (True = no data).

    No data is black in every channel. Note: in 1935/1958/1968 *water* is also black (the source
    mosaics were clipped to land), and very dark woodland shadow can be exactly 0 too.
    """
    rgb = np.asarray(Image.open(Path(site_dir) / f"{year}.png").convert("RGB"))
    nodata = (rgb == 0).all(axis=2)
    img = (rgb.astype(np.float32) @ [0.299, 0.587, 0.114]).round().astype(np.uint8) if grey else rgb
    return img, nodata


def downsample(a, factor):
    """Average `factor` x `factor` blocks (A2 used factor 2: 1.16 m -> 2.3 m)."""
    if factor == 1:
        return a
    h, w = (a.shape[0] // factor) * factor, (a.shape[1] // factor) * factor
    a = a[:h, :w].astype(np.float32)
    shp = (h // factor, factor, w // factor, factor) + a.shape[2:]
    return a.reshape(shp).mean(axis=(1, 3)).round().astype(np.uint8)


# ============================================================================ tile positions
def sample_positions(nodata, size, n=None, stride=None, strategy="random", img=None,
                     max_nodata=0.0, seed=0):
    """Top-left (row, col) positions of `size` x `size` tiles (in the image's own pixels).

    strategy:
      "grid"     every tile on a non-overlapping grid (stride = size unless given)
      "random"   n random positions, never overlapping each other
      "texture"  n positions spread evenly over texture levels (needs `img`), so rare busy
                 tiles (buildings, hedgerows) are not swamped by plain fields. This is a
                 simple version of how A2 found enough built-up tiles (it also sampled near
                 towns and roads, which you can't do without coordinates).
    Tiles with more than `max_nodata` (fraction) no-data pixels are skipped. A few isolated black
    pixels are normal (deep shadow); a block of them is a gap, or water in 1935-1968.
    """
    H, W = nodata.shape
    stride = stride or size
    grid = [(r, c) for r in range(0, H - size + 1, stride) for c in range(0, W - size + 1, stride)]
    ok = [(r, c) for r, c in grid if nodata[r:r + size, c:c + size].mean() <= max_nodata]
    rng = np.random.default_rng(seed)
    if strategy == "grid" or n is None or n >= len(ok):
        return ok if strategy == "grid" or n is None else [ok[i] for i in rng.permutation(len(ok))]
    if strategy == "random":
        return [ok[i] for i in sorted(rng.choice(len(ok), n, replace=False))]
    if strategy == "texture":
        if img is None:
            raise ValueError("strategy='texture' needs img")
        g = img if img.ndim == 2 else img.mean(axis=2)
        tex = np.array([np.abs(np.diff(g[r:r + size, c:c + size].astype(np.float32), axis=1)).mean() for r, c in ok])
        bins = np.quantile(tex, np.linspace(0, 1, 6))
        which = np.clip(np.digitize(tex, bins[1:-1]), 0, 4)
        out = []
        for b in range(5):
            idx = np.where(which == b)[0]
            take = min(len(idx), n // 5 + (b < n % 5))
            out += [ok[i] for i in rng.choice(idx, take, replace=False)]
        return sorted(out)
    raise ValueError(f"unknown strategy {strategy!r}")


# ============================================================================ cutting
def cut_tiles(site_dirs, year, size=64, factor=1, n=300, strategy="random", max_nodata=0.01,
              out_dir="tiles", context_years=None, context_scale=3, seed=0):
    """Cut tiles from one or more sites and save them, with context crops for the labeller.

    size     tile size in OUTPUT pixels (after downsampling by `factor`)
    factor   1 = keep the image set's resolution; 2 = average 2 x 2 (A2 did this)
    context_years  years to show next to each tile when labelling (default: every year);
                   crops are `context_scale` x the tile's width, centred on it, at the same spot
    Writes  out_dir/tiles.npz (images + index)  and  out_dir/context/<id>_<year>.jpg
    Returns the index as a list of dicts.
    """
    out = Path(out_dir); (out / "context").mkdir(parents=True, exist_ok=True)
    imgs, index = [], []
    for k, site in enumerate(site_dirs):
        site = Path(site); name = site.name
        img, nod = load_year(site, year)
        img_d, nod_d = downsample(img, factor), downsample(nod.astype(np.uint8) * 255, factor) > 0
        pos = sample_positions(nod_d, size, n=n, strategy=strategy, img=img_d,
                               max_nodata=max_nodata, seed=seed + k)
        years = context_years or site_years(site)
        ctx = {y: load_year(site, y, grey=False)[0] for y in years}
        S = size * factor                        # tile width in image-set pixels
        for r, c in pos:
            tid = f"{name}_{year}_{r * factor}_{c * factor}"
            imgs.append(img_d[r:r + size, c:c + size])
            index.append({"id": tid, "site": name, "year": year, "row": r * factor, "col": c * factor,
                          "size_px": S, "factor": factor})
            for y, a in ctx.items():
                _context_crop(a, r * factor, c * factor, S, context_scale).save(
                    out / "context" / f"{tid}_{y}.jpg", quality=72)
        print(f"{name}: {len(pos)} tiles from {year}")
    np.savez_compressed(out / "tiles.npz", X=np.stack(imgs), ids=np.array([d["id"] for d in index]))
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0])); w.writeheader(); w.writerows(index)
    print(f"wrote {len(index)} tiles to {out}/")
    return index


def _context_crop(a, r, c, S, scale, px=200):
    """A crop `scale` x the tile around it, padded with grey at the image edge, tile outlined."""
    half = S * scale // 2; cr, cc = r + S // 2, c + S // 2
    H, W = a.shape[:2]
    canvas = np.full((2 * half, 2 * half, 3), 128, np.uint8)
    r0, c0 = cr - half, cc - half
    rs, cs = max(r0, 0), max(c0, 0); re_, ce = min(r0 + 2 * half, H), min(c0 + 2 * half, W)
    canvas[rs - r0:re_ - r0, cs - c0:ce - c0] = a[rs:re_, cs:ce]
    im = Image.fromarray(canvas).resize((px, px), Image.LANCZOS)
    from PIL import ImageDraw
    k = px / (2 * half); a0 = (half - S // 2) * k; a1 = (half + S // 2) * k
    ImageDraw.Draw(im).rectangle([a0, a0, a1, a1], outline=(255, 214, 0), width=2)
    return im


# ============================================================================ labelling page
def make_labeller(tiles_dir="tiles", classes=("class_a", "class_b"), out_html=None, context_years=None,
                  per_page=48, title="Tile labelling"):
    """Write a self-contained HTML labelling page. Open it in any browser (double-click it).

    * keys 1-9 set a class for the hovered tile, 0 = skip; click cycles; Enter = page done
    * the side panel shows the hovered tile in context in every year you cut context for
    * progress is kept in the browser; "Export labels" downloads labels_<name>.json
    * "Import" merges a labels file back in (to continue on another computer, or to check a
      teammate's labels)
    Split the work by page between teammates, then pass every labels_*.json to build_dataset().
    """
    tiles_dir = Path(tiles_dir)
    z = np.load(tiles_dir / "tiles.npz"); X, ids = z["X"], z["ids"].astype(str)
    years = context_years or sorted({p.stem.rsplit("_", 1)[1] for p in (tiles_dir / "context").glob("*.jpg")})
    tiles = []
    for x, tid in zip(X, ids):
        tiles.append({"id": tid, "img": _b64(Image.fromarray(x)),
                      "ctx": {str(y): _b64file(tiles_dir / "context" / f"{tid}_{y}.jpg") for y in years
                              if (tiles_dir / "context" / f"{tid}_{y}.jpg").exists()}})
    keys = {str(i + 1): c for i, c in enumerate(classes)}
    html = _LABELLER.replace("__TITLE__", title).replace("__DATA__", json.dumps(
        {"tiles": tiles, "keys": keys, "per": per_page, "years": [str(y) for y in years]}))
    out_html = Path(out_html or tiles_dir / "labeller.html")
    out_html.write_text(html)
    print(f"wrote {out_html} ({out_html.stat().st_size / 1e6:.1f} MB, {len(tiles)} tiles, classes {keys})")
    return out_html


def _b64(im):
    b = io.BytesIO(); im.save(b, "PNG"); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def _b64file(p):
    return "data:image/jpeg;base64," + base64.b64encode(Path(p).read_bytes()).decode()


# ============================================================================ dataset
def read_labels(paths):
    """Merge one or more exported labels_*.json. Returns {tile_id: [labels from each file]}."""
    out = {}
    for p in sorted(set(sum((glob.glob(str(q)) for q in paths), []))):
        d = json.load(open(p))
        for tid, lab in d["labels"].items():
            out.setdefault(tid, []).append(lab)
    return out


def agreement(paths):
    """Tiles labelled by more than one person: fraction where they agree, and the confusions.

    Label 50 or so tiles twice (two teammates, or yourself a week later) and report this in your
    project: it tells you how much accuracy you can honestly expect.
    """
    L = {t: v for t, v in read_labels(paths).items() if len(v) > 1}
    if not L:
        print("no tile has more than one label"); return None
    same = sum(len(set(v)) == 1 for v in L.values())
    pairs = {}
    for v in L.values():
        if len(set(v)) > 1:
            k = " vs ".join(sorted(set(v))); pairs[k] = pairs.get(k, 0) + 1
    print(f"{len(L)} tiles labelled more than once; agreement {same / len(L):.0%}")
    for k, n in sorted(pairs.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {n}")
    return same / len(L)


def build_dataset(tiles_dir, label_paths, classes, test_sites=(), val_frac=0.2, out="dataset.npz", seed=0):
    """Labelled tiles -> train / val / test arrays, split by SITE.

    Tiles from `test_sites` form the test set (a place the model never saw, like Kings County in
    A2). The rest is split into train/val per class. Tiles labelled "skip", or where labellers
    disagree, are dropped. Saves X_* (uint8), y_*, id_*, classes, mean, std (training set).
    """
    tiles_dir = Path(tiles_dir)
    z = np.load(tiles_dir / "tiles.npz"); X, ids = z["X"], z["ids"].astype(str)
    site = {r["id"]: r["site"] for r in csv.DictReader(open(tiles_dir / "index.csv"))}
    L = read_labels(label_paths)
    cidx = {c: i for i, c in enumerate(classes)}
    keep, y, dropped = [], [], {"skip": 0, "disagree": 0, "unlabelled": 0}
    for i, tid in enumerate(ids):
        v = L.get(tid)
        if not v: dropped["unlabelled"] += 1; continue
        if len(set(v)) > 1: dropped["disagree"] += 1; continue
        if v[0] not in cidx: dropped["skip"] += 1; continue
        keep.append(i); y.append(cidx[v[0]])
    keep, y = np.array(keep), np.array(y)
    is_test = np.array([site[ids[i]] in set(test_sites) for i in keep])
    rng = np.random.default_rng(seed); tr, va = [], []
    for c in range(len(classes)):
        m = rng.permutation(np.where((y == c) & ~is_test)[0]); nv = int(round(val_frac * len(m)))
        va += list(m[:nv]); tr += list(m[nv:])
    te = np.where(is_test)[0]
    parts = {"train": np.array(tr, int), "val": np.array(va, int), "test": te}
    arr = {}
    for k, p in parts.items():
        arr[f"X_{k}"] = X[keep[p]]; arr[f"y_{k}"] = y[p]; arr[f"id_{k}"] = ids[keep[p]]
    mean, std = float(arr["X_train"].mean()), float(arr["X_train"].std())
    np.savez_compressed(out, classes=np.array(classes), mean=mean, std=std, **arr)
    print(f"wrote {out}: " + ", ".join(f"{k} {len(p)} {np.bincount(y[p], minlength=len(classes)).tolist()}"
                                       for k, p in parts.items()) + f"; dropped {dropped}")
    return out


# ============================================================================ mapping a whole site
def sliding_map(predict, img, size, stride=None, factor=1, nodata=None, max_nodata=0.05, batch=256):
    """Apply a tile classifier across a whole image. Returns (class map, probability map).

    predict  a function: uint8 tiles (N, size, size) -> class probabilities (N, n_classes).
             Wrap your model so it does your own normalisation, e.g. for an A2-style network:
                 def predict(t):
                     x = ((torch.as_tensor(t, dtype=torch.float32) - MEAN) / STD).unsqueeze(1)
                     model.eval()
                     with torch.no_grad(): return torch.softmax(model(x), 1).numpy()
    Each output cell covers `stride` x `stride` downsampled pixels; cells whose tile has more
    than `max_nodata` (fraction) no-data pixels get class -1.
    """
    a = downsample(img, factor); stride = stride or size
    nod = None if nodata is None else downsample(nodata.astype(np.uint8) * 255, factor) > 0
    rows = range(0, a.shape[0] - size + 1, stride); cols = range(0, a.shape[1] - size + 1, stride)
    pos = [(r, c) for r in rows for c in cols]
    probs = []
    for i in range(0, len(pos), batch):
        t = np.stack([a[r:r + size, c:c + size] for r, c in pos[i:i + batch]])
        probs.append(np.asarray(predict(t)))
    P = np.concatenate(probs).reshape(len(rows), len(cols), -1)
    M = P.argmax(-1)
    if nod is not None:
        bad = np.array([nod[r:r + size, c:c + size].mean() > max_nodata for r, c in pos]).reshape(M.shape)
        M[bad] = -1
    return M, P


# ============================================================================ labeller page (HTML)
_LABELLER = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>
:root{--ink:#1c1c1c;--mut:#666;--bg:#f6f5f2}*{box-sizing:border-box}
body{margin:0;font:14px/1.4 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--ink);background:var(--bg)}
header{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center}
h1{font-size:16px;margin:0 8px 0 0}.chip{display:inline-flex;gap:6px;align-items:center;padding:3px 9px;border-radius:14px;border:2px solid;font-weight:600;background:#fff}
kbd{font:600 12px ui-monospace,monospace;background:#eee;border-radius:4px;padding:0 5px}.n{font-weight:400;color:var(--mut)}
.nav{margin-left:auto;display:flex;gap:8px;align-items:center}button,label.btn{font:inherit;padding:6px 12px;border-radius:6px;border:1px solid #bbb;background:#fff;cursor:pointer}
button.primary{background:#1c1c1c;color:#fff;border-color:#1c1c1c}button:disabled{opacity:.4}
main{max-width:1500px;margin:0 auto;padding:14px 16px 40px}.wrap{display:grid;grid-template-columns:1fr 520px;gap:16px;align-items:start}
.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}.t{position:relative;border:5px solid #8a8a8a;border-radius:6px;overflow:hidden;cursor:pointer;aspect-ratio:1;background:#000}
.t img{width:100%;height:100%;display:block}.t .lab{position:absolute;left:0;top:0;padding:1px 6px;font:700 12px ui-monospace,monospace;color:#fff;border-bottom-right-radius:5px;background:#8a8a8a}
.t.hov{outline:3px solid #000;outline-offset:1px}.t.unl{border-style:dashed}
#side{position:sticky;top:70px;background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 12px}
#ctx{display:grid;grid-template-columns:1fr 1fr;gap:8px}#ctx figure{margin:0}#ctx img{width:100%;border-radius:4px}#ctx figcaption{font-size:12px;color:var(--mut)}
details{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px 12px;margin-bottom:12px}.status{color:var(--mut);font-size:13px}
@media (max-width:1100px){.wrap{grid-template-columns:1fr}#side{position:static}}
</style></head><body>
<header><h1>__TITLE__</h1><span id="chips"></span>
<div class="nav"><span id="pg"></span><button id="prev">←</button><button id="done" class="primary">✓ Page done (Enter)</button><button id="next">→</button>
<button id="exp">Export labels</button><label class="btn">Import<input id="imp" type="file" accept=".json" multiple hidden></label></div></header>
<main><details open><summary><b>How to label</b></summary><ul>
<li>Hover a tile and press a number key (<kbd>0</kbd> = skip: unclear, mixed, no data, ink marks). Clicking cycles through the classes.</li>
<li>The panel on the right shows the same spot in every year, 3× the tile's width; the yellow square is the tile. Label what you see in the tile's own year; other years are only a guide.</li>
<li>Tiles start unlabelled (dashed border). Press <b>Enter</b> when a page is done. Progress stays in this browser; <b>Export labels</b> often, and commit the file to your repository.</li>
<li>Decide your class rules as a team and write them down <i>before</i> you start (e.g. "woodland = trees cover about ¾ of the tile"). Label 50 tiles twice to measure how consistent you are.</li></ul></details>
<div class="wrap"><div class="grid" id="grid"></div><aside id="side"><div id="ctx"></div><p class="status" id="sid"></p></aside></div>
<p class="status" id="status"></p></main>
<script>
const D=__DATA__, T=D.tiles, K=D.keys, PER=D.per, NP=Math.ceil(T.length/PER);
const NAMES=Object.values(K), COL=['#2f7d3b','#c98a0b','#2a6fb5','#c0392b','#7b4fb5','#178f8f','#b5577b','#6b6b2f','#3d3d8f'];
const color=n=>n==='skip'?'#8a8a8a':COL[NAMES.indexOf(n)%COL.length];
const STORE='tilekit:'+document.title+':'+T.length;
let lab={}, done=new Set(), page=0, hov=null, who='';
try{const s=JSON.parse(localStorage.getItem(STORE)||'null'); if(s){lab=s.lab||{}; done=new Set(s.done||[]); who=s.who||'';}}catch(e){}
function save(){try{localStorage.setItem(STORE,JSON.stringify({lab,done:[...done],who}));}catch(e){document.getElementById('status').textContent='This browser is not keeping progress: export your labels before closing.';}}
function chips(){const c={}; for(const v of Object.values(lab)) c[v]=(c[v]||0)+1;
 document.getElementById('chips').innerHTML=Object.entries(K).map(([k,n])=>`<span class="chip" style="border-color:${color(n)}"><kbd>${k}</kbd>${n} <span class="n">${c[n]||0}</span></span>`).join(' ')+` <span class="chip" style="border-color:#8a8a8a"><kbd>0</kbd>skip <span class="n">${c.skip||0}</span></span>`;}
function paint(d,id){const v=lab[id]; d.className='t'+(v?'':' unl')+(hov===d?' hov':''); d.style.borderColor=v?color(v):'#8a8a8a'; const l=d.querySelector('.lab'); l.textContent=v||'?'; l.style.background=v?color(v):'#8a8a8a';}
function side(t){document.getElementById('ctx').innerHTML=D.years.filter(y=>t.ctx[y]).map(y=>`<figure><img src="${t.ctx[y]}" alt=""><figcaption>${y}</figcaption></figure>`).join('');
 document.getElementById('sid').textContent=t.id;}
function render(){const g=document.getElementById('grid'); g.innerHTML='';
 T.slice(page*PER,(page+1)*PER).forEach(t=>{const d=document.createElement('div'); d.innerHTML=`<img src="${t.img}" alt=""><span class="lab"></span>`; d.dataset.id=t.id; paint(d,t.id);
  d.onmouseenter=()=>{hov=d; paint(d,t.id); side(t);}; d.onmouseleave=()=>{hov=null; paint(d,t.id);};
  d.onclick=()=>{const opts=[...NAMES,'skip']; lab[t.id]=opts[(opts.indexOf(lab[t.id])+1)%opts.length]; paint(d,t.id); chips(); save();}; g.appendChild(d);});
 if(T[page*PER]) side(T[page*PER]);
 document.getElementById('pg').textContent=`Page ${page+1} / ${NP}`+(done.has(page)?' ✓':''); chips(); window.scrollTo(0,0);}
document.addEventListener('keydown',e=>{if(!hov) {if(e.key==='Enter'){pageDone();} return;} const id=hov.dataset.id;
 if(e.key==='0'){lab[id]='skip';} else if(K[e.key]){lab[id]=K[e.key];} else if(e.key==='Enter'){pageDone(); return;} else return; paint(hov,id); chips(); save(); e.preventDefault();});
function pageDone(){done.add(page); save(); if(page<NP-1) page++; render();}
document.getElementById('done').onclick=pageDone; document.getElementById('prev').onclick=()=>{if(page>0){page--;render();}}; document.getElementById('next').onclick=()=>{if(page<NP-1){page++;render();}};
document.getElementById('exp').onclick=()=>{ if(!who){who=(window.prompt?prompt('Your name (goes in the file name):',''):'')||'me'; save();}
 const blob=new Blob([JSON.stringify({labeller:who,classes:NAMES,labels:lab},null,1)],{type:'application/json'});
 const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='labels_'+who.replace(/\W+/g,'_')+'.json'; document.body.appendChild(a); a.click(); a.remove();};
document.getElementById('imp').onchange=async e=>{for(const f of e.target.files){const d=JSON.parse(await f.text()); Object.assign(lab,d.labels||{});} save(); render(); document.getElementById('status').textContent='Imported.';};
render();
</script></body></html>"""


# ============================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("cut", help="cut tiles (+ context crops) from one or more site folders")
    c.add_argument("--sites", nargs="+", required=True); c.add_argument("--year", type=int, required=True)
    c.add_argument("--size", type=int, default=64); c.add_argument("--factor", type=int, default=1)
    c.add_argument("--n", type=int, default=300, help="tiles per site"); c.add_argument("--out", default="tiles")
    c.add_argument("--strategy", default="random", choices=["random", "grid", "texture"])
    c.add_argument("--max-nodata", type=float, default=0.01); c.add_argument("--context", default=None,
                   help="comma-separated years for the labeller's context panel (default: all)")
    l = sp.add_parser("labeller", help="write the HTML labelling page")
    l.add_argument("--tiles", default="tiles"); l.add_argument("--classes", required=True)
    l.add_argument("--title", default="Tile labelling"); l.add_argument("--out", default=None)
    d = sp.add_parser("dataset", help="labels -> train/val/test .npz (split by site)")
    d.add_argument("--tiles", default="tiles"); d.add_argument("--labels", nargs="+", required=True)
    d.add_argument("--classes", required=True); d.add_argument("--test-sites", default="")
    d.add_argument("--out", default="dataset.npz")
    g = sp.add_parser("agreement", help="how often labellers agree"); g.add_argument("--labels", nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd == "cut":
        ctx = [int(y) for y in a.context.split(",")] if a.context else None
        cut_tiles(a.sites, a.year, a.size, a.factor, a.n, a.strategy, a.max_nodata, a.out, ctx)
    elif a.cmd == "labeller":
        make_labeller(a.tiles, a.classes.split(","), a.out, title=a.title)
    elif a.cmd == "dataset":
        build_dataset(a.tiles, a.labels, a.classes.split(","), [s for s in a.test_sites.split(",") if s], out=a.out)
    elif a.cmd == "agreement":
        agreement(a.labels)


if __name__ == "__main__":
    main()
