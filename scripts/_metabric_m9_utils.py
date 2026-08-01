from __future__ import annotations
import csv, hashlib, json, re
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

def project_root(): return Path.cwd().resolve()
def load_config(root):
    return json.loads((root/"metabric_m9_config.json").read_text(encoding="utf-8"))
def out_dir(root,cfg):
    p=(root/cfg["output_dir"]).resolve(); p.mkdir(parents=True,exist_ok=True); return p
def figure_dir(root,cfg):
    p=(root/cfg["figure_dir"]).resolve(); p.mkdir(parents=True,exist_ok=True); return p
def manuscript_dir(root,cfg):
    p=(root/cfg["manuscript_dir"]).resolve(); p.mkdir(parents=True,exist_ok=True); return p
def rel(root,path):
    try:return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:return path.resolve().as_posix()
def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def write_csv(path,rows,fieldnames=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        fields=list(fieldnames or ["empty"])
        with path.open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        return
    fields=list(fieldnames or [])
    if not fields:
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
def print_table(rows,columns,max_rows=None):
    if not rows: print("<empty>"); return
    shown=list(rows if max_rows is None else rows[:max_rows])
    widths={c:len(c) for c in columns}; rendered=[]
    for row in shown:
        item={c:str(row.get(c,"")) for c in columns}; rendered.append(item)
        for c in columns: widths[c]=min(58,max(widths[c],len(item[c])))
    print("  ".join(c[:widths[c]].ljust(widths[c]) for c in columns))
    print("  ".join("-"*widths[c] for c in columns))
    for row in rendered: print("  ".join(row[c][:widths[c]].ljust(widths[c]) for c in columns))
    if max_rows is not None and len(rows)>max_rows:
        print(f"... {len(rows)-max_rows} additional rows written to CSV")
def fast_harrell(time,event,risk):
    time=np.asarray(time,float); event=np.asarray(event,int); risk=np.asarray(risk,float)
    valid=np.isfinite(time)&np.isfinite(event)&np.isfinite(risk)
    if valid.sum()<3 or event[valid].sum()==0:return float("nan")
    return float(concordance_index(time[valid],-risk[valid],event_observed=event[valid]))
def binary_auc(time,event,risk,horizon):
    time=np.asarray(time,float); event=np.asarray(event,int); risk=np.asarray(risk,float)
    cases=(event==1)&(time<=horizon); controls=time>horizon; valid=cases|controls
    if valid.sum()<10 or cases.sum()==0 or controls.sum()==0:return float("nan")
    return float(roc_auc_score(cases[valid].astype(int),risk[valid]))
def bootstrap_repeated_oof(frame,repeat_col,sample_col,time_col,event_col,clinical_col,model_col,repetitions,seed,horizon):
    repeats=sorted(frame[repeat_col].astype(int).unique())
    samples=sorted(frame[sample_col].astype(str).unique()); n=len(samples)
    arrays={}
    for repeat in repeats:
        s=frame[frame[repeat_col].astype(int)==repeat].copy()
        s[sample_col]=s[sample_col].astype(str)
        s=s.set_index(sample_col).loc[samples]
        arrays[repeat]={
            "time":s[time_col].to_numpy(float),"event":s[event_col].to_numpy(int),
            "clinical":s[clinical_col].to_numpy(float),"model":s[model_col].to_numpy(float)}
    rng=np.random.default_rng(seed); rows=[]
    for b in range(1,repetitions+1):
        idx=rng.integers(0,n,n); rc=[]
        for repeat in repeats:
            d=arrays[repeat]
            cc=fast_harrell(d["time"][idx],d["event"][idx],d["clinical"][idx])
            mc=fast_harrell(d["time"][idx],d["event"][idx],d["model"][idx])
            ca=binary_auc(d["time"][idx],d["event"][idx],d["clinical"][idx],horizon)
            ma=binary_auc(d["time"][idx],d["event"][idx],d["model"][idx],horizon)
            rc.append((cc,mc,ca,ma))
        rows.append({
            "bootstrap":b,
            "mean_clinical_c_index":float(np.nanmean([x[0] for x in rc])),
            "mean_model_c_index":float(np.nanmean([x[1] for x in rc])),
            "delta_c_index":float(np.nanmean([x[1]-x[0] for x in rc])),
            "mean_clinical_auc_5y":float(np.nanmean([x[2] for x in rc])),
            "mean_model_auc_5y":float(np.nanmean([x[3] for x in rc])),
            "delta_auc_5y":float(np.nanmean([x[3]-x[2] for x in rc]))})
    metadata={"unique_patients":n,"repeats":len(repeats),"repeat_values":repeats}
    return rows,metadata
def summarize(rows,group,metrics):
    out=[]
    for metric in metrics:
        v=np.asarray([float(r[metric]) for r in rows]); v=v[np.isfinite(v)]
        out.append({**group,"metric":metric,"repetitions":len(v),"mean":float(v.mean()),
            "sd":float(v.std(ddof=1)),"median":float(np.median(v)),
            "ci_low":float(np.quantile(v,.025)),"ci_high":float(np.quantile(v,.975)),
            "fraction_positive":float(np.mean(v>0))})
    return out
def read_feature_list(path):
    text=path.read_text(encoding="utf-8-sig",errors="replace"); vals=[]
    for token in re.split(r"[\s,;]+",text):
        token=token.strip().strip('"').strip("'")
        if token and token.lower() not in {"gene","genes","feature","features"} and token not in vals:
            vals.append(token)
    return vals
def gene_like_tokens(value):
    if pd.isna(value):return []
    out=[]
    for token in re.split(r"[;,|/\s]+",str(value).upper()):
        token=token.strip()
        if 2<=len(token)<=30 and re.match(r"^[A-Z0-9][A-Z0-9._-]*$",token) and not token.startswith("CG") and token not in {"NA","NAN","NONE","UNKNOWN","CHR","GENE"} and any(c.isalpha() for c in token):
            out.append(token)
    return sorted(set(out))
def claim_status(low,high):
    if low>0:return "INCREMENTAL_UTILITY_SUPPORTED"
    if high<0:return "LOWER_DISCRIMINATION_THAN_CLINICAL"
    return "NO_RELIABLE_INCREMENTAL_UTILITY"
