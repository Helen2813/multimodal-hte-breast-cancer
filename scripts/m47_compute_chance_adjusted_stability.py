from __future__ import annotations
from itertools import combinations
import numpy as np, pandas as pd
from _metabric_m9_utils import load_config,out_dir,print_table,project_root,write_csv

def pair_rows(selected,candidates,groups,label):
    gs={k:set(g["feature"].astype(str)) for k,g in selected.groupby(groups)}
    gc={k:set(g["feature"].astype(str)) for k,g in candidates.groupby(groups)}
    outer=groups[:-1]; outer_keys=sorted({k[:-1] if isinstance(k,tuple) else () for k in gs})
    rows=[]
    for ok in outer_keys:
        keys=sorted([k for k in gs if (k[:-1] if isinstance(k,tuple) else ())==ok],key=lambda k:k[-1])
        for a,b in combinations(keys,2):
            A,B=gs[a],gs[b]; universe=gc.get(a,set())|gc.get(b,set())
            n=len(universe); ka,kb=len(A),len(B); overlap=len(A&B); union=len(A|B)
            exp=ka*kb/n if n else float("nan"); den=min(ka,kb)-exp
            adj=(overlap-exp)/den if np.isfinite(exp) and den>0 else float("nan")
            row={"analysis":label,"candidate_union":n,"selected_a":ka,"selected_b":kb,
                "observed_overlap":overlap,"expected_overlap_random":exp,
                "raw_jaccard":overlap/union if union else 1.0,
                "chance_adjusted_overlap":adj,"fold_a":a[-1],"fold_b":b[-1]}
            for col,val in zip(outer,ok): row[col]=val
            rows.append(row)
    return rows

def main():
    root=project_root(); cfg=load_config(root); out=out_dir(root,cfg)
    print("="*124); print("METABRIC M9.47 - CHANCE-ADJUSTED FEATURE-STABILITY AUDIT"); print("="*124)
    ms=pd.read_csv(root/cfg["files"]["m8_modality_selected"],low_memory=False)
    mc=pd.read_csv(root/cfg["files"]["m8_modality_candidates"],low_memory=False)
    rows=pair_rows(ms,mc,["modality","repeat","fold"],"modality_specific")
    cs=pd.read_csv(root/cfg["files"]["m7_combined_selected"],low_memory=False)
    cc=pd.read_csv(root/cfg["files"]["m7_combined_candidates"],low_memory=False)
    rows+=pair_rows(cs,cc,["repeat","fold"],"combined_reconstructed")
    write_csv(out/"m47_chance_adjusted_pairwise_stability.csv",rows)
    summary=[]
    labels=sorted({r.get("modality","Multimodal") for r in rows})
    for label in labels:
        sub=[r for r in rows if r.get("modality","Multimodal")==label]
        raw=np.asarray([r["raw_jaccard"] for r in sub],float)
        obs=np.asarray([r["observed_overlap"] for r in sub],float)
        exp=np.asarray([r["expected_overlap_random"] for r in sub],float)
        adj=np.asarray([r["chance_adjusted_overlap"] for r in sub],float)
        summary.append({"modality":label,"pairwise_comparisons":len(sub),
            "mean_raw_jaccard":float(np.nanmean(raw)),
            "mean_observed_overlap":float(np.nanmean(obs)),
            "mean_expected_random_overlap":float(np.nanmean(exp)),
            "mean_chance_adjusted_overlap":float(np.nanmean(adj)),
            "median_chance_adjusted_overlap":float(np.nanmedian(adj)),
            "fraction_adjusted_overlap_positive":float(np.mean(adj>0))})
    write_csv(out/"m47_chance_adjusted_stability_summary.csv",summary)
    print_table(summary,["modality","pairwise_comparisons","mean_raw_jaccard","mean_observed_overlap","mean_expected_random_overlap","mean_chance_adjusted_overlap","fraction_adjusted_overlap_positive"])
    print("\nPASS: raw stability calibrated against random overlap."); return 0
if __name__=="__main__": raise SystemExit(main())
