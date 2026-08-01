from __future__ import annotations
import pandas as pd
from _metabric_m9_utils import bootstrap_repeated_oof,load_config,out_dir,print_table,project_root,summarize,write_csv

def main():
    root=project_root(); cfg=load_config(root); out=out_dir(root,cfg); s=cfg["bootstrap"]
    print("="*124); print("METABRIC M9.46 - PAIRED PATIENT BOOTSTRAP OF REPEATED OOF PREDICTIONS"); print("="*124)
    all_rows=[]; summaries=[]; metadata=[]
    frame=pd.read_csv(root/cfg["files"]["m8_modality_predictions"],low_memory=False)
    frame["repeat"]=frame["repeat"].astype(int)
    for i,modality in enumerate(sorted(frame["modality"].unique())):
        subset=frame[frame["modality"]==modality].copy()
        rows,meta=bootstrap_repeated_oof(
            subset,"repeat","sample_id","time_months","event",
            "clinical_risk","clinical_modality_risk",
            int(s["repetitions"]),int(s["seed"])+i,float(s["five_year_months"]))
        all_rows.extend([{"analysis":"modality_specific","modality":modality,**r} for r in rows])
        summaries.extend(summarize(rows,{"analysis":"modality_specific","modality":modality},[
            "mean_clinical_c_index","mean_model_c_index","delta_c_index",
            "mean_clinical_auc_5y","mean_model_auc_5y","delta_auc_5y"]))
        metadata.append({"analysis":"modality_specific","modality":modality,**meta})
        d=next(r for r in summaries if r["analysis"]=="modality_specific" and r["modality"]==modality and r["metric"]=="delta_c_index")
        print(f"{modality:12s} n={meta['unique_patients']:4d} repeats={meta['repeats']:2d} delta C={d['mean']:+.4f} CI=[{d['ci_low']:+.4f},{d['ci_high']:+.4f}]")
    frame=pd.read_csv(root/cfg["files"]["m7_combined_predictions"],low_memory=False)
    frame["repeat"]=frame["repeat"].astype(int)
    rows,meta=bootstrap_repeated_oof(
        frame,"repeat","sample_id","time_months","event","clinical_risk","model_risk",
        int(s["repetitions"]),int(s["seed"])+100,float(s["five_year_months"]))
    all_rows.extend([{"analysis":"combined_reconstructed","modality":"Multimodal",**r} for r in rows])
    summaries.extend(summarize(rows,{"analysis":"combined_reconstructed","modality":"Multimodal"},[
        "mean_clinical_c_index","mean_model_c_index","delta_c_index",
        "mean_clinical_auc_5y","mean_model_auc_5y","delta_auc_5y"]))
    metadata.append({"analysis":"combined_reconstructed","modality":"Multimodal",**meta})
    write_csv(out/"m46_oof_patient_bootstrap_2000.csv",all_rows)
    write_csv(out/"m46_oof_patient_bootstrap_summary.csv",summaries)
    write_csv(out/"m46_oof_bootstrap_metadata.csv",metadata)
    print("\nBootstrap summary")
    print_table(summaries,["analysis","modality","metric","mean","sd","ci_low","ci_high","fraction_positive"])
    print("\nPASS: paired patient bootstrap completed."); return 0
if __name__=="__main__": raise SystemExit(main())
