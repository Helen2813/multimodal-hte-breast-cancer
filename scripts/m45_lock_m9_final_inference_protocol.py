from __future__ import annotations
import json
from _metabric_m9_utils import load_config,out_dir,print_table,project_root,rel,sha256,write_csv

def main():
    root=project_root(); cfg=load_config(root); out=out_dir(root,cfg)
    print("="*124); print("METABRIC M9.45 - FINAL INFERENCE AND CLAIM PROTOCOL LOCK"); print("="*124)
    required=[
        root/cfg["files"]["m8_protocol"],root/cfg["files"]["m8_report"],
        root/cfg["files"]["m8_modality_predictions"],root/cfg["files"]["m8_modality_selected"],
        root/cfg["files"]["m8_modality_candidates"],root/cfg["files"]["m8_modality_universe"],
        root/cfg["files"]["m7_combined_predictions"],root/cfg["files"]["m7_combined_selected"],
        root/cfg["files"]["m7_combined_candidates"],root/cfg["files"]["m7_track_a_deltas"]]
    checks=[{"check":rel(root,p),"observed":sha256(p) if p.exists() else "","pass":p.exists()} for p in required]
    if not all(r["pass"] for r in checks): raise RuntimeError("M9 required inputs are incomplete.")
    report=json.loads((root/cfg["files"]["m8_report"]).read_text(encoding="utf-8"))
    checks.append({"check":"M8 completion decision","observed":report["metabric_m8_decision"],
        "pass":report["metabric_m8_decision"]=="M8_MODALITY_GENE_PATHWAY_ANALYSIS_COMPLETE"})
    if not all(r["pass"] for r in checks): raise RuntimeError("M9 scientific preflight failed.")
    protocol={
        "protocol_id":"","status":"METABRIC_M9_FINAL_INFERENCE_PROTOCOL_LOCKED",
        "locked_before_bootstrap_results":True,
        "sampling_uncertainty":{
            "method":"paired patient bootstrap of locked repeated OOF predictions, averaging paired contrasts over repeats",
            "repetitions":cfg["bootstrap"]["repetitions"],"seed":cfg["bootstrap"]["seed"],
            "primary_metric":"delta Harrell C-index","secondary_metric":"delta 5-year AUC",
            "limitation":"conditional evaluation bootstrap; not full-pipeline model refitting"},
        "stability_calibration":{
            "method":"chance-adjusted overlap using the union of paired fold candidate sets"},
        "methylation_mapping_audit":{
            "purpose":"separate platform non-overlap from incomplete local probe annotation"},
        "claim_rule":{
            "positive":"95% interval entirely above zero",
            "negative":"95% interval entirely below zero",
            "otherwise":"no reliable incremental utility",
            "primary_metric_controls_main_claim":True},
        "boundaries":[
            "Repeated-split quantiles remain algorithmic variability summaries.",
            "Gene/pathway recurrence cannot override a null or negative primary performance contrast.",
            "Methylation concordance is not estimable when too few historical genes are assayable.",
            "No original ITE manuscript file is modified."]}
    payload=json.dumps(protocol,sort_keys=True)
    h=__import__("hashlib").sha256(payload.encode()).hexdigest()
    protocol["protocol_id"]=f"METABRIC_M9_{h[:16].upper()}"
    p=out/"m45_m9_protocol.json"; p.write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    write_csv(out/"m45_protocol_checks.csv",checks)
    write_csv(out/"m45_input_hash_manifest.csv",[
        {"path":rel(root,x),"sha256":sha256(x),"size_bytes":x.stat().st_size} for x in required+[p]])
    print_table(checks,["check","observed","pass"])
    print(f"\nProtocol ID: {protocol['protocol_id']}"); print(json.dumps(protocol,indent=2))
    print("\nPASS: M9 final-inference protocol locked."); return 0
if __name__=="__main__": raise SystemExit(main())
