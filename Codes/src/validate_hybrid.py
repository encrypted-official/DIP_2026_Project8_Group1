"""Choose a PCA-to-Hough switch threshold on one split and evaluate on another."""

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from src.angle_utils import axis_error
from src.estimate_hybrid import distance_to_image_axis

def score(errors):
    e=np.asarray([x for x in errors if x is not None and np.isfinite(x)],float)
    return {"valid":int(len(e)),"mae_deg":float(e.mean()) if len(e) else None,
            "median_error_deg":float(np.median(e)) if len(e) else None,
            "p90_error_deg":float(np.percentile(e,90)) if len(e) else None,
            "within_10_deg_percent":float((e<=10).mean()*100) if len(e) else None}

def load_join(meta_path,pred_path):
    m=pd.read_csv(meta_path); p=pd.read_csv(pred_path)
    need={"crop_id","gt_axis_deg"}; 
    if not need.issubset(m): raise ValueError(f"Metadata requires {need}")
    if not {"crop_id","pca_axis_deg","hough_axis_deg"}.issubset(p): raise ValueError("Prediction CSV must contain crop_id,pca_axis_deg,hough_axis_deg")
    return m.merge(p,on="crop_id",how="inner",validate="one_to_one")

def apply(df,threshold):
    out=[]
    for _,r in df.iterrows():
        p,h,t=r.pca_axis_deg,r.hough_axis_deg,r.gt_axis_deg
        if pd.isna(t): continue
        if pd.notna(p) and distance_to_image_axis(p)<=threshold: chosen=p
        elif pd.notna(h): chosen=h
        else: chosen=p
        out.append(axis_error(chosen,t) if pd.notna(chosen) else None)
    return out

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-metadata",required=True);ap.add_argument("--train-predictions",required=True)
    ap.add_argument("--test-metadata",required=True);ap.add_argument("--test-predictions",required=True)
    ap.add_argument("--out",default="results/hybrid_validation");ap.add_argument("--thresholds",default="5,10,12,15,18,20,25,30")
    a=ap.parse_args();tr=load_join(a.train_metadata,a.train_predictions);te=load_join(a.test_metadata,a.test_predictions)
    thresholds=[float(x) for x in a.thresholds.split(",")]
    candidates=[{"threshold_deg":t,**score(apply(tr,t))} for t in thresholds]
    candidates=[x for x in candidates if x["mae_deg"] is not None]
    if not candidates: raise SystemExit("No valid training predictions.")
    best=min(candidates,key=lambda x:x["mae_deg"]); test_score=score(apply(te,best["threshold_deg"]))
    result={"selected_on":"train_split_only","selected_threshold_deg":best["threshold_deg"],
            "train_candidates":candidates,"train_selected":best,"held_out_test":test_score,
            "warning":"Use genuinely separate source images for train and test; do not tune threshold on the held-out split."}
    o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True);o.with_suffix(".json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()