"""Analyse orientation error by crop size and orientation category."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from src.angle_utils import axis_error

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata",required=True);ap.add_argument("--predictions",required=True)
    ap.add_argument("--out",default="results/crop_size_analysis");a=ap.parse_args()
    m=pd.read_csv(a.metadata);p=pd.read_csv(a.predictions)
    j=m.merge(p[["crop_id","pred_axis_deg"]],on="crop_id",how="left",validate="one_to_one")
    if not {"crop_width","crop_height"}.issubset(j): raise SystemExit("Metadata lacks crop_width/crop_height. Re-extract crops with the updated extractor.")
    j["short_side"]=j[["crop_width","crop_height"]].min(axis=1)
    j["size_group"]=pd.cut(j.short_side,[-np.inf,19,39,79,np.inf],labels=["very_small_lt20","small_20_39","medium_40_79","large_ge80"])
    j["axis_error_deg"]=[axis_error(p,t) if pd.notna(p) and pd.notna(t) else np.nan for p,t in zip(j.pred_axis_deg,j.gt_axis_deg)]
    j["orientation_group"]=j.gt_axis_deg.map(lambda x:"axis_aligned" if min(float(x)%90,90-float(x)%90)<=15 else "diagonal")
    rows=[]
    for key,g in j.groupby("size_group",observed=False):
        e=g.axis_error_deg.dropna()
        rows.append({"group":str(key),"total":len(g),"valid":len(e),"coverage_percent":round(100*len(e)/len(g),2) if len(g) else 0,
                     "mae_deg":round(e.mean(),3) if len(e) else None,"median_error_deg":round(e.median(),3) if len(e) else None,
                     "p90_error_deg":round(e.quantile(.9),3) if len(e) else None,"within_10_deg_percent":round((e<=10).mean()*100,2) if len(e) else None})
    o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True)
    j.to_csv(o.with_name(o.name+"_per_crop.csv"),index=False)
    pd.DataFrame(rows).to_csv(o.with_name(o.name+"_summary.csv"),index=False)
    by=j.groupby("orientation_group",dropna=False).axis_error_deg.agg(["count","mean","median"]).reset_index()
    by.to_csv(o.with_name(o.name+"_orientation.csv"),index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__":main()
