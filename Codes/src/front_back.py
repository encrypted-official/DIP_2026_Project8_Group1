"""Legacy end-region feature extractor retained for compatibility.

The final directed-heading workflow is implemented in `heading_direction.py`
and `run_heading_pipeline.py`. Use that integrated pipeline for final metrics.
"""
import argparse
from pathlib import Path
import cv2,numpy as np,pandas as pd
from src.estimate_hybrid import estimate_axis_hybrid
from src.angle_utils import heading_direction_label

def axis_profile_features(image, axis_deg):
    """Return finite positive/negative end intensity statistics along an axis."""
    if image is None or axis_deg is None or pd.isna(axis_deg):
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    if min(h, w) < 5:
        return None
    theta = np.radians(float(axis_deg))
    ux, uy = np.cos(theta), np.sin(theta)
    vx, vy = -uy, ux
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    half_len = 0.42 * min(w, h)
    half_width = max(1.0, 0.08 * min(w, h))
    longitudinal = np.linspace(0.18, 0.42, 9)
    transverse = np.linspace(-half_width, half_width, 7)
    def sample(sign):
        vals = []
        for t in longitudinal:
            for q in transverse:
                x = int(round(cx + sign * t * (2 * half_len) * ux + q * vx))
                y = int(round(cy + sign * t * (2 * half_len) * uy + q * vy))
                if 0 <= x < w and 0 <= y < h:
                    vals.append(float(gray[y, x]))
        return np.asarray(vals, dtype=np.float32) if vals else None
    pos, neg = sample(1), sample(-1)
    if pos is None or neg is None:
        return None
    return {"positive_end_mean": float(pos.mean()), "negative_end_mean": float(neg.mean()),
            "positive_end_std": float(pos.std()), "negative_end_std": float(neg.std()),
            "end_mean_difference": float(pos.mean() - neg.mean()),
            "end_std_difference": float(pos.std() - neg.std())}

def extract_end_features(image,axis_deg,end_fraction=.28):
    if image is None:return None
    h,w=image.shape[:2]
    if h<5 or w<5:return None
    # Rotate so the estimated vehicle axis is horizontal.
    M=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),-float(axis_deg),1.0)
    rot=cv2.warpAffine(image,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    gray=cv2.cvtColor(rot,cv2.COLOR_BGR2GRAY) if rot.ndim==3 else rot.copy()
    # Compare the left and right longitudinal end regions.
    span=max(1,int(w*end_fraction)); left=gray[:, :span]; right=gray[:, w-span:]
    def feats(a):
        edges=cv2.Canny(a,50,150)
        return {"mean":float(a.mean()),"std":float(a.std()),
                "edge_density":float((edges>0).mean()),
                "bright_fraction":float((a>=200).mean()),
                "dark_fraction":float((a<=55).mean())}
    lf,rf=feats(left),feats(right)
    return {"left_mean":lf["mean"],"right_mean":rf["mean"],
            "mean_difference":lf["mean"]-rf["mean"],
            "left_std":lf["std"],"right_std":rf["std"],
            "left_edge_density":lf["edge_density"],"right_edge_density":rf["edge_density"],
            "left_bright_fraction":lf["bright_fraction"],"right_bright_fraction":rf["bright_fraction"],
            "left_dark_fraction":lf["dark_fraction"],"right_dark_fraction":rf["dark_fraction"],
            "feature_note":"Left/right are arbitrary after axis normalization; front/back label requires labelled examples."}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata",required=True);ap.add_argument("--predictions",default=None)
    ap.add_argument("--out",default="results/front_back_features.csv");ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args();m=pd.read_csv(a.metadata)
    pred=None
    if a.predictions:pred=pd.read_csv(a.predictions).set_index("crop_id")
    if a.limit>0:m=m.head(a.limit)
    rows=[]
    for _,r in m.iterrows():
        im=cv2.imread(str(r.crop_path))
        angle=None
        if pred is not None and r.crop_id in pred.index:
            v=pred.loc[r.crop_id,"pred_axis_deg"]
            if pd.notna(v):angle=float(v)
        if angle is None and im is not None:angle,_=estimate_axis_hybrid(im)
        f=extract_end_features(im,angle) if angle is not None else None
        row = {"crop_id":r.crop_id,"source_image":r.get("source_image", "unknown"),"axis_deg":angle,**(f or {}),**(axis_profile_features(im, angle) or {})}
        # heading_deg is the dataset's directed orientation annotation. The label
        # is 1 when the positive estimated-axis end points toward the annotated heading.
        if angle is not None and "heading_deg" in m.columns and pd.notna(r.get("heading_deg")):
            row["heading_sign"] = heading_direction_label(r["heading_deg"], angle)
        else:
            row["heading_sign"] = np.nan
        rows.append(row)
    o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(o,index=False)
    print(f"Saved end-feature measurements for {len(rows)} crops to {o}")
    if "heading_sign" in pd.DataFrame(rows).columns and pd.DataFrame(rows)["heading_sign"].notna().any():
        print("heading_sign is derived from the dataset heading_deg annotation and estimated axis; verify its semantic meaning before calling it front/rear.")
    else:
        print("No directed heading labels available; feature extraction only, no front/rear direction label.")
if __name__=="__main__":main()
