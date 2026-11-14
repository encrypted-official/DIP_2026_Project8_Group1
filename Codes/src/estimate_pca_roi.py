"""PCA orientation with centre-ellipse edge-point filtering."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from src.preprocess import edge_map

def estimate_axis_pca_roi(image, low_threshold=30, high_threshold=90,
                          blur_size=5, use_clahe=False, ellipse_scale=0.72,
                          min_points=20):
    if image is None: return None
    edges=edge_map(image,low_threshold,high_threshold,blur_size,use_clahe)
    if edges is None: return None
    h,w=edges.shape
    ys,xs=np.where(edges>0)
    if len(xs)<min_points: return None
    nx=(xs-(w-1)/2)/max(w*ellipse_scale/2,1)
    ny=(ys-(h-1)/2)/max(h*ellipse_scale/2,1)
    keep=(nx*nx+ny*ny)<=1.0
    xs,ys=xs[keep],ys[keep]
    if len(xs)<min_points: return None
    pts=np.column_stack((xs,ys)).astype(float)
    cov=np.cov(pts,rowvar=False)
    vals,vecs=np.linalg.eigh(cov)
    if not np.isfinite(vals).all() or vals[-1]<=1e-9 or vals[-1]/max(vals[0],1e-9)<1.05:
        return None
    vx,vy=vecs[:,-1]
    return float(np.degrees(np.arctan2(vy,vx))%180)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata",required=True); ap.add_argument("--out",default="results/pca_roi_predictions.csv")
    ap.add_argument("--limit",type=int,default=0); ap.add_argument("--ellipse-scale",type=float,default=.72)
    ap.add_argument("--blur-size",type=int,default=5); ap.add_argument("--low",type=int,default=30)
    ap.add_argument("--high",type=int,default=90); ap.add_argument("--clahe",action="store_true")
    a=ap.parse_args(); m=pd.read_csv(a.metadata)
    if a.limit>0:m=m.head(a.limit)
    rows=[]
    for _,r in m.iterrows():
        im=cv2.imread(str(r.crop_path))
        p=estimate_axis_pca_roi(im,a.low,a.high,a.blur_size,a.clahe,a.ellipse_scale)
        rows.append({"crop_id":r.crop_id,"crop_path":r.crop_path,"pred_axis_deg":p,"method":"pca_roi"})
    o=Path(a.out);o.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(o,index=False)
    print(f"Saved {len(rows)} predictions to {o}")
if __name__=="__main__":main()
