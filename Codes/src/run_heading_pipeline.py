"""Evaluate rule-based directed heading using silhouette taper analysis only."""
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from src.angle_utils import axis_error
from src.heading_direction import estimate_heading_sign, heading_from_axis_and_sign, circular_heading_error, heading_sign_from_ground_truth

METHOD_FILES = {"pca_edges":"pca.csv", "hough_edges":"hough.csv", "otsu_moments":"moments.csv", "pca_hough_hybrid":"hybrid.csv", "pca_roi":"pca_roi.csv"}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-metadata', required=True, help='Retained for split audit; no training is performed')
    ap.add_argument('--test-metadata', required=True)
    ap.add_argument('--train-pred-dir', required=True, help='Retained for protocol compatibility')
    ap.add_argument('--test-pred-dir', required=True)
    ap.add_argument('--out', default='results/directed_heading')
    ap.add_argument('--ambiguity-threshold', type=float, default=.08)
    a=ap.parse_args()
    train=pd.read_csv(a.train_metadata); test=pd.read_csv(a.test_metadata)
    overlap=set(train.source_image.dropna().astype(str)) & set(test.source_image.dropna().astype(str))
    if overlap: raise ValueError(f'Source-image leakage: {len(overlap)} source images occur in both splits')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    all_rows=[]; summaries=[]
    for method,filename in METHOD_FILES.items():
        path=Path(a.test_pred_dir)/filename
        if not path.exists():
            print(f'SKIP {method}: missing {path}'); continue
        pred=pd.read_csv(path)
        joined=test.merge(pred[['crop_id','pred_axis_deg']],on='crop_id',how='left',validate='one_to_one')
        rows=[]
        for r in joined.itertuples(index=False):
            rec={"crop_id":r.crop_id,"source_image":getattr(r,'source_image','unknown'),"class_name":getattr(r,'class_name','unknown'),"method":method,"pred_axis_deg":r.pred_axis_deg,"gt_axis_deg":r.gt_axis_deg,"heading_deg":r.heading_deg}
            sign=confidence=None; features={}
            if pd.notna(r.pred_axis_deg):
                im=cv2.imread(str(r.crop_path))
                if im is not None: sign,confidence,features=estimate_heading_sign(im,float(r.pred_axis_deg),a.ambiguity_threshold)
            rec.update(features); rec['predicted_heading_sign']=sign; rec['direction_confidence']=confidence
            rec['ground_truth_heading_sign']=heading_sign_from_ground_truth(r.heading_deg, r.gt_axis_deg) if pd.notna(r.heading_deg) and pd.notna(r.gt_axis_deg) else np.nan
            rec['direction_sign_correct']=(int(sign)==int(rec['ground_truth_heading_sign'])) if sign is not None and pd.notna(rec['ground_truth_heading_sign']) else np.nan
            rec['pred_heading_deg']=heading_from_axis_and_sign(r.pred_axis_deg,sign) if sign is not None else np.nan
            rec['axis_error_deg']=axis_error(r.pred_axis_deg,r.gt_axis_deg) if pd.notna(r.pred_axis_deg) and pd.notna(r.gt_axis_deg) else np.nan
            rec['heading_error_deg']=circular_heading_error(rec['pred_heading_deg'],r.heading_deg) if pd.notna(rec['pred_heading_deg']) and pd.notna(r.heading_deg) else np.nan
            rows.append(rec)
        frame=pd.DataFrame(rows); frame.to_csv(out/f'{method}_directed_predictions.csv',index=False); all_rows.append(frame)
        valid=frame.dropna(subset=['heading_error_deg']); err=valid.heading_error_deg.to_numpy(float)
        resolved_sign = frame.dropna(subset=['direction_sign_correct'])['direction_sign_correct']
        summaries.append({"method":method,"test_crops":len(frame),"direction_resolved":len(valid),"heading_coverage_percent":round(100*len(valid)/max(1,len(frame)),2),"heading_mae_deg":round(float(np.mean(err)),3) if len(err) else None,"heading_median_error_deg":round(float(np.median(err)),3) if len(err) else None,"heading_p90_error_deg":round(float(np.percentile(err,90)),3) if len(err) else None,"within_5_deg_percent":round(float(np.mean(err<=5)*100),2) if len(err) else None,"within_10_deg_percent":round(float(np.mean(err<=10)*100),2) if len(err) else None,"within_15_deg_percent":round(float(np.mean(err<=15)*100),2) if len(err) else None,"direction_sign_accuracy_percent":round(float(resolved_sign.mean()*100),2) if len(resolved_sign) else None,"direction_sign_evaluated":int(len(resolved_sign)),"ambiguous_or_invalid":int(len(frame)-len(valid))})
        print(f'{method}: resolved {len(valid)}/{len(frame)} test crops')
    if not all_rows: raise RuntimeError('No method completed; check test prediction CSVs')
    combined=pd.concat(all_rows,ignore_index=True); combined.to_csv(out/'all_methods_directed_heading_per_crop.csv',index=False)
    summary_frame=pd.DataFrame(summaries); summary_frame.to_csv(out/'summary_by_method.csv',index=False)
    summary_frame.to_csv(out/'summary_by_method_metrics.csv',index=False)
    combined['gt_axis_style']=combined.gt_axis_deg.map(lambda x: 'unknown' if pd.isna(x) else ('axis_aligned' if min(float(x)%180, abs(float(x)%180-90), 180-(float(x)%180)) <= 15 else 'diagonal'))
    for filename, keys in [('summary_by_method_vehicle_class.csv',['method','class_name']),('summary_by_method_axis_style.csv',['method','gt_axis_style'])]:
        rows=[]
        for group_key, part in combined.groupby(keys, dropna=False):
            if not isinstance(group_key, tuple): group_key=(group_key,)
            valid_part=part.dropna(subset=['heading_error_deg']); errors=valid_part.heading_error_deg.to_numpy(float)
            row=dict(zip(keys,group_key)); row.update({'test_crops':len(part),'direction_resolved':len(valid_part),'heading_coverage_percent':round(100*len(valid_part)/max(1,len(part)),2),'heading_mae_deg':round(float(np.mean(errors)),3) if len(errors) else np.nan,'heading_median_error_deg':round(float(np.median(errors)),3) if len(errors) else np.nan,'heading_p90_error_deg':round(float(np.percentile(errors,90)),3) if len(errors) else np.nan,'direction_sign_accuracy_percent':round(float(valid_part.direction_sign_correct.mean()*100),2) if valid_part.direction_sign_correct.notna().any() else np.nan})
            rows.append(row)
        pd.DataFrame(rows).to_csv(out/filename,index=False)
    manifest={"task":"directed_vehicle_heading_estimation","direction_method":"deterministic Otsu foreground segmentation + terminal silhouette-width/taper comparison","machine_learning_used":False,"deep_learning_used":False,"training_performed":False,"split_unit":"source_image","train_rows_audit_only":len(train),"test_rows":len(test),"ambiguity_threshold":a.ambiguity_threshold,"front_assumption":"narrower/tapered terminal silhouette is presumed to be the front","limitations":"This is a hand-designed geometric heuristic, not a reliable universal front/rear detector. Symmetric, occluded, low-resolution or poorly segmented vehicles may be ambiguous or misclassified. Report direction coverage and inspect overlays.","methods_completed":sorted(combined.method.unique().tolist())}
    (out/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2)); print(f'Outputs saved to {out}')
if __name__=='__main__': main()
