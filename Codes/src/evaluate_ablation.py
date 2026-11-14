"""Evaluate training-selected preprocessing configurations on held-out metadata."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,pandas as pd
from src.ablation import estimate,metrics
from src.angle_utils import axis_error

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--metadata',required=True);ap.add_argument('--configs',required=True)
 ap.add_argument('--out',default='results/ablation_test');ap.add_argument('--top-k',type=int,default=5);a=ap.parse_args()
 meta=pd.read_csv(a.metadata);configs=json.loads(Path(a.configs).read_text(encoding='utf-8'))
 if a.top_k>0:configs=configs[:a.top_k]
 if not configs:raise ValueError('No selected configurations found')
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);summaries=[];details=[]
 for i,cfg in enumerate(configs,1):
  config={k:cfg[k] for k in ('method','blur_size','clahe','low_threshold','high_threshold') if k in cfg}
  errors=[]
  for _,r in meta.iterrows():
   im=cv2.imread(str(r.crop_path));pred=None
   if im is not None:
    try:pred=estimate(im,config)
    except (ValueError,cv2.error,np.linalg.LinAlgError):pred=None
   err=axis_error(float(pred),float(r.gt_axis_deg)) if pred is not None and pd.notna(r.gt_axis_deg) else None
   errors.append(err);details.append({'crop_id':r.crop_id,**config,'pred_axis_deg':pred,'error_deg':err})
  summaries.append(metrics(errors,len(meta),config['method'],{k:v for k,v in config.items() if k!='method'}))
  print(f'[{i}/{len(configs)}] {config["method"]}: MAE={summaries[-1]["mae_deg"]}, coverage={summaries[-1]["coverage_percent"]}%')
 pd.DataFrame(summaries).to_csv(out/'ablation_test_summary.csv',index=False)
 pd.DataFrame(details).to_csv(out/'ablation_test_per_crop.csv',index=False)
 (out/'protocol.json').write_text(json.dumps({'source_configs':str(a.configs),'test_metadata':str(a.metadata),'top_k':len(configs),'note':'Configs selected on training data; these are held-out evaluations.'},indent=2),encoding='utf-8')
if __name__=='__main__':main()
