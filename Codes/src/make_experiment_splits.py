"""Create leakage-aware train/test metadata splits and a balanced ablation subset."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def axis_style(a, tol):
    if pd.isna(a): return "unknown"
    x=float(a)%180
    return "axis_aligned" if min(x,abs(x-90),180-x)<=tol else "diagonal"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',required=True);p.add_argument('--out',default='data/final_splits')
    p.add_argument('--test-size',type=float,default=.2);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--ablation-per-stratum',type=int,default=100);p.add_argument('--axis-tolerance',type=float,default=15)
    a=p.parse_args();m=pd.read_csv(a.metadata)
    required={'crop_id','crop_path','gt_axis_deg','source_image'}
    missing=required-set(m.columns)
    if missing: raise ValueError(f'Metadata missing required split columns: {sorted(missing)}')
    if m.crop_id.duplicated().any(): raise ValueError('crop_id must be unique')
    m=m.copy();m['axis_style']=m.gt_axis_deg.map(lambda x:axis_style(x,a.axis_tolerance))
    if 'class_name' not in m:m['class_name']='unknown'
    groups=m.source_image.fillna(m.crop_id).astype(str)
    if not 0 < a.test_size < 1: raise ValueError('test-size must be between 0 and 1')
    unique_groups=np.array(sorted(groups.unique()))
    rng=np.random.default_rng(a.seed); rng.shuffle(unique_groups)
    n_test=max(1, min(len(unique_groups)-1, int(round(len(unique_groups)*a.test_size))))
    test_groups=set(unique_groups[:n_test])
    test=m.loc[groups.isin(test_groups)].copy(); train=m.loc[~groups.isin(test_groups)].copy()
    if set(train.source_image.dropna()) & set(test.source_image.dropna()): raise RuntimeError('Source-image leakage detected')
    # Stratified bounded subset keeps class x axis-style strata represented for the 42-config grid.
    pieces=[]
    for _,g in train.groupby(['class_name','axis_style'],dropna=False):
        pieces.append(g.sample(n=min(len(g),a.ablation_per_stratum),random_state=a.seed))
    ablation=pd.concat(pieces).sort_values('crop_id') if pieces else train.iloc[:0]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    train.to_csv(out/'train_metadata.csv',index=False);test.to_csv(out/'test_metadata.csv',index=False)
    ablation.to_csv(out/'ablation_train_sample.csv',index=False)
    report={'rows_total':len(m),'rows_train':len(train),'rows_test':len(test),
      'source_images_train':int(train.source_image.nunique()),'source_images_test':int(test.source_image.nunique()),
      'ablation_sample_rows':len(ablation),'test_fraction_actual':round(len(test)/len(m),4),
      'split_unit':'source_image','seed':a.seed,'axis_tolerance_deg':a.axis_tolerance}
    pd.Series(report).to_json(out/'split_manifest.json',indent=2)
    print(pd.Series(report).to_string())
if __name__=='__main__':main()
