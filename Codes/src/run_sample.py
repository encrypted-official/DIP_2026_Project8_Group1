"""Run all final-week estimators on a small, already-extracted metadata sample."""
import argparse,subprocess,sys
from pathlib import Path

def run(cmd):
    print("\\n> "+" ".join(map(str,cmd)),flush=True)
    subprocess.run(list(map(str,cmd)),check=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata",required=True);ap.add_argument("--out",default="results/final_sample")
    ap.add_argument("--limit",type=int,default=100);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    common=[sys.executable]
    run(common+["-m","src.estimate_pca","--metadata",a.metadata,"--out",out/"pca.csv"])
    run(common+["-m","src.estimate_edges","--metadata",a.metadata,"--out",out/"hough.csv"])
    run(common+["-m","src.estimate_moments","--metadata",a.metadata,"--out",out/"moments.csv"])
    run(common+["-m","src.estimate_hybrid","--metadata",a.metadata,"--out",out/"hybrid.csv","--limit",str(a.limit)])
    run(common+["-m","src.estimate_pca_roi","--metadata",a.metadata,"--out",out/"pca_roi.csv","--limit",str(a.limit)])
    for method in ("pca","hough","moments","hybrid","pca_roi"):
        run(common+["-m","src.evaluate","--metadata",a.metadata,"--predictions",out/f"{method}.csv","--out",out/method])
    run(common+["-m","src.crop_size_analysis","--metadata",a.metadata,"--predictions",out/"hybrid.csv","--out",out/"crop_size"])
    run(common+["-m","src.front_back","--metadata",a.metadata,"--predictions",out/"hybrid.csv","--out",out/"front_back_features.csv","--limit",str(a.limit)])
    print(f"\\nSample pipeline finished. Results: {out}")
if __name__=="__main__":main()
