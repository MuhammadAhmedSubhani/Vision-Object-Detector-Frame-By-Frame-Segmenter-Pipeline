"""Box-level evaluation using per-frame, per-class greedy IoU matching.
Ground-truth CSV columns: frame,class_name,x1,y1,x2,y2
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

def iou(a,b):
    iw=max(0,min(a[2],b[2])-max(a[0],b[0]))
    ih=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=iw*ih
    union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
    return inter/union if union else 0.0

def read_csv(path):
    groups=defaultdict(list)
    with open(path,newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key=(int(r["frame"]),r["class_name"].strip().lower())
            groups[key].append(tuple(float(r[k]) for k in ("x1","y1","x2","y2")))
    return groups

def evaluate(pred, gt, threshold):
    tp=fp=fn=0; matched=[]
    for key in set(pred)|set(gt):
        ps, gs=pred.get(key,[]),gt.get(key,[])
        pairs=sorted(((iou(p,g),i,j) for i,p in enumerate(ps) for j,g in enumerate(gs)),reverse=True)
        used_p=set(); used_g=set()
        for score,i,j in pairs:
            if score<threshold: break
            if i not in used_p and j not in used_g:
                used_p.add(i); used_g.add(j); matched.append(score)
        tp+=len(used_p); fp+=len(ps)-len(used_p); fn+=len(gs)-len(used_g)
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0
    return {"precision":round(precision,4),"recall":round(recall,4),"f1":round(f1,4),
            "tp":tp,"fp":fp,"fn":fn,"iou_threshold":threshold,
            "mean_matched_iou":round(sum(matched)/len(matched),4) if matched else 0,
            "frames_evaluated":len({k[0] for k in set(pred)|set(gt)})}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--predictions",default="outputs/predictions.csv")
    p.add_argument("--ground-truth",required=True)
    p.add_argument("--iou",type=float,default=.5)
    p.add_argument("--output",default="outputs/evaluation.json")
    a=p.parse_args()
    result=evaluate(read_csv(a.predictions),read_csv(a.ground_truth),a.iou)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
