
"""data.py - synthetic fallback so the project runs without downloads; real loader provided."""
from pathlib import Path
import numpy as np
FEATURES=["feat_%02d"%i for i in range(12)]
def make_synthetic(n=4000,seed=42):
    rng=np.random.default_rng(seed); d=len(FEATURES); X=rng.normal(size=(n,d))
    w=rng.normal(size=d)*(rng.random(d)<0.5); logits=X@w+0.6*X[:,0]*X[:,1]-1.4
    y=(rng.random(n)<1/(1+np.exp(-logits))).astype(int)
    return {"X":X,"y":y,"features":FEATURES}
def load_real(csv_name,target):
    import pandas as pd; df=pd.read_csv(Path("data/raw")/csv_name)
    num=df.drop(columns=[target]).select_dtypes("number")
    return {"X":num.to_numpy(),"y":df[target].astype(int).to_numpy(),"features":list(num.columns)}
if __name__=="__main__":
    d=make_synthetic(); print("X",d["X"].shape,"pos",int(d["y"].sum()))
