"""R3-3: empirical check of the temporal-median reference floor on the Sentinel-1 panel stacks (split-half medians).

    python revision/rev15_panel_reference_floor.py   (reads a152_stage3_panel/evalset/<site>.npz; set EV below)
"""
import numpy as np, json, glob, os, collections, sys
EV=os.path.expanduser('~/mnt/03_Benzetim_ve_Panel_Verileri/SAR/kodlar/outputs/a152_stage3_panel/evalset')
rng=np.random.default_rng(20261112)
def theory_sd(h, n=200000):
    x=np.random.default_rng(h).exponential(size=(n,h)); m=np.median(x,axis=1); return float(np.std(np.log(m)))
TH={}
out=[]
for f in sorted(glob.glob(EV+'/*.npz')):
    if f.endswith('_sar2sar.npz'): continue
    site=os.path.basename(f)[:-4]; z=np.load(f)
    ids=z['ids']; noisy=z['noisy']; stable=z['stable']
    loc=collections.defaultdict(list)
    for i,k in enumerate(ids):
        loc[k.split('Z_',1)[1]].append(i)
    for key,idx in loc.items():
        k=len(idx)
        if k<4: continue
        h=k//2
        st=np.logical_and.reduce(stable[idx])
        if st.mean()<0.2: continue
        for rep in range(5):
            p=rng.permutation(idx); a=np.median(noisy[p[:h]],axis=0); b=np.median(noisy[p[h:2*h]],axis=0)
            a=a/a[st].mean(); b=b/b[st].mean()
            d=np.log(np.clip(a[st],1e-8,None))-np.log(np.clip(b[st],1e-8,None))
            out.append(dict(site=site,loc=key,dates=k,half=h,rms=float(np.sqrt(np.mean(d**2)))))
import pandas as pd
d=pd.DataFrame(out)
for h in sorted(d.half.unique()): TH[int(h)]=theory_sd(int(h))
d['theory']=d.half.map(lambda h: np.sqrt(2)*TH[int(h)])
d['inflation']=d.rms/d.theory
d.to_csv(os.path.expanduser('~/rev15/rev15_split_half.csv'),index=False)
print(d.groupby('site')[['dates','half','rms','theory','inflation']].median().round(3).to_string())
print('ALL', d[['rms','theory','inflation']].median().round(3).to_dict(), 'n_loc', d.groupby(['site','loc']).ngroups)
print('theory sd by h', {h:round(v,4) for h,v in TH.items()})
