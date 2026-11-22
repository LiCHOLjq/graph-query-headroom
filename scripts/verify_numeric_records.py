"""Validate all released repeated-call measurements against retained medians.

Checks numeric records and retained audit flags, not graph predicate legality or
exact-reference rankings: those would require intentionally omitted graph data.
"""
from pathlib import Path
import argparse, gzip, json
import numpy as np
import pandas as pd

def verify(root):
    d=root/'data'; report={}
    p=d/'prospective';plan=json.loads((p/'plan/preanalysis_plan.json').read_text())
    raw=pd.concat([pd.read_csv(p/'results'/c/'raw.csv.gz').assign(config=c) for c in plan['corpora']],ignore_index=True)
    med=pd.read_csv(p/'results/costs.csv')
    check(raw,med,['config','seed','query_id','arm'])
    assert len(raw)==204800 and len(med)==25600
    assert raw.groupby(['query_id','arm','position']).size().eq(1).all()
    assert raw.groupby('query_id').answer_hash.nunique().eq(1).all()
    report['prospective']={'queries':3200,'measured_calls':len(raw),'median_rows':len(med),'answer_hashes_agree':True}
    count=0;rows=0
    for f in sorted((d/'factorial').glob('*/*/*/raw.jsonl')):
        r=pd.read_json(f,lines=True);m=pd.read_csv(f.with_name('costs.csv'))
        r=r.rename(columns={'dataset':'config'})
        keys=['session','cell','config','seed','query_id','policy']
        check(r,m,keys);assert len(r)==3072 and len(m)==384
        assert r.groupby(['query_id','policy']).size().eq(8).all()
        assert r.answer_matches_retained.all()
        count+=len(r);rows+=len(m)
    assert count==122880 and rows==15360
    report['factorial']={'queries':960,'sessions':2,'cells':4,'measured_calls':count,'median_rows':rows}
    count=0;rows=0
    for f in sorted((d/'external/results').glob('*/raw.jsonl')):
        r=pd.read_json(f,lines=True);m=pd.read_csv(f.with_name('costs.csv'))
        check(r,m,['corpus','query_id','seed','policy']);assert len(r)==9600 and len(m)==960
        assert r.groupby(['query_id','policy','order']).size().eq(2).all()
        q=pd.read_csv(f.with_name('quality.csv'));assert q.guarded_invalid_count.eq(0).all()
        for component in ['bitmap_ms','search_ms']:
            z=r.groupby(['query_id','policy'])[component].median().reset_index().merge(m,on=['query_id','policy'],suffixes=('_raw','_stored'))
            assert np.allclose(z[component+'_raw'],z[component+'_stored'],equal_nan=True,atol=1e-12,rtol=1e-12)
        count+=len(r);rows+=len(m)
    assert count==28800 and rows==2880
    report['external']={'queries':576,'measured_calls':count,'median_rows':rows}
    s=pd.read_csv(d/'scan/observations.csv');assert len(s)==3840 and s.query_id.nunique()==960
    assert s.status.isin(['completed','censored']).all()
    assert s.loc[s.status.eq('censored'),'cost_ms'].isna().all()
    assert s.status.eq('censored').sum()==488
    report['scan']={'queries':960,'attempts':len(s),'completed':int(s.status.eq('completed').sum()),'censored':488,'no_imputed_costs':True}
    return {'status':'PASS','scope':'Numeric records and recorded audit flags; no independent graph or answer-ID verification.',**report}

def check(raw,stored,keys):
    v=raw.groupby(keys,sort=True).wall_ms.median().rename('recomputed_ms').reset_index()
    v=v.merge(stored,on=keys,validate='one_to_one')
    assert len(v)==len(stored)
    assert np.allclose(v.recomputed_ms,v.cost_ms,atol=1e-12,rtol=1e-12)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path);a=p.parse_args()
    result=verify(a.root)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
