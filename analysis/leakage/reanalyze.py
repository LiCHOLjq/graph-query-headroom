"""Refit the fixed diagnostic feature-availability controls on numerical caches.

Result/selectivity and direct-cost features are intentionally invalid online
inputs. They are diagnostic controls, not proposed deployable improvements.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
ROOT=Path(__file__).resolve().parents[2]
DATA=Path(__file__).resolve().parent/'data'
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
HERE=a.output;HERE.mkdir(parents=True,exist_ok=True)
def read_csv(path):return pd.read_csv(path)
def posthoc(stage,meta,quality):
    z=pd.read_csv(DATA/f'posthoc_features_{stage}.csv',dtype={'query_id':str})
    return z.set_index('query_id').reindex(meta.index)
def leakage(a,quality):
    raw=pd.concat([read_csv(ROOT/'analysis/retained/data/costs.csv'),a],ignore_index=True);raw.query_id=raw.query_id.astype(str)
    features=sorted(c for c in raw if c.startswith('f_') and c not in {'f_noise','f_overlap'});raw[features]=raw[features].fillna(0)
    rows=[];folds=[];availability=[]
    for stage,sd in raw.groupby('stage'):
        policies=sorted(sd.policy.unique());Ydf=sd.pivot(index='query_id',columns='policy',values='cost_ms')[policies]
        meta=sd.drop_duplicates('query_id').set_index('query_id').loc[Ydf.index];Y=Ydf.to_numpy();base=meta[features]
        ph=posthoc(stage,meta,quality);assert ph.notna().all().all()
        ph.to_csv(HERE/f'posthoc_features_{stage}.csv')
        availability.append({'stage':stage,'legal_features':features,'posthoc_quality_selectivity_features':list(ph),'target_leak_columns':['cost_'+p for p in policies], 'posthoc_counters_not_used':'Per-policy work counters are retained but not added; this ablation uses result/quality statistics, separately from target costs.'})
        for mode,X in [('admissible',base.to_numpy()),('posthoc_quality_selectivity',np.column_stack([base.to_numpy(),ph.to_numpy()])),('direct_cost_target_leak',np.column_stack([base.to_numpy(),Y]))]:
            for fold in sorted(meta.config.unique()):
                test=meta.config.eq(fold).to_numpy();train=~test
                folds.append(dict(stage=stage,mode=mode,fold=fold,train_n=int(train.sum()),test_n=int(test.sum()),test_keys_sha256=hashlib.sha256('\n'.join(map(str,meta.index[test])).encode()).hexdigest()))
                for kind in ['cost_tree','penalty_tree','uncertainty_tree']:
                    target=np.log(Y[train]) if kind=='cost_tree' else Y[train]/Y[train].min(axis=1,keepdims=True)
                    model=DecisionTreeRegressor(max_depth=4,min_samples_leaf=20,random_state=1837).fit(X[train],target)
                    if kind=='uncertainty_tree':
                        vals=model.tree_.value[:,:,0].copy();leaf=model.apply(X[train])
                        for lid in np.unique(leaf):
                            z=target[leaf==lid];vals[lid]=z.mean(axis=0)+.5*z.std(axis=0,ddof=1)/np.sqrt(len(z))
                        ids=vals[model.apply(X[test])].argmin(axis=1)
                    else:ids=model.predict(X[test]).argmin(axis=1)
                    tm=meta.loc[test];yt=Y[test];status_id=policies.index(sd.status_quo.iloc[0])
                    for i,(qid,r) in enumerate(tm.iterrows()):
                        rows.append(dict(stage=stage,config=fold,cluster=r.cluster,query_id=qid,feature_mode=mode,strategy=kind,chosen_policy=policies[ids[i]],chosen_ms=yt[i,ids[i]],oracle_ms=yt[i].min(),status_ms=yt[i,status_id]))
    pred=pd.DataFrame(rows);pred.to_csv(HERE/'leakage_predictions.csv',index=False)
    fold=pd.DataFrame(folds);assert fold.groupby(['stage','fold']).test_keys_sha256.nunique().max()==1;fold.to_csv(HERE/'leakage_folds.csv',index=False)
    (HERE/'feature_availability.json').write_text(json.dumps(availability,indent=2))
    sums=[]
    for (stage,mode,strategy),g in pred.groupby(['stage','feature_mode','strategy']):
        sums.append(dict(stage=stage,feature_mode=mode,strategy=strategy,queries=len(g),oracle_ratio=g.chosen_ms.sum()/g.oracle_ms.sum(),ratio_vs_status=g.chosen_ms.sum()/g.status_ms.sum(),mean_cost_ms=g.chosen_ms.mean(),mean_paired_difference_us=1000*(g.chosen_ms-g.status_ms).mean(),maximum_regret_ms=(g.chosen_ms-g.oracle_ms).max()))
    sm=pd.DataFrame(sums);sm.to_csv(HERE/'leakage_summary.csv',index=False)
    old=read_csv(ROOT/'analysis/retained/data/predictions.csv');old.query_id=old.query_id.astype(str)
    check=pred[pred.feature_mode=='admissible'].merge(old,on=['stage','query_id','strategy'],suffixes=('_new','_old'))
    assert len(check)==len(pred[pred.feature_mode=='admissible'])
    agree=(check.chosen_policy_new==check.chosen_policy_old).mean()
    assert agree==1, f'Legal baseline predictions differ: {agree}'
    return sm


leakage(read_csv(ROOT/'analysis/retained/data/costs_bcr.csv'),None)
