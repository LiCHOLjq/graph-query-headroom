"""Analyze all prespecified cells only after both complete sessions exist."""
from pathlib import Path
import argparse,hashlib,json,platform,sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=HERE/'results');p.add_argument('--output',type=Path,default=HERE/'analysis');p.add_argument('--pair-only',action='store_true');a=p.parse_args()
CELLS=['pair_balanced_warm','pair_balanced_no_prewarm','pair_random_warm','pair_random_no_prewarm','rich_balanced_warm','rich_balanced_no_prewarm']
if a.pair_only:CELLS=CELLS[:4]
expected_cost_rows=15360 if a.pair_only else 30720
expected_quality_rows=7680 if a.pair_only else 11520
expected_measured=122880 if a.pair_only else 245760
expected_warmups=61440 if a.pair_only else 122880
DATASETS=['openalex_db','amazon_product','ldbc_snb','ogbn_arxiv_full','ogbn_products']
assert (a.results/'completion.json').exists(),'Joint analysis requires both completed sessions'
assert not (a.results/'failure.json').exists(),'An incomplete/failed experiment is not analyzed as complete'
files=[];costs=[];qualities=[];completions=[]
for session in ['A','B']:
  for corpus in DATASETS:
    for cell in CELLS:
      path=a.results/session/corpus/cell
      done=json.loads((path/'completion.json').read_text())
      assert done['status']=='complete' and done['queries']==192 and done['repeats']==8
      assert done['legality_failures']==0
      completions.append(done)
      for name in ['costs.csv','quality.csv','completion.json','environment.json','schedule.json']:
          f=path/name;files.append({'path':str(f.relative_to(a.results)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
      costs.append(pd.read_csv(path/'costs.csv'))
      q=pd.read_csv(path/'quality.csv');q['session']=session;q['cell']=cell;q['config']=corpus;qualities.append(q)
d=pd.concat(costs,ignore_index=True);quality=pd.concat(qualities,ignore_index=True)
assert len(d)==expected_cost_rows and len(quality)==expected_quality_rows
assert sum(z['measured_calls'] for z in completions)==expected_measured
assert sum(z['untimed_prewarm_calls'] for z in completions)==expected_warmups
assert quality.groupby('query_id').recall.nunique().max()==1
assert quality.groupby('query_id').eligible_count.nunique().max()==1
assert d.repeats.eq(8).all() and d.cost_ms.gt(0).all()
assert quality.legal.all() and quality.all_policies_repeats_match_retained.all()
a.output.mkdir(parents=True,exist_ok=True)
d.to_csv(a.output/'costs.csv',index=False);quality.to_csv(a.output/'quality.csv',index=False)
pd.DataFrame(completions).to_csv(a.output/'completion_audit.csv',index=False)

def paired_stats(values,meta):
    z=pd.DataFrame({'difference_ms':values,'cluster':meta.cluster},index=meta.index)
    blocks=z.groupby('cluster').difference_ms.agg(['sum','size','mean'])
    result={'queries':len(z),'blocks':len(blocks),'mean_difference_us':1000*z.difference_ms.mean(),
            'median_difference_us':1000*z.difference_ms.median(),'block_min_us':1000*blocks['mean'].min(),
            'block_max_us':1000*blocks['mean'].max(),'ci95_low_us':np.nan,'ci95_high_us':np.nan}
    if len(blocks)>=10:
        ix=np.random.default_rng(90171).integers(len(blocks),size=(10000,len(blocks)))
        boot=blocks['sum'].to_numpy()[ix].sum(axis=1)/blocks['size'].to_numpy()[ix].sum(axis=1)
        result['ci95_low_us'],result['ci95_high_us']=1000*np.quantile(boot,[.025,.975])
    return result

matrices={};metas={};summaries=[];fixed=[];winner_counts=[]
for (session,cell),g in d.groupby(['session','cell']):
    Y=g.pivot(index='query_id',columns='policy',values='cost_ms').sort_index()
    meta=g.drop_duplicates('query_id').set_index('query_id').loc[Y.index]
    q=quality[(quality.session==session)&(quality.cell==cell)].set_index('query_id').loc[Y.index]
    meta=meta.join(q[['eligible_count','recall']]);meta['empty']=meta.eligible_count.eq(0)
    assert len(Y)==960 and not Y.isna().any().any()
    matrices[session,cell]=Y;metas[session,cell]=meta
    for corpus in ['ALL',*DATASETS]:
      ids=meta.index if corpus=='ALL' else meta.index[meta.config.eq(corpus)]
      for stratum in ['all','empty','nonempty']:
        ix=ids if stratum=='all' else ids[meta.loc[ids,'empty'].eq(stratum=='empty')]
        y=Y.loc[ix];m=meta.loc[ix];e=y.Eager;o=y.min(axis=1);o2=y[['Eager','RACE']].min(axis=1)
        summaries.append(dict(session=session,cell=cell,corpus=corpus,stratum=stratum,queries=len(y),policies=len(y.columns),
            eager_mean_ms=e.mean(),oracle_mean_ms=o.mean(),eager_oracle_ratio=e.sum()/o.sum(),
            oracle_budget_mean_us=1000*(e-o).mean(),oracle_budget_median_us=1000*(e-o).median(),
            oracle_budget_p95_us=1000*(e-o).quantile(.95),oracle_budget_max_us=1000*(e-o).max(),
            pair_subset_oracle_ratio=e.sum()/o2.sum(),pair_to_full_oracle_ratio=o2.sum()/o.sum(),
            added_oracle_budget_us=1000*(o2-o).mean(),macro_recall=m.recall.mean()))
      for policy in Y.columns:
        y=Y.loc[ids];cost=y[policy]
        fixed.append(dict(session=session,cell=cell,corpus=corpus,policy=policy,queries=len(y),mean_ms=cost.mean(),
             median_ms=cost.median(),p95_ms=cost.quantile(.95),sum_ms=cost.sum(),ratio_vs_eager=cost.sum()/y.Eager.sum(),
             oracle_ratio=cost.sum()/y.min(axis=1).sum()))
      wins=Y.loc[ids].idxmin(axis=1).value_counts()
      for policy in Y.columns:
        winner_counts.append(dict(session=session,cell=cell,corpus=corpus,policy=policy,wins=int(wins.get(policy,0)),queries=len(ids)))
pd.DataFrame(summaries).to_csv(a.output/'cell_summary.csv',index=False)
pd.DataFrame(fixed).to_csv(a.output/'fixed_policy_summary.csv',index=False)
pd.DataFrame(winner_counts).to_csv(a.output/'winner_counts.csv',index=False)

contrasts=[];block_rows=[]
for session in ['A','B']:
    meta=metas[session,'pair_balanced_warm']
    for measure in ['Eager','RACE','empirical_oracle','Eager_oracle_gap']:
        vals={}
        for cell in CELLS[:4]:
            y=matrices[session,cell]
            vals[cell]=y.min(axis=1) if measure=='empirical_oracle' else (y.Eager-y.min(axis=1) if measure=='Eager_oracle_gap' else y[measure])
        c={
          'warm_minus_no_prewarm_balanced':vals['pair_balanced_warm']-vals['pair_balanced_no_prewarm'],
          'warm_minus_no_prewarm_random':vals['pair_random_warm']-vals['pair_random_no_prewarm'],
          'balanced_minus_random_warm':vals['pair_balanced_warm']-vals['pair_random_warm'],
          'balanced_minus_random_no_prewarm':vals['pair_balanced_no_prewarm']-vals['pair_random_no_prewarm'],
          'prewarm_by_order_difference_in_differences':(vals['pair_balanced_warm']-vals['pair_balanced_no_prewarm'])-(vals['pair_random_warm']-vals['pair_random_no_prewarm'])}
        for name,diff in c.items():
          for corpus in ['ALL',*DATASETS]:
            ids=meta.index if corpus=='ALL' else meta.index[meta.config.eq(corpus)]
            contrasts.append(dict(session=session,measure=measure,contrast=name,corpus=corpus,**paired_stats(diff.loc[ids],meta.loc[ids])))
          for cluster,g in pd.DataFrame({'difference_ms':diff,'cluster':meta.cluster}).groupby('cluster'):
            block_rows.append(dict(session=session,measure=measure,contrast=name,cluster=cluster,queries=len(g),mean_difference_us=1000*g.difference_ms.mean()))
pd.DataFrame(contrasts).to_csv(a.output/'factorial_contrasts.csv',index=False)
pd.DataFrame(block_rows).to_csv(a.output/'factorial_seed_blocks.csv',index=False)

cross=[];cross_cases=[]
for cell in CELLS:
    A=matrices['A',cell];B=matrices['B',cell];meta=metas['A',cell]
    assert A.index.equals(B.index) and A.columns.equals(B.columns)
    wa=A.idxmin(axis=1);wb=B.idxmin(axis=1)
    charged=pd.Series(B.to_numpy()[np.arange(len(B)),A.to_numpy().argmin(axis=1)],index=B.index)
    for qid in A.index:
        cross_cases.append(dict(cell=cell,query_id=qid,config=meta.loc[qid,'config'],cluster=meta.loc[qid,'cluster'],
            winner_a=wa[qid],winner_b=wb[qid],eager_b_ms=B.loc[qid,'Eager'],oracle_b_ms=B.loc[qid].min(),frozen_a_on_b_ms=charged[qid]))
    for corpus in ['ALL',*DATASETS]:
        ids=meta.index if corpus=='ALL' else meta.index[meta.config.eq(corpus)]
        aa=wa.loc[ids];bb=wb.loc[ids];pa=aa.value_counts(normalize=True);pb=bb.value_counts(normalize=True)
        chance=sum(pa.get(pol,0)*pb.get(pol,0) for pol in A.columns);agreement=aa.eq(bb).mean()
        y=B.loc[ids];f=charged.loc[ids]
        cross.append(dict(cell=cell,corpus=corpus,eager_b_oracle_b=y.Eager.sum()/y.min(axis=1).sum(),
            frozen_a_oracle_b=f.sum()/y.min(axis=1).sum(),frozen_a_eager_b=f.sum()/y.Eager.sum(),
            observed_agreement=agreement,marginal_independence_agreement=chance,kappa=(agreement-chance)/(1-chance) if chance<1 else np.nan,
            **paired_stats(f-y.Eager,meta.loc[ids])))
pd.DataFrame(cross).to_csv(a.output/'cross_session_summary.csv',index=False)
pd.DataFrame(cross_cases).to_csv(a.output/'cross_session_cases.csv',index=False)
(a.output/'analysis_manifest.json').write_text(json.dumps({'inputs':files,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'both_sessions_complete':True,'queries':960,'measured_calls':expected_measured,'pair_only_iteration':a.pair_only,'bootstrap_repetitions':10000,'bootstrap_seed':90171,
    'analysis_environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__},
    'interval':'paired corpus-seed percentile intervals when 15 blocks; three-block per-corpus ranges remain descriptive',
    'claims':'retained-query repeated-session condition effects; no cold-cache or prospective claim'},indent=2),encoding='utf8')
print(pd.DataFrame(summaries).query("corpus == 'ALL' and stratum == 'all'").to_string(index=False))
print(pd.DataFrame(cross).query("corpus == 'ALL'").to_string(index=False))
