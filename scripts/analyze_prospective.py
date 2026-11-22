"""Apply the frozen analysis to completed prospective measurements."""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')
def holm(p):
    p=np.asarray(p);ix=np.argsort(p,kind='stable');out=np.empty(len(p));out[ix]=np.minimum(1,np.maximum.accumulate(p[ix]*(len(p)-np.arange(len(p)))));return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=HERE/'results_r01');ap.add_argument('--frozen',type=Path,default=HERE/'frozen_r01');ap.add_argument('--output',type=Path,default=HERE/'analysis_r01');a=ap.parse_args();t0=time.perf_counter()
    plan=json.loads((a.frozen/'preanalysis_plan.json').read_text());done=json.loads((a.results/'COMPLETE.json').read_text());assert done['status']=='PASS'
    env=json.loads((a.results/'environment.json').read_text());assert env['plan_sha256']==sha(a.frozen/'preanalysis_plan.json')
    p=pd.read_csv(a.results/'costs.csv',dtype={'query_id':str});arms=plan['arms'];assert len(p)==3200*8 and not p.duplicated(['query_id','arm']).any();assert p.repeats.eq(8).all()
    raw=pd.concat([pd.read_csv(a.results/name/'raw.csv.gz').assign(config=name) for name in plan['corpora']],ignore_index=True)
    assert len(raw)==204800 and raw.groupby(['query_id','arm','position']).size().eq(1).all() and raw.groupby(['query_id','repeat','arm']).size().eq(1).all()
    assert raw.groupby('query_id').answer_hash.nunique().eq(1).all()
    check=raw.groupby(['query_id','arm']).wall_ms.median().rename('recomputed').reset_index().merge(p,on=['query_id','arm'],validate='one_to_one');assert np.allclose(check.recomputed,check.cost_ms,rtol=1e-12,atol=1e-12)
    costs=p.pivot(index=['config','seed','query_id'],columns='arm',values='cost_ms')[arms];assert not costs.isna().any().any()
    blocks=costs.groupby(level=['config','seed']).sum();assert len(blocks)==50
    B=plan['inference']['bootstrap_replicates'];rng=np.random.default_rng(plan['inference']['bootstrap_seed']);draws=np.zeros((B,len(arms)))
    corpus_draws={}
    for name in plan['corpora']:
        z=blocks.loc[name].sort_index().to_numpy();assert z.shape==(10,8)
        ix=rng.integers(0,10,size=(B,10));d=z[ix].sum(axis=1);draws+=d;corpus_draws[name]=d
    total=costs.sum().to_numpy();base=total[0];ratio=total/base;bsratio=draws/draws[:,[0]]
    deltas=blocks.to_numpy()[:,1:]-blocks.to_numpy()[:,[0]];observed=np.abs(deltas.sum(axis=0));counts=np.zeros(7,dtype=int)
    trng=np.random.default_rng(plan['inference']['test_seed'])
    for _ in range(20):
        signs=trng.choice(np.array([-1,1],dtype=np.int8),size=(5000,50));counts+=(np.abs(signs@deltas)>=observed-1e-12).sum(axis=0)
    ps=(counts+1)/100001;adj=holm(ps);rows=[];configrows=[];seedrows=[];tails=[]
    oracle=costs[['Eager','RACE']].min(axis=1);quality=[]
    for name in plan['corpora']:
        q=json.loads((a.results/name/'quality.json').read_text());assert len(q)==640 and all(x['all_outputs_equal'] and x['all_outputs_legal'] for x in q);quality.extend(dict(config=name,**x) for x in q)
    quality=pd.DataFrame(quality)
    for j,arm in enumerate(arms):
        values=costs[arm];diff=values-costs.Eager;lo,hi=np.quantile(bsratio[:,j],[.025,.975]);dlo,dhi=np.quantile((draws[:,j]-draws[:,0])/len(costs),[.025,.975])
        reverse=draws[:,0]/draws[:,j];upper=float(np.quantile(reverse,1-.05/7))
        cfg_ratio=[]
        for name,g in costs.groupby(level='config'):
            br=corpus_draws[name];cr=br[:,j]/br[:,0];cl,ch=np.quantile(cr,[.025,.975]);cfg_ratio.append(g[arm].sum()/g.Eager.sum())
            configrows.append(dict(config=name,arm=arm,queries=len(g),mean_ms=g[arm].mean(),median_ms=g[arm].median(),ratio_vs_Eager=g[arm].sum()/g.Eager.sum(),ratio_ci_low=cl,ratio_ci_high=ch,mean_difference_ms=(g[arm]-g.Eager).mean(),mean_fixed_policy_oracle_ms=g[['Eager','RACE']].min(axis=1).mean()))
        rows.append(dict(arm=arm,queries=len(costs),clusters=50,total_ms=total[j],mean_ms=values.mean(),median_ms=values.median(),ratio_vs_Eager=ratio[j],ratio_ci_low=lo,ratio_ci_high=hi,mean_difference_ms=diff.mean(),difference_ci_low_ms=dlo,difference_ci_high_ms=dhi,equal_corpus_macro_ratio=np.mean(cfg_ratio),p_signflip_mc=float(ps[j-1]) if j else 1.,p_holm7=float(adj[j-1]) if j else 1.,Eager_vs_arm_ratio=base/total[j],Eager_vs_arm_simult_upper=upper,noninferior_at_1pct=bool(upper<=1.01) if j else True))
        for (name,seed),g in costs.groupby(level=['config','seed']):seedrows.append(dict(config=name,seed=seed,arm=arm,queries=len(g),sum_ms=g[arm].sum(),ratio_vs_Eager=g[arm].sum()/g.Eager.sum(),mean_difference_ms=(g[arm]-g.Eager).mean()))
        excess=values-oracle
        tails.append(dict(arm=arm,mean_fixed_oracle_excess_ms=excess.mean(),median_fixed_oracle_excess_ms=excess.median(),p95_fixed_oracle_excess_ms=excess.quantile(.95),p99_fixed_oracle_excess_ms=excess.quantile(.99),max_fixed_oracle_excess_ms=excess.max(),min_fixed_oracle_excess_ms=excess.min(),negative_fraction=float(excess.lt(0).mean())))
    summary=pd.DataFrame(rows);cfg=pd.DataFrame(configrows);seed=pd.DataFrame(seedrows);tails=pd.DataFrame(tails)
    positions=raw.groupby(['arm','position']).agg(timed_requests=('wall_ms','size'),mean_ms=('wall_ms','mean'),median_ms=('wall_ms','median')).reset_index()
    components=raw.assign(outer_minus_session_ms=raw.wall_ms-raw.engine_session_ms).groupby('arm').agg(mean_wall_ms=('wall_ms','mean'),mean_session_ms=('engine_session_ms','mean'),mean_outer_minus_session_ms=('outer_minus_session_ms','mean'),median_outer_minus_session_ms=('outer_minus_session_ms','median')).reset_index()
    choice=p.groupby(['config','arm','chosen_policy']).size().rename('queries').reset_index()
    qualitysum=quality.groupby('config').agg(queries=('query_id','size'),mean_recall=('recall','mean'),exact_matches=('exact_match','sum'),empty_queries=('empty_exact','sum')).reset_index()
    a.output.mkdir(parents=True,exist_ok=True)
    for df,name in [(summary,'summary.csv'),(cfg,'configuration_summary.csv'),(seed,'seed_results.csv'),(tails,'fixed_oracle_excess.csv'),(positions,'position_audit.csv'),(components,'timing_components.csv'),(choice,'policy_choices.csv'),(qualitysum,'quality_summary.csv')]:df.to_csv(a.output/name,index=False,float_format='%.17g')
    subset=summary[summary.arm!='Eager'];best=subset.loc[subset.total_ms.idxmin()];audit_pass=bool(subset.noninferior_at_1pct.all())
    report={'status':'PASS','preanalysis_plan_sha256':sha(a.frozen/'preanalysis_plan.json'),'input_cost_sha256':sha(a.results/'costs.csv'),'final_queries':len(costs),'timed_requests':len(raw),'all_raw_medians_recomputed':True,'every_arm_every_position_per_query':True,'every_query_all_answers_equal':True,'frozen_decision':'Eager','same_graph_new_query_scope':True,'new_mean_fixed_oracle_headroom_ms':float((costs.Eager-oracle).mean()),'new_median_fixed_oracle_headroom_ms':float((costs.Eager-oracle).median()),'best_non_Eager_arm_post_selection':best.arm,'Eager_ratio_to_best_observed_non_Eager_arm':float(best.Eager_vs_arm_ratio),'largest_simultaneous_upper_Eager_vs_non_Eager':float(subset.Eager_vs_arm_simult_upper.max()),'engineering_margin':.01,'simultaneous_noninferiority_within_1pct':audit_pass,'interpretation':'The frozen Eager decision meets the predeclared engineering noninferiority criterion for these same-graph new queries.' if audit_pass else 'The predeclared 1% noninferiority criterion is not met. Report this as unresolved or contrary evidence, not as validation.','bootstrap_replicates':B,'bootstrap_strata':5,'bootstrap_blocks_per_stratum':10,'signflip_mc_patterns':100000,'holm_family':7,'seconds':time.perf_counter()-t0,'notes':['Fixed-policy empirical oracle uses the within-query minimum of noisy Eager/RACE medians and can be optimistic.','Router excess over separately timed fixed-policy oracle can be negative from timing variation; values are not clipped.','Outer-minus-session time includes feature extraction, dispatch, bindings and return construction; it is not an isolated inference benchmark.','A best-arm point is post-selection; simultaneous bounds, not that point alone, govern the frozen audit criterion.','Resident inputs and matched warmup scope do not include cold startup, network or cache-cold production behavior.']}
    write(a.output/'analysis_audit.json',report)
    lines=['# Prospective same-graph new-query result','',report['interpretation'],'',f"Queries: {len(costs)}; independent query-seed blocks: 50 across five fixed graph strata; timed requests: {len(raw)}.",'','| Arm | Full-path mean ms | Ratio / Eager | 95% paired block CI | Holm p |','|---|---:|---:|---:|---:|']
    for r in summary.itertuples():lines.append(f'| {r.arm} | {r.mean_ms:.6f} | {r.ratio_vs_Eager:.6f} | [{r.ratio_ci_low:.6f}, {r.ratio_ci_high:.6f}] | {r.p_holm7:.6g} |')
    lines+=['',f"Frozen decision: Eager. Worst simultaneous one-sided Eager/alternative upper bound: {report['largest_simultaneous_upper_Eager_vs_non_Eager']:.6f}; predeclared threshold: 1.01.",'','This is new-query validation on the same five graphs and generator family. It is not an unseen-graph result. Timed requests include routing feature acquisition and selected execution; immutable loading, network and quality audits are excluded.','',*['- '+x for x in report['notes']]]
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
