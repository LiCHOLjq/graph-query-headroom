"""Describe all one-shot attempts without imputing censored native costs."""
from pathlib import Path
import hashlib,json,platform,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ROOT=a.data;OUT=a.output;OUT.mkdir(parents=True,exist_ok=True)
done=json.loads((ROOT/'completion.json').read_text());assert done['status']=='complete'
assert not (ROOT/'failure.json').exists()
d=pd.read_csv(ROOT/'observations.csv');policies=['Eager','RACE','Lazy','MemoLazy']
assert len(d)==3840 and d.groupby(['dataset','policy']).size().eq(192).all()
assert d.groupby('query_id').size().eq(4).all() and d.query_id.nunique()==960
completed=d.status.eq('completed');censored=d.status.eq('censored')
assert (completed|censored).all() and d.loc[censored,'cost_ms'].isna().all()
assert d.loc[censored,'pure_call_lower_bound_ms'].isna().all()
assert d.loc[completed,'cost_ms'].gt(0).all()
assert d.loc[completed,'output_matches_retained'].astype(str).str.lower().eq('true').all()
rows=[]
for corpus,g in [('ALL',d),*list(d.groupby('dataset'))]:
    for policy in policies:
        v=g[g.policy.eq(policy)];cost=v.loc[v.status.eq('completed'),'cost_ms']
        rows.append(dict(corpus=corpus,policy=policy,attempts=len(v),completed=len(cost),censored=int(v.status.eq('censored').sum()),
            completion_fraction=len(cost)/len(v),completed_within_100ms=int(cost.le(100).sum()),completed_late=int(cost.gt(100).sum()),
            observed_mean_ms=cost.mean(),observed_median_ms=cost.median(),observed_p95_ms=cost.quantile(.95),observed_max_ms=cost.max()))
summary=pd.DataFrame(rows);summary.to_csv(OUT/'completion_summary.csv',index=False)
coverage=[];subset=[]
for corpus,g in [('ALL',d),*list(d.groupby('dataset'))]:
    y=g.pivot(index='query_id',columns='policy',values='cost_ms')[policies]
    full=y.notna().all(axis=1);eager_known=y.Eager.notna()
    coverage.append(dict(corpus=corpus,queries=len(y),all_four_complete=int(full.sum()),any_censored=int((~full).sum()),
        eager_complete=int(eager_known.sum()),full_workload_native_oracle_identified=False,
        full_workload_empirical_oracle_cost_ms=None,reason='Censored native call costs remain unknown; this is a one-shot fork regime'))
    if full.any():
        z=y.loc[full]
        subset.append(dict(corpus=corpus,queries=len(z),total_queries=len(y),scope='conditional on all four policy calls completing; selection-biased diagnostic',
            eager_observed_oracle_ratio=z.Eager.sum()/z.min(axis=1).sum(),
            pair_subset_oracle_over_four=z[['Eager','RACE']].min(axis=1).sum()/z.min(axis=1).sum(),
            observed_oracle_budget_us=1000*(z.Eager-z.min(axis=1)).mean()))
pd.DataFrame(coverage).to_csv(OUT/'query_completion_coverage.csv',index=False)
pd.DataFrame(subset).to_csv(OUT/'all_completed_subset_diagnostic.csv',index=False)
race=[];raw_count=0;errors=0
for path in ROOT.glob('*/raw.jsonl'):
    for line in path.read_text().splitlines():
        r=json.loads(line);raw_count+=1;errors+=r['status']=='error'
        if r['status']=='censored':race.append(r)
assert raw_count==3840 and errors==0
audit=dict(status='complete_budgeted_scan',attempts=raw_count,queries=960,completed=int(completed.sum()),censored=int(censored.sum()),
    explicit_prewarm_calls=0,reference_queries_audited=960,returned_outputs_matching_retained=int(completed.sum()),
    censored_outputs_not_claimed_correct=int(censored.sum()),
    end_timestamp_present_after_join_censored=sum(r['end_timestamp_present_after_join'] for r in race),
    watchdog_observed_elapsed_ms_median=float(np.median([r['watchdog_observed_elapsed_ms'] for r in race])) if race else None,
    full_uncensored_four_policy_benchmark_completed=False,full_native_oracle_identified=False,
    input_sha256=hashlib.sha256((ROOT/'observations.csv').read_bytes()).hexdigest(),
    analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    analysis_environment={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},
    interpretation='No censored cost is imputed. Conditional observed costs and all-completed subsets do not describe unconditional latency or full-workload headroom.')
(OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
plt.rcParams.update({'font.size':8.5,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(3.35,2.65),constrained_layout=True);s=summary[summary.corpus.eq('ALL')].set_index('policy').loc[policies]
left=np.zeros(4)
for col,label,color in [('completed_within_100ms','Returned: measured ≤100 ms','#0072B2'),('completed_late','Returned: measured >100 ms','#E69F00'),('censored','Censored by supervisor','#999999')]:
    v=100*s[col].to_numpy()/s.attempts.to_numpy();ax.barh(np.arange(4),v,left=left,color=color,label=label,height=.55);left+=v
ax.set_yticks(np.arange(4),policies);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Policy/query attempts (%)')
ax.set_title('Budgeted four-policy scan',fontsize=9,loc='left');ax.spines[['top','right']].set_visible(False)
fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncols=1,fontsize=7.5,frameon=False)
for ext in ['pdf','png','svg']:fig.savefig(OUT/f'fig_rich_completion.{ext}',dpi=300)
plt.close(fig)
(OUT/'figure_caption.md').write_text('Completion coverage for all 960 retained queries and four original policies (3,840 one-shot attempts). Each child inherits a resident immutable Engine through fork; its predicate state is fresh and no explicit prewarm is performed. The parent applies a nominal 100-ms supervised completion-publication budget, with possible watchdog/publication races and observation overshoot. Returned calls retain their measured cost, including start/state instrumentation and call-internal COW effects; censored calls have no imputed native cost. Completion categories are descriptive, and the scan does not identify the full native four-policy oracle or replace the incomplete repeated benchmark.\n',encoding='utf8')
print(summary[summary.corpus.eq('ALL')].to_string(index=False));print(json.dumps(audit,indent=2));print(pd.DataFrame(coverage).to_string(index=False))
