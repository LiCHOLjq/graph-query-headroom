"""Summarize complete external-calibration records without selecting a best ef."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=args.root.resolve();out=args.output;out.mkdir(parents=True,exist_ok=True)
corpora=['openalex_db','amazon_product','ogbn_arxiv_full']
costs=[];quality=[];provenance=[]
for corpus in corpora:
    folder=root/'results'/corpus;done=json.loads((folder/'completion.json').read_text())
    assert done['status']=='complete' and done['queries']==192 and done['repeats']==10
    for name,suffix in [('raw','jsonl'),('costs','csv'),('quality','csv')]:
        digest=hashlib.sha256((folder/(name+'.'+suffix)).read_bytes()).hexdigest()
        assert digest==done[name+'_sha256'],(corpus,name)
    costs.append(pd.read_csv(folder/'costs.csv'));quality.append(pd.read_csv(folder/'quality.csv'));provenance.append(done)
c=pd.concat(costs,ignore_index=True);q=pd.concat(quality,ignore_index=True)
x=c.merge(q,on=['corpus','query_id','seed','policy'],validate='one_to_one',suffixes=('','_quality'))
assert (x.eligible_count==x.eligible_count_quality).all()
assert (x.guarded_invalid_count==0).all()
rows=[];seed_rows=[]
for corpus,group in x.groupby('corpus',sort=False):
    base=group[group.policy=='Eager'].set_index('query_id')
    for policy,g in group.groupby('policy',sort=False):
        g=g.set_index('query_id').loc[base.index]
        nonempty=g.reference_count>0
        total=g.reference_count.sum()
        rows.append({'corpus':corpus,'policy':policy,'queries':len(g),'empty_queries':int((~nonempty).sum()),
             'mean_total_ms':g.cost_ms.mean(),'median_total_ms':g.cost_ms.median(),
             'nonempty_mean_total_ms':g.loc[nonempty,'cost_ms'].mean(),
             'nonempty_median_total_ms':g.loc[nonempty,'cost_ms'].median(),
             'empty_mean_total_ms':g.loc[~nonempty,'cost_ms'].mean(),
             'mean_bitmap_ms':g.bitmap_ms.mean(),'mean_search_ms':g.search_ms.mean(),
             'ratio_vs_current_Eager':g.cost_ms.sum()/base.cost_ms.sum(),
             'nonempty_ratio_vs_current_Eager':g.loc[nonempty,'cost_ms'].sum()/base.loc[nonempty,'cost_ms'].sum() if nonempty.any() else float('nan'),
             'mean_eligible_count':g.eligible_count.mean(),'median_eligible_count':g.eligible_count.median(),
             'macro_recall':g.recall.mean(),'nonempty_macro_recall':g.loc[nonempty,'recall'].mean(),
             'micro_recall':float((g.recall*g.reference_count).sum()/total) if total else 1.0,
             'raw_macro_recall':g.raw_recall.mean(),'mean_raw_returned_precision':g.raw_returned_precision.mean(),
             'mean_guarded_returned_precision':g.returned_precision.mean(),
             'raw_illegal_queries':int((g.raw_invalid_count>0).sum()),'raw_illegal_ids':int(g.raw_invalid_count.sum()),
             'guarded_illegal_queries':int((g.guarded_invalid_count>0).sum()),
             'raw_duplicate_queries':int((g.raw_duplicate_count>0).sum()),
             'underfilled_queries':int(g.underfilled.sum()),'eligible_ge_k_underfilled_queries':int(g.eligible_ge_k_underfilled.sum()),
             'mean_returned_size':g.returned_size.mean(),'mean_raw_returned_size':g.raw_returned_size.mean(),
             'exact_ordered_fraction':g.exact_ordered_match.mean(),'exact_set_fraction':g.exact_set_match.mean(),
             'raw_exact_ordered_fraction':g.raw_exact_ordered_match.mean(),'raw_exact_set_fraction':g.raw_exact_set_match.mean()})
        for seed,sg in g.groupby('seed'):
            b=base.loc[sg.index]
            seed_rows.append({'corpus':corpus,'policy':policy,'seed':int(seed),'queries':len(sg),
                'ratio_vs_current_Eager':sg.cost_ms.sum()/b.cost_ms.sum(),'mean_total_ms':sg.cost_ms.mean(),
                'macro_recall':sg.recall.mean(),'raw_illegal_queries':int((sg.raw_invalid_count>0).sum())})
summary=pd.DataFrame(rows);seeds=pd.DataFrame(seed_rows)
summary.to_csv(out/'summary.csv',index=False);seeds.to_csv(out/'seed_summary.csv',index=False)
audit={'status':'PASS','corpora':corpora,'query_cases':576,'cost_quality_rows':len(x),
       'timed_calls':sum(p['measurements'] for p in provenance),'guarded_legality_failures':0,
       'all_declared_ef_settings_reported':True,'external_baseline':'ACORN-1 + explicit eligibility selector + output guard',
       'no_exact_external_baseline_claim':True,'inference':'descriptive three-seed per-corpus summaries; no p-values',
       'phase_note':'mean of component medians can differ from mean of total medians',
       'summary_sha256':hashlib.sha256((out/'summary.csv').read_bytes()).hexdigest(),
       'seed_summary_sha256':hashlib.sha256((out/'seed_summary.csv').read_bytes()).hexdigest()}
(out/'summary_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit))
print(summary[['corpus','policy','mean_total_ms','ratio_vs_current_Eager','macro_recall','raw_illegal_queries','eligible_ge_k_underfilled_queries']].to_string(index=False))
