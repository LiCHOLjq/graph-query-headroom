"""Reproduce the retained-stage aggregate audit without refitting or altering experiments.

python reanalyze.py --refresh-inputs  # first run in the research workspace
python reanalyze.py                  # thereafter, use adjacent frozen inputs
Outputs: CSV/JSON/README in this directory; two TeX inputs and two figures in
the version's source directory. No other experiment or manuscript is written.
"""
from pathlib import Path
import argparse, hashlib, itertools, json, platform, shutil, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
SOURCE = HERE.parents[1] / 'outputs' / 'retained_figures'
ORDER = ['status_quo','train_best','most_frequent','threshold','cost_tree','penalty_tree','uncertainty_tree']
LABEL = dict(zip(ORDER,['Status quo','Training-best','Frequent-winner','Rule/default','Log-cost','Penalty','Uncertainty']))
SHORT = dict(zip(ORDER,['Status quo','Train. best','Most freq.','Rule/default','Log-cost','Penalty','Uncertainty']))
PUBLIC = {'JQ8':'BCR','JQ10':'EGR','JQ13':'RPE','JQ13_legacy':'RPE legacy'}
STAGES = ['BCR','EGR','RPE','RPE legacy']
PRETTY = {'amazon_product':'Amazon','ldbc_snb':'LDBC SNB','ogbn_arxiv_full':'ogbn-arxiv','ogbn_products':'ogbn-products','openalex_db':'OpenAlex','syn_base':'Synthetic','support_churn':'SupportChurn','hetionet':'Hetionet','openalex':'OpenAlexCites','syn_small':'Synthetic small','syn_large':'Synthetic large','syn_clean':'Synthetic clean'}
COLORS = dict(zip(ORDER,['#556677','#7A5195','#009E73','#D89000','#0072B2','#D55E00','#CC79A7']))
BOOTSTRAP = 2000
RNG_SEED = 18061115

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def config_label(x): return PRETTY.get(x,x.replace('n100000_','100k, ').replace('n10000_','10k, ').replace('deg','d='))
def tex(x): return str(x).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def num(x): return f'{x:.6f}' if abs(x)<.1 else f'{x:.4f}'
def write_csv(frame,name): frame.to_csv(HERE/name,index=False,float_format='%.17g')

def refresh():
    root=next(p for p in HERE.parents if (p/'experiments/results/predictions.csv').is_file())
    paths={'predictions.csv':'results/predictions.csv','summary.csv':'results/summary.csv','costs.csv':'data/costs.csv','costs_bcr.csv':'data/costs_jq8.csv','router_timings.csv':'results/router_timings.csv','native_overhead.csv':'results/native_overhead.csv','folds.csv':'results/folds.csv'}
    DATA.mkdir(exist_ok=True); manifest={}
    for name,rel in paths.items():
        source=root/'experiments'/rel; shutil.copy2(source,DATA/name)
        manifest[name]={'original_relative_path':'experiments/'+rel,'sha256':sha(source),'bytes':source.stat().st_size}
    (HERE/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')

def exact_test(g):
    d=g.groupby('cluster',sort=True).apply(lambda z:(z.chosen_ms-z.status_ms).sum(),include_groups=False).to_numpy()
    obs=abs(d.sum()); null=np.abs(np.array(list(itertools.product([-1,1],repeat=len(d))))@d)
    p=float(np.mean(null>=obs-1e-12)) if np.any(d) else 1.
    m=int(np.count_nonzero(d))
    return dict(p_raw=p,clusters=len(d),nonzero_blocks=m,positive_blocks=int((d>0).sum()),negative_blocks=int((d<0).sum()),zero_blocks=int((d==0).sum()),nominal_raw_resolution=2./2**len(d),effective_raw_lower_bound=2./2**m if m else 1.)

def bootstrap(g,rng):
    z=g.assign(delta=g.chosen_ms-g.status_ms).groupby('cluster').agg(delta=('delta','sum'),status=('status_ms','sum'),n=('status_ms','size')).to_numpy()
    a=z[rng.integers(0,len(z),size=(BOOTSTRAP,len(z)))].sum(axis=1)
    dm=np.quantile(a[:,0]/a[:,2],[.025,.975]); relative=np.quantile(a[:,0]/a[:,1],[.025,.975])
    return dict(delta_mean_ci_low_ms=dm[0],delta_mean_ci_high_ms=dm[1],relative_delta_ci_low=relative[0],relative_delta_ci_high=relative[1])

def holm_table(t,family):
    z=t.copy().sort_values(['p_raw','stage','strategy'],kind='stable').reset_index(drop=True)
    z['family']=family; z['family_size']=len(z); z['rank']=np.arange(1,len(z)+1)
    z['multiplier']=len(z)-np.arange(len(z)); z['rank_scaled_p']=z.p_raw*z.multiplier
    z['p_adjusted']=np.minimum(1,np.maximum.accumulate(z.rank_scaled_p))
    return z

def save_figure(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(SOURCE/f'{name}.{ext}',dpi=300,facecolor='white',metadata={'Creator':'Retained-data aggregate audit'} if ext=='pdf' else None)
    plt.close(fig)

def plot_intervals(t):
    fig,axs=plt.subplots(3,1,figsize=(3.45,4.5))
    fig.subplots_adjust(left=.28,right=.985,bottom=.12,top=.95,hspace=.51)
    titles=['BCR: bounded-candidate retrieval','EGR: exact graph-constrained retrieval','RPE: regular-path endpoints']
    for j,(stage,ax) in enumerate(zip(STAGES,axs)):
        g=t[t.stage==stage].set_index('strategy').loc[ORDER]
        low,high=g.lower.min(),g.upper.max()
        xmin=1. if stage=='BCR' else min(1.,low-.05)
        xmax=high+(high-xmin)*(.62 if stage=='BCR' else .26)
        for i,(kind,r) in enumerate(g.iterrows()):
            ax.errorbar(r.estimate,i,xerr=[[r.estimate-r.lower],[r.upper-r.estimate]],fmt='s' if stage=='RPE' else 'o',ms=3,color=COLORS[kind],elinewidth=.8,capsize=1.7)
            ax.text(xmax-(xmax-xmin)*.015,i,f'{r.estimate:.5f}' if stage=='BCR' else f'{r.estimate:.3f}',fontsize=7,ha='right',va='center',color=COLORS[kind])
        ax.axvline(1,color='#66727C',ls=':',lw=.8);ax.set_xlim(xmin,xmax);ax.set_ylim(6.6,-.7)
        ax.set_yticks(range(7),[SHORT[k] for k in ORDER],fontsize=7.3)
        ax.tick_params(axis='y',length=0,pad=3);ax.tick_params(axis='x',labelsize=7,pad=2)
        ax.set_xticks([1.,1.001,1.002,1.003,1.004] if stage=='BCR' else ([1.,1.2,1.4,1.6] if stage=='EGR' else [1.,1.2,1.4]))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f' if stage=='BCR' else '%.1f'))
        ax.set_title(chr(97+j)+'  '+titles[j],loc='left',fontsize=8,pad=5)
        ax.grid(axis='x',alpha=.17);ax.spines['left'].set_visible(False)
    fig.text(.5,.052,'Execution cost / oracle',ha='center',fontsize=7.2)
    fig.text(.5,.023,'BCR/EGR: 95% seed-block CI; RPE: three-seed range',ha='center',fontsize=6.9)
    save_figure(fig,'fig03_results')

def plot_weights(w):
    fig,axs=plt.subplots(1,3,figsize=(7.2,3.45),gridspec_kw={'width_ratios':[1.05,1.5,1.1]})
    fig.subplots_adjust(left=.115,right=.965,bottom=.17,top=.875,wspace=.84)
    for j,(stage,ax) in enumerate(zip(STAGES,axs)):
        g=w[w.stage==stage].copy()
        if stage=='EGR': g=g.iloc[np.argsort([int(x.split('_')[0][1:])*100+int(x.split('deg')[1]) for x in g.config])]
        y=np.arange(len(g));ax.barh(y-.16,100*g.status_weight,height=.29,label='Status cost share',color='#2266A0')
        ax.barh(y+.16,100*g.oracle_weight,height=.29,label='Oracle cost share',color='#D89000')
        ax.set_yticks(y,[config_label(x) for x in g.config],fontsize=7)
        ax.invert_yaxis();ax.tick_params(axis='y',length=0,pad=3);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.set_xlabel('Cost share (%)',fontsize=8);ax.set_title(chr(97+j)+'  '+stage,loc='left',fontsize=9)
        ax.set_xlim(0,100 if stage=='RPE' else (45 if stage=='BCR' else 35))
        ax.tick_params(axis='x',labelsize=7)
        if stage=='RPE':
            k=list(g.config).index('syn_base');ax.text(98,k-.16,f'{100*g.iloc[k].status_weight:.2f}%',fontsize=7,va='center',ha='right')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,1.0),frameon=False,ncol=2,fontsize=8)
    fig.text(.5,.025,'Denominator weights for cost ratios; equal configuration weights are reported separately.',ha='center',fontsize=8)
    save_figure(fig,'fig08_cost_weights')

def plot_legacy(seeds,absolute):
    fig,ax=plt.subplots(figsize=(7.2,2.6));fig.subplots_adjust(left=.09,right=.99,bottom=.22,top=.82)
    rows=[]
    for stage,off,color,label in [('RPE',-.18,'#0072B2','RPE: strong terminal-aware cohort'),('RPE legacy',.18,'#B6C3CC','RPE: legacy synthetic-size cohort')]:
        a=absolute[(absolute.scope=='stage')&(absolute.stage==stage)].set_index('strategy').loc[ORDER]
        z=seeds[(seeds.scope=='stage')&(seeds.stage==stage)].groupby('strategy').oracle_ratio.agg(['min','max']).loc[ORDER]
        y=a.aggregate_oracle_ratio.to_numpy();lo=z['min'].to_numpy();hi=z['max'].to_numpy()
        ax.bar(np.arange(7)+off,y,.34,color=color,label=label,yerr=np.vstack([y-lo,hi-y]),error_kw=dict(ecolor='#37474F',elinewidth=.8,capsize=2))
        rows.extend(dict(stage=stage,strategy=k,estimate=y[i],lower=lo[i],upper=hi[i],interval_type='three_seed_descriptive_range') for i,k in enumerate(ORDER))
    ax.axhline(1,color='#66727C',ls=':',lw=.8);ax.set_ylabel('Execution cost / cohort oracle')
    ax.set_xticks(np.arange(7),['Status\nquo','Training\nbest','Most\nfrequent','Rule /\ndefault','Log-cost\ntree','Penalty\ntree','Uncertainty\ntree'],fontsize=8)
    ax.set_ylim(0,1.75);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,fontsize=8)
    fig.text(.5,.025,'Whiskers: descriptive range of three seed ratios; each cohort has its own oracle.',ha='center',fontsize=8)
    save_figure(fig,'fig06_legacy_compare');write_csv(pd.DataFrame(rows),'figure6_ranges.csv')

def generate_tex(absolute,weights,macro,tails,paired,seeds,intervals,tests,policy,details):
    main=[r'\subsection{Absolute budgets, weighting, and paired uncertainty}',r'\label{sec:aggregate-audit}',
    r'Our aggregate ratios are ratios of sums, not averages of per-case ratios. For configuration $g$, let $C_g$, $B_g$, and $O_g$ be summed selector, status-quo, and oracle costs. Then $C/B=\sum_g(B_g/B)(C_g/B_g)$ and $C/O=\sum_g(O_g/O)(C_g/O_g)$. These denominators induce different cost weights (Fig.~\ref{fig:costweights}). We additionally report the arithmetic mean of configuration ratios, assigning each held-out configuration equal weight; this is descriptive sensitivity to the workload mixture, not a new deployment distribution.',
    r'\begin{table}[t]',r'\centering\small',r'\caption{Absolute per-query execution costs (ms). The budget is the mean paired gap $B-O$; it is not the difference of medians. Selector means, medians, tails, and inference costs appear in the supplement.}',r'\label{tab:absolute-budget}',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{lrrrrr}',r'\toprule',r' & \multicolumn{2}{c}{Status quo} & \multicolumn{2}{c}{Oracle} & Mean gap\\',r'Stage & Mean & Median & Mean & Median & $B-O$\\\midrule']
    for stage in STAGES[:3]:
        b=absolute[(absolute.scope=='stage')&(absolute.stage==stage)&(absolute.strategy=='status_quo')].iloc[0]
        o=absolute[(absolute.scope=='stage')&(absolute.stage==stage)&(absolute.strategy=='oracle')].iloc[0]
        main.append(f'{stage} & {b.mean_ms:.4f} & {b.median_ms:.4f} & {o.mean_ms:.4f} & {o.median_ms:.4f} & {b.mean_regret_ms:.6f}'+r'\\')
    main += [r'\bottomrule\end{tabular}\end{table}',
    r'RPE has 936 cases each in Hetionet, OpenAlexCites, and Synthetic, and 1,800 in SupportChurn; its cases are not equally divided among configurations. Synthetic contributes '+f'{details["rpe_synthetic_status_weight"]*100:.4f}'+r'\% of status-quo cost and '+f'{details["rpe_synthetic_oracle_weight"]*100:.4f}'+r'\% of oracle cost. Frequent-winner savings fall from '+f'{details["rpe_frequency_weighted_savings"]*100:.4f}'+r'\% under status-cost weighting to '+f'{details["rpe_frequency_macro_savings"]*100:.4f}'+r'\% under equal configuration weighting. On Hetionet that selector is '+f'{details["rpe_hetionet_frequency_regression"]*100:.4f}'+r'\% slower than Incremental+, and every non-reference selector regresses there. Thus the aggregate finding does not support replacing Incremental+ across all RPE workloads.',
    r'On Synthetic the training-best choice is ObservedOnly+, whereas frequent-winner chooses NoSchema+; their status ratios are '+f'{details["rpe_synthetic_train_ratio"]:.6f}'+r' and '+f'{details["rpe_synthetic_frequency_ratio"]:.6f}'+r'. Conversely, held-out Hetionet sends the training-selected fixed choices to NoSchema+ and the learned trees to NoSchema+ or ObservedOnly+, all costlier in aggregate than Incremental+. Hetionet has higher stored mean outdegree (47.845) and transition counts up to 45, compared with 5.189/5 for OpenAlexCites and 8/14 for Synthetic. These are descriptive differences and observed choice patterns, not an identified cause of the regression.',
    r'In EGR the six 100k-node configurations with degree at least eight carry '+f'{details["egr_six_config_status_weight"]*100:.4f}'+r'\% of status-quo cost. The log-cost tree\textquotesingle s aggregate cost ratio against Forward is '+f'{details["egr_logcost_weighted_ratio"]:.6f}'+r', versus an equal-configuration mean ratio of '+f'{details["egr_logcost_macro_ratio"]:.6f}'+r'. Cost weighting therefore changes the size, but not the existence, of its regression.',
    r'Paired selector-minus-reference intervals resample the same seed blocks jointly, conditional on the fitted held-out choices (2,000 replicates); they do not refit the selectors. The BCR and EGR numeric bounds and regret quantiles are supplied in the supplement. For RPE we replace nominal bootstrap confidence intervals by all three seed ratios and their descriptive range. Three seed blocks cannot establish conventional 5\% exact sign-flip evidence.',
    r'The primary multiplicity family contains 18 comparisons (six non-reference selectors in three stages); the original 24-comparison family, including legacy RPE, is retained as sensitivity analysis. EGR\textquotesingle s rule improves all ten blocks, while each learned selector regresses in all ten. Their exact two-sided $p=2/2^{10}=0.001953125$ becomes 0.03515625 under the primary family and 0.046875 under the original family. These are the observed Holm step-down results: the adjusted value is $\min\{1,\max_{j\leq r}(m-j+1)p_{(j)}\}$, so $m$ times a stage\textquotesingle s raw resolution is not a universal adjusted-$p$ floor.',
    r'\begin{figure*}[t]\centering',r'\includegraphics[width=\textwidth]{fig08_cost_weights.pdf}',r'\caption{Configuration shares of summed status-quo and oracle cost, the respective weights in aggregate status and oracle ratios. No milliseconds are pooled across stages or hardware. Small SupportChurn bars reflect 0.0122\% and 0.0136\% shares, not missing cases. Exact shares and equal-configuration summaries are reported in the supplement.}\label{fig:costweights}\end{figure*}']
    (SOURCE/'review_aggregate.tex').write_text('\n'.join(main)+'\n',encoding='utf8')
    supp=[r'\section{Aggregate and uncertainty audit}',
    r'This re-analysis freezes the original out-of-fold choices and paired per-case costs. It does not retrain models or rerun historical workloads. All values are available at full precision in the accompanying CSV files. Means weight cases equally; stage ratios weight configurations by their denominator costs; macro ratios weight configurations equally. Confidence intervals for BCR/EGR resample whole seed blocks; RPE ranges are descriptive. The paired gaps and regret are computed before summarization. A small difference of rounded printed means need not reproduce a microsecond-scale gap.',
    r'\paragraph{Reproduction and files.} Run \texttt{python analysis/aggregate/reanalyze.py} from the version directory. The script uses frozen inputs under \texttt{analysis/aggregate/data}; hashes are in \texttt{input\_manifest.json}. Absolute stage/configuration means and medians for every selector and the oracle are in \texttt{absolute\_costs.csv}; all regret quantiles are in \texttt{regret\_tails.csv}; exact weighting in \texttt{configuration\_weights.csv}; the primary and sensitivity Holm steps in \texttt{holm\_stepdown.csv}; and per-seed details in \texttt{seed\_values.csv}.',
    r'\paragraph{Timing block size.} Each Python block runs $\min(128,n_{\rm fold})$ single-request predictions, after up to 20 warm-up requests; seven block averages yield the median. Thus BCR uses 128, EGR 90, and primary/legacy RPE 128 requests per Python block. Each native block makes $1000n_{\rm fold}$ decisions (192,000 for BCR, 90,000 for EGR, 936,000 or 1,800,000 for primary RPE), with nine hot-loop blocks. Native measurement excludes feature acquisition and state costs. Fixed selectors have zero charged inference in the retained replay; this is an accounting lower bound, not a measured zero dispatch latency.',
    r'\paragraph{Test resolution.} With $b$ nonzero independently flippable blocks, the two-sided raw lower bound is $2/2^b$, with further ties able to raise the attained minimum. Nominal block counts are 15, 10, and 3 for BCR, EGR, and RPE. In the observed BCR comparisons many block differences are exactly zero: active counts are reported in \texttt{holm\_stepdown.csv}. Holm adjustment depends on the ordered whole family and preceding steps; there is no stage-only adjusted floor equal to $mp_{\min}$. In these data the four EGR raw values tie for first, so the first factor controls all four (18 or 24). The legacy family is retained solely for comparability with the previous report.']
    def table(caption,header,rows,spec):
        supp.extend([r'\begin{table}[!htbp]\centering\scriptsize',r'\caption{'+caption+'}',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{'+spec+'}',r'\hline',header+r'\\\hline',*rows,r'\hline\end{tabular}\end{table}'])
    for stage in STAGES[:3]:
        a=absolute[(absolute.stage==stage)&(absolute.scope=='stage')].set_index('strategy')
        rows=[]
        for k in ORDER+['oracle']:
            r=a.loc[k];rows.append(f'{LABEL.get(k,"Oracle")} & {num(r.mean_ms)} & {num(r.median_ms)} & {num(r.mean_regret_ms)} & {r.python_mean_us:.3f}'+r'\\')
        table(stage+r' absolute per-query costs (ms) and mean charged Python inference ($\mu$s). Zero for fixed choices/oracle is accounting only.',r'Selector & Mean & Median & Mean regret & Python',rows,'lrrrr')
    for metric,label in [('mean_ms','mean'),('median_ms','median')]:
        rows=[]
        for stage in STAGES[:3]:
            sub=absolute[(absolute.stage==stage)&(absolute.scope=='config')]
            for cfg,g in sub.groupby('config',sort=True):
                g=g.set_index('strategy');rows.append(stage+' / '+tex(config_label(cfg))+' & '+' & '.join(num(g.loc[k,metric]) for k in ORDER+['oracle'])+r'\\')
        table('Per-configuration '+label+' execution cost (ms). TB: training-best; FW: frequent-winner; LC: log-cost; P: penalty; U: uncertainty.',r'Configuration & Status & TB & FW & Rule & LC & P & U & Oracle',rows,'lrrrrrrrr')
    rows=[]
    for r in weights[weights.stage.isin(STAGES[:3])].itertuples():
        rows.append(f'{r.stage} / {tex(config_label(r.config))} & {r.queries} & {num(r.status_sum_ms)} & {num(r.oracle_sum_ms)} & {100*r.status_weight:.4f} & {100*r.oracle_weight:.4f} & {num(r.mean_oracle_gap_ms)}'+r'\\')
    table(r'Configuration totals (ms), cost weights (\%), and mean oracle gap (ms).',r'Configuration & Cases & Status total & Oracle total & Status wt. & Oracle wt. & Gap',rows,'lrrrrrr')
    rows=[]
    for r in macro[macro.stage.isin(STAGES[:3])].itertuples():
        rows.append(f'{r.stage} / {LABEL[r.strategy]} & {r.aggregate_status_ratio:.6f} & {r.macro_status_ratio:.6f} & {r.median_config_status_ratio:.6f} & {r.macro_oracle_ratio:.6f}'+r'\\')
    table('Cost-weighted and equal-configuration summaries; no cross-stage millisecond pooling.',r'Stage / selector & $C/B$ & Macro $C_g/B_g$ & Median $C_g/B_g$ & Macro $C_g/O_g$',rows,'lrrrr')
    for stage in STAGES[:3]:
        rows=[]
        for r in tails[(tails.scope=='stage')&(tails.stage==stage)].itertuples():
            rows.append(f'{LABEL[r.strategy]} & {num(r.regret_p50_ms)} & {num(r.regret_p90_ms)} & {num(r.regret_p95_ms)} & {num(r.regret_p99_ms)} & {num(r.regret_max_ms)}'+r'\\')
        table(stage+' per-query absolute oracle regret (ms), including zero-regret cases.',r'Selector & Median & 90th & 95th & 99th & Maximum',rows,'lrrrrr')
    rows=[]
    for r in paired[paired.scope.eq('stage')].itertuples():
        rows.append(f'{r.stage} / {LABEL[r.strategy]} & {r.delta_mean_ms:.7f} & [{r.delta_mean_ci_low_ms:.7f}, {r.delta_mean_ci_high_ms:.7f}] & {100*r.relative_delta:.5f} & [{100*r.relative_delta_ci_low:.5f}, {100*r.relative_delta_ci_high:.5f}]'+r'\\')
    table(r'Paired selector-minus-status differences; positive means slower. BCR/EGR percentile 95\% intervals from 2,000 seed-block resamples, conditional on fitted choices.',r'Stage / selector & Mean (ms) & CI (ms) & Relative (\%) & CI (\%)',rows,'lrrrr')
    rows=[]
    for r in intervals.itertuples():
        rows.append(f'{r.stage} / {LABEL[r.strategy]} & {r.estimate:.7f} & {r.lower:.7f} & {r.upper:.7f} & '+('3-seed range' if r.stage=='RPE' else '95\% bootstrap CI')+r'\\')
    table('Numeric Figure 3 coordinates. RPE endpoints are min/max seed ratios, not confidence bounds.',r'Stage / selector & Point & Lower & Upper & Interval type',rows,'lrrrl')
    rows=[]
    sr=seeds[(seeds.scope=='stage')&(seeds.stage=='RPE')]
    for k in ORDER:
        g=sr[sr.strategy==k].sort_values('seed'); rows.append(LABEL[k]+' & '+' & '.join(f'{x:.7f}' for x in g.oracle_ratio)+' & '+f'[{g.oracle_ratio.min():.7f}, {g.oracle_ratio.max():.7f}]'+r'\\')
    table('RPE oracle cost ratios for every independent seed; no nominal coverage is attached to the range.',r'Selector & Seed 17 & Seed 29 & Seed 43 & Range',rows,'lrrrr')
    rows=[]
    hp=tests[tests.family=='primary18'].set_index(['stage','strategy']); old=tests[tests.family=='original24'].set_index(['stage','strategy'])
    for (stage,k),r in hp.iterrows():
        rows.append(f'{stage} / {LABEL[k]} & {int(r.positive_blocks)}/{int(r.negative_blocks)}/{int(r.zero_blocks)} & {r.p_raw:.8f} & {r.p_adjusted:.8f} & {old.loc[(stage,k),"p_adjusted"]:.8f}'+r'\\')
    table('Exact paired sign flips and Holm sensitivity. Signs are positive/negative/zero selector-minus-status block sums.',r'Stage / selector & Block signs & Raw $p$ & Holm 18 & Holm 24',rows,'lrrrr')
    supp.insert(5,r'\paragraph{Hetionet and synthetic transfer.} The choices and per-policy cost distributions in \texttt{policy\_configuration\_costs.csv} and \texttt{choice\_counts.csv} explain the observed regression algebraically, without identifying a causal mechanism. For held-out Hetionet both training-selected fixed selectors choose NoSchema+ on all 936 cases; learned selectors choose NoSchema+ on 312 and ObservedOnly+ on 624. For held-out Synthetic the aggregate-cost selector chooses ObservedOnly+ on all 936 cases, while frequency chooses NoSchema+ on all 936. This disagreement is a transfer result between regimes, not evidence that frequency uniformly dominates aggregate-cost selection.')
    # Filenames are monospaced but may break at underscores in narrow lines.
    (SOURCE/'review_aggregate_supp.tex').write_text(('\n'.join(supp)+'\n').replace(r'\_',r'\_\allowbreak '),encoding='utf8')

def main():
    start=time.perf_counter(); ap=argparse.ArgumentParser();ap.add_argument('--refresh-inputs',action='store_true');args=ap.parse_args()
    SOURCE.mkdir(parents=True,exist_ok=True)
    if args.refresh_inputs:refresh()
    manifest=json.loads((HERE/'input_manifest.json').read_text())
    for name,meta in manifest.items(): assert sha(DATA/name)==meta['sha256'],name
    p=pd.read_csv(DATA/'predictions.csv',dtype={'query_id':str});s=pd.read_csv(DATA/'summary.csv');c=pd.concat([pd.read_csv(DATA/'costs.csv',dtype={'query_id':str}),pd.read_csv(DATA/'costs_bcr.csv',dtype={'query_id':str})],ignore_index=True)
    py=pd.read_csv(DATA/'router_timings.csv');native=pd.read_csv(DATA/'native_overhead.csv')
    for df in [p,s,c,py,native]:df['stage']=df.stage.map(PUBLIC)
    assert not p.duplicated(['stage','query_id','strategy']).any()
    assert not c.duplicated(['stage','query_id','policy']).any()
    # Validate the entire replay against original paired execution costs.
    check=p.merge(c[['stage','query_id','policy','cost_ms']],left_on=['stage','query_id','chosen_policy'],right_on=['stage','query_id','policy'],validate='many_to_one')
    assert len(check)==len(p) and np.allclose(check.chosen_ms,check.cost_ms,rtol=1e-13,atol=1e-13)
    oc=c.groupby(['stage','query_id']).cost_ms.min().rename('recomputed_oracle')
    check=p.join(oc,on=['stage','query_id']);assert np.allclose(check.oracle_ms,check.recomputed_oracle,rtol=1e-13,atol=1e-13)
    ref=c[c.policy.eq(c.status_quo)].set_index(['stage','query_id']).cost_ms.rename('recomputed_status')
    check=p.join(ref,on=['stage','query_id']);assert np.allclose(check.status_ms,check.recomputed_status,rtol=1e-13,atol=1e-13)
    assert p.groupby(['stage','query_id']).strategy.nunique().eq(7).all()
    rng=np.random.default_rng(RNG_SEED); absol=[];tails=[];paired=[];seeds=[];stats=[]
    for scope in ['stage','config']:
        keys=['stage','strategy']+(['config'] if scope=='config' else [])
        for key,g in p.groupby(keys,sort=True):
            stage,kind=key[:2];cfg=key[2] if scope=='config' else 'ALL';base=dict(scope=scope,stage=stage,strategy=kind,config=cfg,queries=len(g),clusters=g.cluster.nunique())
            regret=g.chosen_ms-g.oracle_ms; relative=g.chosen_ms/g.oracle_ms-1
            absol.append(dict(**base,sum_ms=g.chosen_ms.sum(),mean_ms=g.chosen_ms.mean(),median_ms=g.chosen_ms.median(),mean_regret_ms=regret.mean(),status_mean_ms=g.status_ms.mean(),oracle_mean_ms=g.oracle_ms.mean(),mean_oracle_gap_ms=(g.status_ms-g.oracle_ms).mean(),python_mean_us=1000*g.python_overhead_ms.mean(),aggregate_oracle_ratio=g.chosen_ms.sum()/g.oracle_ms.sum(),aggregate_status_ratio=g.chosen_ms.sum()/g.status_ms.sum()))
            tail=dict(**base)
            for q in [.5,.9,.95,.99]:tail[f'regret_p{int(q*100)}_ms']=regret.quantile(q);tail[f'penalty_p{int(q*100)}']=relative.quantile(q)
            tail.update(regret_max_ms=regret.max(),regret_mean_ms=regret.mean(),regret_sd_ms=regret.std(ddof=1),penalty_max=relative.max(),zero_regret_fraction=float(regret.eq(0).mean()));tails.append(tail)
            # Common resampling indices across selectors in the same stratum:
            # identical choices must yield identical new paired intervals.
            stratum_seed=RNG_SEED+int(hashlib.sha256((stage+'|'+cfg).encode()).hexdigest()[:8],16)
            if stage in ['BCR','EGR'] and kind!='status_quo':paired.append(dict(**base,delta_mean_ms=(g.chosen_ms-g.status_ms).mean(),relative_delta=g.chosen_ms.sum()/g.status_ms.sum()-1,**bootstrap(g,np.random.default_rng(stratum_seed))))
            for cluster,z in g.groupby('cluster',sort=True):seeds.append(dict(**base,cluster=cluster,seed=int(z.seed.iloc[0]),block_queries=len(z),chosen_sum_ms=z.chosen_ms.sum(),status_sum_ms=z.status_ms.sum(),oracle_sum_ms=z.oracle_ms.sum(),mean_ms=z.chosen_ms.mean(),delta_mean_ms=(z.chosen_ms-z.status_ms).mean(),oracle_ratio=z.chosen_ms.sum()/z.oracle_ms.sum(),status_ratio=z.chosen_ms.sum()/z.status_ms.sum()))
            if scope=='stage' and kind!='status_quo':stats.append(dict(stage=stage,strategy=kind,**exact_test(g)))
            if kind=='status_quo':
                obase=base.copy();obase['strategy']='oracle';absol.append(dict(**obase,sum_ms=g.oracle_ms.sum(),mean_ms=g.oracle_ms.mean(),median_ms=g.oracle_ms.median(),mean_regret_ms=0.,status_mean_ms=g.status_ms.mean(),oracle_mean_ms=g.oracle_ms.mean(),mean_oracle_gap_ms=(g.status_ms-g.oracle_ms).mean(),python_mean_us=0.,aggregate_oracle_ratio=1.,aggregate_status_ratio=g.oracle_ms.sum()/g.status_ms.sum()))
    absolute=pd.DataFrame(absol);tails=pd.DataFrame(tails);paired=pd.DataFrame(paired);seeds=pd.DataFrame(seeds);stats=pd.DataFrame(stats)
    baseline=p[p.strategy.eq('status_quo')];weights=[];macro=[];policy=[]
    for stage,g in baseline.groupby('stage'):
        for cfg,z in g.groupby('config'):
            weights.append(dict(stage=stage,config=cfg,queries=len(z),case_weight=len(z)/len(g),status_sum_ms=z.status_ms.sum(),oracle_sum_ms=z.oracle_ms.sum(),status_weight=z.status_ms.sum()/g.status_ms.sum(),oracle_weight=z.oracle_ms.sum()/g.oracle_ms.sum(),mean_oracle_gap_ms=(z.status_ms-z.oracle_ms).mean()))
    weights=pd.DataFrame(weights)
    for (stage,kind),g in absolute[(absolute.scope=='config')&~absolute.strategy.eq('oracle')].groupby(['stage','strategy']):
        a=absolute[(absolute.scope=='stage')&(absolute.stage==stage)&(absolute.strategy==kind)].iloc[0]
        macro.append(dict(stage=stage,strategy=kind,configurations=len(g),aggregate_status_ratio=a.aggregate_status_ratio,aggregate_oracle_ratio=a.aggregate_oracle_ratio,macro_status_ratio=g.aggregate_status_ratio.mean(),median_config_status_ratio=g.aggregate_status_ratio.median(),macro_oracle_ratio=g.aggregate_oracle_ratio.mean(),geomean_config_status_ratio=np.exp(np.log(g.aggregate_status_ratio).mean())))
        w=weights[weights.stage==stage].set_index('config').loc[g.config]
        assert np.isclose(np.dot(w.status_weight,g.aggregate_status_ratio),a.aggregate_status_ratio,rtol=1e-13)
        assert np.isclose(np.dot(w.oracle_weight,g.aggregate_oracle_ratio),a.aggregate_oracle_ratio,rtol=1e-13)
    macro=pd.DataFrame(macro)
    selector_weights=absolute[absolute.scope=='config'][['stage','strategy','config','queries','sum_ms']].copy()
    selector_weights['selector_cost_weight']=selector_weights.sum_ms/selector_weights.groupby(['stage','strategy']).sum_ms.transform('sum')
    for (stage,cfg,pol),g in c.groupby(['stage','config','policy']):policy.append(dict(stage=stage,config=cfg,policy=pol,queries=len(g),sum_ms=g.cost_ms.sum(),mean_ms=g.cost_ms.mean(),median_ms=g.cost_ms.median()))
    policy=pd.DataFrame(policy)
    choices=p.groupby(['stage','config','strategy','chosen_policy']).size().rename('cases').reset_index()
    tests=pd.concat([holm_table(stats,'original24'),holm_table(stats[stats.stage!='RPE legacy'],'primary18')],ignore_index=True)
    original=tests[tests.family=='original24'].merge(s[s.scope.eq('stage')][['stage','strategy','p_signflip','p_holm']],on=['stage','strategy'],validate='one_to_one')
    assert np.allclose(original.p_raw,original.p_signflip,rtol=0,atol=1e-15)
    assert np.allclose(original.p_adjusted,original.p_holm,rtol=0,atol=1e-15)
    intervals=[];original_intervals=[]
    for r in s[(s.scope=='stage')&s.stage.isin(STAGES[:3])].itertuples():
        z=seeds[(seeds.scope=='stage')&(seeds.stage==r.stage)&(seeds.strategy==r.strategy)]
        original_intervals.append(dict(stage=r.stage,strategy=r.strategy,estimate=r.oracle_cost_ratio,lower=r.oracle_ratio_ci_low,upper=r.oracle_ratio_ci_high,interval_type='historical_independent_draws_95pct_seed_block_bootstrap',clusters=len(z)))
        if r.stage=='RPE': low,high=z.oracle_ratio.min(),z.oracle_ratio.max()
        else:
            # Same stage-block resamples for every selector, including references.
            stratum_seed=RNG_SEED+int(hashlib.sha256((r.stage+'|ALL').encode()).hexdigest()[:8],16)
            z=z.sort_values('cluster');a=z[['chosen_sum_ms','oracle_sum_ms']].to_numpy()
            ix=np.random.default_rng(stratum_seed).integers(0,len(a),size=(BOOTSTRAP,len(a)))
            res=a[ix].sum(axis=1);low,high=np.quantile(res[:,0]/res[:,1],[.025,.975])
        intervals.append(dict(stage=r.stage,strategy=r.strategy,estimate=r.oracle_cost_ratio,lower=low,upper=high,interval_type='three_seed_descriptive_range' if r.stage=='RPE' else 'common_draws_95pct_seed_block_bootstrap',clusters=len(z)))
    intervals=pd.DataFrame(intervals)
    write_csv(pd.DataFrame(original_intervals),'original_figure3_intervals.csv')
    # Query-weighted costs and fold medians are deliberately separate quantities.
    overhead=py.groupby(['stage','strategy']).agg(python_fold_median_us=('python_median_ms',lambda x:1000*x.median()),python_fold_min_us=('python_median_ms',lambda x:1000*x.min()),python_fold_max_us=('python_median_ms',lambda x:1000*x.max()),training_fold_median_ms=('train_ms','median')).reset_index()
    overhead=overhead.merge(native.groupby(['stage','strategy']).agg(native_fold_median_ns=('median_ns','median'),native_fold_min_ns=('median_ns','min'),native_fold_max_ns=('median_ns','max')).reset_index(),on=['stage','strategy'])
    syn=weights[(weights.stage=='RPE')&(weights.config=='syn_base')].iloc[0]; fm=macro[(macro.stage=='RPE')&(macro.strategy=='most_frequent')].iloc[0];lc=macro[(macro.stage=='EGR')&(macro.strategy=='cost_tree')].iloc[0]
    cfg=absolute[(absolute.scope=='config')&(absolute.stage=='RPE')].set_index(['config','strategy']);e=weights[weights.stage=='EGR'];eligible=e.config.str.match('n100000_deg(8|10|12|16|24|32)$')
    details=dict(rpe_synthetic_status_weight=syn.status_weight,rpe_synthetic_oracle_weight=syn.oracle_weight,rpe_frequency_weighted_savings=1-fm.aggregate_status_ratio,rpe_frequency_macro_savings=1-fm.macro_status_ratio,rpe_hetionet_frequency_regression=cfg.loc[('hetionet','most_frequent'),'aggregate_status_ratio']-1,rpe_synthetic_train_ratio=cfg.loc[('syn_base','train_best'),'aggregate_status_ratio'],rpe_synthetic_frequency_ratio=cfg.loc[('syn_base','most_frequent'),'aggregate_status_ratio'],egr_six_config_status_weight=e.loc[eligible,'status_weight'].sum(),egr_logcost_weighted_ratio=lc.aggregate_status_ratio,egr_logcost_macro_ratio=lc.macro_status_ratio)
    for df,name in [(absolute,'absolute_costs.csv'),(weights,'configuration_weights.csv'),(selector_weights,'selector_configuration_weights.csv'),(macro,'macro_summary.csv'),(tails,'regret_tails.csv'),(paired,'paired_differences.csv'),(seeds,'seed_values.csv'),(tests,'holm_stepdown.csv'),(intervals,'figure3_intervals.csv'),(policy,'policy_configuration_costs.csv'),(choices,'choice_counts.csv'),(overhead,'overhead_summary.csv')]:write_csv(df,name)
    feature_summary=c.groupby(['stage','config'])[[x for x in c.columns if x.startswith('f_')]].agg(['min','max']);feature_summary.columns=['_'.join(x) for x in feature_summary.columns];write_csv(feature_summary.reset_index(),'configuration_features.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'v006-aggregate-fixed'})
    plot_intervals(intervals);plot_weights(weights);plot_legacy(seeds,absolute);generate_tex(absolute,weights,macro,tails,paired,seeds,intervals,tests,policy,details)
    audit=dict(status='PASS',prediction_rows=len(p),paired_cases=len(baseline),primary_cases=len(baseline[baseline.stage!='RPE legacy']),chosen_costs_verified_against_long_table=True,status_costs_recomputed=True,oracle_costs_recomputed=True,aggregate_weight_identities_verified=True,original_raw_and_holm_values_reproduced=True,bootstrap_replicates=BOOTSTRAP,bootstrap_seed=RNG_SEED,details=details,elapsed_seconds=time.perf_counter()-start,environment={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},outputs={x.name:sha(x) for x in HERE.glob('*.csv')},figure_outputs={x.name:sha(x) for prefix in ['fig03_results','fig06_legacy_compare','fig08_cost_weights'] for x in SOURCE.glob(prefix+'.*')})
    (HERE/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
