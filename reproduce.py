"""One-command offline reproduction of numerical analyses and reference tables."""
from pathlib import Path
import argparse, hashlib, json, os, platform, shutil, subprocess, sys, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'recomputed');a=p.parse_args()
    out=a.output.resolve()
    if out==ROOT or out in ROOT.parents:raise SystemExit('Output cannot replace the repository or an ancestor')
    if out.exists() and any(out.iterdir()):raise SystemExit('Use an empty output directory')
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    manifest=json.loads((ROOT/'MANIFEST_SHA256.json').read_text())
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,('Release file changed',name)
    commands=[]
    def run(name,args):
        command=[sys.executable,*map(str,args)]
        with (out/(name+'.log')).open('w',encoding='utf8') as f:
            r=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1','MPLBACKEND':'Agg'})
        assert r.returncode==0,(name,'failed; inspect its log')
        commands.append({'analysis':name,'returncode':r.returncode})
    run('raw_audit',[ROOT/'scripts/verify_numeric_records.py','--root',ROOT,'--output',out/'raw_audit.json'])
    # The original retained script writes adjacent outputs; an isolated copy keeps
    # all released inputs and references unchanged.
    retained=out/'retained_work/analysis/retained'
    shutil.copytree(ROOT/'analysis/retained',retained)
    run('retained',[retained/'reanalyze.py'])
    run('prospective',[ROOT/'scripts/analyze_prospective.py','--results',ROOT/'data/prospective/results','--frozen',ROOT/'data/prospective/plan','--output',out/'prospective'])
    run('factorial',[ROOT/'scripts/analyze_factorial.py','--pair-only','--results',ROOT/'data/factorial','--output',out/'factorial'])
    run('scan',[ROOT/'scripts/analyze_scan.py','--data',ROOT/'data/scan','--output',out/'scan'])
    run('external',[ROOT/'scripts/analyze_external.py','--root',ROOT/'data/external','--output',out/'external'])
    run('leakage',[ROOT/'analysis/leakage/reanalyze.py','--output',out/'leakage'])
    folders={'retained':retained,**{k:out/k for k in ['prospective','factorial','scan','external','leakage']}}
    comparisons=[]
    for kind,folder in folders.items():
        for ref in sorted((ROOT/'reference'/kind).glob('*.csv')):
            target=folder/ref.name
            if not target.exists():
                # Historical intervals are retained provenance, not newly plotted outputs.
                assert ref.name=='original_figure3_intervals.csv',ref.name
                continue
            x=pd.read_csv(ref);y=pd.read_csv(target)
            assert x.shape==y.shape and x.columns.equals(y.columns),(kind,ref.name,'shape/columns')
            for col in x:
                if pd.api.types.is_numeric_dtype(x[col]) and pd.api.types.is_numeric_dtype(y[col]):
                    assert np.allclose(x[col],y[col],rtol=1e-10,atol=1e-12,equal_nan=True),(kind,ref.name,col)
                else:
                    assert x[col].fillna('<NA>').astype(str).equals(y[col].fillna('<NA>').astype(str)),(kind,ref.name,col)
            comparisons.append({'analysis':kind,'file':ref.name,'rows':len(x),'status':'PASS'})
    audit=json.loads((out/'prospective/analysis_audit.json').read_text())
    assert audit['simultaneous_noninferiority_within_1pct'] is False
    assert abs(audit['largest_simultaneous_upper_Eager_vs_non_Eager']-1.018103231012578)<1e-12
    report={'status':'PASS','seconds':time.perf_counter()-start,'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'numeric_rtol':1e-10,'numeric_atol':1e-12,'commands':commands,'reference_tables':comparisons,'frozen_1pct_criterion':'NOT_MET','largest_simultaneous_upper':audit['largest_simultaneous_upper_Eager_vs_non_Eager']}
    (out/'reproduction_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':'PASS','tables':len(comparisons),'seconds':report['seconds'],'criterion':'NOT_MET'},indent=2))

if __name__=='__main__':main()
