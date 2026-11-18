"""Additional draws for an explicitly selected showcase; original random results retained."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'

BASE=Path('/data/wjx/PARIS/runs/timing_predict_20261007')
SOURCE=BASE/'single_model_seed17_20261117_v1'
PRIOR=BASE/'paris_compare_train1000_test200_20261117_v1'
TRUTH=BASE/'datasets/paris_1204_unified_20261012_v2_flowlibs/reconciled_labels/evaluation_labels.parquet'
ROOT=BASE/'paris_single_xgb_showcase_splits_20261118_v1'
SEEDS=[101,131,151,181,211]
TEST=dict(ariane133=43,ariane136=25,bp=41,bp_be=12,bp_fe=11,bp_multi=42,swerv_wrapper=26)
EXCLUDE={'ariane133','bp','bp_multi','swerv_wrapper'}
KEY=['design','candidate_id']


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(tmp,p)


def status(phase,**kw):save(ROOT/'STATUS.json',dict(status='running',phase=phase,time=time.time(),pid=os.getpid(),**kw))


def disk():assert shutil.disk_usage('/data').free>=60*2**30,'Disk below60GiB; stop new tasks'


def tau(a,b):
    import numpy as np
    from scipy.stats import kendalltau
    t=float(kendalltau(a,b,variant='b').statistic)
    return t if np.isfinite(t) else None


def verify_inputs(plan):
    for p,h in plan['inputs'].items():assert sha(p)==h,p


def plan():
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    assert not (ROOT/'PROTOCOL.json').exists(),'Already planned; do not restart'
    disk()
    old=read(SOURCE/'FROZEN.json');fp=SOURCE/'prepared/ALL_GR_FEATURES.parquet'
    assert sha(fp)==read(SOURCE/'PREPARED_HASHES.json')[str(fp)]
    f=pd.read_parquet(fp);assert len(f)==1204 and not f.duplicated(KEY).any()
    previous=read(PRIOR/'PROTOCOL.json');source=read(SOURCE/'PROTOCOL.json')
    assert previous['single_params']==source['parameters'] and previous['single_grid']==source['grid']
    seeds=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed);splits=[]
        for d,n in TEST.items():
            targets=sorted(x['candidate_id'] for x in old['target_ids'] if x['design']==d)
            history=sorted(x['candidate_id'] for x in old['history_ids'] if x['design']==d)
            order=rng.permutation(targets).tolist();test=sorted(order[:n]);unused=sorted(order[n:n+1]) if d in EXCLUDE else []
            training=sorted(set(f.loc[f.design.eq(d),'candidate_id'])-set(test)-set(unused))
            assert set(history)<=set(training) and not set(training)&set(test)
            splits.append(dict(design=d,training_ids=training,test_ids=test,unused_ids=unused,original_history_ids=history))
        assert sum(len(p['training_ids']) for p in splits)==1000 and sum(len(p['test_ids']) for p in splits)==200
        assert sum(len(p['unused_ids']) for p in splits)==4
        seeds.append(dict(seed=seed,plans=splits))
    files=[fp,SOURCE/'FEATURE_SCHEMA.json',SOURCE/'FROZEN.json',SOURCE/'PROTOCOL.json',PRIOR/'PROTOCOL.json',TRUTH,ROOT/'code/experiment.py']
    protocol=dict(status='frozen',created=time.time(),seeds=SEEDS,splits=seeds,training=1000,test=200,unused=4,
        params=previous['single_params'],grid=previous['single_grid'],feature_count=1002,
        target='log1p(-DR_TNS)-log1p(-GR_TNS), historical raw-residual recipe unchanged',
        split_policy='Same prior1000/200 policy: preserve original719 histories in training; stratified200 from original485 seed17 targets;4 unused',
        selection='3fold shuffledseed17 CV within training only; mean tau-b, then logcostMAE, then fixed grid order',
        aggregation='per-seed seven-design equal-weight Macro tau-b; report all five seeds and sample std(ddof=1)',
        scope='Exploratory extra seeds for a result-selected showcase requested after inspecting prior results; all original and extra results retained; not an unbiased five-run estimate',
        new_DR=0,new_STA=0,gpu=False,cpu_threads=2,automatic_model_enable=False,
        versions=dict(numpy=np.__version__,pandas=pd.__version__,xgboost=xgb.__version__),inputs={str(p):sha(p) for p in files})
    save(ROOT/'PROTOCOL.json',protocol)


def freeze(seed):
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import KFold
    p=read(ROOT/'PROTOCOL.json');verify_inputs(p);disk()
    case=next(v for v in p['splits'] if v['seed']==seed)
    out=ROOT/f'seed_{seed}';out.mkdir(exist_ok=False);(out/'models').mkdir()
    schema=read(SOURCE/'FEATURE_SCHEMA.json')['columns'];f=pd.read_parquet(SOURCE/'prepared/ALL_GR_FEATURES.parquet')
    predictions=[];folds=[];choices=[];cv=[];receipts=[]
    for i,sp in enumerate(case['plans']):
        disk();d=sp['design'];status('training',seed=seed,design=d,model_index=SEEDS.index(seed)*7+i+1,total_models=35)
        # Per-seed child process reads only this split's training labels.
        y=pd.read_parquet(TRUTH,columns=KEY+['final_tns'],filters=[('design','==',d),('candidate_id','in',sp['training_ids'])])
        assert len(y)==len(sp['training_ids']) and set(y.candidate_id)==set(sp['training_ids']) and not y.duplicated(KEY).any()
        assert np.isfinite(y.final_tns).all() and (y.final_tns<=0).all()
        y.to_parquet(out/f'{d}_TRAIN_LABELS.parquet',index=False)
        tr=f[f.design.eq(d)].merge(y,on=KEY,validate='one_to_one').sort_values('candidate_id').reset_index(drop=True)
        te=f[f.design.eq(d)&f.candidate_id.isin(sp['test_ids'])].sort_values('candidate_id').reset_index(drop=True)
        assert len(tr)==len(sp['training_ids']) and len(te)==len(sp['test_ids'])
        x=tr[schema].to_numpy(np.float32);xt=te[schema].to_numpy(np.float32)
        anchor=np.log1p(-tr.grt_l2_tns.to_numpy(float));truth=np.log1p(-tr.final_tns.to_numpy(float));at=np.log1p(-te.grt_l2_tns.to_numpy(float))
        split=list(KFold(3,shuffle=True,random_state=17).split(x));options=[]
        for k,(fit,valid) in enumerate(split):
            folds.append(dict(design=d,fold=k,training_ids=tr.iloc[fit].candidate_id.tolist(),validation_ids=tr.iloc[valid].candidate_id.tolist()))
        for j,conf in enumerate(p['grid']):
            current=[]
            for k,(fit,valid) in enumerate(split):
                m=xgb.XGBRegressor(**p['params'],**conf);m.fit(x[fit],truth[fit]-anchor[fit])
                z=m.predict(x[valid]).astype(float)+anchor[valid]
                row=dict(design=d,configuration=j,fold=k,tau_b=tau(z,truth[valid]),log_mae=float(np.abs(z-truth[valid]).mean()))
                cv.append(row);current.append(row)
            options.append(dict(configuration=j,tau_b=float(np.mean([v['tau_b'] if v['tau_b'] is not None else 0. for v in current])),log_mae=float(np.mean([v['log_mae'] for v in current]))))
        pick=max(options,key=lambda z:(round(z['tau_b'],12),-z['log_mae'],-z['configuration']))
        m=xgb.XGBRegressor(**p['params'],**p['grid'][pick['configuration']]);m.fit(x,truth-anchor)
        file=out/'models'/(d+'.json');m.save_model(file)
        pred=m.predict(xt).astype(float)+at
        reload=xgb.XGBRegressor(n_jobs=2);reload.load_model(file);np.testing.assert_array_equal(pred,reload.predict(xt).astype(float)+at)
        z=te[KEY].copy();z['predicted_log_cost']=pred;z['predicted_tns']=-np.expm1(np.clip(pred,0,50));predictions.append(z)
        te[KEY+schema].to_parquet(out/f'{d}_TEST_GR.parquet',index=False)
        receipts.append(dict(design=d,model_sha256=sha(file),training_ids=sp['training_ids'],test_ids=sp['test_ids'],reload_equal=True))
        choices.append(dict(design=d,selection=pick,options=options))
    pd.concat(predictions,ignore_index=True).to_parquet(out/'PREDICTIONS.parquet',index=False)
    save(out/'FOLDS.json',folds);save(out/'SELECTIONS.json',choices);save(out/'MODELS.json',receipts);pd.DataFrame(cv).to_csv(out/'TRAINING_CV.csv',index=False)
    verify_inputs(p)
    save(out/'FROZEN.json',dict(status='complete',seed=seed,training=1000,test=200,test_truth_read=False,frozen_at=time.time(),
        files={str(q.relative_to(out)):sha(q) for q in out.rglob('*') if q.is_file()}))


def evaluate():
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    p=read(ROOT/'PROTOCOL.json');verify_inputs(p)
    f=pd.read_parquet(SOURCE/'prepared/ALL_GR_FEATURES.parquet');schema=read(SOURCE/'FEATURE_SCHEMA.json')['columns']
    records=[];all_predictions=[]
    # This process starts only after all five split prediction freezes exist.
    assert all((ROOT/f'seed_{seed}'/'FROZEN.json').exists() for seed in SEEDS)
    for case in p['splits']:
        seed=case['seed'];out=ROOT/f'seed_{seed}';frozen=read(out/'FROZEN.json')
        assert frozen['test_truth_read'] is False
        for n,h in frozen['files'].items():assert sha(out/n)==h
        pred=pd.read_parquet(out/'PREDICTIONS.parquet');assert len(pred)==200
        folds=read(out/'FOLDS.json')
        for sp in case['plans']:
            d=sp['design'];train=set(sp['training_ids']);test=set(sp['test_ids']);assert not train&test
            for fold in [v for v in folds if v['design']==d]:
                a,b=set(fold['training_ids']),set(fold['validation_ids']);assert not a&b and a|b==train and not (a|b)&test
            snapshot=pd.read_parquet(out/f'{d}_TRAIN_LABELS.parquet');assert set(snapshot.candidate_id)==train
            x=pd.read_parquet(out/f'{d}_TEST_GR.parquet');m=xgb.XGBRegressor(n_jobs=2);m.load_model(out/'models'/(d+'.json'))
            again=np.log1p(-x.grt_l2_tns.to_numpy(float))+m.predict(x[schema].to_numpy(np.float32)).astype(float)
            z=pred[pred.design.eq(d)].sort_values('candidate_id').reset_index(drop=True);np.testing.assert_array_equal(again,z.predicted_log_cost)
            y=pd.read_parquet(TRUTH,columns=KEY+['final_tns'],filters=[('design','==',d),('candidate_id','in',sp['test_ids'])])
            z=z.merge(y,on=KEY,validate='one_to_one').merge(f[KEY+['fast_l1_tns','grt_l2_tns']],on=KEY,validate='one_to_one')
            assert len(z)==len(test) and set(z.candidate_id)==test
            for name,values in [('CTS',z.fast_l1_tns),('GR',z.grt_l2_tns),('PARIS',z.predicted_tns)]:
                rank_tau=tau(-z.predicted_log_cost,z.final_tns) if name=='PARIS' else tau(values,z.final_tns)
                records.append(dict(seed=seed,design=d,training=len(train),test=len(test),method=name,tau_b=rank_tau,tns_mae_ns=float(np.abs(values-z.final_tns).mean())))
            z['seed']=seed;all_predictions.append(z)
    out=ROOT/'final_v1';out.mkdir(exist_ok=False)
    per=pd.DataFrame(records);per.to_csv(out/'PER_DESIGN_PER_SEED.csv',index=False)
    macro=per.groupby(['seed','method'],sort=False).agg(tau_b=('tau_b','mean'),n_designs=('design','nunique')).reset_index();assert (macro.n_designs==7).all()
    macro.to_csv(out/'MACRO_PER_SEED.csv',index=False)
    meanstd=per.groupby(['design','method'],sort=False).agg(mean_tau_b=('tau_b','mean'),std_tau_b=('tau_b','std')).reset_index();meanstd.to_csv(out/'PER_DESIGN_SUMMARY.csv',index=False)
    summary=macro.groupby('method',sort=False).agg(mean_tau_b=('tau_b','mean'),std_tau_b=('tau_b','std'),min_tau_b=('tau_b','min'),max_tau_b=('tau_b','max')).reset_index();summary.to_csv(out/'MACRO_SUMMARY.csv',index=False)
    pd.concat(all_predictions,ignore_index=True).to_parquet(out/'CANDIDATES.parquet',index=False)
    save(out/'VALIDATION.json',dict(status='passed',seeds=SEEDS,training_per_seed=1000,test_per_seed=200,models=35,
        candidate_train_test_disjoint=True,cv_training_only=True,all_models_reloaded_predictions_bitwise_equal=True,
        all_five_prediction_freezes_before_evaluation=True,all_splits_reported=True,std_ddof=1,source_inputs_unchanged=True))
    save(out/'complete.json',dict(status='complete',outputs={q.name:sha(q) for q in out.iterdir() if q.is_file()}))
    save(ROOT/'STATUS.json',dict(status='complete',phase='evaluated',finished=time.time(),seeds=SEEDS))
    print(summary.to_string(index=False),flush=True)


def run():
    plan();status('planned',seeds=SEEDS)
    for seed in SEEDS:
        subprocess.run([sys.executable,__file__,'freeze','--seed',str(seed)],check=True)
    status('independent_evaluation')
    subprocess.run([sys.executable,__file__,'evaluate'],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['run','freeze','evaluate']);p.add_argument('--seed',type=int);a=p.parse_args()
    try:
        freeze(a.seed) if a.phase=='freeze' else globals()[a.phase]()
    except Exception as e:
        save(ROOT/'STATUS.json',dict(status='failed',phase=a.phase,error=str(e),time=time.time()))
        (ROOT/('FAILURE_'+a.phase+'.txt')).write_text(traceback.format_exc());raise
