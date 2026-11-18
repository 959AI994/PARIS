"""Portable replay of the five archived PARIS evaluation groups."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.model_selection import KFold
import xgboost as xgb

KEY=['design','candidate_id']


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def tau(a,b):
    value=float(kendalltau(a,b,variant='b').statistic)
    if not np.isfinite(value):raise ValueError('Undefined Kendall tau-b')
    return value


def reproduce(root, output=None, refit=False, groups=None):
    root=Path(root).resolve();manifest=read(root/'MANIFEST.json')
    for name,h in read(root/'SHA256.json').items():
        path=(root/name).resolve()
        assert path.is_relative_to(root) and sha(path)==h,name
    recipe=read(root/'RECIPE.json');schema=read(root/'FEATURE_SCHEMA.json')['columns']
    features=pd.read_parquet(root/'data/GR_FEATURES.parquet');labels=pd.read_parquet(root/'data/LABELS.parquet')
    assert len(features)==1204 and len(labels)==1204 and not features.duplicated(KEY).any() and not labels.duplicated(KEY).any()
    metrics=[];prediction_tables=[];refit_checks=[]
    for group in manifest['groups']:
        display=str(group['display_id'])
        if groups and display not in groups:continue
        folder=root/group['directory'];split=read(folder/'SPLIT.json');choices={p['design']:p for p in read(folder/'SELECTIONS.json')}
        stored=pd.read_parquet(folder/'PREDICTIONS.parquet');assert len(stored)==200
        folds=read(folder/'FOLDS.json')
        for plan in split['plans']:
            design=plan['design'];train=set(plan['training_ids']);test=set(plan['test_ids']);unused=set(plan['unused_ids'])
            assert not train&test and not (train|test)&unused
            f=features[features.design.eq(design)]
            assert train|test|unused==set(f.candidate_id)
            for fold in [z for z in folds if z['design']==design]:
                a,b=set(fold['training_ids']),set(fold['validation_ids']);assert not a&b and a|b==train and not(a|b)&test
            te=f[f.candidate_id.isin(test)].sort_values('candidate_id').reset_index(drop=True)
            xt=te[schema].to_numpy(np.float32);anchor_test=np.log1p(-te.grt_l2_tns.to_numpy(float))
            model=xgb.XGBRegressor(n_jobs=2,device='cpu');model.load_model(folder/'models'/(design+'.json'))
            predicted=anchor_test+model.predict(xt).astype(float)
            expected=stored[stored.design.eq(design)].sort_values('candidate_id').reset_index(drop=True)
            assert te.candidate_id.tolist()==expected.candidate_id.tolist()
            np.testing.assert_array_equal(predicted,expected.predicted_log_cost.to_numpy())
            np.testing.assert_array_equal(-np.expm1(np.clip(predicted,0,50)),expected.predicted_tns.to_numpy())
            if refit:
                # Training labels are filtered before joining; test values enter only after prediction.
                y=labels[labels.design.eq(design)&labels.candidate_id.isin(train)]
                tr=f.merge(y,on=KEY,validate='one_to_one').sort_values('candidate_id').reset_index(drop=True)
                assert len(tr)==len(train) and set(tr.candidate_id)==train
                x=tr[schema].to_numpy(np.float32);anchor=np.log1p(-tr.grt_l2_tns.to_numpy(float));target=np.log1p(-tr.final_tns.to_numpy(float))
                options=[]
                for index,params in enumerate(recipe['grid']):
                    rows=[]
                    for fit,valid in KFold(3,shuffle=True,random_state=17).split(x):
                        m=xgb.XGBRegressor(**recipe['params'],**params);m.fit(x[fit],target[fit]-anchor[fit])
                        z=anchor[valid]+m.predict(x[valid]).astype(float)
                        rows.append((tau(z,target[valid]),float(np.abs(z-target[valid]).mean())))
                    options.append(dict(configuration=index,tau_b=float(np.mean([v[0] for v in rows])),log_mae=float(np.mean([v[1] for v in rows]))))
                pick=max(options,key=lambda z:(round(z['tau_b'],12),-z['log_mae'],-z['configuration']))
                assert pick['configuration']==choices[design]['configuration'],(display,design,'CV selection differs')
                m=xgb.XGBRegressor(**recipe['params'],**recipe['grid'][pick['configuration']]);m.fit(x,target-anchor)
                again=anchor_test+m.predict(xt).astype(float)
                np.testing.assert_array_equal(again,predicted)
                refit_checks.append(dict(group=display,design=design,bitwise_equal=True))
            truth=labels[labels.design.eq(design)&labels.candidate_id.isin(test)]
            table=expected.merge(truth,on=KEY,validate='one_to_one').merge(te[KEY+['fast_l1_tns','grt_l2_tns']],on=KEY,validate='one_to_one')
            assert len(table)==len(test)
            table['group']=display;prediction_tables.append(table)
            for method,pred in [('CTS',table.fast_l1_tns),('GR',table.grt_l2_tns),('PARIS',-table.predicted_log_cost)]:
                metrics.append(dict(group=display,actual_seed=group['actual_seed'],design=design,training=len(train),test=len(test),method=method,tau_b=tau(pred,table.final_tns)))
    per=pd.DataFrame(metrics);assert not per.empty
    expected=pd.read_csv(root/'EXPECTED_PER_DESIGN.csv',dtype={'group':str})
    merged=per.merge(expected,on=['group','design','method'],validate='one_to_one',suffixes=('_replay','_expected'))
    assert len(merged)==len(per)
    np.testing.assert_allclose(merged.tau_b_replay,merged.tau_b_expected,rtol=0,atol=1e-14)
    macro=per.groupby(['group','method']).tau_b.mean().unstack()
    summary={m:dict(mean=float(macro[m].mean()),std=float(macro[m].std(ddof=1)) if len(macro)>1 else None) for m in ['CTS','GR','PARIS']}
    result=dict(status='passed',groups=macro.index.tolist(),models_verified=len(per)//3,all_predictions_bitwise_equal=True,
                all_metrics_match=True,refit=refit,refit_models=len(refit_checks),summary=summary)
    if output:
        out=Path(output);out.mkdir(parents=True,exist_ok=False)
        per.to_csv(out/'PER_DESIGN.csv',index=False);macro.to_csv(out/'MACRO_PER_GROUP.csv')
        pd.concat(prediction_tables,ignore_index=True).to_parquet(out/'PREDICTIONS.parquet',index=False)
        (out/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path);parser.add_argument('--refit',action='store_true');parser.add_argument('--groups',nargs='+')
    args=parser.parse_args();reproduce(args.root,args.output,args.refit,args.groups)
