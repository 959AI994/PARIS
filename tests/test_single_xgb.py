import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paris_tool.algorithms.single_xgb import fit, features, infer, ALGORITHM
from paris_tool.candidate_features import ARC, NET
from paris_tool import models
from paris_tool.ui import App
from paris_tool.utils import verify


def dataset(root, n=40):
    root.mkdir()
    ids=[f'c{i:03d}' for i in range(n)]
    x=np.arange(n,dtype=float)
    pd.DataFrame(dict(design=['chip']*n,candidate_id=ids,fast_l1_tns=-x-2,grt_l2_tns=-x-3,
                      electrical_signal=np.sin(x/4))).to_parquet(root/'features.parquet',index=False)
    rows=[];nets=[]
    for i,cid in enumerate(ids):
        for transition in ['rise','fall']:
            row=dict(design='chip',candidate_id=cid,net_name='n',driver_pin='d',sink_pin='s',transition=transition,corner='nominal')
            row.update({k:.1+i*.001 for k in ARC});row['gr_slack']=-1.
            rows.append(row)
        row=dict(design='chip',candidate_id=cid,net_name='n',driver_pin='d',corner='nominal')
        row.update({k:.1+i*.001 for k in NET});nets.append(row)
    pd.DataFrame(rows).to_parquet(root/'source_sink.parquet',index=False)
    pd.DataFrame(nets).to_parquet(root/'net_load.parquet',index=False)
    pd.DataFrame(dict(design=['chip']*n,candidate_id=ids,final_tns=-(x+3)*np.exp(.04*np.sin(x/4)))).to_parquet(root/'labels.parquet',index=False)
    context=dict(version=1,pdk_id='test',flow_id='verified-test',corner='nominal',feature_version='gr1',label_version='dr1',
                 time_unit='ns',label_source='measured_dr_sta')
    for name in ['PARIS_TRAINING.json','FEATURE_CONTEXT.json']:(root/name).write_text(json.dumps(context))
    return ids,context


def test_real_single_estimator_roundtrip_and_label_boundary(tmp_path):
    ids,context=dataset(tmp_path/'data')
    model=tmp_path/'model'
    fit(tmp_path/'data',model,'chip',ids[:24],context)
    card=verify(model)
    assert card['algorithm']==ALGORITHM and card['learned_estimators_per_design']==1
    assert len(list((model/'models').glob('*.json')))==1
    assert set(card['models']['chip']['training_ids'])==set(ids[:24])
    frame,_=features(tmp_path/'data',json.loads((model/'FEATURE_SCHEMA.json').read_text())['columns'],ids=ids[24:])
    first=infer(model,frame)
    archive=tmp_path/'weights.tar.gz';models.export(model,archive)
    restored=models.import_bundle(archive,tmp_path/'imported')
    np.testing.assert_array_equal(first.predicted_tns,infer(restored,frame).predicted_tns)
    with pytest.raises(ValueError,match='训练候选'):
        infer(model,features(tmp_path/'data',ids=ids[:2])[0])
    changed=pd.read_parquet(tmp_path/'data/features.parquet');changed['final_tns']=-1.
    changed.to_parquet(tmp_path/'data/features.parquet',index=False)
    with pytest.raises(ValueError,match='DR'):
        features(tmp_path/'data')


def test_gui_single_training_fixed_holdout_and_snapshot_isolation(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import shutil
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=100*2**30))
    ids,context=dataset(tmp_path/'data')
    fit(tmp_path/'data',tmp_path/'base','chip',ids[:24],context)
    app=App(tmp_path/'ui',[tmp_path],model=tmp_path/'base')
    assert app.single_model
    info=app.iteration.inspect(tmp_path/'data',tmp_path/'base')
    assert info['algorithm']==ALGORITHM and info['training']==28 and info['validation']==12
    assert not set(info['training_ids']) & set(info['validation_ids'])
    job=app.iteration.start(tmp_path/'data',tmp_path/'base')
    app.iteration.worker.join(timeout=90)
    result=app.iteration.state(job['id'])
    assert result['status']=='complete',result.get('error')
    output=Path(result['output'])
    snapshot=pd.read_parquet(output/'training_labels.parquet')
    assert set(snapshot.candidate_id)==set(info['training_ids'])
    assert not set(snapshot.candidate_id)&set(info['validation_ids'])
    protocol=json.loads((output/'model/TRAINING_PROTOCOL.json').read_text())
    assert protocol['validation_labels_used'] is False
    for fold in json.loads((output/'model/FOLDS.json').read_text()):
        assert not set(fold['training_ids'])&set(fold['validation_ids'])
        assert set(fold['training_ids']+fold['validation_ids'])==set(info['training_ids'])
    again=app.iteration.inspect(tmp_path/'data',tmp_path/'base')
    assert again['validation_ids']==info['validation_ids'] and again['new_training']==0


def test_benchmark_five_split_payload_uses_all_seeds():
    from importlib.resources import files
    payload=json.loads(files('paris_tool').joinpath('resources/benchmark.json').read_text())
    if 'five_splits' not in payload:
        pytest.skip('Five-split experiment not yet published')
    five=payload['five_splits']
    assert len(five['seeds'])==5 and len(set(five['seeds']))==5
    assert len(five['per_seed'])==5 and len(five['rows'])==7
    assert five['display_ids']==[17,29,41,211,101]
    assert [r['display_id'] for r in five['per_seed']]==five['display_ids']
    assert five['per_seed'][0]['seed']==20261117
    assert sum(r['training'] for r in five['rows'])==1000
    assert sum(r['test'] for r in five['rows'])==200
    for method in ['CTS','GR','PARIS']:
        values=np.array([r[method] for r in five['per_seed']])
        assert abs(values.mean()-five['macro'][method])<1e-14
        assert abs(values.std(ddof=1)-five['std'][method])<1e-14
        assert abs(np.mean([r[method] for r in five['rows']])-five['macro'][method])<1e-14

    replacement=five.get('selection_record',{}).get('reference_replacement')
    if replacement:
        assert 17 not in five['seeds'] and replacement['new_seed']==20261117
        reference=next(r for r in five['per_seed'] if r['seed']==20261117)
        for method in ['CTS','GR','PARIS']:
            assert abs(reference[method]-payload['macro'][method])<1e-14
