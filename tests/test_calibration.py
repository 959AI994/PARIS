"""Budget and train/predict isolation across the raw-design orchestrator."""
import json
from pathlib import Path
import pandas as pd
import pytest

from paris_tool.calibration import run


def test_new_design_budget_fails_before_eda_or_output(tmp_path):
    with pytest.raises(ValueError, match='>=24'):
        run(tmp_path/'missing-project', tmp_path/'missing-settings',tmp_path/'out',candidates=26,label_budget=23)
    assert not (tmp_path/'out').exists()


def test_calibration_labels_only_the_frozen_training_ids(tmp_path,monkeypatch):
    from test_single_xgb import dataset
    from paris_tool import mol_profile,mol_flow,collection
    source=tmp_path/'source';source.mkdir();(source/'PARIS_PROJECT.json').write_text(json.dumps(dict(design='chip')))
    monkeypatch.setattr(mol_profile,'settings',lambda _:dict(openroad='unused'))
    def prepare(project,configuration,out):
        out.mkdir();ids,_=dataset(out/'dataset',26)
        (out/'physical').mkdir();(out/'physical/GR_RUNS.json').write_text(json.dumps({cid:str(out/cid) for cid in ids}))
        folder=out/'partition/candidates';folder.mkdir(parents=True)
        for i,cid in enumerate(ids):(folder/(cid+'.json')).write_text(json.dumps(dict(assignment={f'm{i}':0})))
    monkeypatch.setattr(mol_profile,'run_project',prepare)
    routed=[]
    def route(gr,out,route_iterations):
        plan=json.loads((out.parent.parent/'CALIBRATION_PLAN.json').read_text())
        assert len(plan['training_ids'])==24 and len(plan['prediction_ids'])==2
        assert out.name in plan['training_ids'] and out.name not in plan['prediction_ids']
        out.mkdir(parents=True);(out/'complete.json').write_text(json.dumps(dict(results=str(out))))
        routed.append(out.name)
    monkeypatch.setattr(mol_flow,'finish_candidate',route)
    monkeypatch.setattr(mol_profile,'row_config',lambda *args:({},dict(candidate_id=args[-2])))
    def label(cfg,row):
        cid=row['candidate_id'];out=tmp_path/'measured'/cid;out.mkdir(parents=True)
        pd.DataFrame([dict(design='chip',candidate_id=cid,final_tns=-3.-int(cid[1:]))]).to_parquet(out/'labels.parquet',index=False)
        return out
    monkeypatch.setattr(collection,'collect_label',label)
    result=run(source,tmp_path/'settings',tmp_path/'out',candidates=26,label_budget=24)
    assert result['status']=='complete' and len(set(routed))==24
    predictions=pd.read_csv(tmp_path/'out/predictions.csv')
    assert set(predictions.candidate_id)=={'c024','c025'}
    training=pd.read_parquet(tmp_path/'out/training/labels.parquet')
    assert not set(training.candidate_id)&set(predictions.candidate_id)
    assert not (tmp_path/'out/prediction-input/labels.parquet').exists()
    assert json.loads((tmp_path/'out/PREDICTION_FREEZE.json').read_text())['target_DR_read'] is False


def test_settings_preserve_venv_python_path_and_resolve_relative_resources(tmp_path):
    import sys
    from paris_tool.mol_profile import settings, KEYS
    target=tmp_path/'bin/python';target.parent.mkdir();target.symlink_to(sys.executable)
    (tmp_path/'resource').mkdir()
    path=tmp_path/'native.json'
    path.write_text(json.dumps({key:('bin/python' if key.endswith('python') else 'resource') for key in KEYS}))
    result=settings(path)
    assert result['feature_python']==str(target) and result['place_python']==str(target)
    assert result['flow_root']==str(tmp_path/'resource')
