"""Raw design to a design-specific model and predictions on unlabelled candidates."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time

from .utils import digest, fresh, save


def subset(source, destination, ids, labels=None):
    import pandas as pd
    destination = fresh(destination)
    wanted = set(ids)
    if not wanted:
        raise ValueError('Candidate subset is empty')
    for name in ['features', 'source_sink', 'net_load']:
        frame = pd.read_parquet(Path(source) / (name + '.parquet'))
        frame = frame[frame.candidate_id.isin(wanted)]
        if set(frame.candidate_id) != wanted:
            raise ValueError('Candidate subset differs across GR tables')
        frame.to_parquet(destination / (name + '.parquet'), index=False)
    context = json.loads((Path(source) / 'FEATURE_CONTEXT.json').read_text())
    save(destination / 'FEATURE_CONTEXT.json', context)
    if labels is not None:
        if set(labels.candidate_id) != wanted:
            raise ValueError('Training labels differ from selected candidates')
        labels.to_parquet(destination / 'labels.parquet', index=False)
        save(destination / 'PARIS_TRAINING.json', {**context, 'label_source':'measured_dr_sta', 'label_version':'dr1', 'version':1})
    return destination


def run(project, configuration, output, candidates=36, seed=17, label_budget=0,
        model=None, route_iterations=1):
    import pandas as pd
    from .mol_profile import settings, run_project, row_config
    from .mol_flow import finish_candidate
    from .collection import collect_label
    from .algorithms.single_xgb import fit, Backend

    # Fail before running any EDA or opening labels when a budget is invalid.
    if candidates < 2 or label_budget < 0 or route_iterations < 1:
        raise ValueError('Require >=2 candidates, a non-negative label budget and >=1 route iteration')
    if model and label_budget:
        raise ValueError('Choose an existing model or a new calibration budget')
    if not model and (label_budget < 24 or label_budget > candidates - 2):
        raise ValueError('New-design calibration requires >=24 labels and >=2 unlabelled candidates')
    cfg = settings(configuration)
    out = fresh(output)
    state = dict(status='running', phase='prepare', started=time.time(), label_budget=label_budget)
    save(out / 'STATUS.json', state)
    try:
        source = out / 'project'
        shutil.copytree(Path(project).resolve(), source)
        p = json.loads((source / 'PARIS_PROJECT.json').read_text())
        p['partition'] = {**p.get('partition', {}), 'count':candidates, 'seed':seed}
        save(source / 'PARIS_PROJECT.json', p)
        prepared = out / 'prepared'
        run_project(source, configuration, prepared)
        mapping = json.loads((prepared / 'physical/GR_RUNS.json').read_text())
        ids = sorted(mapping)
        if len(ids) != candidates:
            raise ValueError(f'Requested {candidates} distinct candidates; got {len(ids)}')
        train_ids, prediction_ids = ids[:label_budget], ids[label_budget:]
        assignments = {}
        for cid in ids:
            assignment = json.loads((prepared / 'partition/candidates' / (cid + '.json')).read_text())['assignment']
            canonical = json.dumps(assignment, sort_keys=True, separators=(',', ':')).encode()
            assignments[cid] = __import__('hashlib').sha256(canonical).hexdigest()
        if len(set(assignments.values())) != len(ids):
            raise ValueError('Two candidates have identical physical assignments')
        save(out / 'CALIBRATION_PLAN.json', dict(training_ids=train_ids, prediction_ids=prediction_ids,
             assignment_sha256=assignments, policy='first sorted IDs before labels or scores',
             input_features={n:digest(prepared / 'dataset' / n) for n in ['features.parquet','source_sink.parquet','net_load.parquet']}))
        if train_ids:
            state['phase'] = 'labels'; save(out / 'STATUS.json', state)
            def label(cid):
                if shutil.disk_usage(out).free < 60 * 1024**3:
                    raise ValueError('Less than 60 GiB free; no new DR started')
                dr_run = out / 'dr' / cid
                finish_candidate(mapping[cid], dr_run, route_iterations=route_iterations)
                receipt = json.loads((dr_run / 'complete.json').read_text())
                results = Path(receipt['results'])
                c, row = row_config(prepared / 'partition/mol', mapping[cid],
                    prepared / 'features/collection', prepared / 'features/metadata' / cid,
                    cfg['openroad'], cid, p['design'])
                row.update(dr_odb=results/'6_1_fill.odb', dr_sdc=results/'6_1_fill.sdc')
                return cid, collect_label(c, row)
            completed = []; tables = []
            with ThreadPoolExecutor(max_workers=2) as pool:
                for future in as_completed([pool.submit(label,cid) for cid in train_ids]):
                    cid, folder = future.result(); completed.append(cid)
                    tables.append(pd.read_parquet(folder/'labels.parquet'))
                    save(out/'LABEL_PROGRESS.json', dict(complete=sorted(completed),total=len(train_ids)))
            training = subset(prepared/'dataset', out/'training', train_ids, pd.concat(tables,ignore_index=True))
            state['phase']='train';save(out/'STATUS.json',state)
            model=out/'model'
            context=json.loads((training/'PARIS_TRAINING.json').read_text())
            fit(training,model,p['design'],train_ids,context)
        else:
            model=Path(model).resolve()
        prediction = subset(prepared/'dataset',out/'prediction-input',prediction_ids)
        state['phase']='predict';save(out/'STATUS.json',state)
        result=Backend().predict(model,prediction,out/'predictions.csv')
        save(out/'PREDICTION_FREEZE.json',dict(status='frozen',model=str(model),
             candidate_ids=prediction_ids,predictions_sha256=digest(out/'predictions.csv'),
             target_DR_read=False))
        state.update(status='complete',phase='complete',model=str(model),predictions=result['output'],
                     training=len(train_ids),prediction_candidates=len(prediction_ids),finished=time.time())
        save(out/'STATUS.json',state)
        return state
    except Exception as exc:
        state.update(status='failed',error=str(exc));save(out/'STATUS.json',state)
        raise
