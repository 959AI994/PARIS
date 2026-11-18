"""Cumulative per-design single-XGBoost training in the existing workspace UI."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .iteration import Iteration as LegacyIteration, stable, now
from .algorithms.single_xgb import ALGORITHM, KEY, GR_FILES, CONTEXT, features
from . import models
from .utils import digest, save, verify, run

FILES = GR_FILES + ["labels.parquet", "PARIS_TRAINING.json"]
STEPS = {"snapshot":"固定数据与验证集", "train":"训练PARIS模型", "predict":"生成验证预测", "evaluate":"比较模型版本"}


def metric(score, truth, predicted_tns=None):
    t = float(kendalltau(score, -truth, variant="b").statistic)
    return dict(tau_b=t if np.isfinite(t) else None, n=len(truth),
                tns_mae_ns=float(np.abs(predicted_tns-truth).mean()) if predicted_tns is not None else None)


class Iteration(LegacyIteration):
    def __init__(self, app):
        super().__init__(app)
        # Preserve old records on disk; this product view lists single-XGB versions.
        self.jobs = {k:v for k,v in self.jobs.items() if v.get("inspection",{}).get("algorithm") == ALGORITHM}

    def reference(self, value):
        source = self.app.path(value)
        if source.is_file():
            checksum = digest(source)
            imported = self.app.workspace / "models" / checksum
            if not imported.exists():
                imported.parent.mkdir(exist_ok=True)
                models.import_bundle(source, imported, expected=checksum)
            source = imported
        root = models.find_root(source)
        self.app.path(root, directory=True)
        card = verify(root)
        if card.get("algorithm") != ALGORITHM:
            raise ValueError("请选择PARIS模型目录或模型包")
        return root, card

    def inspect(self, dataset, reference, design=None, template=None):
        root = self.app.path(dataset, directory=True)
        base, card = self.reference(reference)
        for name in FILES:
            self.app.path(root/name)
        context = json.loads((root/"PARIS_TRAINING.json").read_text())
        needed = CONTEXT + ["label_version", "label_source"]
        if context.get("version") != 1 or any(not isinstance(context.get(k),str) or not context[k].strip() for k in needed):
            raise ValueError("请填写数据的PDK、流程、角点、特征/标签版本与单位")
        if context["time_unit"] != "ns" or context["label_source"] != "measured_dr_sta":
            raise ValueError("训练需要单位ns的实测DR STA标签")
        y = pd.read_parquet(root/"labels.parquet")[KEY+["final_tns"]]
        if y.empty or y.duplicated(KEY).any() or y[KEY].isna().any().any() or not np.isfinite(y.final_tns).all() or (y.final_tns > 0).any():
            raise ValueError("标签缺失、重复或非有效TNS")
        counts = y.groupby("design",sort=True).size().to_dict()
        design = design or next(iter(counts))
        if design not in counts or not all(c.isalnum() or c in "_.-" for c in design) or design in {'.','..'}:
            raise ValueError("请选择有效design")
        frame, schema = features(root, design=design)
        labels = y[y.design.eq(design)].sort_values('candidate_id')
        if set(labels.candidate_id)-set(frame.candidate_id):
            raise ValueError("实测标签缺少对应GR特征")
        identities = labels.candidate_id.tolist()
        purpose = context.get('purpose','development')
        if purpose not in ['development','engineering_smoke']:
            raise ValueError("数据purpose无效")
        line = stable(dict(algorithm=ALGORITHM,design=design,context={k:context[k] for k in needed},purpose=purpose))[:16]
        p = self.root/'lines'/line/'PROTOCOL.json'
        prior = json.loads(p.read_text()) if p.exists() else None
        validation = prior['validation_ids'] if prior else sorted(identities,key=lambda i:stable([design,i,17]))[:max(12,math.ceil(len(identities)*.2))]
        training = sorted(set(identities)-set(validation))
        blockers = []
        if set(validation)-set(identities):blockers.append('固定验证候选不能从累计数据中消失')
        if prior and set(prior['training_ids'])-set(training):blockers.append('累计数据需要包含原训练候选')
        if len(training)<24 or len(validation)<12:blockers.append('至少36例实测样本，训练≥24、验证≥12')
        values = {r.candidate_id:stable(float(r.final_tns)) for r in labels.itertuples()}
        if prior and any(values.get(k)!=v for k,v in prior['label_hashes'].items()):blockers.append('已有标签变化，请更新label_version')
        import pyarrow.parquet as pq
        return dict(algorithm=ALGORITHM,dataset=str(root),reference=str(base),reference_sha256=digest(base/'SHA256.json'),
            reference_name=card['name'],design=design,template='single-xgb',templates=['single-xgb'],
            designs=[dict(name=k,n=int(v)) for k,v in counts.items()],line=line,context={k:context[k] for k in needed},purpose=purpose,
            files=[dict(name=n,rows=pq.ParquetFile(root/n).metadata.num_rows,columns=len(pq.ParquetFile(root/n).schema_arrow.names)) for n in FILES[:-1]],
            labeled=len(identities),training=len(training),validation=len(validation),training_ids=training,validation_ids=validation,
            label_hashes=values,new_training=len(set(training)-set(prior['training_ids'])) if prior else len(training),
            protocol_exists=bool(prior),ready=not blockers,blockers=blockers,feature_count=len(schema))

    def config(self):
        d = super().config()
        d['lines'] = [line for line in d['lines'] if json.loads((self.root/'lines'/line['line']/'PROTOCOL.json').read_text()).get('algorithm')==ALGORITHM]
        d['steps'] = [dict(id=k,title=v) for k,v in STEPS.items()]
        d['algorithm'] = ALGORITHM
        return d

    def work(self, rid):
        r = self.jobs[rid]
        out, info = self.root/'jobs'/rid, r['inspection']
        dataset = Path(info['dataset'])
        try:
            r['status']='running'
            self.event(r,'snapshot','running','检查累计样本并固定验证候选')
            import shutil
            if shutil.disk_usage(out).free < 60*2**30:raise ValueError('可用空间低于60GiB')
            hashes={n:digest(dataset/n) for n in FILES}
            check=self.inspect(info['dataset'],info['reference'],info['design'])
            for k in ['training_ids','validation_ids','context','label_hashes','reference_sha256']:
                if check[k]!=info[k]:raise ValueError('检查后输入变化：'+k)
            line=self.root/'lines'/info['line'];line.mkdir(parents=True,exist_ok=True)
            p=line/'PROTOCOL.json'
            protocol=json.loads(p.read_text()) if p.exists() else dict(algorithm=ALGORITHM,design=info['design'],context=info['context'],
                validation_ids=info['validation_ids'],validation_scope='fixed same-design development holdout')
            val, _=features(dataset,design=info['design'],ids=info['validation_ids'])
            signatures={cid:hashlib.sha256(pd.util.hash_pandas_object(part.sort_index(axis=1).reset_index(drop=True),index=False).to_numpy().tobytes()).hexdigest()
                        for cid,part in val.sort_values('candidate_id').groupby('candidate_id',sort=True)}
            if protocol.get('validation_gr_hashes') and signatures!=protocol['validation_gr_hashes']:
                raise ValueError('固定验证候选的GR输入变化，请更新feature_version')
            protocol.update(training_ids=info['training_ids'],label_hashes=info['label_hashes'],validation_gr_hashes=signatures)
            save(p,protocol)
            pd.read_parquet(dataset/'labels.parquet',filters=[('design','==',info['design']),('candidate_id','in',info['training_ids'])])[KEY+['final_tns']].to_parquet(out/'training_labels.parquet',index=False)
            manifest={k:info[k] for k in ['dataset','design','training_ids','validation_ids','reference','context']}
            manifest.update(training_labels=str(out/'training_labels.parquet'),input_hashes=hashes)
            manifest['training_labels_sha256']=digest(out/'training_labels.parquet')
            save(out/'TRAIN.json',manifest)
            self.event(r,'snapshot','complete',f"固定 {info['training']} 训练候选 / {info['validation']} 验证候选")
            self.event(r,'train','running','使用训练数据拟合PARIS模型')
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
            cmd=[sys.executable,'-m','paris_tool.single_iteration_worker']
            run(cmd+['fit','--manifest',out/'TRAIN.json','--output',out/'model'],out,env,86400,'fit')
            model_hash=digest(out/'model/SHA256.json');r.update(model=str(out/'model'),model_sha256=model_hash)
            self.event(r,'train','complete','设计模型已保存')
            self.event(r,'predict','running','仅读取验证候选GR生成预测')
            run(cmd+['predict','--manifest',out/'TRAIN.json','--bundle',out/'model','--output',out/'validation_predictions.parquet'],out,env,3600,'predict')
            pred_hash=digest(out/'validation_predictions.parquet');r['prediction_sha256']=pred_hash
            current=line/'CURRENT.json';parent=json.loads(current.read_text()) if current.exists() else None
            baseline=Path(parent['model']) if parent else Path(info['reference'])
            run(cmd+['predict','--baseline','--manifest',out/'TRAIN.json','--bundle',baseline,'--output',out/'baseline_predictions.parquet'],out,env,3600,'baseline')
            self.event(r,'predict','complete','预测已封存')
            self.event(r,'evaluate','running','比较固定候选的排序与TNS误差')
            truth=pd.read_parquet(dataset/'labels.parquet',filters=[('design','==',info['design']),('candidate_id','in',info['validation_ids'])])[KEY+['final_tns']]
            pred=pd.read_parquet(out/'validation_predictions.parquet')
            e=truth.merge(pred,on=KEY,validate='one_to_one').merge(val[KEY+['grt_l2_tns']],on=KEY,validate='one_to_one')
            if len(e)!=info['validation']:raise ValueError('验证预测候选不完整')
            target=e.final_tns.to_numpy(float)
            metrics=dict(GR=metric(-e.grt_l2_tns.to_numpy(float),target,e.grt_l2_tns.to_numpy(float)),
                         candidate=metric(e.predicted_log_cost.to_numpy(float),target,e.predicted_tns.to_numpy(float)))
            if (out/'baseline_predictions.parquet').exists():
                old=pd.read_parquet(out/'baseline_predictions.parquet')[KEY+['predicted_log_cost','predicted_tns']].rename(columns={'predicted_log_cost':'current_cost','predicted_tns':'current_tns'})
                e=e.merge(old,on=KEY,validate='one_to_one')
                if len(e)!=info['validation']:raise ValueError('当前版本预测不完整')
                metrics['current']=metric(e.current_cost.to_numpy(float),e.final_tns.to_numpy(float),e.current_tns.to_numpy(float))
            elif parent:raise ValueError('当前版本无法比较')
            candidate=metrics['candidate']
            eligible=candidate['tau_b'] is not None and all(v['tau_b'] is not None and candidate['tau_b']>=v['tau_b']-1e-12 for k,v in metrics.items() if k!='candidate')
            if info['purpose']=='engineering_smoke':eligible=False
            for n,h in hashes.items():
                if digest(dataset/n)!=h:raise ValueError('训练期间输入变化：'+n)
            if digest(out/'model/SHA256.json')!=model_hash or digest(out/'validation_predictions.parquet')!=pred_hash:raise ValueError('模型或预测封存后变化')
            e.to_csv(out/'VALIDATION_CANDIDATES.csv',index=False)
            save(out/'VALIDATION.json',dict(status='passed',metrics=metrics,eligible=eligible,validation_ids=info['validation_ids'],
                scope=protocol['validation_scope'],prediction_sha256=pred_hash,model_sha256=model_hash,gate='tau-b no worse than GR and comparable current model'))
            r.update(metrics=metrics,eligible=eligible,parent=parent,validation_sha256=digest(out/'VALIDATION.json'))
            self.event(r,'evaluate','complete','验证完成，结果与权重已保存')
            r['status']='complete'
        except Exception as e:
            r.update(status='failed',error=str(e));self.event(r,r['phase'],'failed',str(e))
        finally:
            with self.app.lock:
                r['finished_at']=time.time();self.save(r);self.active=None
