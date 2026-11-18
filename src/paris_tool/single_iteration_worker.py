"""Training and GR-only validation prediction in separate processes."""
import argparse
import json
from pathlib import Path

from .algorithms.single_xgb import fit, features, infer
from .utils import digest, save, verify


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['fit','predict']);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--bundle',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',action='store_true')
    a=p.parse_args();m=json.loads(a.manifest.read_text())
    if a.phase=='fit':
        if digest(m['training_labels']) != m['training_labels_sha256']:
            raise ValueError('Training label snapshot changed')
        for n,h in m['input_hashes'].items():
            if digest(Path(m['dataset'])/n)!=h:raise ValueError('Input changed: '+n)
        fit(m['dataset'],a.output,m['design'],m['training_ids'],m['context'],m['training_labels'])
    else:
        card=verify(a.bundle)
        overlap=m['design'] not in card['designs'] or bool(set(card['models'].get(m['design'],{}).get('training_ids',[]))&set(m['validation_ids']))
        overlap=overlap or bool(card.get('context') and any(card['context'].get(k)!=m['context'].get(k) for k in ['pdk_id','flow_id','corner','feature_version','time_unit']))
        if overlap:
            if not a.baseline:raise ValueError('Validation candidates overlap model training or design unknown')
            save(a.output.with_suffix('.skip.json'),dict(reason='当前基础模型没有可比较的独立验证预测'))
            return
        schema=json.loads((a.bundle/'FEATURE_SCHEMA.json').read_text())['columns']
        frame,_=features(m['dataset'],schema,design=m['design'],ids=m['validation_ids'])
        result=infer(a.bundle,frame);result.to_parquet(a.output,index=False)


if __name__=='__main__':main()
