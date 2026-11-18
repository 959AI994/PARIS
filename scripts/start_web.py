#!/usr/bin/env python3
"""Start the web workspace with the bundled real model and candidate example."""
import argparse,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);p.add_argument('--workspace',type=Path,default=root/'work/web')
p.add_argument('--model',type=Path,default=root/'models/seven-designs');p.add_argument('--dataset',type=Path,default=root/'examples/gr-candidates');p.add_argument('--profile',type=Path)
p.add_argument('--root',type=Path,action='append');p.add_argument('--open',action='store_true')
a=p.parse_args();args=[sys.executable,'-m','paris_tool.cli','ui','--port',str(a.port),'--workspace',str(a.workspace),'--model',str(a.model),'--dataset',str(a.dataset)]
for path in a.root or [root]:args+=['--root',str(path)]
profile=a.profile or root/'native-profile.json'
if profile.exists():args+=['--preparation-profile',str(profile)]
if a.open:args+=['--open']
raise SystemExit(subprocess.call(args))
