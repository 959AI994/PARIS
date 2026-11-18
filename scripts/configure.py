#!/usr/bin/env python3
"""Write native tool settings and the web preparation profile for this checkout."""
import argparse,json,shutil,sys
from pathlib import Path
from paris_tool.mol_profile import write_profile

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--openroad',default=shutil.which('openroad'))
p.add_argument('--yosys',default=shutil.which('yosys'))
p.add_argument('--place-python',default=sys.executable)
p.add_argument('--output',type=Path,default=root/'native.json')
a=p.parse_args()
if not a.openroad or not a.yosys:p.error('Pass --openroad /path/openroad --yosys /path/yosys')
for relative in ['vendor/OpenROAD-3D/flow/scripts_3D/init_odb.tcl',
                 'vendor/Place-MoL/DREAMPlace/install/dreamplace/Params.py']:
    if not (root/relative).is_file():p.error('Missing '+relative+'; run python scripts/build_place.py first')
config=dict(openroad=str(Path(a.openroad).resolve(strict=True)),yosys=str(Path(a.yosys).resolve(strict=True)),
 place_python=str(Path(a.place_python).absolute()),feature_python=str(Path(sys.executable).absolute()),
 place_mol_root=str(root/'vendor/Place-MoL'),flow_root=str(root/'vendor/OpenROAD-3D/flow'))
if a.output.exists():p.error('Output already exists; choose a new --output')
a.output.write_text(json.dumps(config,indent=2)+'\n')
profile=a.output.with_name(a.output.stem+'-profile.json')
write_profile(a.output,profile)
print(json.dumps(dict(settings=str(a.output),profile=str(profile)),indent=2))
