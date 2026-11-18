#!/usr/bin/env python3
"""Build the vendored CPU DREAMPlace operators using the active Python/Torch ABI."""
import argparse,os,subprocess,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
root=Path(__file__).resolve().parents[1]/'vendor/Place-MoL/DREAMPlace'
p=argparse.ArgumentParser();p.add_argument('--jobs',type=int,default=2);a=p.parse_args()
if a.jobs<1:p.error('--jobs must be positive')
args=['cmake','-S',str(root),'-B',str(root/'build'),'-DCMAKE_BUILD_TYPE=Release',
 f'-DPython_EXECUTABLE={sys.executable}',f'-DCMAKE_CXX_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}',
 '-DCMAKE_DISABLE_FIND_PACKAGE_CUDA=TRUE','-DCMAKE_POLICY_VERSION_MINIMUM=3.5',f'-DCMAKE_INSTALL_PREFIX={root}/install']
subprocess.run(args,check=True);subprocess.run(['cmake','--build',str(root/'build'),'--parallel',str(a.jobs)],check=True)
subprocess.run(['cmake','--install',str(root/'build')],check=True)
print(root/'install')
