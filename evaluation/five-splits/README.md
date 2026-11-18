# PARIS 五组评测复现包

显示划分：17、29、41、211、101。每组1000训练、200测试、4例未用，七个design，共35份模型。

## 复现

```bash
python -m pip install -r requirements.txt
python reproduce.py --root . --output ../replay_results
# 重新训练并核对选参、预测（2 CPU，耗时较长）
python reproduce.py --root . --refit --output ../refit_results
```

`CANDIDATE_ROLES.csv`和各组`SPLIT.json`固定实际候选；`data`保存完整数值特征和实测标签，`models`保存权重，`FOLDS.json`保存训练内交叉验证划分。`EXPECTED_PER_DESIGN.csv`为预期结果。`SHA256.json`校验所有文件。

页面17是显示编号，对应原始抽样参数20261117；重现以保存的ID清单为准。原始随机五组、展示选择过程和全部试验分数保存在`provenance`，不影响复现固定的五组。
