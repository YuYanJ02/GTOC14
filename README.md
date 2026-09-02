# 2026 CTOC14 甲题轨迹优化

近地小行星防御多航天器遍历探测轨道的全局搜索与组合优化。

当前正式提交包含 39 艘连续推力航天器，唯一覆盖 300 个候选目标中的 298 个，
目标函数为 `J = 53.2625020247`，独立逐段复算结果为 `valid=True`。未覆盖目标为
`131, 144`；全部 298 个已覆盖目标各申报一次，174 号小行星也只申报一次。

相较上一份正式结果 `J=73.3434421419`，本轮降低 `20.0809401172`（27.38%）。
搜索不再固定前 20 艘、固定 174 前缀或固定目标序列，而是联合处理地球发射窗口、
真实连续推力状态下的整条小行星序列，以及最终的多航天器路线选择。

## 主要文件

- 正式提交：`output/CTOC14_Result_TeamID.txt`
- 上一份正式结果备份：`output/CTOC14_Result_pre_global_J73.txt`
- 完整方法与结果：[SOLUTION.md](SOLUTION.md)
- 全局发射窗口与冲量级路线搜索：`search_low_mass_global.py`
- 真实状态连续推力多起点搜索：`search_continuous_global.py`
- 全路线池 0–1 集合覆盖：`optimize_fleet_global.py`
- 带可行 incumbent 的大邻域重组：`optimize_fleet_lns.py`
- 删除重复 Event-3 申报：`deduplicate_submission_encounters.py`
- 动力学与提交格式：`ctoc14_core.py`
- 独立动力学校验：`validate_ctoc14.py`
- 提交统计：`summarize_submission.py`

## 快速复核

```bash
python -X utf8 validate_ctoc14.py output/CTOC14_Result_TeamID.txt
python -X utf8 summarize_submission.py output/CTOC14_Result_TeamID.txt
```

校验摘要应为：

```text
valid=True
spacecraft=39 covered=298 J=53.2625020247
```

当前结果是经过独立动力学复算的可行解，不宣称为全局最优解。路线缓存位于
`tmp/global_route_pool.pkl`，属于可再生搜索产物，不需要手动上传到新仓库。
