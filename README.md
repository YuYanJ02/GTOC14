# 2026 CTOC14 甲题轨迹优化

近地小行星防御多航天器遍历探测轨道设计与优化。

当前正式提交包含 29 艘连续推力航天器，覆盖 300 颗候选小行星中的 296 颗，
目标函数为 `J = 73.3434421419`。仓库内独立校验器逐行重新积分的结果为
`valid=True`。

相较仓库最初的 5 艘、75 目标、`J=240` 基线，当前覆盖率达到 98.67%，
目标函数降低 69.44%。相较上一版 `J=82.3629454988`，本轮取消了重复申报的
174 号小行星，并通过反向燃料优化将总初始质量减少 `4678.27 kg`。

## 主要文件

- 正式提交：`output/CTOC14_Result_TeamID.txt`
- 完整方法与结果：[SOLUTION.md](SOLUTION.md)
- 动力学与格式工具：`ctoc14_core.py`
- 单航天器基线求解：`solve_ctoc14.py`
- 固定 174 前缀的早期舰队扩展：`extend_ctoc14_fleet.py`
- 多艘自动迭代：`iterate_ctoc14_fleet.py`
- 地球直接出发搜索：`extend_ctoc14_direct.py`
- 重建并优化地球出发前缀：`optimize_direct_prefixes.py`
- 固定目标序列的反向燃料优化：`optimize_route_fuel_backward.py`
- 轨迹保持质量缩放：`optimize_submission.py`
- 独立校验：`validate_ctoc14.py`
- 提交统计：`summarize_submission.py`

## 快速复核

```bash
python -X utf8 validate_ctoc14.py output/CTOC14_Result_TeamID.txt
python -X utf8 summarize_submission.py output/CTOC14_Result_TeamID.txt
```

校验结果的首行应为：

```text
valid=True
spacecraft=29 covered=296 J=73.3434421419
```

当前未覆盖目标为 `11, 131, 144, 236`。174 号小行星只由 SC1 申报一次；
SC2–20 不再重复申报。当前结果是经过独立动力学复算的可行解，不宣称为全局最优解。
