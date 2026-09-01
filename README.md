# 2026 CTOC14 甲题轨迹优化

近地小行星防御多航天器遍历探测轨道设计与优化。

当前正式提交包含 29 艘连续推力航天器，覆盖 300 颗候选小行星中的 296 颗，
原始目标函数为 `J = 82.3629454988`。仓库内独立校验器逐行重积分结果为
`valid=True`。

相较仓库原始的 5 艘、75 目标、`J=240` 基线，本轮将覆盖率提高到 98.67%，
并将目标函数降低 65.68%。原始提交保存在
`output/CTOC14_Result_baseline_J240.txt`。

## 主要文件

- 正式提交：`output/CTOC14_Result_TeamID.txt`
- 完整方法与结果：[SOLUTION.md](SOLUTION.md)
- 动力学与格式工具：`ctoc14_core.py`
- 单航天器基线求解：`solve_ctoc14.py`
- 固定 174 首站的多航天器扩展：`extend_ctoc14_fleet.py`
- 多艘自动迭代：`iterate_ctoc14_fleet.py`
- 地球直达与发射余速优化：`extend_ctoc14_direct.py`
- 轨迹保持的质量缩放：`optimize_submission.py`
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
spacecraft=29 covered=296 J=82.3629454988
```

当前未覆盖目标为 `11, 131, 144, 236`。这是一个经过独立动力学复算的可行解，
不宣称为全局最优解。
