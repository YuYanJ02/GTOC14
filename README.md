# 2026 CTOC14 甲题可行解

近地小行星防御多航天器遍历探测轨道设计与优化。

当前提交包含 5 艘连续推力航天器，覆盖 75 颗不同小行星，任务代价为
`J = 240`。仓库内独立校验器逐段复算结果为 `valid=True`。

- 最终提交：`output/CTOC14_Result_TeamID.txt`
- 完整方法与结果：[SOLUTION.md](SOLUTION.md)
- 动力学与格式工具：`ctoc14_core.py`
- 单航天器求解：`solve_ctoc14.py`
- 多航天器扩展：`extend_ctoc14_fleet.py`
- 独立校验：`validate_ctoc14.py`

安装依赖后可运行：

```bash
python3 validate_ctoc14.py output/CTOC14_Result_TeamID.txt
```

预期输出包括：

```text
valid=True
spacecraft=5 covered=75 J=240
```

该结果是满足题目约束的可行解，不宣称为全局最优解。
