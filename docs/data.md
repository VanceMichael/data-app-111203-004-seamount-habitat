# 数据契约

海山调查按海山统计网格深度、坡度和底质覆盖，采样权重表示网格代表面积。

主测量 `depth` 的单位为米，主体字段是 `cell`，核算维度是 `seamount`。`observation_id` 是文件内唯一观测编号，主体允许重复观测。数值只接受有限值；空单元格解析为 None，不能直接当成零。额外列不进入当前核算。

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `observation_id` | str | 必填 |
| `cell` | str | 必填 |
| `seamount` | str | 必填 |
| `depth` | float | 必填 |
| `slope` | float | 允许空值，空值不是零 |
| `roughness` | float | 允许空值，空值不是零 |
| `substrate` | str | 允许空值，空值不是零 |
| `survey` | str | 允许空值，空值不是零 |
| `weight` | float | 允许空值，空值不是零 |

时间字段 day、week、tick、hour、minute、cycle、start_day、reported_day、known_at、visible_at、opened、observed 均以各字段名所述的相对时间单位表示整数；不同字段并不必然代表同一种时间。缺少的登记码与浓度等信息保持为空。示例只表达产品数据结构，不能代替实际调查资料。
