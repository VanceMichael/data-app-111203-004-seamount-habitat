# 生境分群模块

把连续地形（depth、slope、roughness）与离散底质（substrate）联合用于网格级生境分群，支持按海山整组留出的外推验证；产物可保存并重新载入，同一模型再次载入后归属一致。

## 流程

1. **重复观测合并**：同一网格的多次观测合并为一条——数值取均值，底质取众数（并列取字典序较小者），代表面积取均值；同一网格出现在多座海山视为数据错误。
2. **预处理只在训练海山上拟合**：缺值用面积加权中位数填补；数值特征按面积加权均值与标准差标准化（标准差为 0 时退化为 1）；底质字典取训练类别排序，预测时未见底质编码为全零并触发弃权。预处理参数随模型产物保存。
3. **面积加权 k 均值**：weight 为网格代表面积（平方千米），作为样本权重参与中心与惯性计算，避免调查网格数量代替真实面积。固定种子多次重启取惯性最小；空簇丢弃，保留簇按面积占比降序重编号；全部网格特征相同时收敛为单簇。
4. **簇数选择**：候选 k 按「归一化训练轮廓系数 + 留出稳定度 + 小簇规模（最小簇面积占比×k，封顶 1）」平均得分，取最高，并列取较小 k，不只追求训练轮廓系数。留出稳定度按 seamount 整组留出（LOSO）：折内模型与全量模型分别预测留出海山，先把重采样簇编号按匈牙利算法对齐到基准簇，再计算调整后兰德指数。不随机拆分网格。
5. **置信诊断与弃权**：置信边际 = (次近距离 − 最近距离) / 次近距离。出现训练未见底质、最近中心距离超过该簇训练期最大距离、或边际低于 min_margin（默认 0.1）时，该网格 cluster 判为 null（无可靠归属），原因写入 abstain_reasons。

## 产物

- **模型 JSON**（`--model`）：预处理参数、簇中心、面积占比、训练期最大距离、min_margin、选簇报告、中文生境特征说明（profiles）、weight 语义原文与内容指纹。
- **归属 JSON**（`--assignments`）：每个网格的簇归属、置信度、距离、边际、信息标记（如缺值已填补）与弃权原因，并回写模型指纹与 weight 语义。

## 使用

```bash
# 训练（自动选簇）并输出逐网格归属
python3 -m analysis_app habitat --model output/habitat-model.json --assignments output/assignments.json

# 固定簇数
python3 -m analysis_app habitat --k 3 --model output/habitat-model.json

# 载入既有模型预测新网格
python3 -m analysis_app habitat --predict --model output/habitat-model.json --input new.csv --assignments output/new-assignments.json
```

## 语义与限制

weight 语义原文写入产物 `weight_semantics` 字段：训练与解释全程按代表面积加权，不以调查网格数量代替真实面积。演示数据为本地合成地形，不对应真实海山；不请求在线地图或任何外部接口。
