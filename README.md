# 跨社交媒体用户身份关联匹配基线

这个仓库先实现一个可快速落地的 TWIN Dataset 身份对齐闭环：标准数据读取、特征抽取、候选账号对二分类、身份簇聚合、API 风格 JSON 输出。

第一版目标是把框架跑通并冲到可用基线，后续再替换 Sentence-BERT、CLIP、时间指纹和深度多模态融合。

## 数据格式

把 TWIN 原始数据转换为：

```text
data/twin_std/
  users.csv
  posts.csv
  images/
```

`users.csv`：

```csv
entityId,platformId,userId,username
u001,twitter,tw001,alice_tw
u001,instagram,ig001,alice_ig
```

`posts.csv`：

```csv
platformId,userId,postId,text,timestamp,imagePath,imageEmbedding
twitter,tw001,p1,"love coffee and travel","2017-01-01 09:00:00",,
instagram,ig001,p2,"coffee time in paris","2017-01-01 10:00:00",,
```

`imageEmbedding` 是可选字段，格式为 JSON 数组。第一版没有图像向量也可以跑通。

## 快速运行

```powershell
pip install -r requirements.txt
python twin_align_baseline.py make-demo-data --outDir ./data/twin_std
python twin_align_baseline.py train --dataDir ./data/twin_std --outDir ./runs/twin_align --negativeRatio 3 --mergeThreshold 0.85
python twin_align_baseline.py predict --modelPath ./runs/twin_align/twin_align_baseline.joblib --userA tw001 --userB ig001 --taskId demo-task-001 --mergeThreshold 0.85
```

训练输出：

```text
runs/twin_align/twin_align_baseline.joblib
runs/twin_align/metrics.json
runs/twin_align/api_response_demo.json
```

## API 服务

```powershell
python twin_align_baseline.py serve --modelPath ./runs/twin_align/twin_align_baseline.joblib --host 127.0.0.1 --port 8000
```

请求：

```http
POST /api/v1/identity/align
Content-Type: application/json
```

```json
{
  "taskId": "demo-task-001",
  "candidates": [
    {
      "sourcePlatformId": "twitter",
      "sourceUserId": "tw001",
      "targetPlatformId": "instagram",
      "targetUserId": "ig001"
    }
  ],
  "alignConfig": {
    "mergeThreshold": 0.85
  }
}
```

返回字段包含：

```json
{
  "taskId": "demo-task-001",
  "modelVersion": "twin-align-baseline-0.1.0",
  "inferenceTimeMs": 1,
  "identityClusters": [
    {
      "entityId": "entity_xxx",
      "confidence": 0.91,
      "members": [
        {"platformId": "instagram", "userId": "ig001"},
        {"platformId": "twitter", "userId": "tw001"}
      ],
      "alignmentEvidence": {},
      "physicalExplanation": []
    }
  ],
  "unmatchedUsers": []
}
```

## 当前特征

- 用户名相似度
- 用户文本 TF-IDF 余弦相似度
- 可选图像 embedding 余弦相似度
- 24 小时活跃分布、周内活跃分布、burstiness
- 写作风格统计
- 发帖频率差异、发帖量差异

## 后续提升路线

1. 加入 CLIP 离线图像向量，写入 `posts.csv.imageEmbedding`。
2. 用 Sentence-BERT 或 DeBERTa 替换 TF-IDF 用户文本向量。
3. 增加难负样本：文本相似、时间相似、用户名相似但非同一人的账号对。
4. 增加 inter-event time 的 KS-CDF 时间指纹相似度。
5. 升级到多塔编码、门控融合、监督对比学习和平台对抗训练。

## 整体框架结构

当前工程已从单文件拆分为 `twin_align/` 包，根目录的 `twin_align_baseline.py` 只作为兼容入口，原来的运行命令保持不变。

```text
Identity-Association/
  twin_align_baseline.py        # 命令行兼容入口，调用 twin_align.cli.main()
  twin_align/
    __init__.py                 # 包基础信息
    constants.py                # 全局常量和特征名
    schemas.py                  # UserProfile 数据结构
    data.py                     # 标准 users.csv/posts.csv 读取与校验
    features.py                 # 文本、图像、时间、风格、用户名特征工程
    pairs.py                    # 正负样本和候选账号对矩阵构造
    model.py                    # 模型训练、评估和 artifact 保存
    inference.py                # 单对预测、身份簇聚合、API JSON 组装
    api.py                      # FastAPI HTTP 服务
    demo_data.py                # 最小 demo 数据生成
    cli.py                      # make-demo-data/train/predict/predict-all/serve 命令
    io_utils.py                 # JSON 保存等通用 I/O 工具
```

核心调用链：

```text
CLI/API
  -> data
  -> features
  -> pairs
  -> model
  -> inference
  -> API-style JSON
```

各 Python 文件顶部都已添加统一中文说明，包含“文件内容、主要职责、前置文件、后置文件”，方便后续维护和项目答辩说明。
## 文档入口

- 项目整体思路说明：`PROJECT_OVERVIEW.md`
- Demo 运行手册：`RUN_DEMO.md`

## GPU 说明

当前这版 baseline 采用 `TF-IDF + HistGradientBoostingClassifier`，训练和推理主流程默认使用 CPU。
如果你的机器是 `NVIDIA RTX 4090 24GB`，本项目可以正常运行，但当前 baseline 不会重点消耗 GPU。
后续如果升级到 `Sentence-BERT`、`CLIP` 或多模态深度学习模型，4090 会更有价值。
