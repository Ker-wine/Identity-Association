
# 跨社交媒体用户身份关联匹配项目核心思路说明

这份文档是给初学者看的，用来解释这个项目从头到尾在做什么、为什么这么做、用了什么框架、模型和方法。

项目目标很简单：  
给定 Twitter 和 Instagram 等不同社交媒体平台上的账号，判断这些账号是不是同一个真实自然人或同一个主体，并把属于同一个人的账号合并成一个稳定的 `entityId`。

---

## 1. 项目要解决什么问题

现实中，一个人可能同时拥有多个社交平台账号，例如：

- Twitter：`alice_tw`
- Instagram：`alice_ig`

这两个账号虽然平台不同、账号 ID 不同、用户名也可能不同，但它们背后可能是同一个人。

本项目要做的事情就是：

1. 读取多平台用户数据。
2. 分析每个账号的用户名、发帖文本、发帖时间、写作风格、图像特征等信息。
3. 计算两个账号之间的相似度。
4. 用机器学习模型判断两个账号是否属于同一个人。
5. 把判断为同一个人的账号合并成一个身份簇，并生成统一的 `entityId`。

第一版目标不是直接做到最高精度，而是先把完整工程流程跑通，做到一个可解释、可训练、可预测、可对接 API 的基线系统。

---

## 2. 使用的数据集

项目面向的数据集是 **TWIN Dataset**。

TWIN Dataset 是一个用于跨社交媒体用户身份关联的数据集，主要包含 Twitter 和 Instagram 两个平台上的用户数据。

在这个项目中，我们先把原始数据统一整理成两个标准文件：


```text
data/twin_std/
  users.csv
  posts.csv
```

### 2.1 users.csv

`users.csv` 存账号级信息：

```csv
entityId,platformId,userId,username
u001,twitter,tw001,alice_tw
u001,instagram,ig001,alice_ig
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `entityId` | 真实身份 ID，同一个人的多个账号共享同一个值 |
| `platformId` | 平台，例如 `twitter`、`instagram` |
| `userId` | 平台账号 ID |
| `username` | 平台用户名 |

训练阶段中，`entityId` 是标签来源。  
如果两个账号的 `entityId` 相同，就表示它们是同一个人。

### 2.2 posts.csv

`posts.csv` 存帖子级信息：

```csv
platformId,userId,postId,text,timestamp,imagePath,imageEmbedding
twitter,tw001,p1,"love coffee and travel","2017-01-01 09:00:00",,
instagram,ig001,p2,"coffee time in paris","2017-01-01 10:00:00",,
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `platformId` | 平台 |
| `userId` | 发帖账号 |
| `postId` | 帖子 ID |
| `text` | 帖子文本，比如 tweet 或 Instagram caption |
| `timestamp` | 发帖时间 |
| `imagePath` | 图片路径，可选 |
| `imageEmbedding` | 图片向量，可选 |

第一版中，图片不是强依赖。即使没有图片向量，也可以先用文本、时间、用户名、写作风格跑通。

---

## 3. 整体工程框架

项目现在被拆成了一个 Python 包：

```text
Identity-Association/
  twin_align_baseline.py
  twin_align/
    constants.py
    schemas.py
    data.py
    features.py
    pairs.py
    model.py
    inference.py
    api.py
    demo_data.py
    cli.py
    io_utils.py
```

整体调用链是：

```text
命令行 CLI / API 请求
  -> 读取数据 data.py
  -> 构建用户画像 features.py
  -> 构造账号对 pairs.py
  -> 训练模型 model.py
  -> 推理匹配 inference.py
  -> 输出 API 风格 JSON
```

可以把这个项目理解成一条流水线：

```text
原始数据
  -> 标准 users.csv/posts.csv
  -> 用户特征
  -> 账号对特征
  -> 二分类模型
  -> 匹配分数
  -> 身份簇 entityId
```

---

## 4. 每个模块做什么

### 4.1 twin_align_baseline.py

这是项目入口文件。

它本身不放复杂逻辑，只负责调用：

```python
from twin_align.cli import main
```

这样做的好处是：  
用户原来执行的命令不用变，但内部代码已经被拆成了更清晰的小文件。

### 4.2 cli.py

负责命令行功能。

支持这些命令：

```powershell
python twin_align_baseline.py make-demo-data
python twin_align_baseline.py train
python twin_align_baseline.py predict
python twin_align_baseline.py predict-all
python twin_align_baseline.py serve
```

也就是说，用户可以通过命令行完成生成样例数据、训练模型、预测账号是否匹配、启动 API 服务等操作。

### 4.3 data.py

负责读取数据。

它会读取：

```text
users.csv
posts.csv
```

并检查字段是否完整，例如必须有：

- `entityId`
- `platformId`
- `userId`
- `username`
- `text`
- `timestamp`

同时，它还会把平台名统一转成小写，例如 `Twitter` 会变成 `twitter`。

### 4.4 features.py

这是项目最核心的特征工程模块。

它负责把每个账号变成一个“用户画像”，并计算两个账号之间的相似度。

当前使用的特征包括：

| 特征 | 作用 |
| --- | --- |
| 用户名相似度 | 判断用户名是否相似 |
| 文本相似度 | 判断两个账号发的内容主题是否相似 |
| 图像相似度 | 如果有图片向量，判断图片内容是否相似 |
| 时间行为相似度 | 判断两个账号是否在相似时间段活跃 |
| 写作风格相似度 | 判断语言风格是否接近 |
| 发帖频率差异 | 判断活跃程度是否接近 |
| 发帖量差异 | 判断内容量级是否接近 |

### 4.5 pairs.py

负责构造训练样本。

机器学习模型需要训练数据，训练数据的形式是：

```text
账号 A + 账号 B + label
```

其中：

- `label = 1`：两个账号是同一个人
- `label = 0`：两个账号不是同一个人

例如：

```text
tw001 + ig001 -> 1
tw001 + ig002 -> 0
```

正样本来自相同 `entityId` 的账号对。  
负样本来自不同 `entityId` 的账号对。

### 4.6 model.py

负责训练机器学习模型。

当前第一版使用的是：

```text
HistGradientBoostingClassifier
```

它是 scikit-learn 里的一个梯度提升树模型。

选择它的原因是：

1. 依赖少，容易安装。
2. 对表格特征效果比较稳定。
3. 不需要 GPU。
4. 适合作为第一版可解释基线。
5. 后面可以继续升级成 XGBoost、LightGBM 或深度学习模型。
 

```text
runs/twin_align/twin_align_baseline.joblib
```

### 4.7 inference.py

负责推理和身份簇生成。

它主要做三件事：

1. 对一个账号对计算匹配分数。
2. 如果分数超过阈值，就认为两个账号属于同一个人。
3. 把多个匹配账号合并成一个身份簇，生成 `entityId`。

例如：

```text
tw001 和 ig001 匹配分数 = 0.91
阈值 = 0.85
0.91 >= 0.85
所以认为它们是同一个人
```

最后输出：

```json
{
  "entityId": "entity_xxx",
  "confidence": 0.91,
  "members": [
    {"platformId": "twitter", "userId": "tw001"},
    {"platformId": "instagram", "userId": "ig001"}
  ]
}
```

### 4.8 api.py

负责提供 HTTP 服务。

启动后可以访问：

```text
GET /health
POST /api/v1/identity/align
```

这样前端、后端或其他系统就可以通过 HTTP 请求来调用身份匹配能力。

---

## 5. 使用的模型

当前第一版模型是：

```text
HistGradientBoostingClassifier
```

这是一个二分类模型。

二分类的意思是：  
模型只回答一个问题：

```text
这两个账号是不是同一个人？
```

答案用概率表示：

```text
0.00 表示几乎不可能是同一个人
1.00 表示非常可能是同一个人
```

例如：

```text
tw001 + ig001 -> 0.91
tw001 + ig002 -> 0.12
```

如果我们设置阈值为 `0.85`，那么：

- `0.91 >= 0.85`，判断为同一个人
- `0.12 < 0.85`，判断为不是同一个人

---

## 6. 为什么不用一开始就上深度学习模型

参考方案中提到了更强的模型，比如：

- Sentence-BERT
- DeBERTa
- CLIP
- 多模态门控融合
- 监督对比学习
- 平台对抗学习

这些方法确实更强，但第一版不直接使用它们，主要原因是：

1. 依赖更重，环境更难配置。
2. 训练成本更高，可能需要 GPU。
3. 工程链路还没跑通时，直接上复杂模型容易排错困难。
4. 项目早期最重要的是先打通数据、训练、推理、API 全流程。

所以第一版采用轻量机器学习模型，先做一个能跑、能解释、能扩展的基线系统。

后续如果要提高准确率，可以逐步升级。

---

## 7. 当前使用的方法

### 7.1 用户级文本合并

一个账号可能有很多帖子。

我们先把同一个账号的所有文本合并起来：

```text
userText = post_1 + post_2 + post_3 + ...
```

这样每个账号就有一个整体文本表示。

### 7.2 TF-IDF 文本特征

当前文本特征使用 TF-IDF。

TF-IDF 可以理解为一种关键词权重算法：

- 一个词在某个用户文本中经常出现，说明它对这个用户重要。
- 一个词在所有用户中都经常出现，说明它区分度不高。

例如：

```text
coffee, travel, paris
```

这些词可能能说明用户兴趣。

然后我们计算两个账号文本向量的余弦相似度：

```text
textSimilarity = cosine(textVectorA, textVectorB)
```

### 7.3 用户名相似度

用户名也很有用。

例如：

```text
alice_tw
alice_ig
```

虽然不完全一样，但明显比较相似。

当前使用两种用户名相似度：

1. 字符串相似度。
2. 三字符片段 Jaccard 相似度。

### 7.4 时间行为特征

一个人的活跃时间通常有一定习惯。

例如：

- 有些人喜欢早上发帖。
- 有些人喜欢晚上发帖。
- 有些人周末更活跃。

所以项目会统计：

| 特征 | 含义 |
| --- | --- |
| 24 小时活跃分布 | 用户一天中哪些小时常发帖 |
| 周内活跃分布 | 用户一周中哪几天常发帖 |
| burstiness | 用户发帖是否集中爆发 |
| 发帖频率 | 单位时间内发帖多少 |

然后计算两个账号的时间行为是否相似。

### 7.5 写作风格特征

不同人写东西有不同习惯。

当前提取的写作风格包括：

- 平均句长
- 标点比例
- 数字比例
- 大写比例
- 特殊字符比例
- 平均词长
- 词数量

这些特征不一定单独决定身份，但可以作为辅助证据。

### 7.6 图像特征

第一版支持图像向量，但不强制提取图片。

如果后面用 CLIP 把图片转成向量，可以写入：

```text
posts.csv.imageEmbedding
```

系统会自动读取并计算图像相似度。

这为后续升级多模态模型预留了接口。

---

## 8. 训练流程

训练流程如下：

```text
1. 读取 users.csv 和 posts.csv
2. 构建每个账号的 UserProfile
3. 根据 entityId 构造正样本
4. 随机采样不同 entityId 的账号对作为负样本
5. 为每个账号对计算 pairFeature
6. 用 pairFeature 和 label 训练二分类模型
7. 保存模型 artifact
8. 输出 metrics.json 和 api_response_demo.json
```

命令：

```powershell
python twin_align_baseline.py train --dataDir ./data/twin_std --outDir ./runs/twin_align
```

训练完成后会得到：

```text
runs/twin_align/twin_align_baseline.joblib
runs/twin_align/metrics.json
runs/twin_align/api_response_demo.json
```

其中：

- `twin_align_baseline.joblib` 是训练好的模型。
- `metrics.json` 是训练评估指标。
- `api_response_demo.json` 是一份 API 返回示例。

---

## 9. 推理流程

推理就是给两个账号，判断它们是不是同一个人。

流程如下：

```text
1. 加载训练好的模型
2. 找到账号 A 和账号 B 的 UserProfile
3. 计算两个账号之间的特征
4. 模型输出匹配概率 confidence
5. 和 mergeThreshold 比较
6. 如果超过阈值，合并为一个身份簇
7. 输出 API 风格 JSON
```

命令：

```powershell
python twin_align_baseline.py predict ^
  --modelPath ./runs/twin_align/twin_align_baseline.joblib ^
  --userA tw001 ^
  --userB ig001
```

输出大概长这样：

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
      "alignmentEvidence": {
        "pairScore": 0.91,
        "featureScores": {}
      },
      "physicalExplanation": [
        "username similarity=0.750",
        "text similarity=0.520"
      ]
    }
  ],
  "unmatchedUsers": []
}
```

---

## 10. API 服务流程

除了命令行，也可以启动 HTTP 服务。

命令：

```powershell
python twin_align_baseline.py serve --modelPath ./runs/twin_align/twin_align_baseline.joblib
```

然后请求：

```http
POST /api/v1/identity/align
```

请求体示例：

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

API 内部仍然走同一套推理流程：

```text
API 请求
  -> predict_pair
  -> build_api_response
  -> JSON 返回
```

---

## 11. entityId 是怎么生成的

当多个账号被判断为同一个人后，系统会把它们合并成一个身份簇。

例如：

```text
twitter:tw001
instagram:ig001
```

系统会把这些成员组合起来，并生成一个稳定的哈希 ID：

```text
entity_xxx
```

这个 `entityId` 就代表一个跨平台统一身份。

---

## 12. alignmentEvidence 和 physicalExplanation 是什么

项目不仅要给出结果，还要尽量解释为什么这么判断。

### 12.1 alignmentEvidence

它保存模型判断时用到的特征分数，例如：

```json
{
  "pairScore": 0.91,
  "featureScores": {
    "username_similarity": 0.75,
    "text_similarity": 0.52,
    "temporal_similarity": 0.81
  }
}
```

### 12.2 physicalExplanation

它是给人看的解释文本，例如：

```text
username similarity=0.750
temporal similarity=0.810
text similarity=0.520
```

这样项目答辩或调试时，不只是看到一个黑盒结果，还能知道模型主要依赖了哪些证据。

---

## 13. 为什么这个方案适合快速落地

这个方案适合作为第一版，是因为：

1. 数据格式简单，只需要 `users.csv` 和 `posts.csv`。
2. 模型轻量，不需要 GPU。
3. 特征可解释，方便调试和答辩。
4. 命令行和 API 都有，方便后续对接系统。
5. 结构已经模块化，后续容易升级。

它不是最终最强方案，但它是一个很适合快速跑通项目的工程基线。

---

## 14. 后续如何提高准确率

后续可以按这个顺序升级：

### 14.1 加入更强文本模型

当前是 TF-IDF，后续可以换成：

- Sentence-BERT
- DeBERTa

这样能更好理解语义相似度。

### 14.2 加入 CLIP 图像特征

如果有用户发布图片，可以用 CLIP 提取图像向量。

这样系统就能比较两个账号发布图片的视觉内容是否相似。

### 14.3 加入难负样本

当前负样本主要是随机负样本。

后续可以加入更难的负样本，例如：

- 用户名很像但不是同一个人
- 文本主题很像但不是同一个人
- 发帖时间很像但不是同一个人

这样模型会更强。

### 14.4 加入时间指纹

可以用更细的时间序列方法，比如 KS-CDF，比较用户发帖间隔分布。

这比简单统计 24 小时分布更精细。

### 14.5 升级成多模态深度模型

最终可以升级为：

```text
文本编码器 + 图像编码器 + 时间编码器 + 门控融合 + 对比学习
```

这就是更接近高精度版本的方向。

---

## 15. 一句话总结

这个项目第一版做的是一个“跨平台账号是否属于同一人”的机器学习基线系统。

它把不同平台账号的数据整理成标准格式，再从用户名、文本、时间、风格、图像等角度提取特征，用二分类模型输出匹配概率，最后把高置信度账号合并成统一身份 `entityId`，并通过命令行和 API 提供完整可运行的工程链路。
