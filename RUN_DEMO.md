# 项目 Demo 运行手册

这份文档专门讲两件事：

1. 如何把这个跨平台身份对齐项目的 demo 跑起来。
2. 如何使用我已经写好的命令行命令。

这份说明默认你在 Windows + PowerShell 环境下操作，项目根目录为：

```text
Identity-Association/
```

---

## 1. 先说结论：这版 demo 怎么跑

最短路径一共 4 步：

```powershell
pip install -r requirements.txt
python twin_align_baseline.py make-demo-data --outDir ./data/twin_std
python twin_align_baseline.py train --dataDir ./data/twin_std --outDir ./runs/twin_align
python twin_align_baseline.py predict --modelPath ./runs/twin_align/twin_align_baseline.joblib --userA tw001 --userB ig001
```

如果这 4 步都成功，说明整个最小 demo 已经跑通了。

---

## 2. 这个 demo 现在跑的是什么

当前这版是一个“可快速落地的 baseline”工程，核心流程是：

```text
users.csv / posts.csv
  -> 构建用户画像
  -> 生成账号对
  -> 提取 pair 特征
  -> 训练二分类模型
  -> 预测两个账号是否属于同一人
  -> 生成 entityId 风格的聚合结果
```

当前使用的核心模型是：

```text
HistGradientBoostingClassifier
```

它属于 `scikit-learn` 的梯度提升树模型，优点是：

- 上手快
- 依赖轻
- 可解释
- 适合先把工程链路跑通

---

## 3. 关于你的 NVIDIA RTX 4090

这一点我直接给你讲清楚，避免你后面踩坑。

### 3.1 当前这版 baseline 是否能在 4090 机器上运行？

可以。

### 3.2 当前这版 baseline 是否会重点使用 4090 GPU？

不会。

原因是当前训练和推理主模型是：

- `TF-IDF`
- `HistGradientBoostingClassifier`

这两部分主要是 CPU 流程，不是 PyTorch GPU 训练流程。  
所以你现在这台带 `NVIDIA RTX 4090 24GB` 的机器当然可以跑这个项目，但这版 demo 还不会把 GPU 真正“吃满”。

### 3.3 那 4090 在这个项目里什么时候有价值？

当你后面升级这些模块时，4090 就很有用了：

- 用 `Sentence-BERT` / `bge` / `DeBERTa` 做文本向量
- 用 `CLIP` 做图片 embedding
- 用多模态深度学习模型替换当前 baseline

也就是说：

- **当前版本：CPU baseline，先跑通工程**
- **后续升级版：4090 用来加速 embedding 和深度模型**

这是我对现在代码的真实说明，不是你环境有问题，而是当前实现本身就是轻量 baseline。

---

## 4. 运行前准备

建议环境：

- Windows 10/11
- Python 3.10 或 3.11
- `pip`
- 项目根目录已打开 PowerShell

你可以先在项目根目录执行：

```powershell
python --version
```

如果能看到 Python 版本号，再继续下面步骤。

---

## 5. 安装依赖

项目依赖在 `requirements.txt` 中。

执行：

```powershell
pip install -r requirements.txt
```

如果你后面还要启 API，需要确保下面两个库也安装成功：

- `fastapi`
- `uvicorn`

它们已经在 `requirements.txt` 里了。

---

## 6. 项目目录结构

当前核心结构如下：

```text
Identity-Association/
  twin_align_baseline.py
  README.md
  PROJECT_OVERVIEW.md
  RUN_DEMO.md
  requirements.txt
  data/
    twin_std/
      users.csv
      posts.csv
  twin_align/
    cli.py
    data.py
    features.py
    pairs.py
    model.py
    inference.py
    api.py
    demo_data.py
    constants.py
    schemas.py
    io_utils.py
```

其中：

- `twin_align_baseline.py` 是总入口
- `twin_align/cli.py` 是命令行命令注册处
- `data/twin_std/` 是标准化数据目录
- `runs/twin_align/` 是训练输出目录

---

## 7. 第一步：生成 demo 数据

如果你还没有准备自己的标准化数据，先用项目内置 demo 数据跑通流程。

执行：

```powershell
python twin_align_baseline.py make-demo-data --outDir ./data/twin_std
```

运行后你会得到：

```text
data/twin_std/users.csv
data/twin_std/posts.csv
```

这一步的作用是：

- 生成最小可运行样例
- 让训练流程先能走通
- 方便验证命令行和模型逻辑没有问题

---

## 8. 第二步：训练模型

执行：

```powershell
python twin_align_baseline.py train `
  --dataDir ./data/twin_std `
  --outDir ./runs/twin_align `
  --negativeRatio 3 `
  --mergeThreshold 0.85 `
  --testSize 0.25 `
  --seed 42 `
  --taskId demo-task-001
```

### 8.1 这条命令在做什么

它会完成整条训练链路：

1. 读取 `users.csv` 和 `posts.csv`
2. 构建每个账号的 `UserProfile`
3. 生成正样本和负样本账号对
4. 提取 pair 特征
5. 训练 `HistGradientBoostingClassifier`
6. 保存模型和评估结果

### 8.2 训练输出文件

默认会在这里生成结果：

```text
runs/twin_align/
```

核心文件包括：

```text
runs/twin_align/twin_align_baseline.joblib
runs/twin_align/metrics.json
runs/twin_align/api_response_demo.json
```

它们的含义是：

- `twin_align_baseline.joblib`：训练好的模型 artifact
- `metrics.json`：训练评估指标
- `api_response_demo.json`：一份示例 API 输出

---

## 9. 第三步：预测单个账号对

训练完成后，可以先验证一个账号对是否匹配。

执行：

```powershell
python twin_align_baseline.py predict `
  --modelPath ./runs/twin_align/twin_align_baseline.joblib `
  --platformA twitter `
  --userA tw001 `
  --platformB instagram `
  --userB ig001 `
  --taskId demo-task-001 `
  --mergeThreshold 0.85
```

### 9.1 返回结果是什么

终端会打印一份 JSON，里面主要有：

- `taskId`
- `modelVersion`
- `inferenceTimeMs`
- `identityClusters`
- `unmatchedUsers`

如果这两个账号被判断为同一个人，那么结果里会出现：

- `entityId`
- `confidence`
- `alignmentEvidence`
- `physicalExplanation`

---

## 10. 第四步：批量预测全部候选对

如果你想一次性看看 Twitter 和 Instagram 全部账号两两之间的预测结果，可以用：

```powershell
python twin_align_baseline.py predict-all `
  --modelPath ./runs/twin_align/twin_align_baseline.joblib `
  --platformA twitter `
  --platformB instagram `
  --taskId demo-task-all `
  --mergeThreshold 0.85 `
  --out ./runs/twin_align/predict_all.json
```

这条命令会：

- 读取模型中的全部用户画像
- 枚举 `platformA` 和 `platformB` 的候选账号对
- 对每一对都做预测
- 最后输出聚合后的 JSON 结果

如果加了 `--out`，还会把结果保存到文件中。

---

## 11. 第五步：启动 API 服务

如果你想让前端或其他系统通过 HTTP 调用身份对齐能力，可以启动服务：

```powershell
python twin_align_baseline.py serve `
  --modelPath ./runs/twin_align/twin_align_baseline.joblib `
  --host 127.0.0.1 `
  --port 8000
```

启动后可用接口：

```text
GET  /health
POST /api/v1/identity/align
```

### 11.1 健康检查

```http
GET http://127.0.0.1:8000/health
```

### 11.2 预测请求示例

```http
POST http://127.0.0.1:8000/api/v1/identity/align
Content-Type: application/json
```

请求体：

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

---

## 12. 所有命令行命令总览

这个项目目前支持以下命令：

### 12.1 查看帮助

```powershell
python twin_align_baseline.py --help
python twin_align_baseline.py train --help
python twin_align_baseline.py predict --help
```

### 12.2 生成 demo 数据

```powershell
python twin_align_baseline.py make-demo-data --outDir ./data/twin_std
```

### 12.3 训练

```powershell
python twin_align_baseline.py train --dataDir ./data/twin_std --outDir ./runs/twin_align
```

### 12.4 预测单对账号

```powershell
python twin_align_baseline.py predict --modelPath ./runs/twin_align/twin_align_baseline.joblib --userA tw001 --userB ig001
```

### 12.5 批量预测

```powershell
python twin_align_baseline.py predict-all --modelPath ./runs/twin_align/twin_align_baseline.joblib --out ./runs/twin_align/predict_all.json
```

### 12.6 启动 API

```powershell
python twin_align_baseline.py serve --modelPath ./runs/twin_align/twin_align_baseline.joblib --host 127.0.0.1 --port 8000
```

---

## 13. 关键参数怎么理解

### `--dataDir`

标准化数据目录，里面至少应有：

- `users.csv`
- `posts.csv`

### `--outDir`

训练输出目录，模型和指标文件会写到这里。

### `--negativeRatio`

负样本采样比例。  
比如设为 `3`，表示每个正样本大约采 3 个负样本。

### `--mergeThreshold`

匹配阈值。  
模型预测分数大于等于这个值时，会更倾向认为两个账号属于同一人。

### `--testSize`

训练集/测试集划分比例。  
例如 `0.25` 表示 25% 用作测试集。

### `--seed`

随机种子，控制采样和划分的可复现性。

---

## 14. 如果你要换成真实 TWIN 数据

你需要先把原始数据整理为项目要求的标准格式：

### `users.csv`

```csv
entityId,platformId,userId,username
u001,twitter,tw001,alice_tw
u001,instagram,ig001,alice_ig
```

### `posts.csv`

```csv
platformId,userId,postId,text,timestamp,imagePath,imageEmbedding
twitter,tw001,p1,"love coffee and travel","2017-01-01 09:00:00",,
instagram,ig001,p2,"coffee time in paris","2017-01-01 10:00:00",,
```

整理完成后，直接执行：

```powershell
python twin_align_baseline.py train --dataDir ./data/twin_std --outDir ./runs/twin_align
```

---

## 15. 常见问题

### 15.1 提示找不到 `python`

说明 Python 没装好，或者没加入系统 PATH。  
先确认：

```powershell
python --version
```

### 15.2 提示缺少 `fastapi` 或 `uvicorn`

重新安装依赖：

```powershell
pip install -r requirements.txt
```

### 15.3 为什么 4090 没有高占用？

因为这版 baseline 主体是 CPU 流程，不是 GPU 深度学习训练。  
这是当前实现设计如此，不是显卡坏了，也不是环境没配对。

### 15.4 当前准确率能直接到 95% 吗？

不能保证。

当前版本目标是：

- 先把工程链路跑通
- 做到可训练、可预测、可解释、可对接 API
- 先争取 80% 左右可用基线

后面如果要冲高准确率，建议升级：

- 文本编码器
- 图像编码器
- 难负样本
- 多模态融合

---

## 16. 一句话建议

如果你现在最重要的是“赶紧把项目跑起来”，就按下面顺序：

```powershell
pip install -r requirements.txt
python twin_align_baseline.py make-demo-data --outDir ./data/twin_std
python twin_align_baseline.py train --dataDir ./data/twin_std --outDir ./runs/twin_align
python twin_align_baseline.py predict --modelPath ./runs/twin_align/twin_align_baseline.joblib --userA tw001 --userB ig001
python twin_align_baseline.py serve --modelPath ./runs/twin_align/twin_align_baseline.joblib --host 127.0.0.1 --port 8000
```

这条线跑通后，再把 demo 数据替换成真实 TWIN 数据，再去做精度优化，就是最稳的推进方式。
