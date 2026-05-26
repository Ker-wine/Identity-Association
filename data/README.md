# A/B 平台数据说明

当前 `data/` 目录已经按 Pascal Sentences 数据源组织为：

```text
data/
  platform_a/
    upload.csv
    images/
  platform_b/
    upload.csv
    images/
  pascal_class_mapping.csv
  twin_std/
    users.csv
    posts.csv
```

## 数据规则

- platform_a有 10 个用户，用户编号为 `1` 到 `10`。
- platform_b有 10 个用户，用户编号也为 `1` 到 `10`。
- 两个平台中相同 `userId` 表示同一个真实用户。
- 每个平台中每个用户只有 1 张图片。
- 每张图片对应 Pascal Sentences 官网提供的 5 句描述文本。

## 类别对应

| userId | Pascal 类别 |
| --- | --- |
| 1 | aeroplane |
| 2 | bicycle |
| 3 | bird |
| 4 | boat |
| 5 | bottle |
| 6 | bus |
| 7 | car |
| 8 | cat |
| 9 | chair |
| 10 | cow |

每一类的第 1 张图片放入 platform_a，第 2 张图片放入 platform_b。

## 重新生成标准训练数据

```powershell
python twin_align_baseline.py prepare-upload-data `
  --inputRoot ./data `
  --platformADir platform_a `
  --platformBDir platform_b `
  --platformAId a_platform `
  --platformBId b_platform `
  --outDir ./data/twin_std
```

生成后的 `data/twin_std/users.csv` 和 `data/twin_std/posts.csv` 可以直接用于训练。
