# API 入参和出参样例

本目录用于保存基于当前 `data/platform_a` 和 `data/platform_b` 数据组织方式的接口样例。

## 文件说明

- `identity_align_request_platform_ab.json`：身份对齐接口入参样例。
- `identity_align_response_platform_ab.json`：身份对齐接口出参样例。

## 当前样例的数据来源

当前 A/B 平台数据来自 Pascal Sentences：

- `a_platform`：每个用户 1 张图片。
- `b_platform`：每个用户 1 张图片。
- 两个平台相同 `userId` 表示同一个真实用户。
- `userId=1` 对应 `aeroplane` 类别。
- `userId=2` 对应 `bicycle` 类别。

样例只放了用户 1 和用户 2，方便阅读。完整 10 个用户的数据在：

```text
data/platform_a/upload.csv
data/platform_b/upload.csv
data/pascal_class_mapping.csv
```

## 关于空字段

当前数据只有图片路径和 Pascal caption 文本，没有真实采集这些字段：

- `avatarFeatureVector`
- `writingStyleVector`
- `socialGraph`
- `ipRegion`
- `deviceFingerprint`
- `imageEmbedding`

所以样例里这些字段使用 `[]`、`{}` 或 `null` 占位。

后续如果加入 CLIP 图片向量提取脚本，可以把 `imageEmbedding` 或 `avatarFeatureVector` 填成真实向量，让 RTX 4090 参与图片特征编码。
