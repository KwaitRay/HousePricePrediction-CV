# 固定训练与验证划分

[assignment.csv](assignment.csv)是当前实验共用的固定索引，共8000行：训练6399张，验证1601张。所有图片都来自原数据集 `house_dataset/train/`。外部 `test/` 的3000张不包含在本索引中。

| 字段 | 含义 |
| --- | --- |
| imageid | 图片文件名；匹配图片和本地train.csv的主键 |
| partition | train为训练，validation为验证 |
| group_id | 重复及已登记近重复图片的分组编号 |
| pixel_hash | 冻结时的像素哈希，用于内容一致性检查 |

同步这两个文件即可共享划分：`assignment.csv`和[metadata.json](metadata.json)。不需要重新随机划分；不要根据CSV行号猜测样本归属。种子编号变化也不改变这里的划分。

元信息记录样本数量、索引摘要、原完整划分摘要与原始数据表摘要。该索引不含价格标签、图片或权重；同学须自行持有相同版本的数据集。

```python
import pandas as pd

index = pd.read_csv('data_split/assignment.csv', dtype=str)
labels = pd.read_csv('/your/house_dataset/train.csv', dtype={'imageid': str})
assert set(index.imageid) == set(labels.imageid)
data = index.merge(labels, on='imageid', how='left', validate='one_to_one', sort=False)
assert data.price.notna().all()
train = data[data.partition == 'train']
validation = data[data.partition == 'validation']
assert len(train) == 6399 and len(validation) == 1601
assert not set(train.imageid) & set(validation.imageid)
assert not set(train.group_id) & set(validation.group_id)
```

这是供协作同步的无标签索引，**不能直接替换现有训练入口的完整split文件**：现有入口还要求价格列及配套元信息。当前本机入口仍通过 `local_paths.json` 读取原 `scripts/experiments/artifacts/split_v1.csv`，本次导出没有修改划分或运行配置。索引的行顺序也与原文件一致。
