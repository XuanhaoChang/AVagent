# CSV/预测结果导出飞书调研

更新日期：2026-07-23。

## 推荐载体

首选飞书多维表格：一条问题点对应一条记录，便于筛选问题类型、置信度、时间区间和样本序号。普通 CSV 仍作为权威离线产物。

飞书官方帮助中心确认多维表格支持[批量导入数据](https://www.feishu.cn/hc/zh-CN/articles/721035791948-%E5%9C%A8%E5%A4%9A%E7%BB%B4%E8%A1%A8%E6%A0%BC%E4%B8%AD%E6%89%B9%E9%87%8F%E5%AF%BC%E5%85%A5%E6%95%B0%E6%8D%AE)，并支持多种[字段类型](https://www.feishu.cn/hc/zh-CN/articles/541575577400-%E4%BD%BF%E7%94%A8%E5%A4%9A%E7%BB%B4%E8%A1%A8%E6%A0%BC%E5%AD%97%E6%AE%B5)。因此先用生成的 `feishu_import.csv` 做手工导入验证，再启用 API。

## 字段映射

| 飞书字段 | 类型建议 | 来源 |
|---|---|---|
| 导入键 | 单行文本，唯一展示 | `序号:问题序号` |
| 序号 | 单行文本 | 原 CSV |
| 问题序号 | 数字 | 扁平化序号 |
| 可定位性/置信度/问题类型 | 单选或文本 | 预测 JSON |
| 问题说明 | 多行文本 | 预测 JSON |
| 时间区间/关键帧秒/BBox | 文本 | 预测 JSON |

## API PoC

`scripts/upload_feishu_bitable.py`：

- 默认仅 dry-run，先报告待上传条数。
- 凭据只从 `FEISHU_APP_ID`、`FEISHU_APP_SECRET`、`FEISHU_APP_TOKEN`、`FEISHU_TABLE_ID` 读取。
- 使用批量新增记录接口，每批最多发送 100 条。
- 对 429 和 5xx 做有限重试。
- 用本地 checkpoint 记录完成的 `导入键`，降低重复上传风险。

飞书应用必须同时具备多维表格访问权限，并被授予目标文档的可编辑权限；飞书官方示例说明缺少应用权限或文档授权会出现 `Forbidden`：[权限说明](https://www.feishu.cn/content/137710114294)。

本地 checkpoint 不能覆盖“服务端成功、客户端写 checkpoint 前崩溃”的极端窗口。正式上线前应在飞书表内对“导入键”做查重或改用服务端可验证的 upsert 工作流。
