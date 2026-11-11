# 原始数据来源记录

- 数据集：Default of Credit Card Clients（UCI 编号 350）
- 获取方式：`ucimlrepo.fetch_ucirepo(id=350)`，接口文件 https://archive.ics.uci.edu/static/public/350/data.csv
- 获取时间（UTC）：2026-11-09 07:40:37
- 本地文件：`credit_default.csv`，30000 行 × 25 列
- SHA-256：`e7becb953a45e8d81f49da13f6982ea3d115848d1859777dccfdcc93792d248c`
- 改动：仅将接口字段名 X1–X23、Y 按 UCI 变量说明改为业务字段名，Y 改名为 target；数值未改动。
- 许可：CC BY 4.0。引用 Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
