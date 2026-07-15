# Metro Vancouver 犯罪数据 — 会议速查表 (Cheatsheet)

> 快照日期 **2026-07-13** · 原始数据在 `data/raw/crime/`(gitignore,本地保留)· 复现脚本在 `src/data/crime_*.py`
> 蒸馏产物:`data/deploy/subarea_safety.parquet`(由 `src/data/build_safety_deploy.py` 生成,只用 StatCan 城市级犯罪率)
> 用途一句话:**StatCan = 官方犯罪率(15 城可比);城市 dashboard = 地图点位(只有部分类别)。两者不能对账。**

---

## 0. 一眼看全(我们手上有什么)

| 层级 | 数据 | 覆盖城市 | 时间 | 可否叫 "crime rate" | 能做社区级? |
|---|---|---|---|---|---|
| **官方城市级** | StatCan 犯罪表 + CSI | **全 15 城** | 1998–2024 | ✅ 是(唯一) | ❌ 只到城市 |
| **事件级(有接口)** | VPD | Vancouver | 2003–2026 | ⚠️ 只是 counts | ✅ 坐标 |
| | Maple Ridge | Maple Ridge | 2021–2026 | ⚠️ 只是 counts | ✅ 社区名 |
| | Langley Township | Langley Twp | 2018–2026 | ❌ 仅财产 | ✅ 坐标 |
| | Coquitlam + PoCo | Coq/PoCo | 2021–2026 | ❌ 仅财产 | ✅ 坐标 |
| | Burnaby | Burnaby | **仅 2025 起** | ❌ 仅财产 | ⚠️ 坐标(匿名) |
| **PDF 补充** | Surrey / Richmond | Surrey/Richmond | 2016–2026 | 部分(PDF 里) | ❌ |

---

## 1. 城市级官方数据 — 全 15 城,这是 baseline

### ① StatCan 表 35-10-0184-01 —— 各城市官方犯罪率 ⭐核心
- **是什么**:BC 各警务辖区、**314 种罪类**、每年的**案件数 + 官方 Rate per 100,000**。1998–2024。
- **文件**:`raw/statcan/35100184_metro_van_subset.csv`(大温子集,241.6 万行,23 个辖区)+ `35100184-eng.zip`(全省存档) + `35100184_MetaData.csv`(罪类定义)
- **来源**:https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3510018401
- **批量下载**:`https://www150.statcan.gc.ca/n1/tbl/csv/35100184-eng.zip`
- **用途**:**唯一能横向比、能合法叫 "crime rate" 的源。** 15 城全覆盖,含官方人口分母。
- **样例(Burnaby 2024)**:全部犯罪 17,975 起 / rate 6,012;财产犯罪 10,623 / rate 3,553。

### ② StatCan 表 35-10-0063-01 —— Crime Severity Index (CSI)
- **是什么**:CSI(总/暴力/非暴力)+ 加权清案率,按辖区,1998–2024。**独立方法,不是从案件数算的**——按判刑严重度加权。
- **文件**:`raw/statcan/35100063_metro_van_subset.csv`(10,074 行)+ `35100063.csv`(全省)
- **来源**:https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=3510006301
- **用途**:严重度视角的跨城市比较(高 CSI = 案更严重,不一定案更多)。

### ③ BC 省府 PSSG 犯罪趋势 XLSX —— 交叉校验
- **是什么**:BC Policing & Security Branch 的分辖区/区域犯罪趋势,2014–2023。
- **文件**:`raw/bc_pssg/*.xlsx`(4 份,主用 `bc-policing-jurisdiction-crime-trends-2014-2023-appendix-i.xlsx`)
- **来源**:https://www2.gov.bc.ca/gov/content/justice/criminal-justice/policing-in-bc/publications-statistics-legislation/crime-police-resource-statistics
- **用途**:StatCan 数字的省级交叉验证。

---

## 2. 事件级数据 — 已抓到公开接口,可每周自动同步

> 全部是"案件点位",**没有人口分母**;除 VPD/Maple Ridge 外**只有财产犯罪**。用于地图展示,不是"犯罪率"。

### ④ VPD (Vancouver) —— 最完整,唯一能做 MLS 社区级 ⭐
- **规模**:**920,282 行,2003–2026**。11 类(含暴力 + 凶杀),24 个 neighbourhood,hundred-block 坐标(UTM Zone 10N)。暴力类坐标已偏移保护隐私。每周日更新。
- **文件**:`raw/vpd/crimedata_csv_all_years.csv`(+ 字段说明 PDF + 许可条款)
- **来源**:https://geodash.vpd.ca/opendata/
- **用途**:与 `community_boundary` 做 spatial join → Vancouver 社区级展示。

### ⑤ Maple Ridge (Ridge Meadows RCMP) —— 最好的 RCMP 源
- **规模**:**37,776 行,2021–2026**。含 **PERSON(人身犯罪)8,441 条** + 财产 + 其他刑事 + 毒品;有 UCR 码和 **11 个 Neighbourhood**(无坐标)。
- **文件**:`raw/maple_ridge/RCMP_Crime_Dashboard_Data.csv`
- **来源**:https://rmpolicedashboards-mapleridge.hub.arcgis.com/ (dashboard)
- **用途**:按社区名聚合(不需 GIS)。注:只覆盖 Maple Ridge,不含 Pitt Meadows。

### ⑥ Township of Langley —— 正式开放数据
- **规模**:**24,294 行,2018–2026**。9 类财产犯罪(含 arson),有坐标。
- **文件**:`raw/langley_township/Property_Crime_-_All.csv`
- **来源**:https://data-tol.opendata.arcgis.com/datasets/property-crimes
- **用途**:财产犯罪地图点位。⚠️ 只是 **Langley Township**,不是 Langley City。

### ⑦ Coquitlam + Port Coquitlam (Coquitlam RCMP) —— 两个独立图层
- **规模**:Coquitlam 6,091 + PoCo 2,888 行,均 **2021–2026**。仅 4 类财产,有真实经纬度 + 街道级地址。
- **文件**:`raw/coquitlam_poco/City_of_Coquitlam_*.csv` 和 `City_of_Port_Coquitlam_*.csv`
- **来源**:https://rcmp.ca/en/bc/coquitlam/property-crime-dashboard
- **用途**:财产犯罪地图点位。

### ⑧ Burnaby (Burnaby RCMP) —— ⚠️ 滚动窗口,必须归档
- **规模**:**3,334 行,只有 2025-01 起**(疑似滚动,旧数据会被删)。仅 5 类财产,位置匿名化。
- **文件**:`raw/burnaby/AnonymizedCrime.csv`
- **来源**:https://gis.burnaby.ca/propertycrime/
- **用途**:财产犯罪地图点位。**必须每周跑脚本归档,否则历史永久丢失。**

---

## 3. PDF 补充 — 没有接口的城市

### ⑨ Surrey (Surrey Police Service) —— 已提取成 CSV
- **是什么**:6 份官方 PDF(10 年趋势 2016–2025、季报、月报)+ **我已把关键表提取成 135 个数据点的 CSV**:三大类年度数量+rate、人口、CSI 对比。
- **文件**:`raw/surrey/reports/*.pdf` + `raw/surrey/extracted_surrey_10yr_trends_2016_2025.csv`
- **来源**:https://www.surreypolice.ca/crime-maps-stats
- **用途**:Surrey 2025 数据(比 StatCan 早半年);原开放数据集已下架。

### ⑩ Richmond (Richmond RCMP) —— 扫描件,待 OCR
- **是什么**:RCMP 给市议会的月度活动报告(已存 2025 年 1–3 月),各类月度数量 + 同比。
- **文件**:`raw/richmond/reports/*.pdf`(**扫描图片,提取需 OCR/人工**)
- **来源**:https://citycouncil.richmond.ca (Community Safety Committee)
- **用途**:月度粒度补充。低优先级——城市级已被 StatCan 覆盖。

### 其余 8 城(无机器可读事件源,城市级看 StatCan)
Delta · New Westminster · White Rock · Port Moody · West Vancouver · North Vancouver(City + District)· Langley City · Pitt Meadows

---

## 4. 会上要强调的 5 条红线 / 注意事项

1. **只有 StatCan 能叫 "crime rate"。** dashboard 都只是财产犯罪子集 + 无人口分母。
2. **Burnaby dashboard ≠ Burnaby 财产犯罪率**:dashboard 5 类 = 2025 全年 2,393 起;StatCan 财产犯罪 2024 = 10,623 起。**差 4 倍,测的不是一回事,别对账。**
3. **社区级只有两条路**:Vancouver(坐标)+ Maple Ridge(社区名)。其余第一版只做城市级。
4. **Burnaby 只有 2025 起的数据且在滚动** → 抓取脚本必须上每周 cron 归档。
5. **Surrey 口径断点**:StatCan 2024 版 Surrey 仍挂 RCMP;2025 版(预计本月底发布)要确认 RCMP→SPS 切换。

## 5. 复现 / 刷新(均从 repo 根目录运行)
- `python -m src.data.crime_fetch_arcgis` —— 重抓 Burnaby/Coquitlam/PoCo/Langley/Maple Ridge(键集分页,存元数据)
- `python -m src.data.crime_filter_statcan` —— 从全省 zip 重建大温子集
- `python -m src.data.build_safety_deploy` —— 蒸馏出 `data/deploy/subarea_safety.parquet`(城市级安全分)
- VPD/StatCan:直接重下载对应链接的 zip
- 详细字段/坑点见 `data/raw/crime/README.md`
