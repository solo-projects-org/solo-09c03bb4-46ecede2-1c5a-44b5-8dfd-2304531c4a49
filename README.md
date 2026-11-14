# 巡测周期筛查:跨夜通量校正 + Lomb–Scargle 候选筛选

面向天文巡测数据员的网页工具:

1. **跨夜通量校正**——先把不同夜晚的目标星通量校正到同一尺度
   (观测通量 = 固有通量 × 当夜透过率),下载按时刻排序的校正后 CSV;
2. **周期候选筛选**——把校正后的序列(或任意单星测光序列)交给原有的
   Lomb–Scargle 周期图,输出最高峰周期、功率谱、相位折叠散点与置换尾概率,
   供人工复核。

## 快速开始

```bash
docker compose up --build
# 打开 http://localhost:8000
```

页面上有三个虚构样例按钮:

- 「载入虚构样例并提交」:单星周期样例 `sample/sample_star.csv`
  (周期约 4.25 天,160 个观测点),直接走周期筛查;
- 「载入虚构样例并校正」:`sample/reference_bridge.csv` +
  `sample/target_star.csv`,参照星跨夜桥接,六夜全部可校准,
  第 7 夜(参照图里没有)列为未校准并被剔除;
- 「载入相互矛盾的样例」:`sample/reference_conflict.csv`,同一份构图,
  但 S_ANCHOR 在第 3 夜偏亮 12%、S_BRIDGE 在第 5 夜偏暗 12%,
  在默认最大相对偏差 0.05 下会列出冲突夜晚 N3/N5 与冲突星号,且不生成文件。

无 Docker 时本地运行:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 跨夜通量校正

### 输入

| 文件 | 必需列 | 约束 |
|---|---|---|
| 参照星测量 | `night_id`、`star_id`、`flux` | 通量须为正的有限数值;同一夜晚对同一参照星只能有一条测量;参照星可跨夜重复 |
| 目标星测量 | `night_id`、`time_day`、`flux` | 通量须为正的有限数值;`time_day` 为有限数值 |

另需填写锚星的 `star_id`、固有通量(正数)与最大相对偏差
(`|观测值－预测值|/预测值`,非负有限数值)。

### 模型与算法

对同一颗参照星 i 在夜晚 j 的观测,取对数后有加性模型

```
ln f_ij = ln F_i + ln T_j
```

把参照测量看成「星—夜」二部图的边,从锚星出发做连通性搜索,只有与锚星
连通的夜晚参与校正。锚星固有通量由用户给定,固定 `ln F_锚星` 以消除整体
尺度自由度,其余星固有通量与各夜透过率在该连通分量内联合求解。

正规方程稀疏、对称正定,用**预条件共轭梯度法**求解(不依赖 scipy)。为让
互相矛盾的参照测量自己显形,求解采用**按相对残差反比的迭代重加权最小二乘**:
普通最小二乘会把矛盾摊薄到整张图,重加权则压住离群测量的影响。

求得解后,逐条计算参与校正的参照测量的预测值与相对偏差;若任一条超过
最大相对偏差,响应中给出**冲突的夜晚列表、冲突星号列表以及每条冲突测量的
观测值/预测值/偏差**,不生成校正文件。

与锚星不连通的夜晚(以及参照数据里根本没有的夜晚)列为**未校准**,其目标
观测不进入结果。剩余观测仍须满足原周期入口规则(至少 10 行、`time_day`
互异、通量有限),否则同样说明原因且不生成文件。

### 输出

- 网页展示各夜透过率、各参照星固有通量、未校准夜晚及其被舍弃的行数;
- `GET /api/calibrations/latest.csv` 下载两列 CSV:`time_day,flux`,
  按 `time_day` 升序,通量已除以当夜透过率;
- 该文件可直接提交给 `/api/jobs` 或页面的「周期候选筛选」表单。

### 手工调用

```bash
curl -F "reference=@sample/reference_bridge.csv" \
     -F "target=@sample/target_star.csv" \
     -F "anchor_star_id=S_ANCHOR" -F "anchor_flux=100" -F "max_rel_dev=0.05" \
     http://localhost:8000/api/calibrations
curl -OJ http://localhost:8000/api/calibrations/latest.csv
```

## 周期筛查

### 输入要求

- CSV 含 `time_day`、`flux` 两列(允许其他列存在,只取这两列);
- 至少 10 行;两列均为有限数值(无空值/NaN/Inf);观测时刻互异;
- 周期范围须满足 `0 < 下限 < 上限`。

### 计算流程

1. 周期范围取倒数得到频率范围 `[1/上限, 1/下限]`,均匀取 **10000** 个频率点;
2. 用**带浮动均值**的 Lomb–Scargle(`astropy.timeseries.LombScargle`,
   `fit_mean=True`)计算功率谱,取最高峰周期;
3. **置换检验**:固定时刻,随机重排光通量 **200** 次,每次重算全频段
   最大功率;尾概率 = `(不低于原峰的次数 + 1) / 201`;
4. 展示最高峰周期、峰值功率、尾概率、功率谱图与按峰周期折叠的相位散点图。

## 并发与生命周期语义

- 每批任务使用 **2 个计算子进程**(各承担 100 次置换,worker 0 另算原始
  功率谱);最多 **2 批并行**,超额提交返回 **HTTP 429**;
- 任务可查询、可取消(`POST /api/jobs/{id}/cancel`);
- 无论成功、失败还是取消,都要等本批全部子进程退出、临时文件清理完毕,
  才公布结局并释放名额;另一批不受影响;
- 取消一经受理即终止子进程,**不会发布迟到结果**;重复取消幂等,不会
  重复释放名额;
- 全部任务状态只存内存(重启即清空);计算期临时文件随收尾删除;
- 校正同步执行、结果只保留最近一次成功校正;新的校正一开始就作废上一次
  结果,失败时不会再提供旧的下载文件。

## API 摘要

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 网页 |
| POST | `/api/calibrations` | 跨夜通量校正(multipart:`reference`、`target`、`anchor_star_id`、`anchor_flux`、`max_rel_dev`)→ 201 / 400 |
| GET | `/api/calibrations/latest` | 最近一次成功校正的报告(透过率、未校准夜晚) |
| GET | `/api/calibrations/latest.csv` | 最近一次成功的校正 CSV(`time_day,flux`,按时刻排序) |
| GET | `/api/samples/reference.csv` | 参照星测量样例(跨夜桥接) |
| GET | `/api/samples/reference_conflict.csv` | 相互矛盾的参照测量样例 |
| GET | `/api/samples/target.csv` | 目标星测量样例(含未校准夜) |
| POST | `/api/jobs` | 提交批处理(multipart:`file`、`period_min`、`period_max`)→ 201 / 400 / 429 |
| GET | `/api/jobs` | 批处理列表与名额占用 |
| GET | `/api/jobs/{id}` | 单批详情(含指标) |
| POST | `/api/jobs/{id}/cancel` | 取消(幂等) |
| GET | `/api/jobs/{id}/plot/power.png` | 功率谱图(成功后) |
| GET | `/api/jobs/{id}/plot/phase.png` | 相位折叠散点图(成功后) |
| GET | `/api/sample.csv` | 单星周期样例数据 |

校正失败时 `detail` 为结构化对象,`code` 取
`reference_conflict`(冲突,含 `conflict_nights`、`conflict_stars`、`conflicts`)、
`target_entry_rules`(校正后不满足周期入口,含 `uncalibrated_nights`)、
`anchor_missing`、`*_nonpositive`、`*_invalid` 等。

示例:

```bash
curl -F "file=@sample/sample_star.csv" -F "period_min=2" -F "period_max=10" \
     http://localhost:8000/api/jobs
```

## 测试

```bash
pip install pytest
python -m pytest tests -q
```

## 目录结构

```
app/main.py                       FastAPI 主程序:校验、任务队列、子进程编排、出图、校正接口
app/calibration.py                跨夜校正:连通性、联合估计、冲突定位、入口规则检查
app/worker.py                     计算子进程:Lomb–Scargle 功率谱 + 置换检验
app/static/index.html             网页(原生 JS,无外部依赖)
sample/make_sample.py             单星周期样例生成脚本(固定种子,可复现)
sample/make_calibration_sample.py 校正样例生成脚本(桥接 + 矛盾 + 目标星)
sample/sample_star.csv            单星样例(160 行,真周期 4.25 天)
sample/reference_bridge.csv       跨夜桥接的参照星测量样例
sample/reference_conflict.csv     相互矛盾的参照星测量样例
sample/target_star.csv            目标星测量样例(7 夜,第 7 夜未校准)
tests/test_calibration.py         校正算法的回归测试
Dockerfile / docker-compose.yml
```

重新生成样例:`python sample/make_sample.py && python sample/make_calibration_sample.py`
