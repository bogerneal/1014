# 1014

print hello world

## 50 Startups：多元線性回歸

依 CRISP-DM 六階段，使用研發、行政、行銷支出及州別預測 Profit。
Model B 包含五個特徵與截距：R&D Spend、Administration、Marketing Spend、
State_Florida、State_New York；California 為基準類別。

### 執行（Windows PowerShell）

在此專案資料夾執行：

```powershell
# 首次建立環境時使用
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 執行完整分析
.\.venv\Scripts\python.exe main.py
```

預設資料為 `data/50_Startups.csv`，結果寫入 `outputs/`。
可使用 `--data 路徑`、`--output 路徑`、`--seed 42` 調整設定。
資料來源：[公開 CSV 副本](https://github.com/content-anu/dataset-multiple-regression/blob/master/50_Startups.csv)。

### 分析流程

1. 商業理解：以 Profit 為連續預測目標。
2. 資料理解：驗證欄位、缺失值、數值、州別，輸出摘要。
3. 資料準備：固定州別編碼、80% 訓練／20% 測試切分。
4. 建模：比較全特徵 Baseline、LassoCV、RFECV、Sequential Forward Selection、
   ElasticNetCV、Exhaustive Feature Selection（全部 31 個非空子集合）。
   各方法均以選出的特徵重新擬合含截距的普通最小平方法線性回歸。
5. 評估：訓練集外層 5-fold CV 的每一折重新執行特徵選擇；選擇器內層使用
   5-fold CV。依外層平均 RMSE 選出方法，再評估保留測試集及 Baseline。
   輸出 MAE、RMSE、R²、VIF、OLS 摘要、影響值與圖表。
6. 新資料預測：凍結訓練集選出的特徵，在完整資料上重新擬合，預測 California
   新公司（研發 10,000、行政 120,000、行銷 150,000）的利潤。

### 實際執行結果（2026-10-14）

已成功執行完整六階段流程。資料共 50 筆，無重複資料；使用 `seed=42`，
訓練集 40 筆、保留測試集 10 筆。以下數值取自本次產生的結果檔案。

#### 訓練集交叉驗證比較

| 方法 | 完整訓練集選出特徵數 | 外層 CV 平均 RMSE | 各折 RMSE 標準差 |
|---|---:|---:|---:|
| LassoCV | 3 | 9,495.81 | 3,174.76 |
| ElasticNetCV | 3 | 9,495.81 | 3,174.76 |
| RFECV | 3 | 9,519.09 | 3,470.16 |
| SFS | 3 | 9,532.69 | 3,273.92 |
| Exhaustive | 3 | 9,532.69 | 3,273.92 |
| Baseline | 5 | 9,550.68 | 3,282.68 |

LassoCV 與 ElasticNetCV 的平均 RMSE 並列最低，本次程式選用 LassoCV。
五種特徵選擇方法在完整訓練集上都保留研發、行政及行銷支出，排除兩個州別欄位；
各驗證折選出的特徵可能不同，詳見 `cv_fold_metrics.csv`。
此處 LassoCV 用於選特徵，最後預測器為普通最小平方法線性回歸。

#### 保留測試集結果

| 模型 | MAE | RMSE | R² |
|---|---:|---:|---:|
| LassoCV 選特徵後的線性回歸 | 6,979.15 | 8,995.91 | 0.9001 |
| 全五特徵 Baseline | 6,961.48 | 9,055.96 | 0.8987 |

選特徵後模型的 RMSE 略低、R² 略高，但 MAE 略高，因此並非所有指標都改善。
方法是依訓練集 CV 選定，而非依測試集成績挑選。

#### 全資料重訓與新公司預測

固定選出的三個特徵，以全部 50 筆資料重新擬合後，原始單位下的模型約為：

```text
Profit = 50,122.192990
       + 0.805715 × R&D Spend
       - 0.026816 × Administration
       + 0.027228 × Marketing Spend
```

範例公司位於 California，研發支出 10,000、行政支出 120,000、行銷支出 150,000，
以未四捨五入的係數預測利潤為 **59,045.64**（與資料中 Profit 相同單位）。
此預測來自全資料重訓模型；上方測試指標來自僅以 40 筆資料訓練的模型。

驗證完成：完整程式執行成功、產生 15 個結果檔，包含四張 PNG 圖表。
結果檔位於本機 `outputs/`，可依上述指令重建；該目錄不納入版本控制。

### 輸出

| 檔案 | 內容 |
|---|---|
| `data_summary.csv`、`split.csv` | 資料摘要及可追蹤的切分 |
| `feature_selection_comparison.csv` | 各方法特徵與 CV 指標 |
| `cv_fold_metrics.csv` | 每折指標與選出特徵 |
| `evaluation.json`、`test_predictions.csv` | 測試集評估與逐筆預測 |
| `vif.csv`、`ols_summary.txt` | 訓練集共線性與 OLS 診斷 |
| `top10_outliers.csv` | 依 Cook's distance 排序的前 10 筆訓練觀測 |
| `feature_selection_comparison.png` | CV RMSE 比較（誤差棒為各折標準差） |
| `actual_vs_predicted.png`、`residuals.png` | 測試集預測與殘差 |
| `top10_outliers_plot.png` | 前 10 筆影響值 |
| `deployment_model.json` | 全資料重訓後的原始單位係數、截距與範例預測 |

資料只有 50 筆，預設測試集只有 10 筆，分數可能隨切分而明顯變動。
LassoCV／ElasticNetCV 的標準化在各外層訓練折內進行，其內部 alpha 搜尋共用
該訓練折的縮放參數；外層驗證資料及測試資料均不參與這些步驟。
特徵選擇後的 OLS p 值僅供探索，未做選擇後推論校正。
高影響觀測不等於錯誤資料，程式不會自動刪除。
測試指標對應訓練集模型，不是最後全資料重訓模型的獨立驗證。
最後一階段輸出可重用的模型係數，未架設 API 或網站。
