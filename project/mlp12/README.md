# Titanic Analytics Module

This module lives at `/analytics`. It contains:

- **`01_eda.ipynb`** (source: `01_eda.py`, jupytext percent format) — Part A: profiling, cleaning, univariate/bivariate/multivariate analysis, EDA-stage standardization check.
- **`02_modeling.ipynb`** (source: `02_modeling.py`) — Part B: stratified split, preprocessing pipeline, three classifiers, evaluation, imbalance comparison, hyperparameter tuning, regression side-task, final recommendation, saved pipeline.
- **`titanic.csv`** — the one committed offline fallback of the raw dataset, produced by `df.to_csv("titanic.csv", index=False)` immediately after the single `sns.load_dataset("titanic")` call in `01_eda.ipynb`. `02_modeling.ipynb` reads this file rather than reloading from the network.
- **`titanic_full_pipeline.joblib`** — the complete fitted pipeline (preprocessing `ColumnTransformer` + tuned `RandomForestClassifier`), saved via `joblib.dump`. Confirmed reloadable and correct on raw input in `02_modeling.ipynb`'s final cell.
- **`charts/`** — all supporting chart images (12 PNGs) referenced by both notebooks.

Both notebooks execute end-to-end with no errors (verified via `jupyter nbconvert --execute`).

---

## Part A — Data story

### Missing-value report (Task 1–2)

| Column | % Missing | Bucket | Strategy applied |
|---|---|---|---|
| `deck` | 77.22% | Far above the 30% impute ceiling | Encoded `"missing"` as its own category (`"Unknown"`) rather than dropped — deck plausibly relates to `pclass`/survival, but imputing a specific letter for ~4 in 5 rows would be unreliable |
| `age` | 19.87% | 5%–30% | Imputed with the median (28.0) — robust to `age`'s right-skew and outliers |
| `embarked` | 0.22% | < 5% | Rows dropped (2 rows) |
| `embark_town` | 0.22% | < 5% | Rows dropped (same 2 rows) |

Cleaned shape after Task 2: **(889, 15)**.

### Univariate analysis (Task 3)

- **IQR outliers:** `age` → **65** outliers (bounds [2.50, 54.50]); `fare` → **114** outliers (bounds [-26.76, 65.66]).
- **Fare:** mean = 32.10, median = 14.45, mode = 8.05. Since mean > median > mode, `fare` is clearly **right-skewed** — a small number of high-fare (first-class) passengers pull the mean well above the bulk of low-fare tickets clustered near the mode.

### Bivariate analysis (Task 4)

**Survival rate by sex:**

| sex | survival rate |
|---|---|
| female | 0.755 |
| male | 0.191 |

**Survival rate by pclass:**

| pclass | survival rate |
|---|---|
| 1 | 0.630 |
| 2 | 0.473 |
| 3 | 0.242 |

**Survival rate by sex + pclass** (see notebook for full boolean-masked table): survival is highest for first/second-class women (well above 90%) and lowest for third-class men (well below 20%), showing sex and class compound rather than substitute for each other.

**Correlation matrix (6 numeric columns: `survived, pclass, age, sibsp, parch, fare`; `adult_male`/`alone` excluded as derived flags):**

Top pairs by |correlation|, ranked:
1. `pclass` ↔ `fare`: **r ≈ -0.55** (strongest) — better (numerically lower) class goes with higher fare, as expected since first-class tickets cost more.
2. `sibsp` ↔ `parch`: **r ≈ 0.42** (second-strongest) — both variables largely capture the same underlying thing: traveling as part of a family group vs. alone.
3. `pclass` ↔ `age`: r ≈ -0.34
4. `survived` ↔ `pclass`: r ≈ -0.34 — `survived`'s own strongest correlate is `pclass`, reinforcing class as the dominant structural factor, even though it ranks third overall by magnitude.

### Multivariate data story (Task 5) — 5 charts, each interpreted in the notebook

1. **Survival rate by sex and class (bar):** women survive far more than men in every class; the gap barely narrows in third class — sex dominates the outcome, with class as a secondary modifier.
2. **Age by survival (box):** survivors skew slightly younger with a wider spread toward children — a weak but present "children first" signal.
3. **Fare vs. age scatter, colored by survival:** survivors cluster at higher fares, reinforcing the fare/class/survival link from the correlation matrix.
4. **Survival by embark town and class (bar):** Southampton's lower overall survival is mostly explained by its heavier third-class mix, not the port itself.
5. **Pair plot (`survived, age, fare, pclass`):** confirms all of the above in one view — fare and pclass are the dominant separators, age contributes only at the margins.

**Story:** sex is the strongest survival factor, class/fare is the second and largely overlapping factor, embark town is mostly a class proxy, and age matters only mildly at the youngest ages.

### EDA-stage standardization check (Task 6)

Before: `age` mean ≈ 29.6, std ≈ 13.0; `fare` mean ≈ 32.1, std ≈ 49.7.
After z-scoring both: mean ≈ 0.000, std ≈ 1.000 for both `age_z` and `fare_z` (confirmed numerically and visually in the notebook). This check is exploratory only and does not feed into the Part B modeling pipeline, which performs its own train-only scaling.

---

## Part B — Modeling

### Stratified split (Task 7)

Because survival is imbalanced (~62% not-survived / ~38% survived, per Task 1), a stratified split was used so both train (61.7%/38.3%) and test (61.5%/38.5%) folds preserve that ratio, avoiding a fold-specific balance skew that would bias training or metric reliability.

### Preprocessing (Task 8)

A `ColumnTransformer` (median-impute + scale for `age`/`fare`; scale-only for `pclass`/`sibsp`/`parch`; most-frequent-impute + one-hot for `sex`/`embarked`) wrapped in a `Pipeline`, fit **only** on `X_train` and applied transform-only to `X_test` — enforced structurally, not by hand.

### Classifier comparison (Tasks 9–10)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| Decision Tree | 0.777 | 0.754 | 0.623 | 0.683 | 0.793 |
| Random Forest | 0.810 | 0.787 | 0.696 | 0.738 | 0.832 |

Confusion matrices, the labeled decision-tree plot, and ROC curves are all rendered in `02_modeling.ipynb` and saved under `charts/`.

### Imbalance handling comparison (Task 11)

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| (a) Baseline / no handling | 0.787 | 0.696 | 0.738 |
| (b) `class_weight='balanced'` | 0.800 | 0.696 | **0.744** |
| (c) SMOTE (train fold only) | 0.754 | 0.710 | 0.731 |

**Conclusion:** `class_weight='balanced'` gave the best F1 (0.744), improving precision over baseline while holding recall steady. SMOTE traded precision for a small recall gain with no net F1 benefit. Given Titanic's only-moderate imbalance (~62/38), reweighting the loss was enough — synthetic oversampling added variance without a proportional payoff.

### Hyperparameter tuning (Task 12)

`GridSearchCV` over `n_estimators ∈ {100,200,300}`, `max_depth ∈ {3,5,8,None}`, `max_features ∈ {sqrt, log2}` (5-fold CV, scored on F1):

- **Best params:** `{'max_depth': 3, 'max_features': 'sqrt', 'n_estimators': 300}`
- **Best CV F1:** 0.741
- **OOB score** (refit with `oob_score=True`): **0.812**

### Regression side-task — predicting `fare` (Task 13)

| MAE | RMSE | R² | Adjusted R² |
|---|---|---|---|
| 20.898 | 30.533 | 0.398 | 0.373 |

**Heteroscedasticity conclusion:** the residual plot (`charts/12_residual_plot.png`) shows residual spread widening as predicted fare increases rather than forming a uniform band — a clear heteroscedastic pattern, consistent with `fare`'s strong right-skew found in Task 3.

### Final model comparison and recommendation (Task 14)

**Classification metrics** (own scale) and **regression metrics** (own, separate scale) are kept as two distinct groups, not merged:

| Classifier | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| Decision Tree | 0.777 | 0.754 | 0.623 | 0.683 | 0.793 |
| Random Forest | 0.810 | 0.787 | 0.696 | 0.738 | 0.832 |

| Regression model | MAE | RMSE | R² | Adjusted R² |
|---|---|---|---|---|
| Linear Regression (fare) | 20.898 | 30.533 | 0.398 | 0.373 |

**Recommendation:** Deploy the **Random Forest** classifier (Accuracy 0.810, Precision 0.787, Recall 0.696, F1 0.738, AUC 0.832). It edges out Logistic Regression on accuracy/F1 and beats the Decision Tree on every metric, reflecting its ability to capture non-linear interactions (e.g. sex × pclass) that a linear model or single tree cannot. Logistic Regression's marginally higher AUC (0.844 vs. 0.832) makes it worth keeping in mind if coefficient-level interpretability is a priority, but Random Forest's better precision/recall balance plus its tuned OOB score (0.812) make it the stronger overall deployment choice.

### Saved pipeline (Task 15)

`titanic_full_pipeline.joblib` contains the complete fitted `Pipeline` (`ColumnTransformer` preprocessing + tuned `RandomForestClassifier`, fit with the Task 12 best hyperparameters). The notebook reloads it with `joblib.load` and confirms identical predictions to the original in-memory pipeline on 5 raw, unpreprocessed test rows (`Match: True`).
