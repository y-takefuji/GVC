import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.cluster import FeatureAgglomeration
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import shap
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Helper: flatten SHAP values to 1D importance
# ─────────────────────────────────────────────
def shap_to_importance(sv, n_features):
    if isinstance(sv, list):
        stacked = np.array([np.abs(s).mean(axis=0) for s in sv])
        return stacked.mean(axis=0)

    sv = np.array(sv)

    if sv.ndim == 2:
        return np.abs(sv).mean(axis=0)

    if sv.ndim == 3:
        axes_match = [i for i, s in enumerate(sv.shape) if s == n_features]

        if len(axes_match) == 0:
            raise ValueError(
                f"No axis of SHAP array {sv.shape} matches n_features={n_features}."
            )

        feature_axis = axes_match[0]

        if feature_axis == 0:
            return np.abs(sv).mean(axis=(1, 2))
        if feature_axis == 1:
            return np.abs(sv).mean(axis=0).mean(axis=1)
        if feature_axis == 2:
            return np.abs(sv).mean(axis=1).mean(axis=0)

    raise ValueError(f"Unexpected SHAP array ndim={sv.ndim}, shape={sv.shape}")


# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
df = pd.read_csv('global_supply_chain_disruption_v1.csv')
print("Shape:", df.shape)

target_col = 'Delay_Days'
print("\nTarget summary:")
print(df[target_col].describe())

# ─────────────────────────────────────────────
# 2. Encode known string columns
#    - first 7 columns
#    - Delivery_Status, Disruption_Event, Mitigation_Action_Taken
#    - exclude target if it appears in that list
# ─────────────────────────────────────────────
df_enc = df.copy()

string_cols = list(dict.fromkeys(
    df.columns[:7].tolist() +
    ['Delivery_Status', 'Disruption_Event', 'Mitigation_Action_Taken']
))

for col in string_cols:
    if col == target_col:
        continue                          # target is numeric, skip
    if col in df_enc.columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        print(f"  Encoded: {col}")

y = df_enc[target_col].values.astype(float)

X             = df_enc.drop(columns=[target_col])
feature_names = X.columns.tolist()
X_arr         = X.values.astype(float)
n_features    = len(feature_names)

print("\nFeature set size:", n_features)
print("Features:", feature_names)

# ─────────────────────────────────────────────
# 3. CV helper — returns R-squared
# ─────────────────────────────────────────────
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_r2(features, regressor='rf'):
    feat_idx = [feature_names.index(f) for f in features]
    X_sub    = X_arr[:, feat_idx]

    if regressor == 'rf':
        reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        reg = XGBRegressor(n_estimators=100, random_state=42,
                           verbosity=0, n_jobs=-1)

    scores = cross_val_score(reg, X_sub, y, cv=cv, scoring='r2')
    return round(scores.mean(), 4)


# ─────────────────────────────────────────────
# 4. Feature Selection Algorithms
# ─────────────────────────────────────────────
results = []

# ── 4a. Random Forest importance ──────────────
print("\n[1/7] Random Forest importance ...")

rf_full = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_full.fit(X_arr, y)
rf_imp  = pd.Series(rf_full.feature_importances_,
                    index=feature_names).sort_values(ascending=False)
top5_rf = rf_imp.index[:5].tolist()
r2_5_rf = cv_r2(top5_rf, 'rf')

highest_rf = rf_imp.index[0]
reduced_rf = [f for f in feature_names if f != highest_rf]
X_red_rf   = X_arr[:, [feature_names.index(f) for f in reduced_rf]]
rf_red     = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_red.fit(X_red_rf, y)
rf_red_imp = pd.Series(rf_red.feature_importances_,
                       index=reduced_rf).sort_values(ascending=False)
top4_rf    = rf_red_imp.index[:4].tolist()

results.append(['RF', r2_5_rf, top5_rf, top4_rf])

# ── 4b. RF + SHAP ─────────────────────────────
print("[2/7] RF + SHAP ...")

rf_shap_full = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_shap_full.fit(X_arr, y)
np.random.seed(42)
idx100        = np.random.choice(len(X_arr), 100, replace=False)
explainer_rf  = shap.TreeExplainer(rf_shap_full)
sv_rf         = explainer_rf.shap_values(X_arr[idx100])
print(f"  sv_rf type={type(sv_rf)}, "
      f"shape={np.array(sv_rf).shape if not isinstance(sv_rf, list) else [s.shape for s in sv_rf]}")
rfshap_imp    = pd.Series(shap_to_importance(sv_rf, n_features),
                          index=feature_names).sort_values(ascending=False)
top5_rfshap   = rfshap_imp.index[:5].tolist()
r2_5_rfshap   = cv_r2(top5_rfshap, 'rf')

highest_rfshap = rfshap_imp.index[0]
reduced_rfshap = [f for f in feature_names if f != highest_rfshap]
X_red_rfshap   = X_arr[:, [feature_names.index(f) for f in reduced_rfshap]]
rf_shap_red    = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_shap_red.fit(X_red_rfshap, y)
explainer_rf2  = shap.TreeExplainer(rf_shap_red)
np.random.seed(42)
idx100_2       = np.random.choice(len(X_red_rfshap), 100, replace=False)
sv_rf2         = explainer_rf2.shap_values(X_red_rfshap[idx100_2])
rfshap_imp2    = pd.Series(shap_to_importance(sv_rf2, len(reduced_rfshap)),
                           index=reduced_rfshap).sort_values(ascending=False)
top4_rfshap    = rfshap_imp2.index[:4].tolist()

results.append(['RF-SHAP', r2_5_rfshap, top5_rfshap, top4_rfshap])

# ── 4c. XGBoost importance ────────────────────
print("[3/7] XGBoost importance ...")

xgb_full = XGBRegressor(n_estimators=100, random_state=42,
                        verbosity=0, n_jobs=-1)
xgb_full.fit(X_arr, y)
xgb_imp  = pd.Series(xgb_full.feature_importances_,
                     index=feature_names).sort_values(ascending=False)
top5_xgb = xgb_imp.index[:5].tolist()
r2_5_xgb = cv_r2(top5_xgb, 'xgb')

highest_xgb = xgb_imp.index[0]
reduced_xgb = [f for f in feature_names if f != highest_xgb]
X_red_xgb   = X_arr[:, [feature_names.index(f) for f in reduced_xgb]]
xgb_red     = XGBRegressor(n_estimators=100, random_state=42,
                            verbosity=0, n_jobs=-1)
xgb_red.fit(X_red_xgb, y)
xgb_red_imp = pd.Series(xgb_red.feature_importances_,
                        index=reduced_xgb).sort_values(ascending=False)
top4_xgb    = xgb_red_imp.index[:4].tolist()

results.append(['XGB', r2_5_xgb, top5_xgb, top4_xgb])

# ── 4d. XGB + SHAP ────────────────────────────
print("[4/7] XGB + SHAP ...")

xgb_shap_full = XGBRegressor(n_estimators=100, random_state=42,
                              verbosity=0, n_jobs=-1)
xgb_shap_full.fit(X_arr, y)
np.random.seed(42)
idx100_x      = np.random.choice(len(X_arr), 100, replace=False)
explainer_xgb = shap.TreeExplainer(xgb_shap_full)
sv_xgb        = explainer_xgb.shap_values(X_arr[idx100_x])
print(f"  sv_xgb type={type(sv_xgb)}, "
      f"shape={np.array(sv_xgb).shape if not isinstance(sv_xgb, list) else [s.shape for s in sv_xgb]}")
xgbshap_imp   = pd.Series(shap_to_importance(sv_xgb, n_features),
                           index=feature_names).sort_values(ascending=False)
top5_xgbshap  = xgbshap_imp.index[:5].tolist()
r2_5_xgbshap  = cv_r2(top5_xgbshap, 'xgb')

highest_xgbshap = xgbshap_imp.index[0]
reduced_xgbshap = [f for f in feature_names if f != highest_xgbshap]
X_red_xgbshap   = X_arr[:, [feature_names.index(f) for f in reduced_xgbshap]]
xgb_shap_red    = XGBRegressor(n_estimators=100, random_state=42,
                                verbosity=0, n_jobs=-1)
xgb_shap_red.fit(X_red_xgbshap, y)
explainer_xgb2  = shap.TreeExplainer(xgb_shap_red)
np.random.seed(42)
idx100_x2       = np.random.choice(len(X_red_xgbshap), 100, replace=False)
sv_xgb2         = explainer_xgb2.shap_values(X_red_xgbshap[idx100_x2])
xgbshap_imp2    = pd.Series(shap_to_importance(sv_xgb2, len(reduced_xgbshap)),
                             index=reduced_xgbshap).sort_values(ascending=False)
top4_xgbshap    = xgbshap_imp2.index[:4].tolist()

results.append(['XGB-SHAP', r2_5_xgbshap, top5_xgbshap, top4_xgbshap])

# ── 4e. Feature Agglomeration ─────────────────
print("[5/7] Feature Agglomeration ...")

n_clusters = 5
fa         = FeatureAgglomeration(n_clusters=n_clusters)
fa.fit(X_arr)
labels     = fa.labels_
rf_fa_full = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_fa_full.fit(X_arr, y)
imp_series = pd.Series(rf_fa_full.feature_importances_, index=feature_names)

cluster_best = []
for c in range(n_clusters):
    cluster_feats = [feature_names[i] for i, lbl in enumerate(labels) if lbl == c]
    if cluster_feats:
        best_feat = imp_series[cluster_feats].idxmax()
        cluster_best.append((imp_series[best_feat], best_feat))

cluster_best.sort(key=lambda x: x[0], reverse=True)
top5_fa  = [f for _, f in cluster_best[:5]]
r2_5_fa  = cv_r2(top5_fa, 'rf')

# Simply drop the highest → remaining 4 keep same order
top4_fa = top5_fa[1:]

results.append(['FA', r2_5_fa, top5_fa, top4_fa])

# ── 4f. HVGS (Variance-based) ─────────────────
print("[6/7] HVGS (Variance-based) ...")

var_series = pd.Series(np.var(X_arr, axis=0),
                       index=feature_names).sort_values(ascending=False)
top5_hvgs  = var_series.index[:5].tolist()
r2_5_hvgs  = cv_r2(top5_hvgs, 'rf')

# Simply drop the highest → remaining 4 keep same order
top4_hvgs = top5_hvgs[1:]

results.append(['HVGS', r2_5_hvgs, top5_hvgs, top4_hvgs])

# ── 4g. Spearman Correlation ───────────────────
print("[7/7] Spearman correlation ...")

spear_scores = {}
for feat in feature_names:
    rho, _ = spearmanr(X[feat].values, y)
    spear_scores[feat] = abs(rho)
spear_series = pd.Series(spear_scores).sort_values(ascending=False)
top5_spear   = spear_series.index[:5].tolist()
r2_5_spear   = cv_r2(top5_spear, 'rf')

highest_spear = spear_series.index[0]
reduced_spear = [f for f in feature_names if f != highest_spear]
spear2_scores = {}
for feat in reduced_spear:
    rho2, _ = spearmanr(X[feat].values, y)
    spear2_scores[feat] = abs(rho2)
spear2_series = pd.Series(spear2_scores).sort_values(ascending=False)
top4_spear    = spear2_series.index[:4].tolist()

results.append(['Spearman', r2_5_spear, top5_spear, top4_spear])

# ─────────────────────────────────────────────
# 5. Build Summary Table & Save
# ─────────────────────────────────────────────
summary_rows = []
for method, r2, t5, t4 in results:
    summary_rows.append({
        'Method':        method,
        'CV5_R2':        f"{r2:.4f}",
        'Top5_Features': '; '.join(t5),
        'Top4_Features': '; '.join(t4)
    })

summary_df = pd.DataFrame(summary_rows,
                           columns=['Method', 'CV5_R2',
                                    'Top5_Features', 'Top4_Features'])
print("\n─── Summary Table ───")
print(summary_df.to_string(index=False))

summary_df.to_csv('result.csv', index=False)
print("\nSaved to result.csv")
