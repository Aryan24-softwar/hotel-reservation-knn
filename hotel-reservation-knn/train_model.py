# coding: utf-8
"""
Train KNN Classifier (main) + compare with other models
Major Project - Hotel Reservation System using KNN Classifier
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection     import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing       import LabelEncoder, StandardScaler
from sklearn.neighbors           import KNeighborsClassifier
from sklearn.linear_model        import LogisticRegression
from sklearn.ensemble            import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree                import DecisionTreeClassifier
from sklearn.svm                 import SVC
from sklearn.naive_bayes         import GaussianNB
from sklearn.metrics             import (accuracy_score, f1_score, roc_auc_score,
                                         confusion_matrix, classification_report,
                                         roc_curve, precision_recall_curve)
from sklearn.utils               import resample

warnings.filterwarnings("ignore")

BASE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(BASE, "data",   "hotel_reservations.csv")
MODEL = os.path.join(BASE, "models")
PLOTS = os.path.join(BASE, "plots")
os.makedirs(MODEL, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

print("=" * 68)
print("   MAJOR PROJECT - HOTEL RESERVATION SYSTEM USING KNN CLASSIFIER")
print("=" * 68)

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
print(f"\nDataset Shape  : {df.shape}")
print(f"Cancellation   : {df['is_canceled'].value_counts().to_dict()}")
print(f"Cancellation % : {df['is_canceled'].mean()*100:.1f}%")

# ── Encode Categoricals ────────────────────────────────────────────────────────
cat_cols = ["hotel","arrival_date_month","meal","market_segment",
            "distribution_channel","reserved_room_type","assigned_room_type",
            "deposit_type","customer_type"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col+"_enc"] = le.fit_transform(df[col])
    encoders[col] = le

feature_cols = [
    "lead_time","stays_in_weekend_nights","stays_in_week_nights","total_nights",
    "adults","children","babies","is_repeated_guest","previous_cancellations",
    "previous_bookings_not_canceled","room_type_match","booking_changes",
    "days_in_waiting_list","adr","required_car_parking_spaces",
    "total_of_special_requests","arrival_date_week_number","arrival_date_day_of_month",
] + [c+"_enc" for c in cat_cols]

X = df[feature_cols]
y = df["is_canceled"]

# ── Balance ────────────────────────────────────────────────────────────────────
df_tmp = X.copy(); df_tmp["target"] = y.values
df_maj = df_tmp[df_tmp["target"]==0]
df_min = df_tmp[df_tmp["target"]==1]
if len(df_min) < len(df_maj):
    df_min = resample(df_min, replace=True, n_samples=len(df_maj), random_state=42)
df_bal = pd.concat([df_maj, df_min])
X_bal  = df_bal[feature_cols]; y_bal = df_bal["target"]

scaler   = StandardScaler()
X_sc     = scaler.fit_transform(X_bal)
X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y_bal, test_size=0.2,
                                           random_state=42, stratify=y_bal)

# ============================================================
# PART A: KNN HYPERPARAMETER TUNING (K selection)
# ============================================================
print("\n" + "="*68)
print("PART A: KNN - Finding Best K Value")
print("="*68)

k_range = range(1, 31)
k_scores_acc = []
k_scores_f1  = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    acc = cross_val_score(knn, X_tr, y_tr, cv=5, scoring="accuracy").mean()
    f1  = cross_val_score(knn, X_tr, y_tr, cv=5, scoring="f1").mean()
    k_scores_acc.append(acc)
    k_scores_f1.append(f1)
    print(f"  K={k:2d}  Accuracy={acc*100:.2f}%  F1={f1:.4f}")

best_k = k_range[np.argmax(k_scores_f1)]
print(f"\nBest K = {best_k}  (F1={max(k_scores_f1):.4f})")

# ── Plot K vs Accuracy/F1 ──────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(k_range, [s*100 for s in k_scores_acc], "b-o", markersize=4, label="Accuracy (%)")
ax.plot(k_range, [s*100 for s in k_scores_f1],  "r-o", markersize=4, label="F1 Score x100")
ax.axvline(best_k, color="green", linestyle="--", lw=2, label=f"Best K={best_k}")
ax.set_xlabel("K (Number of Neighbors)", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("KNN: K Value vs Accuracy & F1 Score (5-Fold CV)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11); ax.set_xticks(list(k_range))
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"01_knn_k_selection.png"), dpi=150)
plt.close()

# ============================================================
# PART B: TRAIN FINAL KNN MODEL
# ============================================================
print("\n" + "="*68)
print(f"PART B: Training Final KNN (K={best_k})")
print("="*68)

knn_final = KNeighborsClassifier(n_neighbors=best_k, metric="minkowski", p=2, weights="uniform")
knn_final.fit(X_tr, y_tr)
knn_preds = knn_final.predict(X_te)
knn_probs = knn_final.predict_proba(X_te)[:,1]
knn_acc   = accuracy_score(y_te, knn_preds)
knn_f1    = f1_score(y_te, knn_preds)
knn_auc   = roc_auc_score(y_te, knn_probs)

print(f"\nKNN Final Results:")
print(f"  Accuracy : {knn_acc*100:.2f}%")
print(f"  F1 Score : {knn_f1:.4f}")
print(f"  AUC-ROC  : {knn_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_te, knn_preds, target_names=["Not Cancelled","Cancelled"]))

# Cross-validation for KNN
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(knn_final, X_sc, y_bal, cv=cv, scoring="accuracy")
print(f"\n10-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# ============================================================
# PART C: COMPARE WITH OTHER CLASSIFIERS
# ============================================================
print("\n" + "="*68)
print("PART C: Model Comparison (KNN vs Others)")
print("="*68)

models = {
    f"KNN (K={best_k})":        knn_final,
    "Logistic Regression":       LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree":             DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest":             RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":         GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM":                       SVC(probability=True, kernel="rbf", random_state=42),
    "Naive Bayes":               GaussianNB(),
}

results = []
all_models_trained = {}

for name, mdl in models.items():
    if name != f"KNN (K={best_k})":
        mdl.fit(X_tr, y_tr)
    preds = mdl.predict(X_te)
    probs = mdl.predict_proba(X_te)[:,1]
    acc   = accuracy_score(y_te, preds)
    f1    = f1_score(y_te, preds)
    auc   = roc_auc_score(y_te, probs)
    results.append({"Model": name, "Accuracy(%)": round(acc*100,2),
                    "F1 Score": round(f1,4), "AUC-ROC": round(auc,4)})
    all_models_trained[name] = mdl
    print(f"\n{name}\n  Accuracy={acc*100:.2f}%  F1={f1:.4f}  AUC={auc:.4f}")

res_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
print("\n\nModel Ranking (by F1 Score):")
print(res_df.to_string(index=False))

# ── Save Models ────────────────────────────────────────────────────────────────
pickle.dump(knn_final,   open(os.path.join(MODEL,"knn_model.pkl"),    "wb"))
pickle.dump(scaler,      open(os.path.join(MODEL,"scaler.pkl"),       "wb"))
pickle.dump(encoders,    open(os.path.join(MODEL,"encoders.pkl"),     "wb"))
pickle.dump(feature_cols,open(os.path.join(MODEL,"features.pkl"),     "wb"))
pickle.dump(best_k,      open(os.path.join(MODEL,"best_k.pkl"),       "wb"))
pickle.dump(res_df,      open(os.path.join(MODEL,"results_df.pkl"),   "wb"))
print("\nAll models/artifacts saved.")

# ============================================================
# PLOTS
# ============================================================

# 2. Cancellation Distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
counts = df["is_canceled"].value_counts()
axes[0].pie([counts[0], counts[1]], labels=["Not Cancelled","Cancelled"],
            colors=["#42A5F5","#EF5350"], autopct="%1.1f%%",
            startangle=90, textprops={"fontsize":12})
axes[0].set_title("Reservation Cancellation Distribution", fontsize=13, fontweight="bold")
axes[1].bar(["Not Cancelled","Cancelled"],[counts[0],counts[1]],
            color=["#42A5F5","#EF5350"], width=0.5, edgecolor="white")
for i,(v) in enumerate([counts[0],counts[1]]):
    axes[1].text(i, v+30, str(v), ha="center", fontweight="bold", fontsize=11)
axes[1].set_title("Cancellation Count", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Number of Reservations")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"02_cancellation_distribution.png"), dpi=150)
plt.close()

# 3. Hotel Type vs Cancellation
fig, ax = plt.subplots(figsize=(9, 5))
hotel_cancel = df.groupby("hotel")["is_canceled"].mean().sort_values()*100
hotel_cancel.plot(kind="bar", color=["#42A5F5","#EF5350"], ax=ax, edgecolor="white")
ax.set_title("Cancellation Rate by Hotel Type", fontsize=14, fontweight="bold")
ax.set_ylabel("Cancellation Rate (%)"); ax.set_xlabel("")
plt.xticks(rotation=0)
for i,v in enumerate(hotel_cancel):
    ax.text(i, v+0.5, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"03_hotel_type_cancellation.png"), dpi=150)
plt.close()

# 4. Lead Time Distribution
fig, ax = plt.subplots(figsize=(10, 5))
for label, color, lbl in [(0,"#42A5F5","Not Cancelled"),(1,"#EF5350","Cancelled")]:
    sns.kdeplot(df[df["is_canceled"]==label]["lead_time"], ax=ax,
                color=color, fill=True, alpha=0.4, label=lbl)
ax.set_title("Lead Time Distribution by Cancellation", fontsize=14, fontweight="bold")
ax.set_xlabel("Lead Time (days)"); ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"04_lead_time_distribution.png"), dpi=150)
plt.close()

# 5. Deposit Type vs Cancellation
fig, ax = plt.subplots(figsize=(9, 5))
dep_cancel = df.groupby("deposit_type")["is_canceled"].mean().sort_values(ascending=False)*100
dep_cancel.plot(kind="bar", color=["#EF5350","#FFA726","#42A5F5"], ax=ax, edgecolor="white")
ax.set_title("Cancellation Rate by Deposit Type", fontsize=14, fontweight="bold")
ax.set_ylabel("Cancellation Rate (%)"); ax.set_xlabel("")
plt.xticks(rotation=0)
for i,v in enumerate(dep_cancel):
    ax.text(i, v+0.5, f"{v:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"05_deposit_cancellation.png"), dpi=150)
plt.close()

# 6. Market Segment vs Cancellation
fig, ax = plt.subplots(figsize=(11, 5))
seg_cancel = df.groupby("market_segment")["is_canceled"].mean().sort_values()*100
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(seg_cancel)))
seg_cancel.plot(kind="barh", color=colors, ax=ax, edgecolor="white")
ax.set_title("Cancellation Rate by Market Segment", fontsize=14, fontweight="bold")
ax.set_xlabel("Cancellation Rate (%)")
for i,v in enumerate(seg_cancel):
    ax.text(v+0.3, i, f"{v:.1f}%", va="center", fontweight="bold", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"06_market_segment_cancellation.png"), dpi=150)
plt.close()

# 7. ADR Boxplot
fig, ax = plt.subplots(figsize=(9, 5))
df["Cancellation"] = df["is_canceled"].map({0:"Not Cancelled",1:"Cancelled"})
sns.boxplot(data=df, x="Cancellation", y="adr",
            palette={"Not Cancelled":"#42A5F5","Cancelled":"#EF5350"}, ax=ax)
ax.set_title("Average Daily Rate (ADR) vs Cancellation", fontsize=14, fontweight="bold")
ax.set_xlabel(""); ax.set_ylabel("ADR (Price per Night)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"07_adr_vs_cancellation.png"), dpi=150)
plt.close()

# 8. Special Requests vs Cancellation Rate
fig, ax = plt.subplots(figsize=(9, 5))
sr_cancel = df.groupby("total_of_special_requests")["is_canceled"].mean()*100
ax.bar(sr_cancel.index, sr_cancel.values, color="#5C6BC0", edgecolor="white")
ax.set_title("Cancellation Rate by Number of Special Requests", fontsize=14, fontweight="bold")
ax.set_xlabel("Total Special Requests"); ax.set_ylabel("Cancellation Rate (%)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"08_special_requests_cancellation.png"), dpi=150)
plt.close()

# 9. Monthly Booking Pattern
fig, ax = plt.subplots(figsize=(12, 5))
month_order = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]
monthly = df.groupby("arrival_date_month")["is_canceled"].mean().reindex(month_order)*100
ax.plot(month_order, monthly.values, "ro-", markersize=8, linewidth=2)
ax.fill_between(range(12), monthly.values, alpha=0.2, color="red")
ax.set_xticks(range(12)); ax.set_xticklabels(month_order, rotation=30, ha="right")
ax.set_title("Monthly Cancellation Rate Pattern", fontsize=14, fontweight="bold")
ax.set_ylabel("Cancellation Rate (%)"); ax.set_ylim(0, 80)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"09_monthly_cancellation.png"), dpi=150)
plt.close()

# 10. Correlation Heatmap
fig, ax = plt.subplots(figsize=(13, 9))
num_cols = ["lead_time","stays_in_weekend_nights","stays_in_week_nights",
            "adults","adr","previous_cancellations","days_in_waiting_list",
            "booking_changes","total_of_special_requests","is_canceled","total_nights"]
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            ax=ax, annot_kws={"size":8})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"10_correlation_heatmap.png"), dpi=150)
plt.close()

# 11. Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(13, 6))
x  = np.arange(len(res_df))
w  = 0.25
ax.bar(x-w,   res_df["Accuracy(%)"],   w, label="Accuracy (%)",   color="#42A5F5")
ax.bar(x,     res_df["F1 Score"]*100,  w, label="F1 Score x100",  color="#66BB6A")
ax.bar(x+w,   res_df["AUC-ROC"]*100,  w, label="AUC-ROC x100",   color="#FFA726")
ax.set_xticks(x); ax.set_xticklabels(res_df["Model"], rotation=20, ha="right", fontsize=9)
ax.set_ylim(0, 115); ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"11_model_comparison.png"), dpi=150)
plt.close()

# 12. KNN Confusion Matrix
cm = confusion_matrix(y_te, knn_preds)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Cancelled","Cancelled"],
            yticklabels=["Not Cancelled","Cancelled"], ax=ax)
ax.set_xlabel("Predicted",fontsize=12); ax.set_ylabel("Actual",fontsize=12)
ax.set_title(f"KNN (K={best_k}) - Confusion Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"12_knn_confusion_matrix.png"), dpi=150)
plt.close()

# 13. ROC Curve (all models)
fig, ax = plt.subplots(figsize=(8, 7))
colors_roc = ["#E53935","#1E88E5","#43A047","#8E24AA","#FB8C00","#00ACC1","#6D4C41"]
for (name,mdl),col in zip(all_models_trained.items(), colors_roc):
    probs_roc = mdl.predict_proba(X_te)[:,1]
    fpr,tpr,_ = roc_curve(y_te, probs_roc)
    auc_val   = roc_auc_score(y_te, probs_roc)
    ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={auc_val:.3f})")
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set_xlabel("False Positive Rate",fontsize=12)
ax.set_ylabel("True Positive Rate",fontsize=12)
ax.set_title("ROC Curves - All Models", fontsize=14, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"13_roc_curves.png"), dpi=150)
plt.close()

# 14. KNN 10-fold CV Scores
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(1,11), cv_scores*100, color="#5C6BC0", edgecolor="white")
ax.axhline(cv_scores.mean()*100, color="red", linestyle="--", lw=2,
           label=f"Mean = {cv_scores.mean()*100:.2f}%")
ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy (%)")
ax.set_title("KNN - 10-Fold Cross-Validation Accuracy", fontsize=14, fontweight="bold")
ax.set_xticks(range(1,11)); ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"14_knn_cross_validation.png"), dpi=150)
plt.close()

# 15. Customer Type vs Cancellation
fig, ax = plt.subplots(figsize=(9, 5))
ct_cancel = df.groupby("customer_type")["is_canceled"].mean().sort_values()*100
ct_cancel.plot(kind="bar", color=["#42A5F5","#66BB6A","#FFA726","#EF5350"],
               ax=ax, edgecolor="white")
ax.set_title("Cancellation Rate by Customer Type", fontsize=14, fontweight="bold")
ax.set_ylabel("Cancellation Rate (%)"); ax.set_xlabel("")
plt.xticks(rotation=15, ha="right")
for i,v in enumerate(ct_cancel):
    ax.text(i, v+0.5, f"{v:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS,"15_customer_type_cancellation.png"), dpi=150)
plt.close()

print("\nAll 15 plots saved to /plots/")
print("\nMAJOR PROJECT Training Complete!")
print(f"\nFinal KNN Model -> K={best_k}, Accuracy={knn_acc*100:.2f}%, F1={knn_f1:.4f}, AUC={knn_auc:.4f}")