import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

path = 'transaction_dataset.csv'
df = pd.read_csv(path)

# Normalize target column handling
if 'FLAG' not in df.columns:
    raise ValueError(f"FLAG column not found. Columns: {list(df.columns)[:20]}")

y = df['FLAG']
X = df.drop(columns=['FLAG'])

# (1) Class distribution and majority baseline
class_counts = y.value_counts(dropna=False).sort_index()
class_props = (class_counts / len(y)).sort_index()
majority_acc = class_counts.max() / len(y)

print('=== (1) Class distribution ===')
for cls, cnt in class_counts.items():
    print(f'class={cls} count={cnt} pct={cnt/len(y):.6f}')
print(f'majority_baseline_accuracy={majority_acc:.6f}')

# (2) Numeric features almost identical to FLAG
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print('\n=== (2) Numeric feature similarity to FLAG ===')
if len(num_cols) == 0:
    print('No numeric feature columns found.')
else:
    y_num = pd.to_numeric(y, errors='coerce')
    rows = []
    for c in num_cols:
        s = pd.to_numeric(X[c], errors='coerce')
        valid = s.notna() & y_num.notna()
        corr = np.nan
        if valid.sum() > 1 and s[valid].nunique() > 1 and y_num[valid].nunique() > 1:
            corr = np.corrcoef(s[valid], y_num[valid])[0,1]
        eq = ((s == y_num) & valid).sum() / len(df)
        rows.append((c, corr, eq))
    rows_sorted_corr = sorted(rows, key=lambda t: abs(-999 if pd.isna(t[1]) else t[1]), reverse=True)
    rows_sorted_eq = sorted(rows, key=lambda t: t[2], reverse=True)
    print('Top 10 by |correlation with FLAG|:')
    for c, corr, eq in rows_sorted_corr[:10]:
        print(f'{c}: corr={corr if not pd.isna(corr) else "nan"}, equality_rate={eq:.6f}')
    print('Top 10 by exact equality rate with FLAG:')
    for c, corr, eq in rows_sorted_eq[:10]:
        print(f'{c}: corr={corr if not pd.isna(corr) else "nan"}, equality_rate={eq:.6f}')

# (3) LogisticRegression on 5 random splits (numeric-only with imputation+scaling)
print('\n=== (3) LogisticRegression 5x random 80/20 splits ===')
if len(num_cols) == 0:
    print('No numeric columns for LR.')
    accs = []
else:
    Xn = X[num_cols]
    accs = []
    for seed in [0,1,2,3,4]:
        X_train, X_test, y_train, y_test = train_test_split(
            Xn, y, test_size=0.2, random_state=seed, stratify=y if y.nunique() > 1 else None
        )
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=2000, n_jobs=None))
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        accs.append(acc)
        print(f'seed={seed} acc={acc:.6f}')
    print(f'mean_acc={np.mean(accs):.6f} std_acc={np.std(accs, ddof=1):.6f}')

# (4) Leakage check: exact duplicate row overlap train/test excluding FLAG
print('\n=== (4) Train/test exact row overlap (excluding FLAG) ===')
# Convert all features to canonical strings for stable row signatures
X_canon = X.copy()
for c in X_canon.columns:
    X_canon[c] = X_canon[c].astype(str)
row_sig = pd.util.hash_pandas_object(X_canon, index=False)
for seed in [0,1,2,3,4]:
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=seed, stratify=y if y.nunique() > 1 else None
    )
    train_set = set(row_sig.iloc[idx_train].tolist())
    test_hashes = row_sig.iloc[idx_test].tolist()
    overlap = sum(1 for h in test_hashes if h in train_set)
    print(f'seed={seed} exact_match_count={overlap} test_size={len(idx_test)} overlap_rate={overlap/len(idx_test):.6f}')
