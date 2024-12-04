from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import cohen_kappa_score, make_scorer


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


qwk_scorer = make_scorer(quadratic_weighted_kappa)


param_grid = {
    'n_estimators': [100, 150, 180, 200],
    'max_depth': [5, 10, 20, 25],
    'min_samples_split': [2, 4, 5],
    'min_samples_leaf': [1, 2, 4],
}

# Khởi tạo mô hình
rf_model = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    max_features='sqrt',
)

# GridSearchCV với 3-fold cross-validation
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring=qwk_scorer,
    cv=3,
    verbose=2,
    n_jobs=-1
)

# Tìm kiếm
grid_search.fit(X, y)

print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)
best_rf_model = grid_search.best_estimator_
