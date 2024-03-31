import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor

data = pd.read_csv("/content/Dataset .csv")

X = data.drop(columns=["Aggregate rating", "Rating color", "Rating text"])
y = data["Aggregate rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ["Restaurant Name", "City", "Locality", "Cuisines", "Currency"]
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_features = ["Average Cost for two", "Votes"]
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

base_models = [
    ("Linear Regression", LinearRegression()),
    ("Ridge Regression", Ridge()),
    ("Lasso Regression", Lasso()),
    ("Elastic Net Regression", ElasticNet()),
    ("Decision Tree Regression", DecisionTreeRegressor()),
    ("Random Forest Regression", RandomForestRegressor()),
    ("Gradient Boosting Regression", GradientBoostingRegressor()),
    ("AdaBoost Regression", AdaBoostRegressor()),
    ("Support Vector Regression", SVR()),
    ("XGBoost Regression", XGBRegressor()),
    ("LightGBM Regression", LGBMRegressor())
]

stacking_regressor = StackingRegressor(
    estimators=base_models,
    final_estimator=RandomForestRegressor(),
    cv=5
)

stacking_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', stacking_regressor)])
stacking_pipeline.fit(X_train, y_train)
y_pred_stacking = stacking_pipeline.predict(X_test)
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
r2_stacking = r2_score(y_test, y_pred_stacking)

print("Stacking Regressor:")
print("Mean Squared Error:", mse_stacking)
print("R-squared:", r2_stacking)
print("\n")

final_estimator = stacking_regressor.final_estimator_
if hasattr(final_estimator, 'feature_importances_'):
    feature_importances = final_estimator.feature_importances_
    feature_names = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(input_features=categorical_features)
    numeric_features.extend(feature_names)
    all_feature_names = numeric_features
    feature_importance_dict = dict(zip(all_feature_names, feature_importances))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    print("Most influential features affecting restaurant ratings:")
    for feature, importance in sorted_features:
        print(f"{feature}: {importance}")
