import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import cross_val_score


housing = pd.read_csv("housing.csv")

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
 labels=[1,2,3,4,5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)


housing = strat_train_set.copy()

housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

print(housing, housing_labels)

num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs =["ocean_proximity"]

num_pipeline = Pipeline([
("imputer", SimpleImputer(strategy="median")),
("scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
("onehot", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
("num", num_pipeline, num_attribs),
("cat", cat_pipeline, cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)


# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Decision Tree
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

# Random Forest
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

# PREDICTIONS
lin_preds = lin_reg.predict(housing_prepared)
tree_preds = tree_reg.predict(housing_prepared)
forest_preds = forest_reg.predict(housing_prepared)

# RMSE (VERSION-SAFE)
#=============================

lin_rmse = sqrt(mean_squared_error(housing_labels, lin_preds))
tree_rmse = sqrt(mean_squared_error(housing_labels, tree_preds))
forest_rmse = sqrt(mean_squared_error(housing_labels, forest_preds))

# OUTPUT
print("Linear Regression RMSE :", lin_rmse)
print("Decision Tree RMSE     :", tree_rmse)
print("Random Forest RMSE     :", forest_rmse)

tree_rmses =-cross_val_score(
    tree_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)
print("Decision Tree CV RMSEs:", tree_rmses)
print("\nCross-Validation Performance (Decision Tree):")
print(pd.Series(tree_rmses).describe())