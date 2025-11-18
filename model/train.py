import pandas as pd
import joblib
import numpy as np

if __name__ == "__main__":
    
    # Load the dataset
    print("=== Loading dataset...")
    df = pd.read_csv('data/housing.csv')

    ### ------------------------------- ###

    print("=== Preprocessing...")
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42,stratify=None)

    # Bin values into discrete intervals.
    df["income_cat"] = pd.cut(df["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])  #cut the median income across bins and give labels to each bin

    from sklearn.model_selection import StratifiedShuffleSplit

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # Provides train/test indices to split data in train/test sets.
    for train_index, test_index in split.split(df, df["income_cat"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

    housing = strat_train_set.drop("median_house_value", axis=1)     #drop copy the orignal datframe into housing
    housing_labels = strat_train_set["median_house_value"].copy()    # splitting the predictor and  target variable

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)  # Since median can be calculated only on NUmerical attributes

    # Now can fit the imputer instance to the training data using the fit() method:
    imputer.fit(housing_num)

    X = imputer.transform(housing_num)         # X = plain NumPy array containing the transformed features
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)   

    housing_cat = housing[["ocean_proximity"]] 

    from sklearn.preprocessing import OrdinalEncoder
    ordinal_encoder=OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)    

    from sklearn.preprocessing import OneHotEncoder
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),      # Imputing Missing values
        ('std_scaler', StandardScaler()),                    # Feature Scaling with Standard Scaler
    ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)

    from sklearn.compose import ColumnTransformer
    num_attribs = list(housing_num)             # columns with numerical attributes
    cat_attribs = ["ocean_proximity"]           # columns with categorical attributes
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),        
        ("cat", OneHotEncoder(), cat_attribs),
    ])
        
    housing_prepared = full_pipeline.fit_transform(housing)

    # Save the pipeline
    joblib.dump(housing.columns.to_list(), "housing_columns.joblib")
    joblib.dump(num_pipeline, "num_pipeline.joblib")
    joblib.dump(imputer, "imputer.joblib")
    joblib.dump(ordinal_encoder, "ordinal_encoder.joblib")
    joblib.dump(cat_encoder, "cat_encoder.joblib")
    joblib.dump(full_pipeline, "full_pipeline.joblib")

    ### ------------------------------- ###

    print("=== Training model...")

    print("======= Dummy Regressor")
    from sklearn.dummy import DummyRegressor
    from sklearn.metrics import mean_squared_error

    dummy_reg = DummyRegressor(strategy="mean")
    dummy_reg.fit(housing_prepared, housing_labels)
    dummy_predictions = dummy_reg.predict(housing_prepared)
    dummy_mse = mean_squared_error(housing_labels, dummy_predictions)
    dummy_rmse = np.sqrt(dummy_mse)    
    print("RMSE:", dummy_rmse)
    print("R-Squared:", dummy_reg.score(housing_prepared, housing_labels))
    joblib.dump(dummy_reg, "dummy_reg_model.joblib")

    print("======= Linear Regression")
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared,housing_labels)    
    housing_predictions=lin_reg.predict(housing_prepared)                 # predicting on training data
    lin_mse = mean_squared_error(housing_labels, housing_predictions)     # calculate mean squared error
    lin_rmse = np.sqrt(lin_mse)                                           # calculate root of mse
    print("RMSE:", lin_rmse)
    print("R-Squared:", lin_reg.score(housing_prepared,housing_labels))   # Return the R-squared    
    joblib.dump(lin_reg, "linear_reg_model.joblib")

    print("======= Decision Tree Regression")    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)
    scores = cross_val_score(forest_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-scores)
    print(forest_rmse_scores)
    print("Mean:", forest_rmse_scores.mean())
    print("Standard deviation:", forest_rmse_scores.std())
    joblib.dump(forest_reg, "forest_reg_model.joblib")

    print("======= Random Forest Regression with Hyperparameter Tuning") 
    from sklearn.model_selection import GridSearchCV
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(
        forest_reg, param_grid, cv=5,
        scoring='neg_mean_squared_error',
        return_train_score=True
    )
    grid_search.fit(housing_prepared, housing_labels)

    print("best params", grid_search.best_params_)
    print("best esimator", grid_search.best_estimator_)
    
    joblib.dump(grid_search.best_estimator_, "final_model.joblib")

    # the evaluation scores are also available
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    print("======= Feature Importance for RFR w/ Hyperparameter Tuning")
    feature_importances = grid_search.best_estimator_.feature_importances_
    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    cat_encoder = full_pipeline.named_transformers_["cat"]                   # use the categorical encoder of full pipeline
    cat_one_hot_attribs = list(cat_encoder.categories_[0])                   # categorical one hot attribute
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    sorted(zip(feature_importances, attributes), reverse=True)

    ### ------------------------------- ###

    print("=== Model Evaluation")
    final_model = grid_search.best_estimator_
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("Final RMSE", final_rmse)
