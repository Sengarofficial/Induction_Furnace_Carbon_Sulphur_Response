import pandas as pd
import joblib
import os 
from sklearn.impute import SimpleImputer
from src.Mlflow_Project.__init__ import logger 
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error



class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        target_column_1 = self.config.target_column['target_1']
        target_column_2 = self.config.target_column['target_2']

        X_train = train_data.drop([target_column_1, target_column_2], axis=1)
        X_test = test_data.drop([target_column_1, target_column_2], axis=1)
        y_train = train_data[[target_column_1, target_column_2]]
        y_test = test_data[[target_column_1, target_column_2]]

        y_imputer = SimpleImputer(strategy='mean')  # You can change the strategy as needed
        y_train_imputed = y_imputer.fit_transform(y_train)
        y_test_imputed = y_imputer.transform(y_test)

        param_grid = {
            'estimator__n_estimators': [50, 100, 150],
            'estimator__max_depth': [None, 5, 10, 15],
            # Other parameters...
        }

        base_regressor = RandomForestRegressor(random_state=42)

        multi_output_regressor = MultiOutputRegressor(base_regressor)

        random_search = RandomizedSearchCV(
            estimator=multi_output_regressor,
            param_distributions=param_grid,
            n_iter=10,
            scoring='neg_mean_squared_error',
            cv=5,
            random_state=42
        )

        random_search.fit(X_train, y_train_imputed)

        best_estimator = random_search.best_estimator_

        predictions = best_estimator.predict(X_test)

        mse = mean_squared_error(y_test_imputed, predictions)
        print(f"Mean Squared Error: {mse}")

        test_x = pd.DataFrame(X_test, columns=X_test.columns)
        test_y = pd.DataFrame(y_test_imputed, columns=[target_column_1, target_column_2])

        # Save the scaled datasets and target variables
        test_x.to_csv(os.path.join(self.config.root_dir, "test_x.csv"), index=False)
        test_y.to_csv(os.path.join(self.config.root_dir, "test_y.csv"), index=False)

        joblib.dump(best_estimator, os.path.join(self.config.root_dir, self.config.model_name))
        joblib.dump(y_imputer, os.path.join(self.config.root_dir, self.config.imputer_name))
