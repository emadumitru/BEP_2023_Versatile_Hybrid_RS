import ast
import copy
import random
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Union
from data_creation import generate_topic_data
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC, SVR

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


# Define paths
PATH_DATA = 'data/data_final/'
PATH_RESULTS = 'data/results/'


class IndividualPredictionsPipeline:
    def __init__(self, path_data: str = PATH_DATA, path_res: str = PATH_RESULTS, load_data: int = 1,
                 load_split: int = 1, split_ratio: float = 0.3, gridsearch: int = 0, data_info: tuple = ()) -> None:
        """
        Initiates an instance of IndividualPredictionsPipeline with given parameters.

        :param path_data: Path from where to load data.
        :param path_data: Path for where to save data.
        :param load_data: If to load the characteristics and mapping. Default is 1.
        :param load_split: If to load the split dataset from files. Default is 1.
        :param split_ratio: The test size ratio when splitting the data. Default is 0.3.
        :param gridsearch: If to perform grid search. Default is 0 as gridsearch takes a while.
        :param data_info: Tuple that can contain the data. If list does not have length 4, then we initialize the data.
        """
        if not load_data:
            load_split = 0
        self.path_data = path_data
        self.path_res = path_res
        self.load_data = load_data
        self.load_split = load_split
        self.split_ratio = split_ratio
        self.gridsearch = gridsearch
        self.X, self.Y, self.Y_pred, self.df_results, self.data_char, self.tasks, self.tasks_pt = self.initiate_data()
        if len(data_info) == 4:
            self.data = data_info
        else:
            self.data = self.get_data_split()

    # Data initialization functions
    def initiate_data(self, re_initiate: int = 0) -> Tuple[
                      pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list, list, list]:
        """
        Initiates data by reading the CSVs' characteristics and mapping.

        :param re_initiate: Binary value that tells is the variables need to be initiated again with new X and Y
        :return: A tuple containing X, Y, Y_pred, df_results, data_char, tasks, tasks_pt.
        """
        if re_initiate:
            x = pd.concat([self.data[0], self.data[1]], ignore_index=True)
            y_long = pd.concat([self.data[2], self.data[3]], ignore_index=True)
        else:
            if self.load_data:
                x = pd.read_csv(self.path_data + 'full_characteristics.csv')
                y_long = pd.read_csv(self.path_data + 'full_mapping.csv')
            else:
                x, y_long = generate_topic_data(load_char=False, load_csv25=False, load_mapping=False, target_rows=1000)
        data_char = list(x.columns)
        tasks = list(y_long.columns)
        tasks_pt = ['pt' + str(i + 1) for i in range(len(tasks))]
        tasks_mapping = dict(zip(tasks, tasks_pt))
        y = y_long.rename(columns=tasks_mapping)
        y_pred = pd.DataFrame(columns=tasks_pt)
        df_results = pd.DataFrame(columns=['name'] + ['model-' + task for task in tasks_pt])
        return x, y, y_pred, df_results, data_char, tasks, tasks_pt

    def get_data_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Gets split data. If self.load_split is 1, loads data from files, otherwise does train-test split.

        :return: A tuple containing X_train, X_test, Y_train, Y_test that will represent self.data
        """
        if self.load_split:
            x_train = pd.read_csv(self.path_data + 'X_train.csv').set_index('Unnamed: 0')
            x_test = pd.read_csv(self.path_data + 'X_test.csv').set_index('Unnamed: 0')
            y_train = pd.read_csv(self.path_data + 'Y_train.csv').set_index('Unnamed: 0')
            y_test = pd.read_csv(self.path_data + 'Y_test.csv').set_index('Unnamed: 0')
        else:
            x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=self.split_ratio)
        return x_train, x_test, y_train, y_test

    # Threshold finiding functions
    def threshold_series_find(self, true: pd.Series, pred: pd.Series) -> float:
        """
        Finds optimal threshold for maximizing F1 score.

        :param true: Series of true labels.
        :param pred: Series of predicted labels.
        :return: optimal threshold.
        """
        thresholds = list(np.arange(0.05, 0.96, 0.01))
        f1_scores = [f1_score(true, (pred >= threshold).astype(int)) for threshold in thresholds]
        optimal_index = np.argmax(f1_scores)
        return thresholds[optimal_index]

    def threshold_find_and_transform(self, test_pred: pd.DataFrame, train_pred: pd.DataFrame,
                                     train_true: pd.DataFrame) -> Tuple[pd.DataFrame, List[float]]:
        """
        Finds optimal thresholds and transform predictions accordingly.

        :param test_pred: DataFrame of test set predictions.
        :param train_pred: DataFrame of training set predictions.
        :param train_true: DataFrame of training set true labels.
        :return: Tuple containing binarized predictions and thresholds.
        """
        y_bin = test_pred.copy()
        thresholds = []
        for task in train_true.columns:
            threshold_task = self.threshold_series_find(train_true[task], train_pred[task])
            y_bin[task] = pd.Series(np.where(test_pred[task] >= threshold_task, 1, 0))
            thresholds.append(threshold_task)
        return y_bin, thresholds

    # Gridserach for best parameters function
    def grid_search(self, parameters: Dict, model: Union[ClassifierMixin, RegressorMixin],
                    exc: Dict) -> Tuple[Dict, Dict]:
        """
        Performs grid search for optimal model parameters.

        :param parameters: Dictionary of parameters for grid search.
        :param model: Scikit-learn model to be optimized.
        :param exc: Default parameters in case of exception.
        :return: Tuple containing dictionaries of the best parameters and best models.
        """
        data = self.data
        gridsearch = GridSearchCV(model, parameters, )
        results = {}
        results_model = {}
        for task in data[2].columns:
            try:
                gridsearch.fit(data[0], data[2][task])
                results[task] = gridsearch.best_params_
                results_model[task] = gridsearch.best_estimator_
            except:
                results[task] = exc
                results_model[task] = model.set_params(**exc)
        return results, results_model

    # Evaluation functions
    def base_evaluation(self, true: pd.DataFrame, pred: pd.DataFrame, name: str) -> Dict[str, float]:
        """
        Perform base evaluation metrics between true and predicted values.

        :param true: DataFrame containing true values.
        :param pred: DataFrame containing predicted values.
        :param name: Name to prefix the evaluation results could be prediction task name or overall.
        :return: Dictionary with evaluation results.
        """
        results = {name + '_accuracy': accuracy_score(true.values.flatten(), pred.values.flatten()),
                   name + '_balanced_accuracy': balanced_accuracy_score(true.values.flatten(), pred.values.flatten()),
                   name + '_f1_score': f1_score(true.values.flatten(), pred.values.flatten()),
                   name + '_precision': precision_score(true.values.flatten(), pred.values.flatten()),
                   name + '_recall': recall_score(true.values.flatten(), pred.values.flatten())}

        return results

    def evaluate_prediction(self, true: pd.DataFrame, pred: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate predictions against true values.

        :param true: DataFrame containing true values.
        :param pred: DataFrame containing predicted values.
        :return: Dictionary with evaluation results.
        """
        results = {}

        for col in true.columns:
            results.update(self.base_evaluation(true[[col]], pred[[col]], str(col)))

        if len(true.values.flatten()) == len(pred.values.flatten()):
            results.update(self.base_evaluation(true, pred, 'overall'))

        return results

    # Content based filtering function
    def run_cbf(self, df_results: pd.DataFrame, thresholds: pd.DataFrame) -> tuple:
        """
        Perform content-based filtering.

        :param df_results: DataFrame to store results.
        :param thresholds: DataFrame to store thresholds.
        :return: Tuple containing updated DataFrame, thresholds, and predicted values.
        """
        data = self.data
        csv_data = pd.concat([data[0], data[1]])
        map_data = pd.concat([data[2], data[3]])
        index_list = data[1].index.tolist()
        column_list = data[0].index.tolist()
        y_pred = pd.DataFrame()
        results = {'name': 'CBF'}
        thres = {'name': 'CBF'}

        pearson_corr = csv_data.T.corr()
        pearson_corr = pearson_corr.loc[index_list, column_list]

        max_pea = pearson_corr.idxmax(axis=1).tolist()

        for i in range(len(max_pea)):
            idx_list = [max_pea[i]]
            values_subset = map_data.loc[idx_list].mean().to_dict()
            y_pred = y_pred.append(values_subset, ignore_index=True)

        if ((y_pred != 0) & (y_pred != 1)).any().any():
            for task in data[2].columns:
                threshold_task = data[2][task].mean()
                thres[task] = threshold_task
                y_pred[task] = pd.Series(np.where(y_pred[task] >= threshold_task, 1, 0))
            thresholds = thresholds.append(thres, ignore_index=True)

        res = self.evaluate_prediction(data[3], y_pred)
        results.update(res)
        df_results = df_results.append(results, ignore_index=True)

        return df_results, thresholds, y_pred

    # Collaborative filtering function
    def run_cf(self, df_results: pd.DataFrame, thresholds: pd.DataFrame, arch: bool) -> tuple:
        """
        Perform collaborative filtering.

        :param df_results: DataFrame to store results.
        :param thresholds: DataFrame to store thresholds.
        :param arch: Flag indicating whether it is part of the architecture or not such that values are not dropped.
        :return: Tuple containing updated DataFrame, thresholds, and predicted values.
        """
        data = self.data
        y_drop = data[3].copy()
        y_true = data[3].copy().reset_index(drop=True)
        if not arch:
            custom_lambda = lambda x: x.apply(lambda val: 0.5 if random.random() < 0.5 else val)
            y_drop = y_drop.apply(custom_lambda, axis=1)
            y_drop = y_drop.rename(columns={i: data[2].columns[i] for i in range(len(data[2].columns))})
        map_data = pd.concat([data[2], y_drop])
        index_list = data[1].index.tolist()
        column_list = data[0].index.tolist()
        y_pred = pd.DataFrame()

        results = {'name': 'CF'}
        thres = {'name': 'CF'}

        similarity = cosine_similarity(map_data)
        similarity = pd.DataFrame(similarity, index=map_data.index, columns=map_data.index)
        similarity = similarity.loc[index_list, column_list]
        max_sim = similarity.idxmax(axis=1).tolist()

        for i in range(len(max_sim)):
            idx_list = [max_sim[i]]
            values_subset = map_data.loc[idx_list].mean().to_dict()
            y_pred = y_pred.append(values_subset, ignore_index=True)

        y_drop = y_drop.reset_index(drop=True)
        mask = y_drop == 0.5
        y_drop[mask] = y_pred[mask]

        if ((y_drop != 0) & (y_drop != 1)).any().any():
            for task in data[2].columns:
                threshold_task = data[2][task].mean()
                thres[task] = threshold_task
                y_drop[task] = pd.Series(np.where(y_drop[task] >= threshold_task, 1, 0))
            thresholds = thresholds.append(thres, ignore_index=True)

        res = self.evaluate_prediction(y_true, y_drop)
        results.update(res)
        df_results = df_results.append(results, ignore_index=True)

        return df_results, thresholds, y_drop

    # Functions for running one or more models
    def run_individual_model(self, models: Union[BaseEstimator, Dict[str, BaseEstimator]], df_results: pd.DataFrame,
                             name: str, thresholds: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run an individual model for each task and evaluate the predictions.

        :param models: A single model or a dictionary of models, where each key corresponds to a task.
        :param df_results: DataFrame to store results.
        :param name: Name of the model or experiment.
        :param thresholds: DataFrame to store thresholds.
        :return: Tuple containing updated DataFrame, thresholds, and predicted values.
        """
        data = self.data
        results = {'name': name}
        y_pred = pd.DataFrame()
        x_pred = pd.DataFrame()

        for task in data[2].columns:
            if type(models) == dict:
                model = models[task]
            else:
                model = models
            model.fit(data[0], data[2][task])
            results['model-' + task] = model
            y_pred[task] = model.predict(data[1])
            x_pred[task] = model.predict(data[0])

        if ((y_pred != 0) & (y_pred != 1)).any().any():
            y_pred, thres = self.threshold_find_and_transform(y_pred, x_pred, data[2])
            thres = dict(zip(data[2].columns, thres))
            thres['name'] = name
            thresholds = thresholds.append(thres, ignore_index=True)

        res = self.evaluate_prediction(data[3], y_pred)
        results.update(res)
        df_results = df_results.append(results, ignore_index=True)

        return df_results, thresholds, y_pred

    def run_all_models(self, save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run all models and save the results.

        :param save: Boolean value to decide if results are saved to file.
        :return: Tuple containing results, thresholds and parameters dataframes.
        """
        data = self.data
        df_results = self.df_results
        df_thresholds = pd.DataFrame()
        df_parameters = pd.DataFrame()

        models_no_grid = [LogisticRegression(random_state=1, max_iter=1000), LinearRegression(),
                          MultinomialNB(), GaussianNB()]
        names_no_grid = ['LogisticRegression', 'LinearRegression', 'MultinomialNB', 'GaussianNB']
        for model, name in zip(models_no_grid, names_no_grid):
            df_results, df_thresholds, y_pred = self.run_individual_model(model, df_results, name, df_thresholds)

        models_grid = [KNeighborsClassifier(), KNeighborsRegressor(), DecisionTreeClassifier(random_state=1),
                       DecisionTreeRegressor(random_state=1), RandomForestClassifier(random_state=1),
                       RandomForestRegressor(random_state=1), SVC(kernel='rbf'), SVR(kernel='rbf')]
        names_grid = ['KNeighborsClassifier', 'KNeighborsRegressor', 'DecisionTreeClassifier', 'DecisionTreeRegressor',
                      'RandomForestClassifier', 'RandomForestRegressor', 'SVC', 'SVR']
        if self.gridsearch:
            grid_parameters = [{'n_neighbors': range(1, 50)}, {'n_neighbors': range(1, 50)},
                               {'max_depth': range(1, 11)},
                               {'max_depth': range(1, 11)},
                               {'n_estimators': [100, 500, 1000], 'max_depth': [5, 10, 15]},
                               {'n_estimators': [100, 500, 1000], 'max_depth': [5, 10, 15]},
                               {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]},
                               {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}]
            excs = [{'n_neighbors': 1}, {'n_neighbors': 1}, {'max_depth': 1}, {'max_depth': 1},
                    {'n_estimators': 100, 'max_depth': 5}, {'n_estimators': 100, 'max_depth': 5},
                    {'n_estimators': [100, 500, 1000], 'max_depth': [5, 10, 15]},
                    {'n_estimators': [100, 500, 1000], 'max_depth': [5, 10, 15]},
                    {'C': 1, 'gamma': 0.1}, {'C': 1, 'gamma': 0.1}]
            for model, model_parameters, exc, name in zip(models_grid, grid_parameters, excs, names_grid):
                param, model = self.grid_search(model_parameters, model, exc)
                param['name'] = name
                df_parameters = df_parameters.append(param, ignore_index=True)
                df_results, df_thresholds, y_pred = self.run_individual_model(model, df_results, name, df_thresholds)
        else:
            df_parameters = pd.read_csv(self.path_res + 'grid_parameters.csv').drop(columns=['Unnamed: 0'])
            for model, name in zip(models_grid, names_grid):
                parameters = df_parameters[df_parameters['name'] == name][data[2].columns].iloc[0].to_dict()
                models = {}
                for task in data[2].columns:
                    param = ast.literal_eval(str(parameters[task]))
                    new_model = copy.deepcopy(model)
                    models[task] = new_model.set_params(**param)
                df_results, df_thresholds, y_pred = self.run_individual_model(models, df_results, name, df_thresholds)

        df_results, df_thresholds, y_pred = self.run_cbf(df_results, df_thresholds)
        df_results, df_thresholds, y_pred = self.run_cf(df_results, df_thresholds, False)

        if save:
            drop_cols = ['model-' + x for x in data[2].columns]
            df_results_drop = df_results.drop(columns=drop_cols)
            df_results_drop.to_csv(self.path_res + 'methods_evaluation.csv', index=False)
            df_thresholds.to_csv(self.path_res + 'thresholds.csv')
            df_parameters.to_csv(self.path_res + 'grid_parameters.csv')

        self.df_results = df_results

        return df_results, df_thresholds, df_parameters
