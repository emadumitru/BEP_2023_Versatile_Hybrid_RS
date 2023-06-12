import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from typing import Union, List, Tuple
from methods_pipeline import IndividualPredictionsPipeline as IPP

PATH_RESULTS = 'data/results/'
PATH_DATA = 'data/data_final/'
PATH_PREDICTIONS = 'data/predictions/'


class ModelEvaluator:
    THRESHOLDS = (0.75, 0.85, 0.95)
    MINIM_MODELS = (7, 5, 3)

    def __init__(self, path_data: str = PATH_DATA, path_results: str = PATH_RESULTS, load_data: int = 1,
                 load_split: int = 1, split_ratio: float = 0.3, gridsearch: int = 0, load_kfold: int = 1,
                 data_info: tuple = ()) -> None:
        """
        Initialize the ModelEvaluator class.

        :param path_data: Path to the data.
        :param path_results: Path to the results.
        :param load_data: Flag indicating whether to load data or not.
        :param load_split: Flag indicating whether to load data split or not.
        :param split_ratio: Ratio for data splitting.
        :param gridsearch: Flag indicating whether to perform grid search or not.
        :param load_kfold: Flag indicating whether to load k-fold results or not.
        :param data_info: Additional data - contains full data in case different data should be used
        """
        self.path_results = path_results
        self.pred_pipeline = IPP(path_data, path_results, load_data, load_split, split_ratio, gridsearch, data_info)
        self.data = self.pred_pipeline.data
        self.models = self.get_full_models()
        self.kfold_results = self.get_kfold_results(load_kfold)
        self.names = self.select_all_models()

    def get_full_models(self) -> pd.DataFrame:
        """
        Gets the actual models to be used in predictions.

        :return: DataFrame containing all the models for each task.
        """
        keep_cols = ['model-' + pt for pt in self.data[3].columns]
        models = self.pred_pipeline.run_all_models()[0].set_index('name')[keep_cols]
        return models

    def kfold_evaluation_train(self) -> pd.DataFrame:
        """
        Perform k-fold evaluation on the training data.

        :return: DataFrame containing the evaluation results.
        """
        data = self.data
        kf = KFold(n_splits=5)
        final_results = []
        tasks_pt = data[2].columns
        drop_cols = ['model-' + pt for pt in tasks_pt]
        for train_index, test_index in kf.split(data[0]):
            x_train, x_test = data[0].iloc[train_index], data[0].iloc[test_index]
            y_train, y_test = data[2].iloc[train_index], data[2].iloc[test_index]
            data_train = (x_train, x_test, y_train, y_test)
            ipp = IPP(data_info=data_train, gridsearch=0)
            df_results, df_thresholds, df_parameters = ipp.run_all_models(save=False)
            df_results = df_results.drop(columns=drop_cols)
            df_results = df_results.set_index('name')
            final_results.append(df_results)

        results = final_results[0].copy()
        for df in final_results[1:]:
            results += df
        results = results / len(final_results)

        results.to_csv(self.path_results + 'evaluation_for_architecture.csv', index=True)

        return results

    def get_kfold_results(self, load_kfold: int) -> pd.DataFrame:
        """
        Get k-fold results. Either load them or call k-fol function.

        :param load_kfold: Flag indicating whether to load k-fold results or not.
        :return: DataFrame containing the k-fold results of interest.
        """
        interest_columns = ['overall_balanced_accuracy'] + [task + '_accuracy' for task in self.data[3].columns]
        if load_kfold:
            results = pd.read_csv(self.path_results + 'evaluation_for_architecture.csv').set_index('name')
        else:
            results = self.kfold_evaluation_train()
        results = results[interest_columns]
        results = results.drop(index=['CF'])

        return results

    def select_models(self, col: str, thresh: float, minim: int) -> List[str]:
        """
        Select models based on a specified column, threshold, and minimum count.

        :param col: Column to filter on.
        :param thresh: Threshold value for performance.
        :param minim: Minimum count for number of models.
        :return: List containing the name of the selected models.
        """
        rows = self.kfold_results[self.kfold_results[col] >= thresh]
        if len(rows) < minim:
            rows = self.kfold_results.sort_values(by=[col]).tail(minim)
        return rows.index.tolist()

    def select_all_models(self) -> List[dict]:
        """
        Select all models based on thresholds and minimum counts.

        :return: List containing dictionaries of selected models for each prediction task.
        """
        l1_models = self.select_models('overall_balanced_accuracy', self.THRESHOLDS[0], self.MINIM_MODELS[0])
        l2_models, l3_models, fin_models = {}, {}, {}
        for task in self.data[3].columns:
            col = task + '_accuracy'
            l2_models[task] = self.select_models(col, self.THRESHOLDS[1], self.MINIM_MODELS[1])
            l3_models[task] = self.select_models(col, self.THRESHOLDS[2], self.MINIM_MODELS[2])
            fin_models[task] = self.select_models(col, 1.1, 1)
        cf_models = {key: ['CF'] for key in fin_models}

        return [l1_models, l2_models, l3_models, fin_models, cf_models]


class Architecture:
    def __init__(self, model_evaluator: ModelEvaluator, path_pred: str = PATH_PREDICTIONS) -> None:
        """
        Initialize the Architecture class.

        :param model_evaluator: ModelEvaluator instance.
        :param path_pred: Path for saving predictions.
        """
        self.path_pred = path_pred
        self.model_evaluator = model_evaluator
        self.ipp = self.model_evaluator.pred_pipeline
        self.data = self.model_evaluator.data
        self.models = self.model_evaluator.models
        self.names = self.model_evaluator.names
        self.layers = self.run_3_layers()

    def make_layer_predictions(self, data: tuple, models: pd.Series) -> List[pd.DataFrame]:
        """
        Make predictions for a layer.

        :param data: Tuple containing the data used for prediction.
        :param models: Models to be used for prediction.
        :return: List of dataframes containing the predictions.
        """
        predictions = []
        ipp = self.ipp
        ipp.data = data
        if 'CBF' in models.index:
            models.drop('CBF', inplace=True)
            predictions.append(ipp.run_cbf(pd.DataFrame(), pd.DataFrame())[2])
        if 'CF' in models.index:
            models.drop('CF', inplace=True)
            predictions.append(ipp.run_cf(pd.DataFrame(), pd.DataFrame(), arch=True)[2])

        for model in models:
            pred = ipp.run_individual_model(model, pd.DataFrame(), '', pd.DataFrame())[2]
            predictions.append(pred)

        return predictions

    def general_layer_task(self, phase_models: pd.Series, task: str, prephase: pd.DataFrame) -> pd.DataFrame:
        """
        Perform prediction for a task in a layer.

        :param phase_models: Models to be used for the task.
        :param task: Prediction task name.
        :param prephase: DataFrame containing predictions from previous layers.
        :return: DataFrame containing updated predictions.
        """
        data = self.data
        phase_predictions = prephase.copy()
        ind_pred = phase_predictions[~phase_predictions[task].isin([0, 1])].index
        phase_data = (data[0], data[1].loc[ind_pred], data[2][[task]], data[3][[task]].loc[ind_pred])

        if not phase_data[1].empty:
            predictions = self.make_layer_predictions(phase_data, phase_models)
            predictions = [df[task] for df in predictions]
            predictions = [np.mean(column) for column in zip(*predictions)]
            phase_predictions.loc[ind_pred, task] = predictions

        return phase_predictions

    def run_layer(self, models: pd.DataFrame, name_models: Union[dict, list], prephase: pd.DataFrame) -> pd.DataFrame:
        """
        Run a layer with the given models and update the predictions.

        :param models: Models for the layer.
        :param name_models: Names of the models used in the layer.
        :param prephase: DataFrame containing predictions from previous layers.
        :return: DataFrame containing updated predictions.
        """
        for task in prephase.columns:
            try:
                phase_names = name_models[task]
            except:
                phase_names = name_models
            phase_models = models.loc[phase_names]
            phase_models = phase_models['model-' + task]
            prephase = self.general_layer_task(phase_models, task, prephase)

        return prephase

    def run_3_layers(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the three layers of the architecture.

        :return: Tuple of DataFrames containing the predictions of each layer.
        """
        data = self.data
        models = self.models
        names = self.names

        prephase = pd.DataFrame(columns=data[2].columns, index=data[3].index)

        layer1 = self.run_layer(models, names[0], prephase)
        layer2 = self.run_layer(models, names[1], layer1)
        layer3 = self.run_layer(models, names[2], layer2)

        data[3].to_csv(self.path_pred + 'true.csv', index=False)
        layer1.to_csv(self.path_pred + 'layer1.csv', index=False)
        layer2.to_csv(self.path_pred + 'layer2.csv', index=False)
        layer3.to_csv(self.path_pred + 'layer3.csv', index=False)
        layer3.to_csv(self.path_pred + 'transformation/layer3.csv', index=True)

        return layer1, layer2, layer3


class FinalizeAnalyze:
    def __init__(self, architecture: Architecture, path_pred: str = PATH_PREDICTIONS,
                 list_choices: list = None) -> None:
        """
        Initialize the FinalizeAnalyze class.

        :param architecture: Architecture instance.
        :param path_pred: Path for saving predictions.
        :param list_choices: List of choices for finalizing the analysis. Default is None.
        """
        if list_choices is None:
            list_choices = []
        self.path_trans = path_pred + 'transformation/'
        self.list_choices = list_choices
        self.architecture = architecture
        self.ipp = self.architecture.ipp
        self.path_res = self.ipp.path_res
        self.layers = self.architecture.layers
        self.data = self.architecture.data
        self.models = self.architecture.models
        self.names = self.architecture.names
        self.transformations = []
        self.multiple_finalize(list_choices)
        self.results_disagreements = pd.DataFrame()
        self.results_prediction = pd.DataFrame()
        self.results_overall = pd.read_csv(self.path_res + 'methods_evaluation.csv')

    def finalize_layer(self, choices: tuple) -> pd.DataFrame:
        """
        Finalize a layer based on the given choices.

        :param choices: Tuple of choices for finalizing the layer.
        :return: Finalized layer DataFrame.
        """
        ci = choices[0]
        meth = choices[1]
        layer = self.layers[2]
        name = 'ci' + str(ci) + '_choice' + str(meth)

        if ci:
            thr = choices[2]
            name += '_thr' + str(thr)
            mapping_function = lambda x: 0 if x <= thr else 1 if x >= 1 - thr else round(x, 3)
            layer = layer.applymap(mapping_function)
            if meth == 1:
                mapping_function = lambda x: 0.5 if x != 0 and x != 1 else x
                layer = layer.applymap(mapping_function)
            elif meth == 2:
                layer = self.architecture.run_layer(self.models, self.names[3], layer)
            elif meth == 3:
                layer = self.architecture.run_layer(self.models, self.names[4], layer)
        else:
            mapping_function = lambda x: round(x, 3) if x != 0 and x != 1 else x
            layer = layer.applymap(mapping_function)
            if meth == 1:
                thr = choices[2]
                name += '_thr' + str(thr)
                mapping_function = lambda x: 0 if x < thr else 1
                layer = layer.applymap(mapping_function)
            elif meth == 2:
                layer = self.architecture.run_layer(self.models, self.names[3], layer)
            elif meth == 3:
                layer = self.architecture.run_layer(self.models, self.names[4], layer)

        layer.to_csv(self.path_trans + name + '.csv', index=False)

        return layer

    def multiple_finalize(self, list_choices: list) -> None:
        """
        Finalize the predictions for multiple choices.

        :param list_choices: List of choices for finalizing the predictions.
        """
        for choices in list_choices:
            self.transformations += [self.finalize_layer(choices)]

    def analyze_disagreements(self, layer: pd.DataFrame, name: str) -> None:
        """
        Analyze the disagreements in the predictions for a layer.

        :param layer: Predictions for a layer.
        :param name: Name of the layer.
        """
        layer = layer.applymap(lambda x: 0 if 0 < x < 1 else 1)
        results = {'name': name}

        cases_disagreement = (layer == 0).any(axis=1).mean()
        total_disagreement = (layer == 0).mean().mean()
        column_disagreement = layer.apply(lambda x: (x == 0).mean())

        results['case_disagreement'] = cases_disagreement
        results['total_disagreement'] = total_disagreement
        for task in layer.columns:
            results[f'{task}_disagreement'] = column_disagreement[task]

        results = pd.DataFrame([results])
        self.results_disagreements = pd.concat([self.results_disagreements, results], ignore_index=True)

    def analyze_disagreements_layers(self) -> None:
        """
        Analyze the disagreements in the predictions for all layers.
        """
        self.analyze_disagreements(self.layers[0], 'layer1')
        self.analyze_disagreements(self.layers[1], 'layer2')
        self.analyze_disagreements(self.layers[2], 'layer3')

        results = self.results_disagreements.set_index('name')

        l4 = (results.loc['layer2'] / results.loc['layer1']).to_frame().T.set_index([['layer2_exclusive']]).reset_index(
            drop=False).rename({'index': 'name'}, axis='columns').loc[0].to_dict()
        l5 = (results.loc['layer3'] / results.loc['layer2']).to_frame().T.set_index([['layer2_exclusive']]).reset_index(
            drop=False).rename({'index': 'name'}, axis='columns').loc[0].to_dict()

        self.results_disagreements = self.results_disagreements.append(l4, ignore_index=True)
        self.results_disagreements = self.results_disagreements.append(l5, ignore_index=True)

    def analyze_prediction(self, layer: pd.DataFrame, name: str) -> None:
        """
        Analyze the accuracy of finalized predictions for a layer.

        :param layer: Predictions for a layer.
        :param name: Name of the layer.
        """
        results = {'name': name}
        overall_pred = []
        overall_true = []
        true = self.data[3]

        for task in layer.columns:
            ind_pred = layer[layer[task].isin([0, 1])].index.tolist()
            layer_task = layer[task].loc[ind_pred].astype(int)
            true_task = true[task].loc[ind_pred]
            overall_pred += layer_task.tolist()
            overall_true += true_task.tolist()
            evaluation = self.ipp.base_evaluation(true_task, layer_task, task)
            results.update({key: str(value) for key, value in evaluation.items()})

        evaluation = self.ipp.base_evaluation(pd.DataFrame(overall_true), pd.DataFrame(overall_pred), 'overall')
        results.update({key: str(value) for key, value in evaluation.items()})
        self.results_prediction = self.results_prediction.append(results, ignore_index=True)

    def analyze_accuracy(self, layer: pd.DataFrame, name: str) -> None:
        """
        Analyze the overall accuracy of the predictions for a layer (including those not finalized).

        :param layer: Predictions for a layer.
        :param name: Name of the layer.
        """
        results = {'name': name}
        true = self.data[3]

        for task in layer.columns:
            ind_no_pred = layer[~layer[task].isin([0, 1])].index
            layer.loc[ind_no_pred, task] = 1 - true.loc[ind_no_pred, task]

        evaluation = self.ipp.evaluate_prediction(true, layer.astype(int))
        results.update(evaluation)
        self.results_overall = self.results_overall.append(results, ignore_index=True)

    def evaluation_architecture(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform the evaluation of the architecture.

        :return: Evaluation results for disagreements, prediction accuracy, and overall accuracy.
        """
        self.analyze_disagreements_layers()

        dfs = list(self.layers) + self.transformations
        names = ['layer' + str(x) for x in range(1, 4)]
        names += ['transformation' + str(choices) for choices in self.list_choices]
        for df, name in zip(dfs, names):
            self.analyze_prediction(df, name)
            self.analyze_accuracy(df, name)

        self.results_disagreements.to_csv(self.path_res + 'evaluation_disagreements.csv', index=False)
        self.results_prediction.to_csv(self.path_res + 'evaluation_agreements_accuracy.csv', index=False)
        self.results_overall.to_csv(self.path_res + 'evaluation_overall_accuracy.csv', index=False)

        return self.results_disagreements, self.results_prediction, self.results_overall
