import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from data_creation import infer_type, get_object_column_types, get_datasets_characteristics, expand_dataframe
from data_creation import check_csv_for_tasks, check_all_csvs_in_df, generate_topic_data
from methods_pipeline import IndividualPredictionsPipeline
from system_architecture import ModelEvaluator, Architecture, FinalizeAnalyze

PATH_TEST_DATA = 'data/test_data/data/'
PATH_TEST_DATA_RESULTS = 'data/test_data/results/'
PATH_TEST_DATA_PREDICTIONS = PATH_TEST_DATA_RESULTS + 'predictions/'




class TestDataCreation():
    def test_infer_type(self):
        # Test identify bool
        input_1 = 'True'
        expected_output_1 = 'bool'
        result_1 = infer_type(input_1)
        assert result_1 == expected_output_1

        # Test identify int
        input_2 = '42'
        expected_output_2 = 'int64'
        result_2 = infer_type(input_2)
        assert result_2 == expected_output_2

        # Test identify float
        input_3 = '3.14'
        expected_output_3 = 'float64'
        result_3 = infer_type(input_3)
        assert result_3 == expected_output_3

        # Test identify string
        input_4 = 'Hello World'
        expected_output_4 = 'string'
        result_4 = infer_type(input_4)
        assert result_4 == expected_output_4

    def test_get_object_column_types(self):
        data = {'Name': ['John', 'Jane', 'Mike', 'Juli', 'Lottie', 'Mark', 'Sem', 'Zven', 'Clara', 'Zora'],
                'Age': [25, 30, 35, 27, 31, 43, 25, 18, 24, 21],
                'Gender': ['M', 'F', 'M', 'F', 'F', 'M', 'M', 'M', 'F', 'F']}
        df = pd.DataFrame(data)

        expected_output = {'Name': 'string', 'Gender': 'categorical'}

        result = get_object_column_types(df)

        assert result == expected_output

    def test_get_datasets_characteristics(self):
        # Create a temporary directory for test datasets
        test_data_path = 'test_data/'
        os.makedirs(test_data_path, exist_ok=True)

        # Create sample CSV files
        csv_data_1 = {'Name': ['John', 'Jane', 'Mike', 'Juli', 'Lottie', 'Mark', 'Sem', 'Zven', 'Clara', 'Zora'],
                      'Age': [25, 30, 35, 27, 31, 43, 25, 18, 24, 21],
                      'Gender': ['M', 'F', 'M', 'F', 'F', 'M', 'M', 'M', 'F', 'F']}
        csv_data_2 = {'Product': ['A', 'B', 'C', 'D'], 'Price': [10.0, 15.5, 20.0, 17.5], 'Quantity': [100, 200, 300, 150],
                      'Category': ['Electronics', 'Clothing', 'Food', 'Gifts']}
        csv_data_3 = {'Name': ['Alice', 'Bob', 'Eve', 'Tanya', 'Mark'], 'Age': [22, 27, 31, 15, 21],
                      'City': ['Chicago', 'Sydney', 'Tokyo', 'London', 'Tokyo'], 'Category': ['A', 'B', 'A', 'B', 'A']}

        target_columns = ['crow', 'ccol', 'object', 'float64', 'int64', 'bool', 'cnan', 'categorical', 'string']
        df_expected = pd.DataFrame(columns=target_columns)
        df_expected.loc['data_1.csv'] = [10, 3, 2, 0, 1, 0, 0, 1, 1]
        df_expected.loc['data_2.csv'] = [4, 4, 2, 1, 1, 0, 0, 0, 2]
        df_expected.loc['data_3.csv'] = [5, 4, 3, 0, 1, 0, 0, 1, 2]

        # Save the sample CSV files to the test directory
        pd.DataFrame(csv_data_1).to_csv(test_data_path  + 'data_1.csv', index=False)
        pd.DataFrame(csv_data_2).to_csv(test_data_path  + 'data_2.csv', index=False)
        pd.DataFrame(csv_data_3).to_csv(test_data_path  + 'data_3.csv', index=False)

        # Call the function
        result = get_datasets_characteristics(test_data_path )

        # Assert that the result equals the expected output
        assert result.equals(df_expected)

        # Clean up the test directory
        for file in os.listdir(test_data_path):
            os.remove(test_data_path + file)
        os.rmdir(test_data_path)

    def test_expand_dataframe(self):
        num_rows_25 = 25
        target_rows = 100

        df_25 = pd.DataFrame({'crow': np.random.randint(1, 5000, num_rows_25),
                              'ccol': np.random.randint(1, 50, num_rows_25),
                              'object': np.random.randint(0, 10, num_rows_25),
                              'float64': np.random.randint(0, 10, num_rows_25),
                              'int64': np.random.randint(0, 10, num_rows_25),
                              'bool': np.random.randint(0, 4, num_rows_25),
                              'cnan': np.random.randint(0, 1000, num_rows_25),
                              'categorical': np.random.randint(0, 10, num_rows_25),
                              'string': np.random.randint(0, 10, num_rows_25)})

        expanded_df = expand_dataframe(df_25, target_rows)

        assert len(expanded_df) == target_rows, "The number of rows do not match the target."
        assert set(df_25.columns) == set(expanded_df.columns), "The columns are not the same."
        assert not expanded_df.isnull().values.any(), "The expanded dataframe contains null values."
        assert (expanded_df >= 0).all().all(), "The expanded dataframe contains negative values."
        assert expanded_df['ccol'].equals(expanded_df[['object', 'float64', 'int64', 'bool', 'categorical', 'string'
                                                       ]].sum(axis=1)), "Column 'ccol' is not the sum."

    def test_check_csv_for_tasks(self):
        # Test all true
        csv_description = pd.Series({'crow': 300, 'ccol': 11, 'object': 10, 'float64': 8, 'int64': 5,
                                     'bool': 2, 'cnan': 1200, 'categorical': 4, 'string': 3})

        expected_results = {"Data Manipulation": True,
                            "Data Visualization": True,
                            "Data Cleaning and Preprocessing": True,
                            "Programming Concepts": True,
                            "Exploratory Data Analysis": True,
                            "Object-Oriented Programming": True}

        actual_results = check_csv_for_tasks(csv_description)

        for task in expected_results.keys():
            assert actual_results[task] == expected_results[task], f"Mismatch found for task: {task}"

        # Test all false
        csv_description = pd.Series({'crow': 50, 'ccol': 2, 'object': 1, 'float64': 1, 'int64': 1,
                                     'bool': 0, 'cnan': 0, 'categorical': 0, 'string': 0})

        expected_results = {"Data Manipulation": False,
                            "Data Visualization": False,
                            "Data Cleaning and Preprocessing": False,
                            "Programming Concepts": False,
                            "Exploratory Data Analysis": False,
                            "Object-Oriented Programming": False}

        actual_results = check_csv_for_tasks(csv_description)

        for task in expected_results.keys():
            assert actual_results[task] == expected_results[task], f"Mismatch found for task: {task}"

    def test_check_all_csvs_in_df(self):
        # Prepare input DataFrame
        df_initial = pd.DataFrame({'crow': [300, 50], 'ccol': [11, 2], 'object': [10, 1], 'float64': [8, 1],
                                   'int64': [5, 1], 'bool': [2, 0], 'cnan': [1200, 0], 'categorical': [4, 0],
                                   'string': [3, 0]}, index=['csv1', 'csv2'])

        expected_results = pd.DataFrame({'crow': [300, 50], 'ccol': [11, 2], 'object': [10, 1], 'float64': [8, 1],
                                         'int64': [5, 1], 'bool': [2, 0], 'cnan': [1200, 0], 'categorical': [4, 0],
                                         'string': [3, 0], "Data Manipulation": [1, 0], "Data Visualization": [1, 0],
                                         'Data Cleaning and Preprocessing': [1, 0], 'Programming Concepts': [1, 0],
                                         'Exploratory Data Analysis': [1, 0],
                                         'Object-Oriented Programming': [1, 0]}, index=['csv1', 'csv2'])

        actual_results = check_all_csvs_in_df(df_initial)

        pd.testing.assert_frame_equal(actual_results, expected_results.astype(int))

    def test_generate_topic_data(self):
        path_final = 'data/data_final/'

        # test when load_char is True
        if os.path.exists(path_final + 'full_characteristics.csv'):
            df_char, df_map = generate_topic_data(load_char=True, load_csv25=False, load_mapping=False, target_rows=1000)
            assert isinstance(df_char, pd.DataFrame)
            assert df_char.shape[0] == 1000  # assuming full_characteristics.csv has 1000 rows

        # test when load_csv25 is True
        if os.path.exists(path_final + 'characteristics_25.csv'):
            df_char, df_map = generate_topic_data(load_char=False, load_csv25=True, load_mapping=False, target_rows=1000)
            assert isinstance(df_char, pd.DataFrame)
            assert df_char.shape[0] == 1000  # assuming characteristics_25.csv has 1000 rows

        # test when load_mapping is True
        if os.path.exists(path_final + 'full_mapping.csv'):
            df_char, df_map = generate_topic_data(load_char=False, load_csv25=False, load_mapping=True, target_rows=1000)
            assert isinstance(df_map, pd.DataFrame)
            assert df_map.shape[0] == 1000  # assuming full_mapping.csv has 1000 rows

        # test when all load flags are False (this will generate new data)
        if os.path.exists('data/data_initial/course/') and os.path.exists('data/data_initial/Kaggle/'):
            df_char, df_map = generate_topic_data(load_char=False, load_csv25=False, load_mapping=False, target_rows=1000)
            assert isinstance(df_char, pd.DataFrame)
            assert df_char.shape[0] == 1000  # target_rows parameter
            assert isinstance(df_map, pd.DataFrame)
            assert df_map.shape[0] == 1000  # target_rows parameter

class TestMethodsPipeline():
    IPP = IndividualPredictionsPipeline(path_res=PATH_TEST_DATA_RESULTS, path_data=PATH_TEST_DATA)

    def test_initiate_data(self):
        data = self.IPP.initiate_data(re_initiate=0)
        assert len(data) == 7
        assert isinstance(data[0], pd.DataFrame)  # X
        assert isinstance(data[1], pd.DataFrame)  # Y
        assert isinstance(data[2], pd.DataFrame)  # Y_pred
        assert isinstance(data[3], pd.DataFrame)  # df_results
        assert isinstance(data[4], list)  # data_char
        assert isinstance(data[5], list)  # tasks
        assert isinstance(data[6], list)  # tasks_pt

    def test_get_data_split(self):
        split_data = self.IPP.get_data_split()
        assert len(split_data) == 4
        assert isinstance(split_data[0], pd.DataFrame)  # X_train
        assert isinstance(split_data[1], pd.DataFrame)  # X_test
        assert isinstance(split_data[2], pd.DataFrame)  # Y_train
        assert isinstance(split_data[3], pd.DataFrame)  # Y_test

    def test_threshold_series_find(self):
        true = pd.Series(np.random.choice([0, 1], size=100))
        pred = pd.Series(np.random.random(size=100))
        threshold = self.IPP.threshold_series_find(true, pred)
        assert 0.05 <= threshold <= 0.95

    def test_threshold_find_and_transform(self):
        true = pd.DataFrame(np.random.choice([0, 1], size=(100, 5)))
        pred = pd.DataFrame(np.random.random(size=(100, 5)))
        bin_pred, thresholds = self.IPP.threshold_find_and_transform(true, pred, true)
        assert all((0 <= th <= 1) for th in thresholds)

    def test_grid_search(self):
        grid_parameters = [{'n_neighbors': range(1, 50)}, {'max_depth': range(1, 11)}]
        excs = [{'n_neighbors': 1}, {'max_depth': 1}]
        models = [KNeighborsClassifier(), DecisionTreeRegressor(random_state=1)]
        for parameters, model, exc in zip(grid_parameters, models, excs):
            results, results_model = self.IPP.grid_search(parameters, model, exc)
            assert isinstance(results, dict)
            assert isinstance(results_model, dict)

    def test_base_evaluation(self):
        true = pd.DataFrame(np.random.choice([0, 1], size=(100, 5)))
        pred = pd.DataFrame(np.random.choice([0, 1], size=(100, 5)))
        results = self.IPP.base_evaluation(true, pred, "test")
        assert "test_accuracy" in results
        assert "test_balanced_accuracy" in results
        assert "test_f1_score" in results
        assert "test_precision" in results
        assert "test_recall" in results

    def test_evaluate_prediction(self):
        true = pd.DataFrame(np.random.choice([0, 1], size=(100, 5)))
        pred = pd.DataFrame(np.random.choice([0, 1], size=(100, 5)))
        results = self.IPP.evaluate_prediction(true, pred)
        assert isinstance(results, dict)
        assert len(results) == 5 * (len(true.columns) + 1)

    def test_run_cbf(self):
        df_results = pd.DataFrame()
        thresholds = pd.DataFrame()
        updated_df, updated_thresholds, y_pred = self.IPP.run_cbf(df_results, thresholds)
        assert isinstance(updated_df, pd.DataFrame)
        assert isinstance(updated_thresholds, pd.DataFrame)
        assert isinstance(y_pred, pd.DataFrame)

    def test_run_cf(self):
        df_results = pd.DataFrame()
        thresholds = pd.DataFrame()
        updated_df, updated_thresholds, y_pred = self.IPP.run_cf(df_results, thresholds, False)
        assert isinstance(updated_df, pd.DataFrame)
        assert isinstance(updated_thresholds, pd.DataFrame)
        assert isinstance(y_pred, pd.DataFrame)

    def test_run_individual_model(self):
        model = LinearRegression()

        results = self.IPP.run_individual_model(model, pd.DataFrame(), "TestModel", pd.DataFrame())
        assert isinstance(results[0], pd.DataFrame)
        assert isinstance(results[1], pd.DataFrame)
        assert isinstance(results[2], pd.DataFrame)

    def test_run_all_models(self):
        ipp = self.IPP
        ipp.gridsearch = 0

        results, thresholds, parameters = ipp.run_all_models()
        assert isinstance(results, pd.DataFrame)
        assert isinstance(thresholds, pd.DataFrame)
        assert isinstance(parameters, pd.DataFrame)

class TestSystemArchitecture():
    ME = ModelEvaluator(path_data=PATH_TEST_DATA, path_results=PATH_TEST_DATA_RESULTS, load_kfold=0)
    ARCH = Architecture(ME, PATH_TEST_DATA_PREDICTIONS)
    FA = FinalizeAnalyze(ARCH, PATH_TEST_DATA_PREDICTIONS, [[0, 0]])

    def test_ModelEvaluator_init(self):
        # Tests that all functions init the correct type of variables
        evaluator = self.ME

        assert evaluator.path_results is not None
        assert isinstance(evaluator.pred_pipeline, IndividualPredictionsPipeline)
        assert evaluator.data is not None
        assert isinstance(evaluator.models, pd.DataFrame)
        assert isinstance(evaluator.kfold_results, pd.DataFrame)
        assert isinstance(evaluator.names, list)
        assert os.path.isfile(evaluator.path_results + 'evaluation_for_architecture.csv')

    def test_select_models(self):
        # Tests that the correct models are chosen
        evaluator = self.ME
        evaluator.kfold_results = pd.read_csv(evaluator.path_results + 'test_select_models_ME.csv')
        # Case less over threshold than minim
        expected = [1, 0, 3]
        returned = evaluator.select_models(col='overall_balanced_accuracy', thresh=0.85, minim=3)
        assert returned == expected
        # Case get over threshold
        expected = [0, 1]
        returned = evaluator.select_models(col='overall_recall', thresh=0.9, minim=2)
        assert returned == expected

    def test_Architecture_init(self):
        arch = self.ARCH

        assert isinstance(arch.model_evaluator, ModelEvaluator)
        assert isinstance(arch.ipp, IndividualPredictionsPipeline)
        assert len(arch.data) == 4
        assert isinstance(arch.layers[0], pd.DataFrame)
        assert isinstance(arch.layers[1], pd.DataFrame)
        assert isinstance(arch.layers[2], pd.DataFrame)
        assert os.path.isfile(arch.path_pred + 'layer1.csv')
        assert os.path.isfile(arch.path_pred + 'transformation/layer3.csv')

    def test_FinalizeAnalyze_init(self):
        fin = self.FA

        assert isinstance(fin.architecture, Architecture)
        assert isinstance(fin.ipp, IndividualPredictionsPipeline)
        assert isinstance(fin.transformations[0], pd.DataFrame)
        fin.evaluation_architecture()
        assert isinstance(fin.results_disagreements, pd.DataFrame)
        assert isinstance(fin.results_prediction, pd.DataFrame)
        assert isinstance(fin.results_overall, pd.DataFrame)
        assert len(fin.results_disagreements) > 1
        assert len(fin.results_prediction) > 1
        assert len(fin.results_overall) > len(pd.read_csv(fin.path_res + 'methods_evaluation.csv'))


