import tkinter as tk
from tkinter import messagebox
import pandas as pd
from system_architecture import ModelEvaluator, Architecture, FinalizeAnalyze
from data_creation import get_datasets_characteristics, check_all_csvs_in_df
from typing import List, Tuple, Union
import re
import time


PATH = 'data/tool/'
PATH_PROG = PATH + 'programming_prediction/'
PATH_OTHER = PATH + 'new_topic/'
PATH_RESULTS_OTHER = PATH_OTHER + 'results/'
PATH_RESULTS_PREDICTING = PATH_PROG + 'results_predicting/'
PATH_RESULTS_TESTING = PATH_PROG + 'results_testing/'
THRESHOLDS = (0.5, 0.2)
USER_INPUT = False
USER_OUT = True
INPUT_VALUES = (1, 1, 0, [[0, 0], [1, 3, 0.2]])


class UIInput:
    ANSWERS = []
    VALUES = [[], []]

    def __init__(self, vals: Tuple[float, float] = THRESHOLDS) -> None:
        """
        Initialize the UIInput class.

        :param vals: Tuple of threshold values.
        """
        self.input_values = self.get_input(vals)

    def get_topic_and_transformations(self) -> List[list]:
        """
        Get the topic and transformations from the user through a GUI.
        """
        def submit() -> None:
            """
            Submit the user's choices and display confirmation message.
            """
            topic_answer = topic_var.get()
            transformation_answer = [choice for choice, trans in transformation_vars.items() if trans.get()]

            self.ANSWERS = [[topic_answer], transformation_answer]
            answer_message = "This is what you chose:\n\n" + "\n\n".join(
                f"Q: {q}\nA: {', '.join(a)}" for q, a in zip(questions, self.ANSWERS))

            result = messagebox.askquestion("Confirmation", answer_message +
                                            "\n\nDo you want to change your answers?", icon='question')

            if result == 'yes':
                reset_answers()
            else:
                additional_message = "You are currently using the last pre-trained data. Retraining will take " \
                                     "significant time but is necessary especially if you use a new dataset. " \
                                     "Do you want to retrain?"
                retrain_result = messagebox.askquestion("Retrain Model", additional_message, icon='question')
                self.ANSWERS.append([retrain_result])
                if root.winfo_exists():  # Check if the root window still exists
                    root.destroy()

        def reset_answers() -> None:
            """
            Reset the user's answers to the default values.
            """
            topic_var.set("Programming task")
            for trans in transformation_vars.values():
                trans.set(False)

        questions = ["Do you want to get programming task for your CSV(s) or test the system?",
                     "How do you want your final transformation to be?"]

        root = tk.Tk()
        root.title("Questionnaire")

        # Topic Question
        topic_label = tk.Label(root, text=questions[0])
        topic_label.pack()
        topic_var = tk.StringVar(root)
        topic_var.set("Programming task")
        topic_options = ["Programming task", "Test the system (programming tasks topic)", "Test the system (new topic)"]
        for option in topic_options:
            topic_radio = tk.Radiobutton(root, text=option, variable=topic_var, value=option)
            topic_radio.pack(anchor="w")

        # Transformation Question
        transformation_label = tk.Label(root, text=questions[1])
        transformation_label.pack()
        transformation_vars = {}
        transformation_options = ["CI + Percentage", "CI + 0.5", "CI + Top Model", "CI + Collaborative Filtering",
                                  "Percentage", "Threshold", "Top Model", "Collaborative Filtering"]
        for option in transformation_options:
            var = tk.BooleanVar(root)
            transformation_check = tk.Checkbutton(root, text=option, variable=var)
            transformation_check.pack(anchor="w")
            transformation_vars[option] = var

        # Submit button
        submit_button = tk.Button(root, text="Submit", command=submit)
        submit_button.pack()

        root.mainloop()

        def transfor_answers(answers: List[List[str]]) -> List[list]:
            """
            Transform the user's answers into a structured format.

            :param answers: User's answers.
            :return: Transformed answers.
            """
            topic_mapping = {"Programming task": [1, 1],
                             "Test the system (programming tasks topic)": [1, 0],
                             "Test the system (new topic)": [0, 0]}

            transformation_mapping = {"CI + Percentage": [1, 0],
                                      "CI + 0.5": [1, 1],
                                      "CI + Top Model": [1, 2],
                                      "CI + Collaborative Filtering": [1, 3],
                                      "Percentage": [0, 0],
                                      "Threshold": [0, 1],
                                      "Top Model": [0, 2],
                                      "Collaborative Filtering": [0, 3]}

            topic_numbers = [topic_mapping[choice] for choice in answers[0]]
            transformation_numbers = [transformation_mapping[choice] for choice in answers[1]]
            retrain_answer = [1 if answer == 'yes' else 0 for answer in answers[2]]

            answers = [topic_numbers, transformation_numbers, retrain_answer]

            return answers

        self.ANSWERS = transfor_answers(self.ANSWERS)
        return self.ANSWERS

    def ask_values(self, value_type: str, initial_value: float, valid_range: Tuple[float, float]) -> List[float]:
        """
        Ask the user for a value or multiple values within a specified range.

        :param value_type: Type of value being asked (e.g., "threshold").
        :param initial_value: Initial value to display.
        :param valid_range: Valid range for the value.
        :return: List of values entered by the user.
        """
        def verify_values() -> None:
            """
            Verify the entered values and close the popup window.
            """
            values = [entry.get() for entry in value_entries]
            try:
                if all(valid_range[0] <= float(value) <= valid_range[1] for value in values):
                    confirmation_message = f"{value_type.capitalize()} values: {values}\n\nAre these values correct?"
                    result = messagebox.askquestion("Confirmation", confirmation_message, icon='question')
                    if result == 'yes':
                        return_values.extend([float(value) for value in values])
                        popup.destroy()
                        root.destroy()
                else:
                    messagebox.showerror("Invalid Values", f"Please enter valid {value_type} values ({valid_range[0]}"
                                                           f" - {valid_range[1]}).")
            except:
                messagebox.showerror("Invalid Values",
                                     f"Please enter valid {value_type} values ({valid_range[0]} - {valid_range[1]}).")

        root = tk.Tk()
        root.withdraw()

        option = messagebox.askquestion(f"{value_type} Values", f"Do you want to use more or change the {value_type} "
                                                                f"value ({initial_value})?", icon='question')
        return_values = []
        value_entries = []

        if option == 'no':
            root.destroy()
            return [initial_value]

        if option == 'yes':
            popup = tk.Toplevel(root)
            popup.title(f"{value_type} Values")

            value_count_label = tk.Label(popup, text=f"Enter the number of {value_type} values:")
            value_count_label.pack()
            value_count_entry = tk.Entry(popup)
            value_count_entry.insert(tk.END, "1")  # Set default value
            value_count_entry.pack()

            def prompt_values() -> None:
                """
                Prompt the user to enter the values.
                """
                num_values = int(value_count_entry.get())
                for i in range(num_values):
                    value_label = tk.Label(popup, text=f"Enter {value_type} value {i + 1}:")
                    value_label.pack()
                    value_entry = tk.Entry(popup)
                    value_entry.insert(tk.END, str(initial_value))  # Set default value
                    value_entry.pack()
                    value_entries.append(value_entry)

                submit_button_promt = tk.Button(popup, text="Submit", command=verify_values)
                submit_button_promt.pack()

            submit_button = tk.Button(popup, text="Submit", command=prompt_values)
            submit_button.pack()

            popup.mainloop()

            return return_values

    def get_input(self, vals: Tuple[float, float]) -> List[Union[int, list]]:
        """
        Get user input for the topic, transformations, and values.

        :param vals: Tuple of threshold values.
        :return: User input values.
        """
        self.get_topic_and_transformations()
        answers = self.ANSWERS
        topic_pt, load_test = answers[0][0]
        retrain = answers[2][0]
        transformations = answers[1]

        needs_ci = [trans for trans in transformations if trans[0] == 1]
        needs_thr = [trans for trans in transformations if trans == [0, 1]]
        other_transformations = [trans for trans in transformations if trans not in (needs_ci + needs_thr)]

        if len(needs_ci) > 0:
            ci = self.ask_values('CI', vals[1], (0, 0.5))
            # noinspection PyTypeChecker
            needs_ci = [my_set + [value] for my_set in needs_ci for value in ci]
        if len(needs_thr) > 0:
            thr = self.ask_values('threshold', vals[0], (0, 1))
            # noinspection PyTypeChecker
            needs_thr = [my_set + [value] for my_set in needs_thr for value in thr]

        transformations = needs_ci + needs_thr + other_transformations

        return [topic_pt, load_test, retrain, transformations]


class UIOutput:
    REVERSE_MAPPING = {(1, 0): 'CI + Percentage', (1, 1): 'CI + 0.5', (1, 2): 'CI + Top Model',
                       (1, 3): 'CI + Collaborative Filtering', (0, 0): 'Percentage', (0, 1): 'Threshold',
                       (0, 2): 'Top Model', (0, 3): 'Collaborative Filtering'}
    
    def __init__(self, predictions: pd.DataFrame, absolute: pd.DataFrame,
                 accuracy: pd.DataFrame, predicting: int) -> None:
        """
        Initialize the UIOutput object.

        :param predictions: DataFrame of predictions.
        :param absolute: DataFrame of predicted accuracy.
        :param accuracy: DataFrame of overall accuracy.
        :param predicting: Indicator for prediction mode. 0 when tool is being tested
        """
        self.pred = predictions
        self.abs = absolute
        self.acc = accuracy
        self.predicting = predicting
        self.results = self.transform_dfs_in_str()

    def transform_dfs_in_str(self) -> str:
        """
        Transform DataFrames into a formatted string representation.

        :return: Formatted string representation of the results.
        """
        results = ''

        def print_predictions(df: pd.DataFrame) -> str:
            """
            Format predictions DataFrame into a string representation.

            :param df: DataFrame of predictions.
            :return: Formatted string representation of predictions.
            """
            result_str = ""

            for idx in df.index:
                df.loc[idx, 'trs'] = idx.split("THR")[0]
                df.loc[idx, 'csv'] = idx.split("THR")[1]
            df = df.set_index('csv')

            for group, sub_df in df.groupby('trs'):
                group_tuple = eval(group)
                if len(group_tuple) == 2:
                    group_name = self.REVERSE_MAPPING[(group_tuple[0], group_tuple[1])]
                else:  # In case there is a threshold
                    group_name = self.REVERSE_MAPPING[(group_tuple[0], group_tuple[1])] + f" ({group_tuple[2]})"

                result_str += group_name + "\n"
                sub_df = sub_df.drop('trs', axis=1)

                for idx, row in sub_df.iterrows():
                    csv_name = idx.split(".")[0]

                    ones_list = list(row[row == 1.0].index)
                    others_list = [(i, v) for i, v in row[(row != 0.0) & (row != 1.0)].items()]

                    if len(ones_list) > 0:
                        result_str += csv_name + " suitable: "
                        result_str += ", ".join(ones_list)
                        result_str += "\n"

                    if len(others_list) > 0:
                        result_str += csv_name + " other: "
                        result_str += ", ".join([f"{i}({v})" for i, v in others_list])
                        result_str += "\n"

                    result_str += "\n"

            return result_str

        def print_metrics(df: pd.DataFrame, title: str, metric: str) -> str:
            """
            Format metrics DataFrame into a string representation.

            :param df: DataFrame of metrics.
            :param title: Title of the metrics.
            :param metric: Metric to display.
            :return: Formatted string representation of metrics.
            """
            result_str = title + "\n\n"
            df = df.set_index('name')

            for idx, row in df.iterrows():
                group_tuple = idx
                group_tuple = group_tuple.split('transformation')[1]
                group_tuple = eval(group_tuple)
                if len(group_tuple) == 2:
                    group_name = self.REVERSE_MAPPING[(group_tuple[0], group_tuple[1])]
                else:  # In case there is a threshold
                    group_name = self.REVERSE_MAPPING[(group_tuple[0], group_tuple[1])] + f" ({group_tuple[2]})"

                result_str += group_name + "\n"
                selected_cols = [col for col in df.columns if metric in col]

                for col in selected_cols:
                    result_str += col + ": " + str(row[col]) + "\n"

                result_str += "\n"

            return result_str

        if self.predicting:
            results += print_predictions(self.pred)
        else:
            results += print_metrics(self.abs, 'Absolute Accuracy', 'balanced_accuracy')
            results += '\n\n'
            results += print_metrics(self.acc, 'Overall Accuracy', 'balanced_accuracy')

        return results

    def display_results_and_exit(self):
        """
        Display the results in a message box and exit the program.
        """
        root = tk.Tk()
        root.withdraw()

        # Show the results in a message box
        messagebox.showinfo("Results", re.sub(r'pt\d+_balanced_accuracy:.*?\n', '', self.results))

        root.destroy()


class Tool:
    PATHS = [PATH_RESULTS_PREDICTING, PATH_PROG, PATH_OTHER, PATH_RESULTS_TESTING, PATH_RESULTS_OTHER]
    
    def __init__(self, user_input: bool = USER_INPUT, input_values: tuple = INPUT_VALUES, user_output: bool = USER_OUT,
                 vals: tuple = THRESHOLDS, load_paths: bool = True, path_data: str = '', path_results: str = '') -> None:
        """
        Initialize the Tool object.

        :param user_input: Indicator for using user input.
        :param input_values: Preloaded input values.
        :param vals: Tuple of thresholds.
        :param load_paths: Indicator for loading paths.
        :param path_data: Path to the data.
        :param path_results: Path to the results.
        """
        if user_input:
            input_values = UIInput(vals).input_values
        self.topic_pt, self.predicting, self.retrain, self.transformations = input_values
        if load_paths:
            self.path_data, self.path_res = self.get_paths()
        else:
            self.path_data = path_data
            self.path_res = path_results
        self.model_evaluator = self.get_data()
        self.architecture = Architecture(self.model_evaluator, self.path_res + 'predictions/')
        self.fin_analysis = FinalizeAnalyze(self.architecture, self.path_res + 'predictions/', self.transformations)
        self.eval_architecture = self.fin_analysis.evaluation_architecture()
        self.results = self.get_predictions()
        self.save_results()
        if user_output:
            self.results.display_results_and_exit()

    def get_paths(self) -> Tuple[str, str]:
        """
        Get the data and results paths based on the topic and predicting indicators.

        :return: Tuple of data and results paths.
        """
        paths = self.PATHS
        if self.topic_pt:
            path_data = paths[1]
            path_res = paths[3]
        else:
            path_data = paths[2]
            path_res = paths[4]
        if self.predicting:
            path_data = paths[1]
            path_res = paths[0]
        return path_data, path_res

    def get_data(self) -> ModelEvaluator:
        """
        Get the data for model evaluation.

        :return: ModelEvaluator object.
        """
        if self.predicting:
            x_train = pd.read_csv(self.path_data + 'full_characteristics.csv')
            y_train = pd.read_csv(self.path_data + 'full_mapping.csv')
            tasks_pt = ['pt' + str(i + 1) for i in range(len(list(y_train.columns)))]
            y_train = y_train.rename(columns=dict(zip(list(y_train.columns), tasks_pt)))
            x_test = get_datasets_characteristics(self.path_data + 'load_test/').fillna(0)
            y_test = check_all_csvs_in_df(x_test).drop(columns=x_test.columns)
            y_test = y_test.rename(columns=dict(zip(list(y_test.columns), tasks_pt)))
            data = (x_train, x_test, y_train, y_test)
            evaluator = ModelEvaluator(path_data=self.path_data, path_results=self.path_res, load_split=0,
                                       gridsearch=self.retrain, load_kfold=1 - self.retrain, data_info=data)
        else:
            evaluator = ModelEvaluator(path_data=self.path_data, path_results=self.path_res, load_split=0,
                                       gridsearch=self.retrain, load_kfold=1-self.retrain)
        return evaluator

    def get_predictions(self) -> UIOutput:
        """
        Get the predictions and evaluation results.

        :return: UIOutput object.
        """
        pred = self.fin_analysis.transformations
        pred = pd.concat([df.set_index(pd.Index([f"{ self.transformations[i]}THR{index}"
                                                 for index in df.index])) for i, df in enumerate(pred)])
        pred = pred.rename(columns=dict(zip(list(pred.columns), self.architecture.ipp.tasks)))
        results = self.eval_architecture
        absolute = results[1].drop(results[1].index[:3])
        accuracy = results[2].drop(results[2].index[:17])

        return UIOutput(pred, absolute, accuracy, int(self.predicting))

    def save_results(self) -> None:
        """
        Save the results to a temporary file (gets rewritten every time)
        """
        with open('data/tool/latest_results.txt', 'a') as file:
            file.write(f'\n\n\nTime:{time.time()}\n' + self.results.results)


def run_system_n_times(n: int, topic_pt: bool = True, retrain: bool = True) -> None:
    """
    This function runs the entire tool many times and gives the average results.

    :param n: Number of iterations
    :param topic_pt: Bool whether this is the topic of programming tasks
    :param retrain: Bool whether to be retrained or not. Recommended True
    """
    all_transformations = [[0, 0], [0, 1, 0.5], [0, 2], [0, 3], [1, 0, 0.2], [1, 1, 0.2], [1, 2, 0.2], [1, 3, 0.2]]
    input_values = (int(topic_pt), 0, int(retrain), all_transformations)  # topic_pt, predicting, retrain, transformations

    tool = Tool(False, input_values, False)
    res = [df.set_index('name').astype(float) for df in tool.eval_architecture]
    for i in range(n - 1):
        tool = Tool(False, input_values, False)
        new_res = [df.set_index('name').astype(float) for df in tool.eval_architecture]
        res = [df + new_df for df, new_df in zip(res, new_res)]
    res = [df / n for df in res]

    res[0].to_csv('data/results/loop/evaluation_disagreements.csv')
    res[1].to_csv('data/results/loop/evaluation_agreements_accuracy.csv')
    res[2].to_csv('data/results/loop/evaluation_overall_accuracy.csv')