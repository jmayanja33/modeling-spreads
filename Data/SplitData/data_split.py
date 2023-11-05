"""
Script to split data into train, validation, and test sets. Split is done using a randomization
to make sure that each prompt has an equal representation of seasons/weeks in each set.
For each season, the weeks are randomly shuffled, with about 75% of the data being place the in training set
 and 12.5% in each of the training and validation sets.

The resulting sets are saved with search stat data into `Data/SplitData/(TestSet.csv, TrainingSet.csv,
or ValidationSet.csv).

"""

import random
import pandas as pd
from root_path import ROOT_PATH


class DataSplitController:
    """Class to spilt data into train, test and validation sets"""

    def __init__(self):
        print("Loading in dataset")
        self.df = pd.read_csv(f"{ROOT_PATH}/Data/expanded_data.csv")
        self.df["index"] = [i for i in range(len(self.df))]
        self.training_rows = []
        self.validation_rows = []
        self.test_rows = []

    def save_data_to_csv(self, rows, set_type):
        """
        Function to take training, test, or validation set rows from the main dataset and create a csv file from them
        :param rows: Training, test, or validation set row indexes
        :param set_type: String which could be any of: 'Training', 'Test', 'Validation'
        :return: None
        """
        print(f"Saving {set_type} Set to csv")
        set_df = self.df.iloc[rows]
        set_df.reset_index(inplace=True, drop=False)
        set_df = set_df.drop(columns='index')
        set_df.to_csv(f"{set_type}Set.csv", index=False)

    def sort_data_with_rotation(self):
        """Function to split data using a rotation"""
        year_counter = 1
        for year in self.df["year"].unique():
            year_df = self.df[self.df["year"] == year]
            print(f"Sorting data for year: {year};  Progress: {year_counter}/{len(year_df)}")

            # Shuffle dates of games
            rows = list(year_df["index"].unique())
            random.shuffle(rows)

            # Split data
            counter = 1
            for row in rows:
                if counter % 8 == 0:
                    self.test_rows.append(row)
                elif counter % 4 == 0:
                    self.validation_rows.append(row)
                else:
                    self.training_rows.append(row)
                counter += 1

        year_counter += 1

    def split_data(self):
        """Function to split and save data"""
        self.sort_data_with_rotation()
        self.save_data_to_csv(self.training_rows, 'Training')
        self.save_data_to_csv(self.test_rows, 'Test')
        self.save_data_to_csv(self.validation_rows, 'Validation')


if __name__ == '__main__':
    data_split_controller = DataSplitController()
    data_split_controller.split_data()
