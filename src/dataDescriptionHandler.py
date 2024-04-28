import concurrent.futures
import os
import shutil

import pandas as pd
from tqdm import tqdm


class DataDescriptionHandler:
    """
    Preprocess the data for the classifier.
    Methods:
        get_family_array: Get all unique values in a specific column of a CSV file.
        load_traffic_data: Load traffic data from the flow files.
    """

    def __init__(self, dataset_dir, data_description, graph_dir, flow_dataset_dir):
        """
        Initialize the preprocess class.
        """
        self.DESCRIPTION_PATH = os.path.join(dataset_dir, data_description)
        self.GRAPH_DIR = graph_dir
        self.FLOW_DATASET_DIR = flow_dataset_dir

    def get_family_array(self, label_column="label"):
        """
        Get all unique values in a specific column of a CSV file.
        """
        try:
            df = pd.read_csv(self.DESCRIPTION_PATH)
            unique_values = df[label_column].unique()
            return unique_values
        except Exception as e:
            print(f"Error: {e}")
            return None

    def load_traffic_data(self):
        """
        Load traffic data from the flow files.
        """
        input_dir = self.GRAPH_DIR
        input_folders = os.listdir(input_dir)
        output_dir = self.FLOW_DATASET_DIR
        try:
            label_info = self.get_label_info()
            results = self._process_flow_files(
                input_folders, input_dir, output_dir, label_info
            )
            df = pd.DataFrame(results, columns=["flow_name", "source_malware", "label"])
            df.to_csv(os.path.join(output_dir, "flow_description.csv"), index=False)
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_label_info(self, name_column="name", label_column="label"):
        """
        Get a dictionary saving label information.
        """
        try:
            df = pd.read_csv(self.DESCRIPTION_PATH)
            label_info = dict(zip(df[name_column], df[label_column]))
            return label_info
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

    def copy_file(self, source_file_path, target_dir, source_file, source_dir, label):
        """
        Copy a file to the target directory and return information about the copied file.
        """
        shutil.copy2(source_file_path, target_dir)
        return [source_file, source_dir, label]

    def _process_flow_files(self, input_folders, input_dir, output_dir, label_info):
        """
        Process flow files and copy them to the appropriate target directories.
        """
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for source_dir in tqdm(input_folders, desc="Processing folders"):
                source_dir_path = os.path.join(input_dir, source_dir)
                source_files = os.listdir(source_dir_path)
                for source_file in source_files:
                    source_file_path = os.path.join(source_dir_path, source_file)
                    target_dir = os.path.join(output_dir, label_info[source_dir])
                    futures.append(
                        executor.submit(
                            self.copy_file,
                            source_file_path,
                            target_dir,
                            source_file,
                            source_dir,
                            label_info[source_dir],
                        )
                    )
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Copying files",
            ):
                results.append(future.result())
        return results
