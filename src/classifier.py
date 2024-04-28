import os
import time
import torch

from apiCalling import APICalling
from dataDescriptionHandler import DataDescriptionHandler
from imageGenerator import ImageGenerator
from pcapProcessor import BytesExtractor, FlowsExtractor, ReportFetcher
from model import (
    BasicCNN,
    CustomVGG16,
    DataProcessor,
    EnhancedLeNet,
    LogManger,
    ModelTrainer,
)


class Classifier:
    def __init__(
        self,
        dataset_dir,
        sample_folder,
        data_description,
        pcap_dir,
        flow_dir,
        bytes_dir,
        graph_dir,
        flow_dataset_dir,
        model_dir,
        log_dir,
        api_token,
        process_file_num,
        base_time,
        request_interval,
        max_wait_time,
        batch_size,
        learning_rate,
        train_ratio,
        epochs,
    ):
        """
        Initialize the classifier
        """
        # input path
        self.DATASET_DIR = dataset_dir
        self.SAMPLE_FOLDER = sample_folder
        self.DATA_DESCRIPTION = data_description
        # output path
        self.PCAP_DIR = pcap_dir
        self.FLOW_DIR = flow_dir
        self.BYTES_DIR = bytes_dir
        self.GRAPH_DIR = graph_dir
        self.FLOW_DATASET_DIR = flow_dataset_dir
        self.MODEL_DIR = model_dir
        self.LOG_DIR = log_dir
        # api token
        self.API_TOKEN = api_token
        # process file num: to avoid disk space overload
        self.PROCESS_FILE_NUM = process_file_num
        # dynamic analysis wait time (sec)
        self.BASE_TIME = base_time
        self.REQUEST_INTERVAL = request_interval
        self.MAX_WAIT_TIME = max_wait_time
        # hyperparameters
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.TRAIN_RATIO = train_ratio
        self.EPOCHS = epochs
        # default analysis report location: don't makdir!
        self.analysis_report_dir = "/home/sandbox/.cuckoo/storage/analyses"
        self.dump_pcap = "dump.pcap"
        # constructor
        self.api_calling = APICalling(self.API_TOKEN)
        self.report_fetcher = ReportFetcher(
            self.analysis_report_dir,
            self.dump_pcap,
            self.PCAP_DIR,
        )
        self.flows_extractor = FlowsExtractor(self.PCAP_DIR, self.FLOW_DIR)
        self.bytes_extractor = BytesExtractor(self.FLOW_DIR, self.BYTES_DIR)
        self.image_generator = ImageGenerator(self.BYTES_DIR, self.GRAPH_DIR)
        self.data_description_handler = DataDescriptionHandler(
            self.DATASET_DIR,
            self.DATA_DESCRIPTION,
            self.GRAPH_DIR,
            self.FLOW_DATASET_DIR,
        )

    def mkdir(self):
        """
        Make directories for the output.
        Folder list: pcap, flow, bytes, graph
        """
        os.makedirs(self.PCAP_DIR, exist_ok=True)
        os.makedirs(self.FLOW_DIR, exist_ok=True)
        os.makedirs(self.BYTES_DIR, exist_ok=True)
        os.makedirs(self.GRAPH_DIR, exist_ok=True)
        os.makedirs(self.FLOW_DATASET_DIR, exist_ok=True)
        family_array = self.data_description_handler.get_family_array()
        for family in family_array:
            os.makedirs(os.path.join(self.FLOW_DATASET_DIR, family), exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

    def dynamic_analysis(self):
        """
        Dynamic analysis for the malware samples.
        """
        # input path
        malware_dir_path = os.path.join(self.DATASET_DIR, self.SAMPLE_FOLDER)
        # get file list to process
        file_list = os.listdir(malware_dir_path)
        processed_list = os.listdir(self.PCAP_DIR)
        file_list_to_process = list(set(file_list) - set(processed_list))
        # count the num of the processed files
        processed_counter = 0
        for file in file_list_to_process:
            # prevent the disk space overload
            if processed_counter > self.PROCESS_FILE_NUM:
                break
            # dynamic analysis
            # get the analysis report id
            task_id = self.api_calling.get_analysis_report_id(malware_dir_path, file)
            processed_counter += 1
            if task_id < 0:
                print("The file is not processed successfully.")
                continue
            # wait for the analysis report
            time.sleep(self.BASE_TIME)
            wait_time_counter = self.BASE_TIME
            while wait_time_counter < self.MAX_WAIT_TIME:
                status_code = self.api_calling.is_analysis_end(task_id)
                # if the analysis is finished
                if status_code == 200:
                    # access the analysis report
                    self.report_fetcher.access_pcap_report(file, task_id)
                    print("Successfully get network analysis report.")
                    break
                time.sleep(self.REQUEST_INTERVAL)
                wait_time_counter += self.REQUEST_INTERVAL
        # clear the report log
        self.api_calling.clear_report_log(self.PROCESS_FILE_NUM + 100)

    def extract_feature(self):
        """
        Extract features from the pcap files.
        """
        # get file list to process
        file_list = os.listdir(self.PCAP_DIR)
        processed_list = os.listdir(self.FLOW_DIR)
        file_list_to_process = list(set(file_list) - set(processed_list))
        for file in file_list_to_process:
            flows = self.flows_extractor.extract_flows(file)
            if flows:
                self.flows_extractor.write_flows_to_pcap(file, flows)
        # Extract bytes
        folder_list = os.listdir(self.FLOW_DIR)
        processed_list = os.listdir(self.BYTES_DIR)
        file_list_to_process = list(set(folder_list) - set(processed_list))
        for folder in file_list_to_process:
            self.bytes_extractor.run(folder)

    def generate_image(self):
        folder_list = os.listdir(self.BYTES_DIR)
        processed_list = os.listdir(self.GRAPH_DIR)
        file_list_to_process = list(set(folder_list) - set(processed_list))
        for folder in file_list_to_process:
            self.image_generator.generate_graph(folder)

    def load_traffic_data(self):
        self.data_description_handler.load_traffic_data()

    def train_model(self):
        # Set the path
        timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())
        model_path = os.path.join(self.MODEL_DIR, timestamp)
        os.makedirs(model_path, exist_ok=True)
        metrics_path = os.path.join(self.LOG_DIR, f"metrics_{timestamp}.csv")
        model_description_path = os.path.join(
            self.LOG_DIR, f"model_description_{timestamp}.txt"
        )
        # Set the model
        data_processor = DataProcessor(self.FLOW_DATASET_DIR)
        num_classes = data_processor.get_num_classes()
        model = EnhancedLeNet(num_classes)
        # model = BasicCNN(num_classes)
        # model = CustomVGG16(num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, val_loader, test_loader = data_processor.get_data_loader(
            self.BATCH_SIZE, model.image_size
        )
        log_manager = LogManger()
        model_trainer = ModelTrainer(
            model,
            log_manager,
            device,
            self.LEARNING_RATE,
            self.EPOCHS,
        )
        model_trainer.train(train_loader, val_loader, model_path)
        model_trainer.test(test_loader)
        log_manager.save_metrics(metrics_path)
        log_manager.save_model_description(
            model_description_path,
            model,
            self.BATCH_SIZE,
            self.LEARNING_RATE,
            self.EPOCHS,
        )
