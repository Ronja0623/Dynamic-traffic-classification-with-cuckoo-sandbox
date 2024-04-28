from classifier import Classifier

"""
Set the parameters
"""
# input path
MALWARE_DATASET_DIR = "dataset"
SAMPLE_FOLDER = "armed"
DATA_DESCRIPTION = "dataset.csv"
# output path: flow dataset
PCAP_DIR = "pcap"
FLOW_DIR = "flow"
BYTES_DIR = "bytes"
GRAPH_DIR = "graph"
FLOW_DATASET_DIR = "flow_dataset"
# output path: malware classification
MODEL_DIR = "model"
LOG_DIR = "log"
# api token
API_TOKEN = "API_TOKEN"  # replace with your own API token
# MAX NUM that the hard disk could be affordable
PROCESS_FILE_NUM = 150
# dynamic analysis wait time (sec)
BASE_TIME = 100  # at least wait for BASE_TIME sec
REQUEST_INTERVAL = 2  # then check if it is finished every REQUEST_INTERVAL sec
MAX_WAIT_TIME = 300  # if wait over MAX_WAIT_TIME sec, skip
# balance the number of the sample in every family
NUM_OF_EACH_FAMILY = 10
# hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.8
EPOCHS = 100
RANDOM_STATE = 42

classifier = Classifier(
    MALWARE_DATASET_DIR,
    SAMPLE_FOLDER,
    DATA_DESCRIPTION,
    PCAP_DIR,
    FLOW_DIR,
    BYTES_DIR,
    GRAPH_DIR,
    FLOW_DATASET_DIR,
    MODEL_DIR,
    LOG_DIR,
    API_TOKEN,
    PROCESS_FILE_NUM,
    BASE_TIME,
    REQUEST_INTERVAL,
    MAX_WAIT_TIME,
    BATCH_SIZE,
    LEARNING_RATE,
    TRAIN_RATIO,
    EPOCHS,
)
# classifier.mkdir()
# classifier.dynamic_analysis()
# classifier.extract_feature()
# classifier.generate_image()
# classifier.load_traffic_data()
classifier.train_model()
