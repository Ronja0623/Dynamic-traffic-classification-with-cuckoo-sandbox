from classifier import Classifier
if __name__ == '__main__':
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
    FLOW_DATASET_DIR = r"D:\Backup\Dataset\Flow\flow_dataset"
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
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-3
    TRAIN_RATIO = 0.8
    EPOCHS = 1
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
    classifier.mkdir()
    # classifier.dynamic_analysis()
    # classifier.extract_feature()
    # classifier.generate_image()
    # classifier.load_traffic_data()
    # classifier.train_model()
    # classifier.train_federated_model(num_clients=2)
    #### tmp
    from model import DataProcessor, LogManager, BasicCNN, EnhancedLeNet, CustomVGG16
    import os
    import time
    import torch
    from federatedLearning import FederatedModelTrainer

    # Set the path
    timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())
    model_path = os.path.join(MODEL_DIR, timestamp)
    os.makedirs(model_path, exist_ok=True)
    metrics_path = os.path.join(LOG_DIR, f"metrics_{timestamp}.csv")
    model_description_path = os.path.join(
        LOG_DIR, f"model_description_{timestamp}.txt"
    )

    # Set the model
    data_processor = DataProcessor(FLOW_DATASET_DIR)
    num_classes = data_processor.get_num_classes()
    # model = EnhancedLeNet(num_classes)
    model = BasicCNN(num_classes)
    # model = CustomVGG16(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = data_processor.get_data_loader(
        BATCH_SIZE, model.image_size
    )

    log_manager = LogManager()
    model_trainer = FederatedModelTrainer(
        model,
        log_manager,
        device,
        LEARNING_RATE,
        EPOCHS,
        num_clients=3,
        use_federated_learning=True,
        use_differential_privacy=False,
        use_homomorphic_encryption=True
    )

    model_trainer.train(train_loader, val_loader, model_path)
    model_trainer.test(test_loader)
    log_manager.save_metrics(metrics_path)
    log_manager.save_model_description(
        model_description_path,
        model,
        BATCH_SIZE,
        LEARNING_RATE,
        EPOCHS,
    )