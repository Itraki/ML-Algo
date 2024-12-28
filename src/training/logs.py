import csv
from datetime import datetime

def init_log_file(file_name):
    if not os.path.exists(file_name):
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "Accuracy", "Timestamp"])

def log_to_csv(file_name, epoch, loss, accuracy):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, loss, accuracy, timestamp])
