import pandas as pd
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
from scapy.all import sniff, IP, TCP  # Import the IP and TCP classes
from tensorflow import keras
import joblib
import warnings

# Ignore scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# Load the trained models and pre-processing tools (replace with your actual model paths)
cnn_model = keras.models.load_model('nn_model.h5')
svm_model = joblib.load('svm_model.sav')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

class RealTimeClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Packet Classification")
        self.root.geometry("600x400")

        self.cnn_results = []
        self.svm_results = []
        self.source_ips = []
        self.dest_ips = []

        # Create widgets
        self.classify_button = ttk.Button(root, text="Classify Now", command=self.start_real_time_classification)
        self.classify_button.pack(pady=10)

        self.loading_var = tk.StringVar()
        self.loading_label = ttk.Label(root, textvariable=self.loading_var)
        self.loading_label.pack(pady=10)

        # Flag to control real-time classification
        self.real_time_classification_running = False

    def start_real_time_classification(self):
        if not self.real_time_classification_running:
            self.real_time_classification_running = True
            self.cnn_results = []
            self.svm_results = []
            self.source_ips = []
            self.dest_ips = []

            def packet_callback(packet):
                if packet.haslayer(IP) and packet.haslayer(TCP):
                    # Extract relevant information from the packet
                    packet_data = {
                        "tcp.time_relative": [packet.time],
                        "frame.protocols": [packet.payload.name],
                        "ip.src": [packet[IP].src],
                        "tcp.srcport": [packet[TCP].sport],
                        "ip.dst": [packet[IP].dst],
                        "tcp.dstport": [packet[TCP].dport],
                        "frame.len": [len(packet)],
                        "tcp.len": [len(packet[TCP])],
                        "tcp.stream": [packet[TCP].stream if hasattr(packet[TCP], 'stream') else None],
                        "tcp.flags": [packet.sprintf("%TCP.flags%")],
                        "tcp.ack": [packet[TCP].ack],
                    }

                    packet_df = pd.DataFrame(packet_data)

                    # Preprocess the packet data
                    categorical_columns = packet_df.select_dtypes(include=['object']).columns
                    numerical_columns = packet_df.select_dtypes(include=['float64']).columns

                    for col in categorical_columns:
                        packet_df[col] = label_encoder.fit_transform(packet_df[col])

                    packet_df[numerical_columns] = scaler.fit_transform(packet_df[numerical_columns])

                    # Classify the packet
                    input_data = packet_df.iloc[0].to_numpy().reshape(1, -1)

                    cnn_probabilities = cnn_model.predict(input_data)
                    cnn_prediction = np.argmax(cnn_probabilities, axis=-1)[0]

                    svm_prediction = svm_model.predict(input_data)[0]

                    # Convert numerical labels back to original labels
                    if int(cnn_prediction) == 0:
                        cnn_label = "normal"
                    elif int(cnn_prediction) == 1:
                        cnn_label = "suspicious"
                    elif int(cnn_prediction) == 2:
                        cnn_label = "unknown"

                    svm_label = str(svm_prediction)

                    # Store source and destination IPs
                    self.source_ips.append(packet[IP].src)
                    self.dest_ips.append(packet[IP].dst)

                    # Print the classification results along with IP addresses
                    print(f"Source IP: {packet[IP].src}")
                    print(f"Destination IP: {packet[IP].dst}")
                    print(f"CNN Prediction: {cnn_label}")
                    print(f"SVM Prediction: {svm_label}")
                    print("\n")

                    # TODO: Add your logic for further processing or display here

                    self.cnn_results.append(cnn_label)
                    self.svm_results.append(svm_label)

                    # Update the loading percentage
                    loading_percentage = len(self.cnn_results) / 10  # Just for display purpose
                    self.root.after(0, self.loading_var.set, f"Classifying... {loading_percentage:.2f}%")

                    # Stop real-time classification if a certain condition is met
                    if len(self.cnn_results) >= 10:
                        self.real_time_classification_running = False
                        self.display_results()

            # Start sniffing packets
            sniff(prn=packet_callback, store=0, timeout=30)  # Sniff for 30 seconds (adjust as needed)

    def display_results(self):
        results_window = tk.Tk()
        results_window.title("Real-Time Classification Results")

        frame = ttk.Frame(results_window)
        frame.grid(row=0, column=0)

        tree = ttk.Treeview(frame, columns=["Source IP", "Destination IP", "CNN Predictions", "SVM Predictions"], show="headings")
        tree.heading("Source IP", text="Source IP")
        tree.heading("Destination IP", text="Destination IP")
        tree.heading("CNN Predictions", text="CNN Predictions")
        tree.heading("SVM Predictions", text="SVM Predictions")
        tree.column("Source IP", width=150)
        tree.column("Destination IP", width=150)
        tree.column("CNN Predictions", width=200)
        tree.column("SVM Predictions", width=200)

        for i in range(len(self.cnn_results)):
            tree.insert("", "end", values=[self.source_ips[i], self.dest_ips[i], self.cnn_results[i], self.svm_results[i]])

        tree.grid(row=0, column=0)

        results_window.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeClassificationApp(root)
    root.mainloop()
