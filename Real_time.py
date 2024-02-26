import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog  # Import simpledialog for getting user input
from scapy.all import sniff, IP, TCP
from tensorflow import keras
import joblib
from PIL import Image, ImageTk
from customtkinter import CTk, CTkLabel, CTkFrame, CTkEntry, CTkButton, set_appearance_mode, set_default_color_theme, CTkFont
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
        self.root.geometry("1920x1080")

        self.cnn_results = []
        self.svm_results = []
        self.source_ips = []
        self.dest_ips = []

        # Create widgets
        self.timeout_label = ttk.Label(root, text="Timeout (seconds):")
        self.timeout_label.pack(pady=5)

        self.timeout_entry = CTkEntry(root, placeholder_text="Enter Sniff Timeout (seconds)")
        self.timeout_entry.pack(pady=5)

        self.results_text = tk.Text(root, height=20, width=80)
        self.results_text.pack()
        
        self.classify_button = ttk.Button(root, text="Classify Now", command=self.start_real_time_classification)
        self.classify_button.pack(pady=10)


        # Flag to control real-time classification
        self.real_time_classification_running = False

    def start_real_time_classification(self):
        if not self.real_time_classification_running:
            self.real_time_classification_running = True
            self.cnn_results = []
            self.svm_results = []
            self.source_ips = []
            self.dest_ips = []

            # Get the sniff timeout from the user
            try:
                timeout = int(self.timeout_entry.get())
            except ValueError:
                tk.messagebox.showerror("Error", "Invalid timeout value. Please enter a valid number.")
                return

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
                    result_text = f"Source IP: {packet[IP].src}\n"
                    result_text += f"Destination IP: {packet[IP].dst}\n"
                    result_text += f"CNN Prediction: {cnn_label}\n"
                    result_text += f"SVM Prediction: {svm_label}\n\n"

                    # Update the Text widget
                    self.results_text.insert(tk.END, result_text)
                    self.results_text.see(tk.END)  # Scroll to the end

                    # TODO: Add your logic for further processing or display here

                    self.cnn_results.append(cnn_label)
                    self.svm_results.append(svm_label)

                    # Stop real-time classification if a certain condition is met
                    if len(self.cnn_results) >= 10:
                        self.real_time_classification_running = False

            # Start sniffing packets
            sniff(prn=packet_callback, store=0, timeout=timeout)  # Sniff for the specified timeout

if __name__ == "__main__":
    root = tk.Tk()
    real_time_app = RealTimeClassificationApp(root)
    root.mainloop()