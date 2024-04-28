import hashlib
import os
import shutil
from collections import defaultdict

from scapy.all import rdpcap, wrpcap


class ReportFetcher:
    """
    Fetch the analysis report
    """

    def __init__(self, analysis_report_dir, dump_pcap, pcap_dir):
        """
        Initialize the pcap report processor
        """
        self.ANALYSIS_REPORT_DIR = analysis_report_dir
        self.DUMP_PCAP = dump_pcap
        self.PCAP_DIR = pcap_dir

    def access_pcap_report(self, file_name, task_id):
        """
        Access the pcap report
        """
        source_file_path = os.path.join(
            self.ANALYSIS_REPORT_DIR, str(task_id), self.DUMP_PCAP
        )
        destination_path = os.path.join(self.PCAP_DIR, f"{file_name}.pcap")
        shutil.copy(source_file_path, destination_path)


class FlowsExtractor:
    """
    Extract flows from pcap files
    """

    def __init__(self, pcap_dir, flow_dir):
        """
        Initialize the flows extractor
        """
        self.PCAP_DIR = pcap_dir
        self.FLOW_DIR = flow_dir

    def extract_flows(self, file_name):
        """
        Extract flows from the pcap file
        """
        input_pcap_path = os.path.join(self.PCAP_DIR, file_name)
        packets = rdpcap(input_pcap_path)
        flows = defaultdict(list)
        for pkt in packets:
            if "IP" in pkt and (pkt.haslayer("TCP") or pkt.haslayer("UDP")):
                flow_key = (
                    pkt["IP"].src,
                    pkt.sport if pkt.haslayer("TCP") else pkt["UDP"].sport,
                    pkt["IP"].dst,
                    pkt.dport if pkt.haslayer("TCP") else pkt["UDP"].dport,
                    "TCP" if pkt.haslayer("TCP") else "UDP",
                )
                flows[flow_key].append(pkt)
        return flows

    def write_flows_to_pcap(self, sample_file_name, flows):
        """
        Write the flows to pcap files
        """
        output_pcap_dir = os.path.join(
            self.FLOW_DIR, os.path.splitext(sample_file_name)[0]
        )
        os.makedirs(output_pcap_dir, exist_ok=True)
        for flow_key, packets in flows.items():
            file_name = "_".join(str(item) for item in flow_key) + ".pcap"
            output_pcap_path = os.path.join(output_pcap_dir, file_name)
            wrpcap(output_pcap_path, packets)


class BytesExtractor:
    """
    Extract bytes from pcap files
    """

    def __init__(self, flow_dir, bytes_dir):
        """
        Initialize the bytes extractor
        """
        self.FLOW_DIR = flow_dir
        self.BYTES_DIR = bytes_dir

    def get_hash_name(self, pcap_file):
        """
        Generate a hash name for the pcap file
        """
        with open(pcap_file, "rb") as f:
            file_content = f.read()
        file_hash = hashlib.md5(file_content).hexdigest()
        return f"{file_hash}.txt"

    def remove_address(self, pkt):
        """
        Remove the address information from the packet
        """
        if "IP" in pkt:
            pkt["IP"].src = "0.0.0.0"
            pkt["IP"].dst = "0.0.0.0"
        if "TCP" in pkt:
            pkt["TCP"].sport = 0
            pkt["TCP"].dport = 0
        if "UDP" in pkt:
            pkt["UDP"].sport = 0
            pkt["UDP"].dport = 0
        return pkt

    def pad_data(self, data, target_length):
        """
        Pad the data to the target length
        """
        current_length = len(data)
        if current_length >= target_length:
            return data[:target_length]
        else:
            padding_length = target_length - current_length
            padding_data = b"\x00" * padding_length
            return data + padding_data

    def trans_to_bytes(self, input_pcap_path, output_dir):
        """
        Transform the pcap file to bytes
        """
        packets = rdpcap(input_pcap_path)
        output_file_path = os.path.join(output_dir, self.get_hash_name(input_pcap_path))
        filtered_packets = [self.remove_address(pkt) for pkt in packets]
        bytes_data = [bytes(pkt) for pkt in filtered_packets]
        combined_bytes = b"".join(bytes_data)
        padded_bytes = self.pad_data(combined_bytes, 28 * 28)
        with open(output_file_path, "wb") as f:
            f.write(padded_bytes)

    def run(self, folder_name):
        """
        Run the bytes extractor
        """
        input_dir = os.path.join(self.FLOW_DIR, folder_name)
        input_pcap_files = os.listdir(input_dir)
        output_dir = os.path.join(self.BYTES_DIR, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        for pcap_file in input_pcap_files:
            input_path = os.path.join(input_dir, pcap_file)
            self.trans_to_bytes(input_path, output_dir)
