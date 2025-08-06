import os
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.utils import to_dense_adj
import glob


class HCPA_BoldSCDataset(Dataset):
    def __init__(self, bold_dir, sc_dir):
        self.bold_dir = bold_dir
        self.sc_dir = sc_dir
        self.label_mapping = {
            "REST": 0,
            "CARIT": 1,
            "FACENAME": 2,
            "VISMOTOR": 3,
        }
        self.data = self._load_data()

    def _load_data(self):
        data = []
        bold_files = [f for f in os.listdir(self.bold_dir) if f.endswith(".csv")]
        for bold_file in bold_files:
            try:
                parts = bold_file.split("_")
                subject_id = parts[0]
                task_type = parts[1].split("-")[1]
                if task_type not in self.label_mapping:
                    continue
                sc_path = os.path.join(self.sc_dir, subject_id, f"{subject_id}_space-T1w_desc-preproc_msmtconnectome.mat")
                if not os.path.exists(sc_path):
                    continue
                mat = loadmat(sc_path)
                if 'aal116_radius2_count_connectivity' not in mat:
                    raise KeyError(f"AAL not found in SC file: {sc_path}")
                sc = mat['aal116_radius2_count_connectivity'].astype(np.float32)

                bold_path = os.path.join(self.bold_dir, bold_file)
                bold_data = pd.read_csv(bold_path).values[:, 1:]
                bold_data = self.pad_sentences(bold_data) if bold_data.shape[0] < 300 else bold_data[:300]
                label = self.label_mapping[task_type]
                data.append((bold_data, sc, label))
            except Exception as e:
                print(f"Error processing file {bold_file}: {e}")
        return data

    def pad_sentences(self, sentence):
        pad_data = torch.cat((torch.tensor(sentence), torch.zeros(300 - sentence.shape[0], sentence.shape[1])), dim=0)
        return pad_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bold, sc, label = self.data[idx]
        bold_tensor = torch.tensor(bold, dtype=torch.float32)
        fc = torch.corrcoef(bold_tensor.T)
        fc_tensor = torch.nan_to_num(fc)
        sc_tensor = torch.tensor(sc, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return bold_tensor, fc_tensor, label_tensor


class TUD(Dataset):
    def __init__(self, root: str, name: str):
        self.dataset = TUDataset(root=root, name=name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        x = data.x.T  
        adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0)
        label = data.y
        return x, adj, label


class HCP_YA_SCDataset(Dataset):
    def __init__(self, bold_dir, sc_dir, label_path, scan="LR"):
        self.bold_dir = bold_dir
        self.sc_dir = sc_dir
        self.label_path = label_path
        self.labels = pd.read_csv(label_path).values.flatten()
        self.bold_files = [f for f in os.listdir(bold_dir) if "WM" in f and scan in f]
        self.data = []
        for bold_file in self.bold_files:
            subject_id = bold_file.split("_")[0].split("-")[1]
            sc_path = os.path.join(sc_dir, f"sub-{subject_id}", f"sub-{subject_id}_space-T1w_desc-preproc_msmtconnectome.mat")
            if not os.path.exists(sc_path):
                continue
            sc_dict = loadmat(sc_path)
            if 'brainnetome246_radius2_count_connectivity' not in sc_dict:
                continue
            sc_matrix = sc_dict['brainnetome246_radius2_count_connectivity']
            bold_path = os.path.join(bold_dir, bold_file)
            bold_data = pd.read_csv(bold_path).values[:, 1:]
            segments, segment_labels = self._process_bold_segments(bold_data, self.labels)
            for segment, label in zip(segments, segment_labels):
                self.data.append((segment, sc_matrix, label))

    def _process_bold_segments(self, bold_data, labels):
        segments = []
        segment_labels = []
        current_label = labels[0]
        start_idx = 0
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                if 1 <= current_label <= 8:
                    segments.append(bold_data[start_idx:i])
                    segment_labels.append(current_label - 1)
                start_idx = i
                current_label = labels[i]
        if 1 <= current_label <= 8:
            segments.append(bold_data[start_idx:])
            segment_labels.append(current_label - 1)
        assert len(segments) == len(segment_labels) == 8
        return segments, segment_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bold_segment, sc_matrix, label = self.data[idx]
        bold_tensor = torch.tensor(bold_segment, dtype=torch.float32)
        fc = torch.corrcoef(bold_tensor.T)
        fc_tensor = torch.nan_to_num(fc)
        sc_tensor = torch.tensor(sc_matrix, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return bold_tensor, fc_tensor, label_tensor


class HCPYA_BoldSCDataset(Dataset):
    def __init__(self, bold_dir, sc_dir):
        self.bold_dir = bold_dir
        self.sc_dir = sc_dir
        self.label_mapping = {
            "EMOTION": 0,
            "GAMBLING": 1,
            "LANGUAGE": 2,
            "MOTOR": 3,
            "RELATIONAL": 4,
            "SOCIAL": 5,
            "WM": 6,
        }
        self.data = self._load_data()

    def _load_data(self):
        data = []
        bold_files = [f for f in os.listdir(self.bold_dir) if f.endswith(".csv") and "LR" in f]
        for bold_file in bold_files:
            try:
                parts = bold_file.split("_")
                subject_id = parts[0]
                task_type = parts[1].split("-")[1]
                if task_type not in self.label_mapping:
                    continue
                sc_path = os.path.join(self.sc_dir, subject_id, f"{subject_id}_space-T1w_desc-preproc_msmtconnectome.mat")
                if not os.path.exists(sc_path):
                    continue
                mat = loadmat(sc_path)
                if 'aal116_radius2_count_connectivity' not in mat:
                    raise KeyError(f"AAL not found in SC file: {sc_path}")
                sc = mat['aal116_radius2_count_connectivity'].astype(np.float32)
                bold_path = os.path.join(self.bold_dir, bold_file)
                bold = pd.read_csv(bold_path, header=0).values[:, 1:]
                bold = self.pad_sentences(bold) if bold.shape[0] < 175 else bold[:175]
                label = self.label_mapping[task_type]
                data.append((bold, sc, label))
            except Exception as e:
                print(f"Error processing file {bold_file}: {e}")
        return data

    def pad_sentences(self, sentence):
        pad_data = torch.cat((torch.tensor(sentence), torch.zeros(175 - sentence.shape[0], sentence.shape[1])), dim=0)
        return pad_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bold, sc, label = self.data[idx]
        bold_tensor = torch.tensor(bold, dtype=torch.float32)
        fc = torch.corrcoef(bold_tensor.T)
        fc_tensor = torch.nan_to_num(fc)
        sc_tensor = torch.tensor(sc, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return bold_tensor, fc_tensor, label_tensor


class ADNI_BoldSCDataset(Dataset):
    def __init__(self, bold_dir, sc_dir, label_csv):
        """
        - bold_dir (str): Path to the directory containing BOLD .txt files.
        - sc_dir (str): Path to the directory containing SC .mat files.
        - label_csv (str): Path to the CSV file containing labels.
        """
        self.bold_dir = bold_dir
        self.sc_dir = sc_dir
        self.labels = pd.read_csv(label_csv)

        self.labels['subject_id'] = self.labels['subject_id'].str.replace('_', '')
        self.label_ids = set(self.labels["subject_id"].astype(str))
        self.labels["DX"] = self.labels["DX"].replace({"CN": 0, "SMC": 0, "EMCI": 0, "LMCI": 1, "AD": 1})

        self.data = []
        for bold_file in os.listdir(bold_dir):
            if bold_file.endswith(".txt"):
                subject_id = bold_file.split("_")[0].replace("sub-", "")
                sc_file = f"sub-{subject_id}_space-T1w_desc-preproc_space-T1w_dhollanderconnectome.mat"

                if subject_id in self.label_ids and os.path.exists(os.path.join(sc_dir, sc_file)):
                    label = self.labels.loc[self.labels["subject_id"] == subject_id, "DX"].values[0]
                    bold_path = os.path.join(self.bold_dir, bold_file)
                    bold_data = np.loadtxt(bold_path)[:140]
                    mat = loadmat(os.path.join(sc_dir, sc_file))
                    sc = mat['aal116_radius2_count_connectivity'].astype(np.float32)
                    sc = self.normalize_matrix(sc)
                    self.data.append((bold_data, sc, label))

    def normalize_matrix(self, matrix):
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        if max_val - min_val == 0:
            return np.zeros(matrix.shape)
        
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
        return normalized_matrix
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bold, sc, label = self.data[idx]

        bold_tensor = torch.tensor(bold, dtype=torch.float32)
        fc = torch.corrcoef(bold_tensor.T)
        fc_tensor = torch.nan_to_num(fc)
        sc = preprocess_adjacency_matrix(sc, 30)
        sc_tensor = torch.tensor(sc, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        fc_tensor = preprocess_adjacency_matrix(fc_tensor, 30)

        return bold_tensor, fc_tensor, sc_tensor, label_tensor


class OASIS_BoldSCDataset(Dataset):
    def __init__(self, bold_dir, sc_dir, label_csv):
        """
        - bold_dir (str): Path to the directory containing BOLD .txt files.
        - sc_dir (str): Path to the directory containing SC .mat files.
        - label_csv (str): Path to the CSV file containing labels.
        """
        self.bold_dir = bold_dir
        self.sc_dir = sc_dir
        self.labels = pd.read_csv(label_csv)

        self.labels['SUBJECT_ID'] = self.labels['SUBJECT_ID'].astype(str)
        self.label_ids = set(self.labels["SUBJECT_ID"].astype(str))
        self.labels["LABEL"] = self.labels["LABEL"].astype(str).str.strip().str.upper().map({"CN": 0, "AD": 1})

        self.data = []
        for bold_file in os.listdir(bold_dir):
            if bold_file.endswith(".txt"):
                subject_id = bold_file.split(".")[0]
                id_part = subject_id.split("_")
                sc_file = f"{id_part[0]}_MR_{id_part[1]}.npy"

                if subject_id in self.label_ids and os.path.exists(os.path.join(sc_dir, sc_file)):
                    label_value = self.labels.loc[self.labels["SUBJECT_ID"] == subject_id, "LABEL"].values[0]
                    if not np.isnan(label_value):
                        label = int(label_value)
                    else:
                        continue 
                    
                    bold_path = os.path.join(self.bold_dir, bold_file)
                    bold_data = np.loadtxt(bold_path)
                    bold_data  = self.pad_sentences(bold_data) if bold_data .shape[0] < 328 else bold_data [:328]
                    sc_path = os.path.join(sc_dir, sc_file)
                    sc = np.load(sc_path)
                    sc = self.normalize_matrix(sc)
                    sc = preprocess_adjacency_matrix(sc, 10)
                    self.data.append((bold_data, sc, label))

    def normalize_matrix(self, matrix):
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        if max_val - min_val == 0:
            return np.zeros(matrix.shape)
        
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
        return normalized_matrix
    
    def pad_sentences(self, sentence):
        pad_data = torch.cat((torch.tensor(sentence), torch.zeros(328 - sentence.shape[0], sentence.shape[1])), dim=0)
        return pad_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bold, sc, label = self.data[idx]

        bold_tensor = torch.tensor(bold, dtype=torch.float32)
        fc = torch.corrcoef(bold_tensor.T)
        fc_tensor = torch.nan_to_num(fc)
        fc = preprocess_adjacency_matrix(fc_tensor, 20)
        sc_tensor = torch.tensor(sc, dtype=torch.float32)
        sc = preprocess_adjacency_matrix(sc, 20)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return bold_tensor, fc_tensor, sc_tensor, label_tensor

def preprocess_adjacency_matrix(adjacency_matrix, percent):
    top_percent = np.percentile(adjacency_matrix.flatten(), 100-percent)
    adjacency_matrix[adjacency_matrix < top_percent] = 0
    return adjacency_matrix


class NIFD_BoldSCDataset(Dataset):
    def __init__(self, bold_dir, sc_dir, label_csv):
        """
        - bold_dir (str): Path to the directory containing BOLD .txt files.
        - sc_dir (str): Path to the directory containing SC .mat files.
        - label_csv (str): Path to the CSV file containing labels.
        """
        self.bold_dir = bold_dir
        self.sc_dir = sc_dir
        self.labels = pd.read_excel(label_csv)

        self.labels['LONI_ID'] = self.labels['LONI_ID'].str.replace('_', '')
        self.label_ids = set(self.labels["LONI_ID"].astype(str))
        self.labels["DX"] = self.labels["DX"].replace({"CON": 0, "L_SD": 1, "BV": 2, "PNFA": 3, "SV": 4})

        self.data = []
        
        for bold_file in os.listdir(bold_dir):
            if bold_file.endswith(".csv"):
                subject_id = bold_file.split("_")[0].replace("sub-", "")
                matching_files = [f for f in os.listdir(sc_dir) if subject_id in f and f.endswith(".mat")]
                if matching_files and subject_id in self.label_ids:
                    sc_file = glob.glob(os.path.join(sc_dir, f"sub-{subject_id}*.mat"))[0]
                    label_value = self.labels.loc[self.labels["LONI_ID"].astype(str) == subject_id, "DX"].values[0]
                    if str(label_value).isdigit():
                        label = int(label_value)
                    else:
                        continue 
                    
                    bold_path = os.path.join(self.bold_dir, bold_file)
                    bold_data = pd.read_csv(bold_path).values[:, 1:]
                    bold_data  = self.pad_sentences(bold_data) if bold_data .shape[0] < 176 else bold_data [:176]     
                    mat = loadmat(sc_file)
                    sc = mat['aal116_radius2_count_connectivity'].astype(np.float32)
                    sc = self.normalize_matrix(sc)
                    self.data.append((bold_data, sc, label))
    
    def pad_sentences(self, sentence):
        pad_data = torch.cat((torch.tensor(sentence), torch.zeros(176 - sentence.shape[0], sentence.shape[1])), dim=0)
        return pad_data
    
    def normalize_matrix(self, matrix):
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        if max_val - min_val == 0:
            return np.zeros(matrix.shape)
        
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
        return normalized_matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bold, sc, label = self.data[idx]

        bold_tensor = torch.tensor(bold, dtype=torch.float32)
        fc = torch.corrcoef(bold_tensor.T)
        fc = preprocess_adjacency_matrix(fc, 30)
        fc_tensor = torch.nan_to_num(fc)
        sc = preprocess_adjacency_matrix(sc, 30)
        sc_tensor = torch.tensor(sc, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return bold_tensor, fc_tensor, sc_tensor, label_tensor


class PPMI_BoldSCDataset(Dataset):
    def __init__(self, bold_dir, sc_dir, label_csv):
        """
        - bold_dir (str): Path to the directory containing BOLD .txt files.
        - sc_dir (str): Path to the directory containing SC .mat files.
        - label_csv (str): Path to the CSV file containing labels.
        """
        self.bold_dir = bold_dir
        self.sc_dir = sc_dir
        self.labels = pd.read_csv(label_csv)
        self.label_ids = set(self.labels["PATNO"].astype(str))
        self.labels["COHORT_DEFINITION"] = self.labels["COHORT_DEFINITION"].replace({"Healthy Control": 0, "Prodromal": 1, "Parkinson's Disease": 2})

        self.data = []
        
        for bold_file in os.listdir(bold_dir):
            if bold_file.endswith(".csv"):
                subject_id = bold_file.split("_")[0].replace("sub-", "")
                matching_files = [f for f in os.listdir(sc_dir) if subject_id in f and f.endswith(".mat")]
                if matching_files and subject_id in self.label_ids:
                    sc_file = glob.glob(os.path.join(sc_dir, f"sub-{subject_id}*.mat"))[0]
                    label = self.labels.loc[self.labels["PATNO"].astype(str) == subject_id, "COHORT_DEFINITION"].values[0]
                    
                    bold_path = os.path.join(self.bold_dir, bold_file)
                    bold_data = pd.read_csv(bold_path).values[:, 1:]
                    bold_data  = self.pad_sentences(bold_data) if bold_data .shape[0] < 239 else bold_data [:239]     
                    mat = loadmat(sc_file)
                    sc = mat['aal116_radius2_count_connectivity'].astype(np.float32)
                    sc = self.normalize_matrix(sc)
                    sc = preprocess_adjacency_matrix(sc, 10)
                    self.data.append((bold_data, sc, label))
    
    def pad_sentences(self, sentence):
        pad_data = torch.cat((torch.tensor(sentence), torch.zeros(239 - sentence.shape[0], sentence.shape[1])), dim=0)
        return pad_data
    
    def normalize_matrix(self, matrix):
        min_val = np.min(matrix)
        max_val = np.max(matrix)
        if max_val - min_val == 0:
            return np.zeros(matrix.shape)
        
        normalized_matrix = (matrix - min_val) / (max_val - min_val)
        return normalized_matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bold, sc, label = self.data[idx]

        bold_tensor = torch.tensor(bold, dtype=torch.float32)
        fc = torch.corrcoef(bold_tensor.T)
        fc = preprocess_adjacency_matrix(fc, 30)
        fc_tensor = torch.nan_to_num(fc)
        sc_tensor = torch.tensor(sc, dtype=torch.float32)
        sc = preprocess_adjacency_matrix(sc, 30)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return bold_tensor, fc_tensor, sc_tensor, label_tensor




