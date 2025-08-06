from data.dataset import *

def create_dataset(data):
    if data == 'HCP-WM':
        dataset = HCP_YA_SCDataset(bold_dir='/HCP-YA-SC_FC/Brainnetome_264/BOLD_interpolated', 
                                    sc_dir='/HCP-YA-SC_FC/HCP-YA-SC',
                                    label_path = '/ram/USERS/jiaqi/benchmark_fmri/data/DFYANG_0914/LR_label/LR_label.csv')
        num_class = 8
        in_channels = 39
    elif data == 'HCP-YA':
        dataset = HCPYA_BoldSCDataset(bold_dir='/HCP-YA-SC_FC/AAL_116/BOLD', 
                                     sc_dir='/HCP-YA-SC_FC/HCP-YA-SC')
        num_class = 7
        in_channels = 175
    elif data == 'HCP-A':
        dataset = HCPA_BoldSCDataset(bold_dir='/HCP-A-SC_FC/AAL_116/BOLD', 
                                     sc_dir='/HCP-A-SC_FC/ALL_SC')
        num_class = 4
        in_channels = 300
    elif data == 'ADNI':
        dataset = ADNI_BoldSCDataset(bold_dir='/ADNI/ADNI_BOLD_SC/AAL90', 
                                     sc_dir='/ADNI/ADNI_BOLD_SC/ADNI_SC',
                                     label_csv = '/ADNI/ADNI_BOLD_SC/subject_info_250.csv')
        num_class = 2
        in_channels = 140
    elif data == 'PPMI':
        dataset = PPMI_BoldSCDataset(bold_dir='/PPMI-SC-FC/AAL_116/BOLD', 
                                     sc_dir='/PPMI-SC-FC/ALL-SC/ses-1',
                                     label_csv = '/PPMI-SC-FC/Participant_Status_11Feb2025.csv')
        num_class = 4
        in_channels = 239
    
    elif data == 'NIFD':
        dataset = NIFD_BoldSCDataset(bold_dir='/NIFD-SC-FC/AAL_116/BOLD', 
                                     sc_dir='/NIFD-SC-FC/ALL-SC/ses-1',
                                     label_csv = '/NIFD-SC-FC/NIFD_Clinical_Data_20200724_updated.xlsx')
        num_class = 5
        in_channels = 176
        
    elif data == 'OASIS':
        dataset = OASIS_BoldSCDataset(bold_dir='/OASIS/OASIS_BOLD_SC/OASIS_BOLD160', 
                                     sc_dir='/OASIS/OASIS_BOLD_SC/OASIS_SC',
                                     label_csv = '/OASIS/OASIS_BOLD_SC/fMRI_label.csv')
        num_class = 2
        in_channels = 328
    return dataset