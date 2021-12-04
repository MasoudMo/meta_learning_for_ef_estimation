import re
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
import SimpleITK as sitk
import cv2
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from scipy.io import loadmat
from math import ceil, isnan


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable size videos

    Parameters
    ----------
    batch (tuple): Contains video and label pairs

    Returns
    -------
    Collated batch with padded videos
    """

    videos = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    videos = pad_sequence(videos, padding_value=0, batch_first=True)
    labels = torch.stack(labels)

    return [videos, labels]


def add_sample_to_dataset(patient_dirs,
                          labels,
                          label_to_add,
                          path_to_add):
    """
    Add sample to dataset based on what view is required

    Parameters
    ----------
    patient_dirs (list): List containing directory path for echo videos
    labels (list): List containing labels
    label_to_add (float or int): Label to add to the labels list
    path_to_add (str): Path to patient data

    """

    labels.append(label_to_add)
    patient_dirs.append(path_to_add)


def add_sample_to_dataset_for_task(patient_dirs,
                                   labels,
                                   label_to_add,
                                   path_to_add,
                                   task):
    """
    Adds sample to dataset based on required task

    Parameters
    ----------
    patient_dirs (list): List containing directory path for echo videos
    labels (list): List containing labels
    label_to_add (float or int): Label to add to the labels list
    path_to_add (str): Path to patient data
    task (str): Task to add data samples for
    """

    if task == 'all_ef':
        add_sample_to_dataset(patient_dirs=patient_dirs,
                              labels=labels,
                              label_to_add=label_to_add,
                              path_to_add=path_to_add)

    elif task == 'high_risk_ef':
        if label_to_add <= 35:
            add_sample_to_dataset(patient_dirs=patient_dirs,
                                  labels=labels,
                                  label_to_add=label_to_add,
                                  path_to_add=path_to_add)

    elif task == 'medium_ef_risk':
        if 35 < label_to_add <= 39:
            add_sample_to_dataset(patient_dirs=patient_dirs,
                                  labels=labels,
                                  label_to_add=label_to_add,
                                  path_to_add=path_to_add)

    elif task == 'slight_ef_risk':
        if 39 < label_to_add <= 54:
            add_sample_to_dataset(patient_dirs=patient_dirs,
                                  labels=labels,
                                  label_to_add=label_to_add,
                                  path_to_add=path_to_add)
    elif task == 'normal_ef':
        if label_to_add >= 55:
            add_sample_to_dataset(patient_dirs=patient_dirs,
                                  labels=labels,
                                  label_to_add=label_to_add,
                                  path_to_add=path_to_add)

    else:
        add_sample_to_dataset(patient_dirs=patient_dirs,
                              labels=labels,
                              label_to_add=label_to_add,
                              path_to_add=path_to_add)

    # TODO: Add image quality (if separate task for each image quality)


class CamusEfDataset(Dataset):
    """
    Dataset class for the Camus EF dataset found at https://www.creatis.insa-lyon.fr/Challenge/camus/
    """

    def __init__(self,
                 datasets_root_path,
                 image_shape,
                 device=None,
                 num_frames=32,
                 task='all_ef',
                 view='all_views'):
        """
        Constructor for the Camus EF dataset dataset class

        Parameters
        ----------
        datasets_root_path (str): Path to directory containing all datasets
        image_shape (int): Shape to resize images to
        device (torch device): device to move data to
        num_frames (int): number of frames to use for each video (If video is shorter than this replicate it to fit)
        task (str): task to create the dataset for (must be one of
                    [all_ef, high_risk_ef, medium_ef_risk, slight_ef_risk, normal_ef, esv, edv, quality])
        view (str): video view to make the dataset for (must be one of [all_views, ap2, ap4])
        """

        super().__init__()

        # Input checks
        assert task in ['all_ef',
                        'high_risk_ef',
                        'medium_ef_risk',
                        'slight_ef_risk',
                        'normal_ef',
                        'esv',
                        'edv',
                        'quality'], 'Specified task is not supported'
        assert view in ['all_views',
                        'ap2',
                        'ap4'], 'Specified view is not supported'

        dataset_path = os.path.join(datasets_root_path, 'camus')

        # Obtain file dirs
        data_dirs = os.listdir(dataset_path)

        # Extract file names and locations for each patient (both data and label) depending on task type
        self.labels = list()
        self.patient_data_dirs = list()
        for i, patient in enumerate(data_dirs):

            # Get the correct pattern to search for based on task
            if task in ['all_ef', 'high_risk_ef', 'medium_ef_risk', 'slight_ef_risk', 'normal_ef']:
                label_pattern = re.compile('(?<=(LVef: ))(.*)')
            elif task == 'esv':
                label_pattern = re.compile('(?<=(LVesv: ))(.*)')
            elif task == 'edv':
                label_pattern = re.compile('(?<=(LVedv: ))(.*)')
            else:
                label_pattern = re.compile('(?<=(ImageQuality: ))(.*)')

            # Text files path to extract the label from
            for label_view in ['ap2', 'ap4']:
                if label_view == 'ap2':
                    label_path = os.path.join(dataset_path, patient, 'Info_2CH.cfg')
                    data_path = os.path.join(dataset_path, patient, patient + '_2CH_sequence.mhd')
                else:
                    label_path = os.path.join(dataset_path, patient, 'Info_4CH.cfg')
                    data_path = os.path.join(dataset_path, patient, patient + '_4CH_sequence.mhd')

                label_file = open(label_path)
                label_str = label_file.read()
                label_file.close()

                # Find the label
                match = label_pattern.findall(label_str)[0][1]

                if task == 'quality':
                    if match == 'Good':
                        match = 100
                    if match == 'Medium':
                        match = 50
                    elif match == 'Poor':
                        match = 25
                else:
                    match = float(label_pattern.findall(label_str)[0][1])

                # Add sample for task
                if view == 'all_views':
                    add_sample_to_dataset_for_task(patient_dirs=self.patient_data_dirs,
                                                   labels=self.labels,
                                                   label_to_add=match,
                                                   path_to_add=data_path,
                                                   task=task)
                elif view == label_view:
                    add_sample_to_dataset_for_task(patient_dirs=self.patient_data_dirs,
                                                   labels=self.labels,
                                                   label_to_add=match,
                                                   path_to_add=data_path,
                                                   task=task)

        # Extract the number of available data
        self.num_samples = len(self.patient_data_dirs)

        # Normalization operation
        self.trans = Compose([ToTensor(),
                              Normalize((0.5), (0.5))])

        self.image_shape = image_shape

        # Other attributes
        self.num_frames = num_frames
        self.device = device
        self.task = task

    def __getitem__(self, idx):
        """
        Get the sample video and label at idx

        Parameters
        ----------
        idx (int): Index to fetch data for

        Returns
        -------
        (Torch tensors) for video and corresponding label
        """

        # Get the label
        if self.task == 'quality':
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # extract video
        cine_vid = self.process_cine(sitk.ReadImage(self.patient_data_dirs[idx]), size=self.image_shape)
        cine_vid = self.trans(np.array(cine_vid, dtype=np.uint8)).unsqueeze(1)

        if cine_vid.shape[0] > self.num_frames:
            cine_vid = cine_vid[:self.num_frames]
        else:
            cine_vid = cine_vid.repeat(ceil(self.num_frames / cine_vid.shape[0]), 1, 1, 1)
            cine_vid = cine_vid[:self.num_frames]

        # Move to correct device
        cine_vid = cine_vid.to(self.device)
        label = label.to(self.device)

        return cine_vid, label

    def __len__(self):
        """
        Provides length of the dataset

        Returns
        -------
            (int): Indicates the number of available samples for the task
        """

        return self.num_samples

    @staticmethod
    def process_cine(cine_vid, size=128):
        """
        Processes echo video

        Parameters
        ----------
        cine_vid (SITK Image): Set of images to rescale
        size (int): The size of output images

        Returns
        -------
            (numpy array): Processed cine video
        """

        processed_vid = sitk.GetArrayFromImage(cine_vid)
        processed_vid = np.moveaxis(processed_vid, 0, -1)
        processed_vid = cv2.resize(processed_vid, (size, size))

        return processed_vid


class EchoNetEfDataset(Dataset):
    """
    Dataset class for EchoNet EF found at https://echonet.github.io/dynamic/
    """

    def __init__(self,
                 datasets_root_path,
                 device=None,
                 num_frames=250,
                 nth_frame=1,
                 task='all_ef'):
        """
        Constructor for the EchoNet EF dataset dataset class

        Parameters
        ----------
        datasets_root_path (str): Path to directory containing data
        device (torch device): device to move data to
        num_frames (int): number of frames to use for each video (If video is shorter than this replicate it to fit)
        nth_frame (int): Only extract every nth frame
        task (str): task to create the dataset for (must be one of
                    [all_ef, high_risk_ef, medium_ef_risk, slight_ef_risk, normal_ef, esv, edv])
        """

        super().__init__()

        dataset_path = os.path.join(datasets_root_path, 'echonet')

        # Input checks
        assert task in ['all_ef',
                        'high_risk_ef',
                        'medium_ef_risk',
                        'slight_ef_risk',
                        'normal_ef',
                        'esv',
                        'edv'], 'Specified task is not supported'

        # CSV file containing file names and labels
        filelist_df = pd.read_csv(os.path.join(dataset_path, 'FileList.csv'))

        # All file paths
        data_dirs = [os.path.join(dataset_path,
                                  'Videos',
                                  file_name + '.avi') for file_name in filelist_df['FileName'].tolist()]

        # Get labels based on task
        if task in ['all_ef', 'high_risk_ef', 'medium_ef_risk', 'slight_ef_risk', 'normal_ef']:
            labels = filelist_df['EF'].tolist()
        elif task == 'esv':
            labels = filelist_df['ESV'].tolist()
        else:
            labels = filelist_df['EDV'].tolist()

        self.labels = list()
        self.patient_data_dirs = list()
        for patient_data_dir, patient_label in zip(data_dirs, labels):
            add_sample_to_dataset_for_task(patient_dirs=self.patient_data_dirs,
                                           labels=self.labels,
                                           label_to_add=patient_label,
                                           path_to_add=patient_data_dir,
                                           task=task)

        # Extract the number of available data
        self.num_samples = len(self.patient_data_dirs)

        # Normalization operation
        self.trans = Compose([ToTensor(),
                              Normalize((0.5), (0.5))])

        # Other attributes
        self.device = device
        self.num_frames = num_frames
        self.nth_frame = nth_frame

    def __getitem__(self, idx):
        """
        Get the sample video and label at idx

        Parameters
        ----------
        idx (int): Index to fetch data for

        Returns
        -------
        (Torch tensors) for video and corresponding label
        """

        # Get the label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Get the video
        cine_vid = self.loadvideo(self.patient_data_dirs[idx])

        # Extract nth frames
        cine_vid = cine_vid[0::self.nth_frame]

        # Transform video
        cine_vid = self.trans(cine_vid).unsqueeze(1)

        if cine_vid.shape[0] > self.num_frames:
            cine_vid = cine_vid[:self.num_frames]
        else:
            cine_vid = cine_vid.repeat(ceil(self.num_frames / cine_vid.shape[0]), 1, 1, 1)
            cine_vid = cine_vid[:self.num_frames]

        # Move to correct device
        cine_vid = cine_vid.to(self.device)
        label = label.to(self.device)

        return cine_vid, label

    def __len__(self):
        """
        Provides length of the dataset

        Returns
        -------
            (int): Indicates the number of available samples for the task
        """

        return self.num_samples

    @staticmethod
    def loadvideo(filename):
        """
        Video loader code from https://github.com/echonet/dynamic/tree/master/echonet with some modifications

        Parameters
        ----------
        filename(str): path to file to extract

        Returns
        ----------
        Loaded video as a numpy array with shape (num_frames, channels, height, width)
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        capture = cv2.VideoCapture(filename)

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        v = np.zeros((frame_height, frame_width, frame_count), np.uint8)

        for count in range(frame_count):
            ret, frame = capture.read()
            if not ret:
                raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            v[:, :, count] = frame

        return v


class LVBiplaneEFDataset(Dataset):
    """
    Dataset class for the LV Biplane
    """

    def __init__(self,
                 datasets_root_path,
                 device=None,
                 num_frames=32,
                 raw_data_summary_csv='Biplane_LVEF_DataSummary.csv',
                 task='all_ef',
                 view='all_views'):
        """
        Constructor for the LV Biplane EF dataset class

        Parameters
        ----------
        datasets_root_path (str): Path to directory containing all datasets
        device (torch device): device to move data to
        num_frames (int): number of frames to use for each video (If video is shorter than this replicate it to fit)
        raw_data_summary_csv (str): CSV file containing data information
        task (str): task to create the dataset for (must be one of
                    [all_ef, high_risk_ef, medium_ef_risk, slight_ef_risk, normal_ef, esv, edv, quality])
        view (str): video view to make the dataset for (must be one of [all_views, ap2, ap4])
        """

        super().__init__()

        dataset_path = os.path.join(datasets_root_path, 'biplane_lvef')

        # Input checks
        assert task in ['all_ef',
                        'high_risk_ef',
                        'medium_ef_risk',
                        'slight_ef_risk',
                        'normal_ef',
                        'esv',
                        'edv'], 'Specified task is not supported'
        assert view in ['all_views',
                        'ap2',
                        'ap4'], 'Specified view is not supported'

        # Obtain data paths and labels
        # Load required CSV file
        reports_dir = os.path.join(dataset_path, 'Reports')
        text_report_paths = os.listdir(reports_dir)
        raw_data_summary = pd.read_csv(os.path.join(dataset_path, raw_data_summary_csv))

        # Define regex strings needed to find Accession number and EF value
        pattern = re.compile('(?<=Accession No:)\s*\d*|(?<=LV EF \(Biplane\):)\s*\d*|(?<=LV EDV index:)\s*\d*\.\d*|(?<=LV ESV index:)\s*\d*\.\d*')

        # Initialize variables
        self.labels = list()
        self.patient_data_dirs = list()

        # Go through each text report and obtain EF value if cine data exists
        for file in text_report_paths:

            label_file =open(os.path.join(reports_dir, file), errors='ignore')
            label_str = label_file.read()

            matches = pattern.findall(label_str)

            if len(matches) == 4:

                # Get labels based on task
                if task in ['all_ef', 'high_risk_ef', 'medium_ef_risk', 'slight_ef_risk', 'normal_ef']:
                    label = np.array(float(matches[1]), dtype=np.float32)
                elif task == 'esv':
                    label = np.array(float(matches[3]), dtype=np.float32)
                else:
                    label = np.array(float(matches[2]), dtype=np.float32)

                # Find the corresponding cine video path using Accession number
                match_df = raw_data_summary.loc[raw_data_summary['AccessionNumber'] ==
                                                int(matches[0])]

                # Append .mat to video paths (preprocessed versions are saved in .mat format)
                if match_df['path'].values.tolist():
                    patient_data_dir = os.path.splitext(os.path.basename(match_df['path'].values.tolist()[0]))[0]+'.mat'

                    if view == 'all_views':
                        patient_data_dir = os.path.join(dataset_path, patient_data_dir)
                        if os.path.exists(patient_data_dir):
                            add_sample_to_dataset_for_task(patient_dirs=self.patient_data_dirs,
                                                           labels=self.labels,
                                                           label_to_add=label,
                                                           path_to_add=patient_data_dir,
                                                           task=task)
                    elif view == 'ap2' and patient_data_dir[0:3] == 'AP2':
                        patient_data_dir = os.path.join(dataset_path, patient_data_dir)
                        if os.path.exists(patient_data_dir):
                            add_sample_to_dataset_for_task(patient_dirs=self.patient_data_dirs,
                                                           labels=self.labels,
                                                           label_to_add=label,
                                                           path_to_add=patient_data_dir,
                                                           task=task)
                    elif view == 'ap4' and patient_data_dir[0:3] == 'AP4':
                        patient_data_dir = os.path.join(dataset_path, patient_data_dir)
                        if os.path.exists(patient_data_dir):
                            add_sample_to_dataset_for_task(patient_dirs=self.patient_data_dirs,
                                                           labels=self.labels,
                                                           label_to_add=label,
                                                           path_to_add=patient_data_dir,
                                                           task=task)

        # Extract the number of available data
        self.num_samples = len(self.patient_data_dirs)

        # Normalization operation
        self.trans = Compose([ToTensor(),
                              Normalize((0.5), (0.5))])

        # Other attributes
        self.device = device
        self.task = task
        self.num_frames = num_frames

    def __getitem__(self, idx):
        """
        Get the sample video and label at idx

        Parameters
        ----------
        idx (int): Index to fetch data for

        Returns
        -------
        (Torch tensors) for video and corresponding label
        """

        # Get the label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # extract video
        cine_vid = loadmat(self.patient_data_dirs[idx], simplify_cells=True)['resized']
        cine_vid = self.trans(np.array(cine_vid, dtype=np.uint8)).unsqueeze(1)

        if cine_vid.shape[0] > self.num_frames:
            cine_vid = cine_vid[:self.num_frames]
        else:
            cine_vid = cine_vid.repeat(ceil(self.num_frames / cine_vid.shape[0]), 1, 1, 1)
            cine_vid = cine_vid[:self.num_frames]

        # Move to correct device
        cine_vid = cine_vid.to(self.device)
        label = label.to(self.device)

        return cine_vid, label

    def __len__(self):
        """
        Provides length of the dataset

        Returns
        -------
            (int): Indicates the number of available samples for the task
        """

        return self.num_samples


class DelEfDataset(Dataset):
    """
    Dataset class for the internal Del LVEF dataset
    """

    def __init__(self,
                 datasets_root_path,
                 device=None,
                 num_frames=250,
                 task='all_ef',
                 view='all_views'):
        """
        Constructor for the Del LVEF dataset class

        Parameters
        ----------
        datasets_root_path (str): Path to directory containing all datasets
        device (torch device): device to move data to
        num_frames (int): number of frames to use for each video (If video is shorter than this replicate it to fit)
        task (str): task to create the dataset for (must be one of
                    [all_ef, high_risk_ef, medium_ef_risk, slight_ef_risk, normal_ef])
        view (str): video view to make the dataset for (must be one of [all_views, ap2, ap4])
        """

        super().__init__()

        dataset_path = os.path.join(datasets_root_path, 'del_lvef')

        # Input checks
        assert task in ['all_ef',
                        'high_risk_ef',
                        'medium_ef_risk',
                        'slight_ef_risk',
                        'normal_ef'], 'Specified task is not supported'
        assert view in ['all_views',
                        'ap2',
                        'ap4'], 'Specified view is not supported'

        # Obtain path to data .mat files based on what view is needed
        mat_paths = list()
        if view == 'ap2' or view == 'all_views':
            temp_paths = os.listdir(os.path.join(dataset_path, 'AP2'))
            temp_paths = [os.path.join(dataset_path, 'AP2', mat_path) for mat_path in temp_paths]
            mat_paths += temp_paths

        if view == 'ap4'or view == 'all_views':
            temp_paths = os.listdir(os.path.join(dataset_path, 'AP4'))
            temp_paths = [os.path.join(dataset_path, 'AP4', mat_path) for mat_path in temp_paths]
            mat_paths += temp_paths

        # Initialize variables
        self.labels = list()
        self.patient_data_dirs = list()

        # Go through each mat file and obtain EF value
        for mat_path in mat_paths:

            # Extract EF label
            label = np.array(loadmat(mat_path, simplify_cells=True)['labels']['vEF'], dtype=np.float32)

            # Add sample if it's for the correct task
            add_sample_to_dataset_for_task(patient_dirs=self.patient_data_dirs,
                                           labels=self.labels,
                                           label_to_add=label,
                                           path_to_add=mat_path,
                                           task=task)

        # Extract the number of available data
        self.num_samples = len(self.patient_data_dirs)

        # Normalization operation
        self.trans = Compose([ToTensor(),
                              Normalize((0.5), (0.5))])

        # Other attributes
        self.device = device
        self.task = task
        self.num_frames = num_frames

    def __getitem__(self, idx):
        """
        Get the sample video and label at idx

        Parameters
        ----------
        idx (int): Index to fetch data for

        Returns
        -------
        (Torch tensors) for video and corresponding label
        """

        # Get the label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # extract video
        cine_vid = loadmat(self.patient_data_dirs[idx], simplify_cells=True)['resized']
        cine_vid = self.trans(np.array(cine_vid, dtype=np.uint8)).unsqueeze(1)

        if cine_vid.shape[0] > self.num_frames:
            cine_vid = cine_vid[:self.num_frames]
        else:
            cine_vid = cine_vid.repeat(ceil(self.num_frames / cine_vid.shape[0]), 1, 1, 1)
            cine_vid = cine_vid[:self.num_frames]

        # Move to correct device
        cine_vid = cine_vid.to(self.device)
        label = label.to(self.device)

        return cine_vid, label

    def __len__(self):
        """
        Provides length of the dataset

        Returns
        -------
            (int): Indicates the number of available samples for the task
        """

        return self.num_samples


class NatEfDataset(Dataset):
    """
    Dataset class for the internal Nat LVEF dataset
    """

    def __init__(self,
                 datasets_root_path,
                 device=None,
                 num_frames=250,
                 task='all_ef',
                 view='all_views'):
        """
        Constructor for the Nat LVEF dataset class

        Parameters
        ----------
        datasets_root_path (str): Path to directory containing all datasets
        device (torch device): device to move data to
        num_frames (int): number of frames to use for each video (If video is shorter than this replicate it to fit)
        task (str): task to create the dataset for (must be one of
                    [all_ef, high_risk_ef, medium_ef_risk, slight_ef_risk, normal_ef, quality])
        view (str): video view to make the dataset for (must be one of [all_views, ap2, ap4])
        """

        super().__init__()

        dataset_path = os.path.join(datasets_root_path, 'nat_lvef')

        # Input checks
        assert task in ['all_ef',
                        'high_risk_ef',
                        'medium_ef_risk',
                        'slight_ef_risk',
                        'normal_ef',
                        'quality'], 'Specified task is not supported'
        assert view in ['all_views',
                        'ap2',
                        'ap4'], 'Specified view is not supported'

        # Obtain path to data .mat files based on what view is needed
        mat_paths = list()
        if view == 'ap2' or view == 'all_views':
            temp_paths = os.listdir(os.path.join(dataset_path, 'AP2'))
            temp_paths = [os.path.join(dataset_path, 'AP2', mat_path) for mat_path in temp_paths]
            mat_paths += temp_paths

        if view == 'ap4'or view == 'all_views':
            temp_paths = os.listdir(os.path.join(dataset_path, 'AP4'))
            temp_paths = [os.path.join(dataset_path, 'AP4', mat_path) for mat_path in temp_paths]
            mat_paths += temp_paths

        # Initialize variables
        self.labels = list()
        self.patient_data_dirs = list()

        # Go through each mat file and obtain EF value
        for mat_path in mat_paths:

            # Extract EF label
            if task in ['all_ef', 'high_risk_ef', 'medium_ef_risk', 'slight_ef_risk', 'normal_ef']:
                label = np.array(loadmat(mat_path, simplify_cells=True)['labels']['vEF'], dtype=np.float32)
            else:
                label = np.array(loadmat(mat_path, simplify_cells=True)['labels']['Quality'], dtype=np.float32)

            if label != -1 and label != 0:
                # Add sample if it's for the correct task
                add_sample_to_dataset_for_task(patient_dirs=self.patient_data_dirs,
                                               labels=self.labels,
                                               label_to_add=label,
                                               path_to_add=mat_path,
                                               task=task)

        # Extract the number of available data
        self.num_samples = len(self.patient_data_dirs)

        # Normalization operation
        self.trans = Compose([ToTensor(),
                              Normalize((0.5), (0.5))])

        # Other attributes
        self.device = device
        self.task = task
        self.num_frames = num_frames

    def __getitem__(self, idx):
        """
        Get the sample video and label at idx

        Parameters
        ----------
        idx (int): Index to fetch data for

        Returns
        -------
        (Torch tensors) for video and corresponding label
        """

        # Get the label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # extract video
        cine_vid = loadmat(self.patient_data_dirs[idx], simplify_cells=True)['resized']
        cine_vid = self.trans(np.array(cine_vid, dtype=np.uint8)).unsqueeze(1)

        if cine_vid.shape[0] > self.num_frames:
            cine_vid = cine_vid[:self.num_frames]
        else:
            cine_vid = cine_vid.repeat(ceil(self.num_frames / cine_vid.shape[0]), 1, 1, 1)
            cine_vid = cine_vid[:self.num_frames]

        # Move to correct device
        cine_vid = cine_vid.to(self.device)
        label = label.to(self.device)

        return cine_vid, label

    def __len__(self):
        """
        Provides length of the dataset

        Returns
        -------
            (int): Indicates the number of available samples for the task
        """

        return self.num_samples


class PocEfDataset(Dataset):
    """
    Dataset class for the internal Poc LVEF dataset
    """

    def __init__(self,
                 datasets_root_path,
                 device=None,
                 num_frames=250,
                 task='all_ef',
                 view='all_views'):
        """
        Constructor for the Nat Poc dataset class

        Parameters
        ----------
        datasets_root_path (str): Path to directory containing all datasets
        device (torch device): device to move data to
        num_frames (int): number of frames to use for each video (If video is shorter than this replicate it to fit)
        task (str): task to create the dataset for (must be one of
                    [all_ef, high_risk_ef, medium_ef_risk, slight_ef_risk, normal_ef, quality])
        view (str): video view to make the dataset for (must be one of [all_views, ap2, ap4])
        """

        super().__init__()

        dataset_path = os.path.join(datasets_root_path, 'poc_lvef')

        # Input checks
        assert task in ['all_ef',
                        'high_risk_ef',
                        'medium_ef_risk',
                        'slight_ef_risk',
                        'normal_ef',
                        'quality'], 'Specified task is not supported'
        assert view in ['all_views',
                        'ap2',
                        'ap4'], 'Specified view is not supported'

        # Obtain path to data .mat files based on what view is needed
        mat_paths = list()
        if view == 'ap2' or view == 'all_views':
            temp_paths = os.listdir(os.path.join(dataset_path, 'AP2'))
            temp_paths = [os.path.join(dataset_path, 'AP2', mat_path) for mat_path in temp_paths]
            mat_paths += temp_paths

        if view == 'ap4'or view == 'all_views':
            temp_paths = os.listdir(os.path.join(dataset_path, 'AP4'))
            temp_paths = [os.path.join(dataset_path, 'AP4', mat_path) for mat_path in temp_paths]
            mat_paths += temp_paths

        # Initialize variables
        self.labels = list()
        self.patient_data_dirs = list()

        # Go through each mat file and obtain EF value
        for mat_path in mat_paths:

            # Extract EF label
            if task in ['all_ef', 'high_risk_ef', 'medium_ef_risk', 'slight_ef_risk', 'normal_ef']:
                label = np.array(loadmat(mat_path, simplify_cells=True)['labels']['vEF'], dtype=np.float32)
            else:
                label = np.array(loadmat(mat_path, simplify_cells=True)['labels']['Quality'], dtype=np.float32)

            if not isnan(label) and label != 0 and label != -1:
                # Add sample if it's for the correct task
                add_sample_to_dataset_for_task(patient_dirs=self.patient_data_dirs,
                                               labels=self.labels,
                                               label_to_add=label,
                                               path_to_add=mat_path,
                                               task=task)

        # Extract the number of available data
        self.num_samples = len(self.patient_data_dirs)

        # Normalization operation
        self.trans = Compose([ToTensor(),
                              Normalize((0.5), (0.5))])

        # Other attributes
        self.device = device
        self.task = task
        self.num_frames = num_frames

    def __getitem__(self, idx):
        """
        Get the sample video and label at idx

        Parameters
        ----------
        idx (int): Index to fetch data for

        Returns
        -------
        (Torch tensors) for video and corresponding label
        """

        # Get the label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # extract video
        cine_vid = loadmat(self.patient_data_dirs[idx], simplify_cells=True)['resized']
        cine_vid = self.trans(np.array(cine_vid, dtype=np.uint8)).unsqueeze(1)

        if cine_vid.shape[0] > self.num_frames:
            cine_vid = cine_vid[:self.num_frames]
        else:
            cine_vid = cine_vid.repeat(ceil(self.num_frames / cine_vid.shape[0]), 1, 1, 1)
            cine_vid = cine_vid[:self.num_frames]

        # Move to correct device
        cine_vid = cine_vid.to(self.device)
        label = label.to(self.device)

        return cine_vid, label

    def __len__(self):
        """
        Provides length of the dataset

        Returns
        -------
            (int): Indicates the number of available samples for the task
        """

        return self.num_samples


if __name__ == '__main__':

    # Example code on usage of datasets
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = NatEfDataset(datasets_root_path='D:/Workspace/RCL/datasets/preprocessed/',
                           device=device,
                           num_frames=32,
                           task='quality',
                           view='all_views')

    dataloader = DataLoader(dataset, batch_size=3, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)

    for data, label in dataloader:
        print(label)
