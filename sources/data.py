import re
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
import SimpleITK as sitk
import cv2
import torch
from torch.nn.utils.rnn import pad_sequence


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
                          path_to_add,
                          view):
    """
    Add sample to dataset based on what view is required

    Parameters
    ----------
    patient_dirs (list): List containing directory path for echo videos
    labels (list): List containing labels
    label_to_add (float or int): Label to add to the labels list
    path_to_add (str): Path to patient directory
    view (str): View to add

    """

    labels.append(label_to_add)

    # Need to replicate the label if both video views are needed
    if view == 'all_views':
        labels.append(label_to_add)

    # Add the patient video directory
    if view == 'all_views':
        patient_dirs.append(os.path.join(path_to_add + '_2CH_sequence.mhd'))
        patient_dirs.append(os.path.join(path_to_add + '_4CH_sequence.mhd'))
    elif view == 'ap2':
        patient_dirs.append(os.path.join(path_to_add + '_2CH_sequence.mhd'))
    elif view == 'ap4':
        patient_dirs.append(os.path.join(path_to_add + '_4CH_sequence.mhd'))


def add_sample_to_dataset_for_task(patient_dirs,
                                   labels,
                                   label_to_add,
                                   path_to_add,
                                   view,
                                   task):
    """
    Adds sample to dataset based on required task

    Parameters
    ----------
    patient_dirs (list): List containing directory path for echo videos
    labels (list): List containing labels
    label_to_add (float or int): Label to add to the labels list
    path_to_add (str): Path to patient directory
    view (str): View to add
    task (str): Task to add data samples for
    """

    if task == 'all_ef':
        add_sample_to_dataset(patient_dirs=patient_dirs,
                              labels=labels,
                              label_to_add=label_to_add,
                              path_to_add=path_to_add,
                              view=view)

    elif task == 'high_risk_ef':
        if label_to_add <= 0.35:
            add_sample_to_dataset(patient_dirs=patient_dirs,
                                  labels=labels,
                                  label_to_add=label_to_add,
                                  path_to_add=path_to_add,
                                  view=view)

    elif task == 'medium_ef_risk':
        if label_to_add <= 0.39:
            add_sample_to_dataset(patient_dirs=patient_dirs,
                                  labels=labels,
                                  label_to_add=label_to_add,
                                  path_to_add=path_to_add,
                                  view=view)

    elif task == 'slight_ef_risk':
        if label_to_add <= 0.54:
            add_sample_to_dataset(patient_dirs=patient_dirs,
                                  labels=labels,
                                  label_to_add=label_to_add,
                                  path_to_add=path_to_add,
                                  view=view)
    elif task == 'normal_ef':
        add_sample_to_dataset(patient_dirs=patient_dirs,
                              labels=labels,
                              label_to_add=label_to_add,
                              path_to_add=path_to_add,
                              view=view)

    # TODO: Add image quality (if separate task for each image quality)


class CamusEfDataset(Dataset):
    """
    Dataset class for the Camus EF dataset found at https://www.creatis.insa-lyon.fr/Challenge/camus/
    """

    def __init__(self,
                 dataset_path,
                 image_shape,
                 device=None,
                 num_frames=None,
                 task='all_ef',
                 view='all_views'):
        """
        Constructor for the Camus EF dataset dataset class

        Parameters
        ----------
        dataset_path (str): Path to directory containing data
        image_shape (int): Shape to resize images to
        device (torch device): device to move data to
        num_frames (int): Number of frames to use for each video (If video is shorter than this replica
        task (str): task to create the dataset for (must be one of
                    [all_ef, high_risk_ef, medium_ef_risk, slight_ef_risk, normal_ef, lvesv, lvedv, quality])
        view (str): video view to make the dataset for (must be one of [all_views, ap2, ap4])
        """

        super().__init__()

        # Input checks
        assert task in ['all_ef',
                        'high_risk_ef',
                        'medium_ef_risk',
                        'slight_ef_risk',
                        'normal_ef',
                        'lvesv',
                        'lvedv',
                        'quality'], 'Specified task is not supported'
        assert view in ['all_views'], 'Specified view is not supported'

        # Obtain file dirs
        data_dirs = os.listdir(dataset_path)

        # Extract file names and locations for each patient (both data and label) depending on task type
        self.labels = list()
        self.patient_data_dirs = list()
        for i, patient in enumerate(data_dirs):

            # Get the correct pattern to search for based on task
            if task in ['all_ef', 'high_risk_ef', 'medium_ef_risk', 'slight_ef_risk', 'normal_ef']:
                label_pattern = re.compile('(?<=(LVef: ))(.*)')
            elif task == 'lvesv':
                label_pattern = re.compile('(?<=(LVesv: ))(.*)')
            elif task == 'lvedv':
                label_pattern = re.compile('(?<=(LVedv: ))(.*)')
            else:
                label_pattern = re.compile('(?<=(ImageQuality: ))(.*)')

            # Text file path to extract the label from
            label_path = os.path.join(dataset_path, patient, 'Info_2CH.cfg')
            label_file = open(label_path)
            label_str = label_file.read()
            label_file.close()

            # Find the label
            match = float(label_pattern.findall(label_str)[0][1])/100

            # Add sample for task
            add_sample_to_dataset_for_task(patient_dirs=self.patient_data_dirs,
                                           labels=self.labels,
                                           label_to_add=match,
                                           path_to_add=os.path.join(dataset_path, patient, patient),
                                           view=view,
                                           task=task)

        # Extract the number of available studies/graphs
        self.num_samples = len(self.patient_data_dirs)

        # Normalization operation
        self.trans = Compose([ToTensor(),
                              Normalize((0.5), (0.5))])

        self.image_shape = image_shape

        # Other attributes
        self.device = device

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
        cine_vid = self.process_cine(sitk.ReadImage(self.patient_data_dirs[idx]), size=self.image_shape)
        cine_vid = self.trans(np.array(cine_vid, dtype=np.uint8))

        # Move to correct device
        cine_vid.to(self.device)
        label.to(self.device)

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


if __name__ == '__main__':

    # Example code on usage of datasets

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = CamusEfDataset(dataset_path='D:/Workspace/RCL/datasets/raw/camus',
                             image_shape=128,
                             device=device,
                             task='all_ef',
                             view='all_views')

    dataloader = DataLoader(dataset, batch_size=3, shuffle=False, drop_last=True, collate_fn=custom_collate_fn)

    for data, label in dataloader:
        print(label)