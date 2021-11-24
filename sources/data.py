import re
import os
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize
import SimpleITK as sitk
import torch
import cv2


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

    #TODO: Add image quality (if separate task for each image quality)


class CamusEfDataset(Dataset):
    """
    Dataset class for the Camus EF dataset found at https://www.creatis.insa-lyon.fr/Challenge/camus/
    """

    def __init__(self,
                 dataset_path,
                 image_shape,
                 device=None,
                 task='all_ef',
                 view='all_views'):
        """
        Constructor for the Camus EF dataset dataset class

        Parameters
        ----------
        dataset_path (str): Path to directory containing data
        image_shape (int): Shape to resize images to
        device (torch device): device to move data to
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
            elif task == 'quality':
                label_pattern = re.compile('(?<=(ImageQuality: ))(.*)')

            # Text file path to extract the label from
            label_path = os.path.join(dataset_path, patient, 'Info_2CH.cfg')
            label_file = open(label_path)
            label_str = label_file.read()
            label_file.close()

            # Find the label
            match = float(label_pattern.findall(label_str)[0])/100

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

    def get(self, idx):
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
        label = self.labels[idx]

        # extract video
        cine_vid = self.process_cine(sitk.ReadImage(self.patient_data_dirs[idx]), size=self.image_shape)
        cine_vid = self.trans(np.array(cine_vid, dtype=np.uint8))

        # Move to correct device
        cine_vid.to(self.device)
        label.to(self.device)

        return cine_vid, label

    def len(self):
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