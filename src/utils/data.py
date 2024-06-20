import torch

# from torch.utils.data import Tuple
from torch import Tensor
import pandas as pd
import numpy as np


class Dataset(torch.utils.data.Dataset):
    """
    A custom dataset class for loading data and labels from CSV files.

    Args:
        labelsFile (str): The path to the CSV file containing the labels.
        rootDir (str): The root directory where the data and label files are located.
        scalers (list): A list of scalers to be applied to each channel of the data and labels.
        output_dim (int): The number of output dimensions.
    """

    def __init__(self, labelsFile, rootDir, scalers, output_dim):
        self.data = pd.read_csv(labelsFile)
        self.rootDir = rootDir
        self.scalers = scalers
        self.output_dim = output_dim

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a sample and its corresponding label from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the sample and its corresponding label.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        samplePath = self.rootDir + "/" + self.data["sample_path"][idx]
        sample = np.load(samplePath)
        labelPath = self.rootDir + "/" + self.data["label_path"][idx]
        label = np.load(labelPath)

        for i in range(self.output_dim):
            sample[..., i] = self.scalers[i].transform(sample[..., i])
            label[..., i] = self.scalers[i].transform(label[..., i])

        return Tensor(sample), Tensor(label)


# class TensorDataset(Dataset[Tuple[Tensor, ...]]):
#     r"""Dataset wrapping tensors.

#     Each sample will be retrieved by indexing tensors along the first dimension.

#     Args:
#         *tensors (Tensor): tensors that have the same size of the first dimension.

#     Attributes:
#         tensors (Tuple[Tensor, ...]): The tensors that make up the dataset.

#     """

#     tensors: Tuple[Tensor, ...]

#     def __init__(self, *tensors: Tensor) -> None:
#         assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
#         self.tensors = tensors

#     def __getitem__(self, index):
#         return tuple(tensor[index] for tensor in self.tensors)

#     def __len__(self):
#         return self.tensors[0].size(0)
