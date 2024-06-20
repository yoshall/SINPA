class StandardScaler:
    """
    Standardizes the input data using mean and standard deviation.

    Parameters:
    - mean (float): The mean value used for standardization.
    - std (float): The standard deviation value used for standardization.
    """

    def __init__(self, mean, std):
        print(mean)
        print(std)
        self.mean = mean
        self.std = std

    def transform(self, data):
        """
        Transforms the input data by subtracting the mean and dividing by the standard deviation.

        Parameters:
        - data (numpy.ndarray): The input data to be transformed.

        Returns:
        - numpy.ndarray: The transformed data.
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Inverse transforms the input data by multiplying with the standard deviation and adding the mean.

        Parameters:
        - data (numpy.ndarray): The input data to be inverse transformed.

        Returns:
        - numpy.ndarray: The inverse transformed data.
        """
        return (data * self.std) + self.mean
