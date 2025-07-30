import torch
import numpy as np
import pandas as pd
import os


def masked_mse(preds, labels, null_val=np.nan, mask=None):
    """
    Calculates the mean squared error (MSE) between the predicted values and the labels,
    considering only the non-null values specified by the mask.

    Args:
        preds (torch.Tensor): The predicted values.
        labels (torch.Tensor): The true labels.
        null_val (float, optional): The null value used to identify missing or invalid values. Defaults to np.nan.
        mask (torch.Tensor, optional): The mask indicating which values to consider. If None, it will be automatically
            generated based on the null_val. Defaults to None.

    Returns:
        torch.Tensor: The mean squared error between the predicted values and the labels, considering only the non-null values.
    """
    if mask == None:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = labels > null_val + 0.1  # +0.1 for potential numerical errors

    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan, mask=None):
    """
    Calculate the root mean squared error (RMSE) between predicted values and true labels,
    considering a mask to exclude certain values.

    Args:
        preds (torch.Tensor): Predicted values.
        labels (torch.Tensor): True labels.
        null_val (float, optional): Value to be considered as null. Defaults to np.nan.
        mask (torch.Tensor, optional): Mask to exclude certain values. Defaults to None.

    Returns:
        torch.Tensor: The root mean squared error (RMSE) between preds and labels.
    """
    if mask == None:
        return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))
    else:
        return torch.sqrt(
            masked_mse(preds=preds, labels=labels, null_val=null_val, mask=mask)
        )


def masked_mae(preds, labels, null_val=np.nan, mask=None):
    """
    Calculates the mean absolute error (MAE) between the predicted values and the true labels,
    taking into account a mask to exclude certain values.

    Args:
        preds (torch.Tensor): The predicted values.
        labels (torch.Tensor): The true labels.
        null_val (float, optional): The null value used in the mask. Defaults to np.nan.
        mask (torch.Tensor, optional): The mask to exclude certain values. Defaults to None.

    Returns:
        torch.Tensor: The masked mean absolute error (MAE) between the predicted values and the true labels.
    """
    if mask == None:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = labels > null_val + 0.1  # +0.1 for potential numerical errors
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan, mask=None):
    """
    Calculate the masked mean absolute percentage error (MAPE) between predictions and labels.

    Args:
        preds (torch.Tensor): The predicted values.
        labels (torch.Tensor): The true values.
        null_val (float, optional): The null value used for masking. Defaults to np.nan.
        mask (torch.Tensor, optional): The mask tensor. If not provided, it will be automatically generated based on the null_val. Defaults to None.

    Returns:
        torch.Tensor: The masked MAPE loss.

    """
    if mask == None:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = labels > null_val + 0.1  # +0.1 for potential numerical errors
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / (labels + 0.1)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def compute_all_metrics(pred, real, null_value=np.nan):
    """
    Compute multiple metrics for evaluating the performance of a prediction.

    Args:
        pred (numpy.ndarray): The predicted values.
        real (numpy.ndarray): The ground truth values.
        null_value (float, optional): The value used to represent missing or invalid data. Defaults to np.nan.

    Returns:
        tuple: A tuple containing the computed metrics (mae, rmse).

    """
    mae = masked_mae(pred, real, null_value).item()
    rmse = masked_rmse(pred, real, null_value).item()
    return mae, rmse