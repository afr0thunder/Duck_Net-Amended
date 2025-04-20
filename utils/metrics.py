import torch

def dice_coef(y_true, y_pred, smooth=1.e-9):
    """
    Calculate dice coefficient.
    Input shape should be Batch x #Classes x Height x Width (BxNxHxW).
    Using Mean as reduction type for batch values.
    """
    intersection = torch.sum(y_true * y_pred, dim=(2, 3))
    union = torch.sum(y_true, dim=(2, 3)) + torch.sum(y_pred, dim=(2, 3))
    return torch.mean((2. * intersection + smooth) / (union + smooth))

def jaccard_index(y_true, y_pred, smooth=1.e-9):
    """
    Calculate jaccard index.
    Input shape should be Batch x #Classes x Height x Width (BxNxHxW).
    Using Mean as reduction type for batch values.
    """
    true_positive = torch.sum(y_true * y_pred, dim=(2, 3))
    false_positive = torch.sum(y_pred, dim=(2, 3)) - true_positive
    false_negative = torch.sum(y_true, dim=(2, 3)) - true_positive
    return torch.mean((true_positive + smooth) / (true_positive + false_negative + false_positive + smooth))

def iou(y_true, y_pred, smooth=1.e-9):
    """
    Calculate IoU index.
    Input shape should be Batch x #Classes x Height x Width (BxNxHxW).
    Using Mean as reduction type for batch values.
    """
    intersection = torch.sum(y_true * y_pred, dim=(2, 3))
    union = torch.sum(y_true, dim=(2, 3)) + torch.sum(y_pred, dim=(2, 3)) - intersection
    return torch.mean((intersection + smooth) / (union + smooth))

def precision(y_true, y_pred, smooth=1.e-9):
    """
    Calculate precision.
    Input shape should be Batch x #Classes x Height x Width (BxNxHxW).
    Using Mean as reduction type for batch values.
    """
    true_positive = torch.sum(y_true * y_pred, dim=(2, 3))
    false_positive = torch.sum(y_pred, dim=(2, 3)) - true_positive
    return torch.mean((true_positive + smooth) / (true_positive + false_positive + smooth))

def recall(y_true, y_pred, smooth=1.e-9):
    """
    Calculate recall.
    Input shape should be Batch x #Classes x Height x Width (BxNxHxW).
    Using Mean as reduction type for batch values.
    """
    true_positive = torch.sum(y_true * y_pred, dim=(2, 3))
    false_negative = torch.sum(y_true, dim=(2, 3)) - true_positive
    return torch.mean((true_positive + smooth) / (true_positive + false_negative + smooth))

def accuracy(y_true, y_pred, smooth=1e-9):
    """
    Calculate accuracy
    Input shape should be Batch x #Classes x Height x Width (BxNxHxW).
    Using Mean as reduction type for batch values.
    """
    true_positive = y_true * y_pred
    true_negative = (1 - y_true) * (1 - y_pred)
    total = torch.ones_like(y_true)

    numerator = torch.sum(true_positive + true_negative, dim=(2, 3))
    denominator = torch.sum(total, dim=(2, 3))

    return torch.mean((numerator + smooth) / (denominator + smooth))