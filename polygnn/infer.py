"""A module for running inference on models trained using the polygnn package."""

import numpy as np

from polygnn_trainer import infer as pt_infer


def init_evaluation(model, include_polymer_fps=False):
    """
    This function initializes the evaluation process for a model.

    Args:
        model (nn.Module): The model to be evaluated
        include_polymer_fps (bool): Flag to include fingerprints in the
            prediction result

    Returns:
        y, y_hat_mean, y_hat_std, y_selectors, polymer_fps
    """
    output = pt_infer.init_evaluation(model)

    if include_polymer_fps:
        polymer_fps = []
    else:
        polymer_fps = None

    return (*output, polymer_fps)


def eval_submodel(
    model, val_loader, device, selector_dim=None, include_polymer_fps=False
):
    """
    An alias to _evaluate in which is_submodel is set to True.
    """
    return _evaluate(
        model,
        val_loader,
        device,
        is_submodel=True,
        selector_dim=selector_dim,
        include_polymer_fps=include_polymer_fps,
    )


def _evaluate(
    model, val_loader, device, is_submodel, selector_dim, include_polymer_fps, **kwargs
):
    """
    Evaluate model on the data contained in val_loader. This function is not
    to be called directly. It is a helper function for eval_submodel.

    Args:
        model (nn.Module): The model to be evaluated
        val_loader (DataLoader): The data to evaluate
        device (torch.device): The device to run the model on
        is_submodel (bool): Flag to indicate if the model is a submodel
        selector_dim (int): The number of selector dimensions
        include_polymer_fps (bool): Flag to include fingerprints in the
            prediction result
        **kwargs: Arguments to pass into the 'forward' method of model

    Returns:
        (tuple): A tuple containing four elements. The first element is an np.ndarray
            of the data labels. The second element is an np.ndarray of the mean
            of the data predictions. The third element is an np.ndarray of the selectors
            for the data points. The fourth element is an np.ndarray of the
            fingerprints.
    """

    (
        y_val,
        y_val_hat_mean,
        y_val_hat_std,
        selectors,
        polymer_fps,
    ) = init_evaluation(model, include_polymer_fps)

    # Loop through validation batches
    for ind, data in enumerate(val_loader):
        data = data.to(device)

        # Obtain the fingerprints
        if include_polymer_fps:
            polymer_fps.extend(
                model.get_polymer_fps(data).detach().cpu().numpy().tolist()
            )

        # Sometimes the batch may have labels associated. Let's check
        if data.y is not None:
            y_val += data.y.detach().flatten().cpu().numpy().tolist()

        # Sometimes the batch may have selectors associated. Let's check
        if selector_dim:
            selectors += data.selector.cpu().numpy().tolist()
        if is_submodel:
            output = model(data).view(
                data.num_graphs,
            )
            y_val_hat_mean += output.flatten().detach().cpu().numpy().tolist()

        # If we are not dealing with a submodel then we have an ensemble.
        # The ensemble will have two outputs: the mean and standard deviation.
        else:
            mean, std = model(data, **kwargs)
            y_val_hat_mean += mean.flatten().detach().cpu().numpy().tolist()
            y_val_hat_std += std.flatten().detach().cpu().numpy().tolist()

    del data  # free memory

    if is_submodel:
        return (
            np.array(y_val),
            np.array(y_val_hat_mean),
            selectors,
            np.array(polymer_fps),
        )
    else:
        return (
            np.array(y_val),
            np.array(y_val_hat_mean),
            np.array(y_val_hat_std),
            selectors,
            np.array(polymer_fps),
        )
