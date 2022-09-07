from tensorflow import keras


def create_data_aug_layer(data_aug_layer):
    """
    Use this function to parse the data augmentation methods for the
    experiment and create the corresponding layers.

    It will be mandatory to support at least the following three data
    augmentation methods (you can add more if you want):
        - `random_flip`: keras.layers.RandomFlip()
        - `random_rotation`: keras.layers.RandomRotation()
        - `random_zoom`: keras.layers.RandomZoom()

    See https://tensorflow.org/tutorials/images/data_augmentation.

    Parameters
    ----------
    data_aug_layer : dict
        Data augmentation settings coming from the experiment YAML config
        file.

    Returns
    -------
    data_augmentation : keras.Sequential
        Sequential model having the data augmentation layers inside.
    """

    data_aug_layers = []

    arguments = {"random_flip":keras.layers.RandomFlip,
                 "random_rotation":keras.layers.RandomRotation,
                 "random_zoom":keras.layers.RandomZoom,
                 "random_contrast":keras.layers.RandomContrast
    }

    if data_aug_layer:
        for key in data_aug_layer:
            if key in arguments:
                layer = arguments[key](**data_aug_layer[key])
                data_aug_layers.append(layer)

    data_augmentation = keras.Sequential(data_aug_layers)

    return data_augmentation
