def preprocessing(frame: np.ndarray, size: int) -> np.ndarray:
    """
    Preparing frame before Encoder.
    The image should be scaled to its shortest dimension at "size"
    and cropped, centered, and squared so that both width and 
    height have lengths "size". Frame must be transposed from
    Height-Width-Channels (HWCs) to Channels-Height-Width (CHW).
    
    :param frame: input frame
    :param size: input size to encoder model
    :returns: resized and cropped frame
    """
    # Adapative resize 
    preprocessed = adaptive_resize(frame, size)
    # Center_crop
    (preprocessed, roi) = center_crop(preprocessed)
    # Transpose frame HWC -> CHW
    preprocessed = preprocessed.transpose((2, 0, 1))[None,] # HWC -> CHW
    return preprocessed, roi