def decoder(encoder_output: List, compiled_model_de: CompiledModel) -> List:
    """
    Decoder inference per set of frames. This function concatenates the embedding layer
    forms the encorder output, transpose the array to match with the decoder input size.
    Calls the network previously configured for the decoder model (compiled_model_de), extracts
    the logits and normalize those to get confidence values along specified axis.
    Decodes top probabilities into corresponding label names
    
    :param: encoder_output: embedding layer for 16 frames
    :param: compiled_model_de: Decoder model network
    :returns: decoded_labels: The k most probable actions from the labels list
              decoded_top_probs: confidence for the k most probable actions
    """
    # Concatenate sample_duration frames in just one array
    decoder_input = np.concatenate(encoder_output, axis=0)
    # Organize input shape vector to the Decoder (shape: [1x16x512]]
    decoder_input = decoder_input.transpose((2, 0, 1, 3))
    decoder_input = np.squeeze(decoder_input, axis=3)
    output_key_de = compiled_model_de.output(0)
    # Get results on action-recognition-0001-decoder model
    result_de = compiled_model_de([decoder_input])[output_key_de]
    # Normalize logits to get confidence values along specified axis
    probs = softmax(results_de - np.max(result_de))
    # Decodes top probabilities into corresponding label names
    decoded_labels, decoded_top_probs = decode_output(probs, labels, top_k=3)
    return decoded_labels, decoded_top_probs