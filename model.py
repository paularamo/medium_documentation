# Initialize inference engine
ie_core = Core()

def model_init(model_path: str) -> Tuple:
    """
    Read the network and weights from file, load the 
    model on the CPU and get input and output names of nodes
    
    :param: model: model architecture path *.xml
    :returns:
             compiled_model: Compiled model
             input_key: Input node for model
             output_key: Output node for model
    """
    
    # Read the network and corresponding weights from file
    model = ie_core.read_model(model=model_path)
    # compile the model for the CPU (you can use GPU or MYRIAD as well)
    compiled_model = ie_core.compile_model(model=model, device_name="CPU")
    #Get input and output names of nodes
    input_keys = compiled_model.input(0)
    output_keys = compiled_model.output(0)
    return input_keys, output_keys, compiled_model