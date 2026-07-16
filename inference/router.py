def standalone_inference(checkpoint_path, input_data, strategy="beam"):
    from inference.solve import CalculusSolverInference
    
    inferencer = CalculusSolverInference(checkpoint_path)
    
    if strategy in ["beam", "standard"]:
        return inferencer.solve(input_data)
        
    raise ValueError("Unknown strategy")