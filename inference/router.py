def get_solver(strategy):
    if strategy == "beam":
        from beam_search import run_beam_search
        return run_beam_search
    elif strategy == "standard":
        from solve import run_solve
        return run_solve
    raise ValueError("Unknown strategy")

def standalone_inference(checkpoint_path, input_data, strategy="beam"):
    solver = get_solver(strategy)
    return solver(checkpoint_path, input_data)