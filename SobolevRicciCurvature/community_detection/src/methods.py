from src.flow import SRCFlowMethod, ORCFlowMethod

def make_method(method, 
                steps, 
                alpha, 
                *, 
                method_kwargs=None):
    '''
    Factory function to create flow method instances.
    - Inputs:
        - method: str, method name ("src-spt", "src-mst", "orc")
        - steps: int, number of flow steps
        - alpha: float, flow parameter
        - method_kwargs: dict, additional method-specific parameters
    - Returns:
        - An instance of the specified flow method class.
    '''
    method_kwargs = method_kwargs or {} 
    method = method.lower().strip()

    if method == "src-spt" or method == "src-mst":
        p = float(method_kwargs["p"])
        src_root = int(method_kwargs.get("src_root", 0))
        print(f"SRC parameters: p={p}, src_root={src_root}")

        return SRCFlowMethod(
            delta=alpha,
            steps=steps,
            p=p,
            method=method,
            src_root=src_root,
        )

    elif method == "orc":
        max_iter = int(method_kwargs.get("max_iter", 200))
        orc_eta = float(method_kwargs.get("orc_eta", 0.1))

        return ORCFlowMethod(
            alpha=alpha,
            steps=steps,
            orc_eta=orc_eta,
            max_iter=max_iter,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
