from torch.nn import Linear, ReLU, Sequential


def get_road_sign_fc(num_ftrs: int = 512) -> Sequential:
    return Sequential(
        Linear(num_ftrs, 256),
        ReLU(),
        Linear(256, 256),
        ReLU(),
        Linear(256, 2),
    )
