from torch.nn import Linear, ReLU, Sequential


def get_road_sign_fc(num_ftrs=512):
    return Sequential(
        Linear(num_ftrs, 256),
        ReLU(),
        Linear(256, 256),
        ReLU(),
        Linear(256, 2),
    )
