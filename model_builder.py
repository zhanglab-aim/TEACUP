import torch

from amber.architect import Operation

from model_bench201.tiny_network import TinyNetwork


def build_bench201_model(arc: str, data_info: dict, lr: float=0.1, weight_decay: float=0.0005,
                         momentum: float=0.9, width: int=16, n_cell: int=5):
    input_op = Operation('input', shape=data_info['input_shape'], name="input")
    output_op = Operation('dense', units=data_info['output_shape'], activation=data_info['output_func'], name="output")

    arc = "%s_%s_%s"%(arc[0], arc[1:3], arc[3:])

    bench201_model_builder = TinyNetwork(
        inputs_op=input_op,
        output_op=output_op,
        model_compile_dict={
            'loss': data_info['loss_func'],
            'optimizer': (
                torch.optim.SGD, {"lr": lr, "weight_decay": weight_decay, "momentum": momentum},
                torch.optim.lr_scheduler.CosineAnnealingLR, {"T_max": 100}
            ),
            # 'metrics': [data_info['reward_func']],
        },
        C=width, N=n_cell,
        dim=len(data_info['input_shape'])-1,
        verbose=False
    )
    model = bench201_model_builder(arc)

    return model


def model_builder(model_space: str, arc: str, data_info: dict, lr: float=0.1, weight_decay: float=0.0005, 
                  momentum: float=0.9, width: int=16, n_cell: int=5, verbose: bool=False):

    assert model_space.startswith('bench201'), f"Couldn't support {model_space} at this time."
    model = build_bench201_model(arc, data_info, lr=lr, weight_decay=weight_decay,
                                 momentum=momentum, width=width, n_cell=n_cell)

    model = model.double()

    return model