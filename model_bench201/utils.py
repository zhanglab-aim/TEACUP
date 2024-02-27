from torch import nn

MODE = 'fan_in'

def dag_str2code(dag):
    # list of to_node (list of in_node). 0: broken; 1: skip-connect; 2: linear or conv
    # e.g. "2_02_002" => [[2], [0, 2], [0, 0, 2]]
    if isinstance(dag, str):
        dag = [[int(edge) for edge in node] for node in dag.split('_')]
    elif isinstance(dag, list):
        assert isinstance(dag[0], list) and len(dag[0]) == 1 # 2nd node has one in-degree
        for i in range(1, len(dag)):
            assert len(dag[i]) == len(dag[i-1]) + 1 # next node has one more in-degree than prev node
    return dag


def _init(model, mode='fan_in'):
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0)


class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x * 0
