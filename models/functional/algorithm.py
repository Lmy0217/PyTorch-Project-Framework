import torch.nn.functional as F


def correct(validity, label, softmax=True, return_label=True):
    if softmax:
        validity = F.softmax(validity, dim=1)
    fake_label = validity.max(1, keepdim=True)[1]
    accuracy = fake_label.eq(label.view_as(fake_label)).sum().item()
    return (fake_label, accuracy) if return_label else accuracy
