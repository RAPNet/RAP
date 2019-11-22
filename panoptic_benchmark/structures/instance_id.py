import torch


class InstanceId(object):
    def __init__(self, instance_id):
        instance_id = torch.as_tensor(instance_id, dtype=torch.int32)
        self.instance_id = instance_id

    def to(self, device):
        return self.instance_id.to(device=device)

    def __getitem__(self, item):
        return self.instance_id[item]

    def __len__(self):
        return self.instance_id.size()[0]

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={} )".format(len(self.instance_id))
        return s
