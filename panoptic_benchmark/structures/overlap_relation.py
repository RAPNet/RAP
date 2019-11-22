import torch


class OverlapRelation(object):
    def __init__(self, overlaps, instance_ids):
        self.overlaps = overlaps
        self.instance_ids = instance_ids

    def __getitem__(self, item):
        result = {}
        if isinstance(item, torch.Tensor):
            if item.dtype is torch.uint8:
                overlaps = [self.overlaps[ind] for ind in range(item.size()[0]) if item[ind]]
            instance_ids = self.instance_ids[item]
            relation = {}
            for rel, ins_id in zip(overlaps, instance_ids):
                ins_id = int(ins_id)
                for i in rel:
                    relation[(i, ins_id)] = torch.Tensor([1])
            result["overlaps"] = overlaps
            result["relations"] = relation
        return result

    def __len__(self):
        return len(self.overlaps)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_relations={} )".format(len(self.overlaps))
        return s
