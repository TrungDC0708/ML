from __future__ import division

from util import *
import pytorch_lightning as pl


def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


class EmptyLayer(pl.LightningModule):
    def __init__(self):
        super().__init__()


class DetectionLayer(pl.LightningModule):
    def __init__(self, anchors):
        super().__init__()
        self.anchors = anchors


class Darknet(pl.LightningModule):
    def __init__(self, cfgfile):
        super().__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info = self.blocks[0]
        self.module_list = nn.ModuleList()
        self.create_modules()

    def create_modules(self):
        prev_filters = 3
        output_filters = []
        for index, x in enumerate(self.blocks[1:]):
            module = nn.Sequential()
            if x["type"] == "convolutional":
                activation = x["activation"]
                try:
                    batch_normalize = int(x["batch_normalize"])
                    bias = False
                except:
                    batch_normalize = 0
                    bias = True

                filters = int(x["filters"])

                if int(x["pad"]):
                    pad = (int(x["size"]) - 1) // 2
                else:
                    pad = 0
                conv = nn.Conv2d(prev_filters, filters, int(x["size"]), int(x["stride"]), pad, bias=bias)
                module.add_module("conv_{0}".format(index), conv)
                if batch_normalize:
                    module.add_module("batch_norm_{0}".format(index), nn.BatchNorm2d(filters))
                if activation == "leaky":
                    module.add_module("leaky_{0}".format(index), nn.LeakyReLU(0.1, inplace=True))
            elif x["type"] == "upsample":
                upsample = nn.Upsample(scale_factor=2, mode="nearest")
                module.add_module("upsample_{}".format(index), upsample)
            elif x["type"] == "route":
                x["layers"] = x["layers"].split(',')
                start = int(x["layers"][0])
                try:
                    end = int(x["layers"][1])
                except:
                    end = 0
                if start > 0:
                    start = start - index
                if end > 0:
                    end = end - index
                route = EmptyLayer()
                module.add_module("route_{0}".format(index), route)
                if end < 0:
                    filters = output_filters[index + start] + output_filters[index + end]
                else:
                    filters = output_filters[index + start]

            elif x["type"] == "shortcut":
                shortcut = EmptyLayer()
                module.add_module("shortcut_{}".format(index), shortcut)

            elif x["type"] == "yolo":
                mask = x["mask"].split(",")
                mask = [int(x) for x in mask]

                anchors = x["anchors"].split(",")
                anchors = [int(a) for a in anchors]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]

                detection = DetectionLayer(anchors)
                module.add_module("Detection_{}".format(index), detection)

            self.module_list.append(module)
            prev_filters = filters
            output_filters.append(filters)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}
        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    def load_weights(self, weightfile="yolov3.weights"):
        fp = open(weightfile, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]
                    num_bn_biases = bn.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def save_darknet_weights(self, path, cutoff=-1):
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()