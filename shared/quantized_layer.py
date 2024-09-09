import torch
from transformers import AutoModelForCausalLM

"""
    This class is used to quantize a full-connection layer of OPT-125m model. 
"""


class QuantizedLinearLayer_Opt125m(torch.nn.Module):
    def __init__(self, original_layer):
        super(QuantizedLinearLayer_Opt125m, self).__init__()
        self.quantized_layer = torch.quantization.QuantStub()
        self.dequantized_layer = torch.quantization.DeQuantStub()
        self.original_layer = original_layer

    def forward(self, x):
        x = self.quantized_layer(x)
        x = self.dequantized_layer(x)
        x = self.original_layer(x)
        return x


# Quantize a full-connection layer of OPT-125m model
def modify_opt125m(model: AutoModelForCausalLM):
    penultimate_layer = model.model.decoder.layers[-2]
    penultimate_layer.fc1 = QuantizedLinearLayer_Opt125m(penultimate_layer.fc1)
