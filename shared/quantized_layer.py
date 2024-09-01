import torch


class QuantizedLinearLayer(torch.nn.Module):
    def __init__(self, original_layer):
        super(QuantizedLinearLayer, self).__init__()
        self.quantized_layer = torch.quantization.QuantStub()
        self.dequantized_layer = torch.quantization.DeQuantStub()
        self.original_layer = original_layer

    def forward(self, x):
        x = self.quantized_layer(x)
        x = self.original_layer(x)
        x = self.dequantized_layer(x)
        return x


# Quantize the full connection layers
def replace_layer(model):
    penultimate_layer = model.model.decoder.layers[-2]
    penultimate_layer.fc1 = QuantizedLinearLayer(penultimate_layer.fc1)
    penultimate_layer.fc2 = QuantizedLinearLayer(penultimate_layer.fc2)
