# Autogenerated by onnx-pytorch.

import glob
import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self._vars = nn.ParameterDict()
    self._regularizer_params = []
    for b in glob.glob(
        os.path.join(os.path.dirname(__file__), "variables", "*.npy")):
      v = torch.from_numpy(np.load(b))
      requires_grad = v.dtype.is_floating_point or v.dtype.is_complex
      self._vars[os.path.basename(b)[:-4]] = nn.Parameter(v, requires_grad=requires_grad)
    self.n_Conv_0 = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 64, 'padding': [2, 2], 'kernel_size': (11, 11), 'stride': [4, 4], 'in_channels': 3, 'bias': True})
    self.n_Conv_0.weight.data = self._vars["learned_0"]
    self.n_Conv_0.bias.data = self._vars["learned_1"]
    self.n_MaxPool_2 = nn.MaxPool2d(**{'dilation': 1, 'kernel_size': [3, 3], 'ceil_mode': False, 'stride': [2, 2], 'return_indices': False, 'padding': [0, 0]})
    self.n_Conv_3 = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 192, 'padding': [2, 2], 'kernel_size': (5, 5), 'stride': [1, 1], 'in_channels': 64, 'bias': True})
    self.n_Conv_3.weight.data = self._vars["learned_2"]
    self.n_Conv_3.bias.data = self._vars["learned_3"]
    self.n_MaxPool_5 = nn.MaxPool2d(**{'dilation': 1, 'kernel_size': [3, 3], 'ceil_mode': False, 'stride': [2, 2], 'return_indices': False, 'padding': [0, 0]})
    self.n_Conv_6 = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 384, 'padding': [1, 1], 'kernel_size': (3, 3), 'stride': [1, 1], 'in_channels': 192, 'bias': True})
    self.n_Conv_6.weight.data = self._vars["learned_4"]
    self.n_Conv_6.bias.data = self._vars["learned_5"]
    self.n_Conv_8 = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 256, 'padding': [1, 1], 'kernel_size': (3, 3), 'stride': [1, 1], 'in_channels': 384, 'bias': True})
    self.n_Conv_8.weight.data = self._vars["learned_6"]
    self.n_Conv_8.bias.data = self._vars["learned_7"]
    self.n_Conv_10 = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 256, 'padding': [1, 1], 'kernel_size': (3, 3), 'stride': [1, 1], 'in_channels': 256, 'bias': True})
    self.n_Conv_10.weight.data = self._vars["learned_8"]
    self.n_Conv_10.bias.data = self._vars["learned_9"]
    self.n_MaxPool_12 = nn.MaxPool2d(**{'dilation': 1, 'kernel_size': [3, 3], 'ceil_mode': False, 'stride': [2, 2], 'return_indices': False, 'padding': [0, 0]})
    self.n_AveragePool_13 = nn.AvgPool2d(**{'kernel_size': [1, 1], 'ceil_mode': False, 'stride': [1, 1], 'count_include_pad': False})
    self.n_Flatten_14 = nn.Flatten(**{'start_dim': 1})

  def forward(self, *inputs):
    actual_input_1, = inputs
    input = self.n_Conv_0(actual_input_1)
    onnx__MaxPool_18 = F.relu(input)
    input_4 = self.n_MaxPool_2(onnx__MaxPool_18)
    input_8 = self.n_Conv_3(input_4)
    onnx__MaxPool_21 = F.relu(input_8)
    input_12 = self.n_MaxPool_5(onnx__MaxPool_21)
    input_16 = self.n_Conv_6(input_12)
    onnx__Conv_24 = F.relu(input_16)
    input_20 = self.n_Conv_8(onnx__Conv_24)
    onnx__Conv_26 = F.relu(input_20)
    input_24 = self.n_Conv_10(onnx__Conv_26)
    onnx__MaxPool_28 = F.relu(input_24)
    input_28 = self.n_MaxPool_12(onnx__MaxPool_28)
    onnx__Flatten_30 = self.n_AveragePool_13(input_28)[:, :]
    input_32 = self.n_Flatten_14(onnx__Flatten_30)
    input_36 = 1.0 * torch.matmul(input_32, torch.transpose(self._vars["learned_10"], 0, 1)) + 1.0 * self._vars["learned_11"]
    onnx__Gemm_33 = F.relu(input_36)
    input_40 = 1.0 * torch.matmul(onnx__Gemm_33, torch.transpose(self._vars["learned_12"], 0, 1)) + 1.0 * self._vars["learned_13"]
    onnx__Gemm_35 = F.relu(input_40)
    output1 = 1.0 * torch.matmul(onnx__Gemm_35, torch.transpose(self._vars["learned_14"], 0, 1)) + 1.0 * self._vars["learned_15"]
    return output1

  def compatible_auto_pad(self, input, kernel_spatial_shape, nn_mod, auto_pad=None, **kwargs):
    input_spatial_shape = input.shape[2:]
    d = len(input_spatial_shape)
    strides = nn_mod.stride
    dilations = nn_mod.dilation
    output_spatial_shape = [math.ceil(float(l) / float(r)) for l, r in zip(input.shape[2:], strides)]
    pt_padding = [0] * 2 * d
    pad_shape = [0] * d
    for i in range(d):
      pad_shape[i] = (output_spatial_shape[i] - 1) * strides[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
      mean = pad_shape[i] // 2
      if auto_pad == b"SAME_UPPER":
        l, r = pad_shape[i] - mean, mean
      else:
        l, r = mean, pad_shape[i] - mean
      pt_padding.insert(0, r)
      pt_padding.insert(0, l)
    return F.pad(input, pt_padding)

@torch.no_grad()
def test_run_model(inputs=[torch.from_numpy(np.random.randn(*[10, 3, 224, 224]).astype(np.float32))]):
  model = Model()
  model.eval()
  rs = model(*inputs)
  print(rs)
  return rs