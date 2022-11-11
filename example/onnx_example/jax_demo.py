# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An ONNX to XLA compiler by JAX-tracing a Numpy-backed ONNX interpreter."""

import sys

import numpy as np
import onnx
from onnx import numpy_helper

import jax
import jax.numpy as jnp
from jax import jit, grad
from jax import lax

jax.config.update("jax_platform_name", "cpu")


def _asarray(proto):
  return numpy_helper.to_array(proto).reshape(tuple(proto.dims))


attr_types = dict(onnx.AttributeProto.AttributeType.items())
attribute_handlers = {
    attr_types['FLOAT']: lambda a: a.f,
    attr_types['INT']: lambda a: a.i,
    attr_types['STRING']: lambda a: a.s,
    attr_types['TENSOR']: lambda a: _asarray(a.t),
    attr_types['FLOATS']: lambda a: a.floats,
    attr_types['INTS']: lambda a: a.ints,
    attr_types['STRINGS']: lambda a: a.strings,
    attr_types['TENSORS']: lambda a: [_asarray(x) for x in a.tensors],
}


def onnx_maxpool(x, kernel_shape, pads=None, strides=None,ceil_mode = 0):
  """Numpy-backed implementation of ONNX MaxPool op."""
  prefix = (1,) * (x.ndim - len(kernel_shape))
  dims = prefix + tuple(kernel_shape)
  pads = tuple(pads) if pads else [0] * len(kernel_shape)
  strides = (prefix + tuple(strides)) if strides else [1] * len(kernel_shape)
  return [lax.reduce_window(x, -jnp.inf, lax.max, dims, strides, 'VALID')]

def onnx_averagepool(x, kernel_shape, pads=None, strides=None):
  """Numpy-backed implementation of ONNX AveragePool op."""
  # items_cnt = prod(kernel_shape) # TODO!!!
  items_cnt = 1
  prefix = (1,) * (x.ndim - len(kernel_shape))
  dims = prefix + tuple(kernel_shape)
  strides = (prefix + tuple(strides)) if strides else [1] * len(kernel_shape)
  
  return [lax.reduce_window(x, 0, lax.add, dims, strides, 'SAME')/items_cnt]

def onnx_conv(x, w, b=0, group=1, kernel_shape=None, pads=None, strides=None,
              dilations=None, auto_pad=None):
  """Numpy-backed implementation of ONNX Conv op."""
  assert group == 1
  kernel_shape = kernel_shape or w.shape
  strides = strides or [1] * (w.ndim - 2)
  if auto_pad:
    auto_pad = 'SAME' if auto_pad.startswith(b'SAME') else 'VALID'
    pads = lax.padtype_to_pads(x.shape[2:], w.shape[2:], strides, auto_pad)
  else:
    if len(pads) == 2:
      pads = pads or [0] * (w.ndim - 2)
    elif len(pads) == 4:
      pads = [(2,2),(2,2)] if pads[0] == 2 else [(1,1),(1,1)]
  lhs_dilation = [1] * (w.ndim - 2)
  rhs_dilation = dilations or [1] * (w.ndim - 2)
  
  b = b.reshape(-1, 1, 1)
  return [lax.conv_with_general_padding(x, w, strides, pads,
                                        lhs_dilation, rhs_dilation) + b]


def onnx_add(a, b, axis=None, broadcast=True):
  """Numpy-backed implementation of ONNX Add op."""
  if broadcast:
    axis = (a.dim - b.ndim) if axis is None else axis % a.ndim
    assert a.shape[axis:][:b.ndim] == b.shape
    b_shape = np.ones(a.ndim, dtype='int64')
    b_shape[axis:axis + b.ndim] = b.shape
    b = jnp.reshape(b, b_shape)
  return [a + b]

def onnx_gemm(a,b,c,alpha,beta,transB):
  # print(a.shape,b.shape,c.shape)
  return  jnp.dot(a,b.T) + c
  
def onnx_flatten(x,axis):
  ret = [jnp.reshape(x,(x.shape[0], -1))]

  return ret 
  

onnx_ops = {
    'Add': onnx_add,
    'Constant': lambda value: [value],
    'Conv': onnx_conv,
    'MatMul': lambda x, y: [jnp.matmul(x, y)],
    'MaxPool': onnx_maxpool,
    'AveragePool': onnx_averagepool,
    'Relu': lambda x: [jnp.maximum(x, 0)],
    'Reshape': lambda x, shape: [jnp.reshape(x, shape)],
    'Flatten': onnx_flatten,
    'Gemm': onnx_gemm
}


def interpret_onnx(graph, *args):
  vals = dict({n.name: a for n, a in zip(graph.input, args)},
              **{n.name: _asarray(n) for n in graph.initializer})
  for node in graph.node:
    args = (vals[name] for name in node.input)
    attrs = {a.name: attribute_handlers[a.type](a) for a in node.attribute}
    
    outputs = onnx_ops[node.op_type](*args, **attrs)
    for name, output in zip(node.output, outputs):
      vals[name] = output
  return [vals[n.name] for n in graph.output]


if __name__ == "__main__":
  model = onnx.load('./alexnet.onnx')

  predict = lambda inputs: interpret_onnx(model.graph, inputs)[0]

  dummy_input = np.random.rand(10,3,224,224)
  dummy_input = jnp.array(dummy_input)
  # print("interpreted:")
  # print(predict(dummy_input))

  compiled_predict = jit(predict).lower(dummy_input).compile()

  print("compiled:")
  print(compiled_predict(dummy_input).shape)

  fun = lambda inputs: jnp.sum(predict(inputs))
  grad_fun = grad(fun)
  jit_grad_fun = jit(grad_fun).lower(dummy_input).compile()
  print("a derivative with respect to inputs:")
  print(jit_grad_fun(dummy_input).shape)
