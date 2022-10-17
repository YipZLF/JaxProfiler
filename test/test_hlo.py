import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import time
import trace
T = 100

# jax.config.update("jax_platform_name", "cpu")
# @jax.named_call
# def MySub(x):


def norm(x):
    with trace.CustomTraceAnnotation():
        x = x + 2
        x = x * 4
        x = x ** 2
    x = x - x.mean(0)
    return x / x.std(0)

np.random.seed(1701)
x = jnp.array(np.random.rand(10000,10))
x_np = np.random.rand(10000,10)
jnorm = norm
# print(jax.make_jaxpr(norm)(x))
s = time.time()
with jax.profiler.trace("./jax_trace",create_perfetto_link=False):
    for i in range(T):
        jnorm_x = jnorm(x).block_until_ready()
e = time.time()
print("JIT Norm time: {}".format((e-s)))
