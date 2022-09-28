import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import time
T = 10000

# jax.config.update("jax_platform_name", "cpu")
@jax.named_call
def MySub(x):
    return x - x.mean(0)

def norm(x):
    x = MySub(x)
    return x / x.std(0)

np.random.seed(1701)
x = jnp.array(np.random.rand(10000,10))
x_np = np.random.rand(10000,10)
jnorm = jit(norm)

s = time.time()
with jax.profiler.trace("./jax_trace",create_perfetto_link=False):
    for i in range(T):
        with jax.profiler.StepTraceAnnotation("TRAINING STEP", step_num=i):
            jnorm_x = jnorm(x).block_until_ready()
e = time.time()
print("JIT Norm time: {}".format((e-s)))
