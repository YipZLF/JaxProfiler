import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import time
T = 10


d = {1:"begin",2:"end"} 
def cb(s):

    print("Time {}".format(time.time()),d[s.item()])
# jax.config.update("jax_platform_name", "cpu")
@jax.named_call
def MySub(x):
    return x - x.mean(0)

def norm(x):
    
    jax.debug.callback(cb,1)
    x = MySub(x)
    jax.debug.callback(cb,2)
    return x / x.std(0)

np.random.seed(1701)
x = jnp.array(np.random.rand(10000,10))


jaxpr = jax.make_jaxpr(norm)(x)
print(jaxpr)
# print(jaxpr)
jit_norm = jax.jit(norm)
jit_norm(x)
for i in range(T):
    print("{}:{}",i,jit_norm(x))
    

# s = time.time()
# with jax.profiler.trace("./jax_trace",create_perfetto_link=False):
#     for i in range(T):
#         with jax.profiler.StepTraceAnnotation("TRAINING STEP", step_num=i):
#             jnorm_x = jnorm(x).block_until_ready()
# e = time.time()
# print("JIT Norm time: {}".format((e-s)))

    