from jax import grad, jit, vmap, pmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import lax
from jax import make_jaxpr
from jax import random
from jax import device_put

seed = 0 
key = random.PRNGKey(seed)

x = random.normal(key,(10,))
print(type(x),x)
