from IPython.display import Markdown, display
from jax import vmap
import array_to_latex as a2l
import jax.numpy as jnp
import numpy as np

def ltx(a, fmt):
    return a2l.to_ltx(np.array(a), frmt = fmt, print_out = False)

def P_a(a):
    p = 0.5
    return a*p + (~a)*(1-p)

def P_b(b):
    p=0.5
    return b*p + (~b)*(1-p)

def P_c_ab(c,a,b):
    return c == (a == b)

def P_abc(a,b,c):
    return P_a(a)*P_b(b)*P_c_ab(c,a,b)

values = jnp.asarray([0,1], dtype = bool)
print(values)
l = vmap(
        vmap(
            vmap(P_abc, in_axes=(None, None, 0)),in_axes=(None,0,None)),
        in_axes=(0, None, None),)(values, values, values)

# axis names (that's what the notation "p(A)" means)
#print(p)

k = vmap(P_abc, in_axes=(None, None, 0))
p = vmap(k, in_axes=(None,0,None))(values,values,values)
print(l)

A = 0
B = 1
C = 2

p_AB = p.sum(axis=C)
p_AC = p.sum(axis=B)
p_BC = p.sum(axis=A)

l_AB = l.sum(axis=C)
l_AC = l.sum(axis=B)
l_BC = l.sum(axis=A)

print(p_AB)
print(p_AC)
print(p_BC)


print(l_AB)
print(l_AC)
print(l_BC)


p_A = p.sum(axis=(B,C))
p_B = p.sum(axis=(A,C))
p_C = p.sum(axis=(A,B))


print(p_A)
print(p_B)
print(p_C)


p_AB_C = (p / p.sum(axis=(A,B), keepdims=True))[:,:,1]
p_AC_B = (p / p.sum(axis=(A,C), keepdims=True))[:,1,:]
p_BC_A = (p / p.sum(axis=(B,C), keepdims=True))[1,:,:]

print(p_AB_C)
print(p_AC_B)
print(p_BC_A)
