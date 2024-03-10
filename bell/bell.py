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
p = vmap(
        vmap(
            vmap(P_abc, in_axes=(None, None, 0)),in_axes=(None,0,None)),
        in_axes=(0, None, None),)(values, values, values)

# axis names (that's what the notation "p(A)" means)

A = 0
B = 1
C = 2

p_AB = p.sum(axis=C)
p_AC = p.sum(axis=B)
p_BC = p.sum(axis=A)

display(
    Markdown(
            "$$p(A,B) ="
            + ltx(p_AB, "{:6.2f}")
            + ", \qquad "
            + "p(A,C) ="
            + ltx(p_AC,"{:6.2f}")
            + ", \qquad "
            + "p(B,C) ="
            + ltx(p_BC, "{:6.2f}")
            + "$$"
     )
)

p_A = p.sum(axis=(B,C))
p_B = p.sum(axis=(A,C))
p_C = p.sum(axis=(A,B))

display(
        Markdown(
            "$$p(A) ="
            + ltx(p_A, "{:4.2f}")
            + ",\qquad "
            + "p(B) ="
            + ltx(p_B,"{:4.2f}")
            + ",\qquad "
            + "p(C) ="
            + ltx(p_C, "{:4.2f}")
            + "$$"
        )
    )

p_AB_C = (p / p.sum(axis=(A,B), keepdims=True))[:,:,1]
p_AC_B = (p / p.sum(axis=(A,C), keepdims=True))[:,1,:]
p_BC_A = (p / p.sum(axis=(B,C), keepdims=True))[1,:,:]

display(
        Markdown(
            "$$p(A,B\mid C) ="
            + ltx(p_AB_C, "{:4.2f}")
            + ",\qquad "
            + "p(A,C\mid B) ="
            + ltx(p_AC_B,"{:4.2f}")
            + ",\qquad "
            + "p(B,C\mid A) ="
            + ltx(p_BC_A, "{:4.2f}")
            + "$$"
        )
    )
