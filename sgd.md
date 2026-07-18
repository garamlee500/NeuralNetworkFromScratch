# Derivation of gradient for Stochastic Gradient Descent
### Garam Lee

## Prerequisites

This document (ab)uses index/einstein notation.


## Definitions
Let $d$ be the depth of the network.


Let $`\boldsymbol x ^{(0)}, \boldsymbol x ^{(1)}, \ldots, \boldsymbol x ^{(d)}`$ be the layers of the neural network. $\boldsymbol x^{(0)}$ is the input layer, and $`\boldsymbol x^{(d)}`$ the output layer.

Let $`f_k`$ be the (element-wise) activation function acting on the $`k`$th layer.

Let $`W^{(k)}`$ be the matrix of weights at the $`k`$th layer, i.e.: 
```math
\boldsymbol x^{(k+1)}=f_k(\boldsymbol x^{(k)}W^{(k)})
```

Define $`y^{(k)}=\boldsymbol x^{(k)}W^{(k)}`$ for convenience - so that $`\boldsymbol x^{(k+1)}= f_k(\boldsymbol y^{(k)})`$.

Let the loss function be $L$ so that our loss is $`L(\boldsymbol x^{(d)})`$. Note $L$ implicitly depends on our target output.

## Derivation

We want to find the gradient of $L$ w.r.t $`W^{(k)}`$, i.e. we want:

```math
\begin{align}
\frac{\partial L}{\partial W^{(k)}_{ij}}&= \frac{\partial L}{\partial y^{(d-1)}_\alpha}\frac{\partial y^{(d-1)}_\alpha}{\partial y^{(d-2)}_\beta }\cdots \frac{\partial y^{(k+1)}_\gamma}
{\partial y_j^{(k)}}\frac{\partial y_j^{(k)}}{\partial W_{ij}^{(k)}} \text {        (This line is not true index notation - don't sum over $j$)}\nonumber
\\
&=\frac{\partial L}{\partial y^{(d-1)}_\alpha}\frac{\partial y^{(d-1)}_\alpha}{\partial y^{(d-2)}_\beta }\cdots \frac{\partial y^{(k+1)}_\gamma}{\partial y_j^{(k)}}x^{(k)}_i
\end{align}
```

Note we can fix index of $`\boldsymbol y^{(k)}`$  to $j$ as $`W_{ij}^{(k)}$ only affects $L$ through $y^{(k)}_j`$


We have that as $`y_a^{(k+1)} = f_k(y_b^{(k)})W^{(k+1)}_{ba}`$:
```math
\begin{align}
\frac{\partial y^{(k+1)}_a}{\partial y^{(k)}_b} = f'_k(y_b^{(k)})W^{(k+1)}_{ba} \text{ (No sum again)}
\end{align}
```

## Implementing Bias

To implement bias, we add a bias node with value $1$ input into each layer. The maths here remains almost the same except $`W^{(k)}`$ now has an extra row compared to $`\boldsymbol y^{(k)}`$ - the last row of $`W`$ can essentially be ignored for (2) as this only adds a constant term to $`y_a^{(k)}`$ (w.r.t $`y_b^{(k-1)}`$). However the $`\boldsymbol x`$ term in (1) does require the extra $1$ as changing the bias weight component would affect $`\boldsymbol y^{(k)}`$.