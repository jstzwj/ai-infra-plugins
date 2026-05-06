# PyTorch - Chapter 38: Special Math Functions (torch.special)

This reference covers the torch.special module for special mathematical functions.

---

## 38.1 Error Functions

```python
torch.special.erf(input)           # Error function: erf(x)
torch.special.erfc(input)          # Complementary: 1 - erf(x)
torch.special.erfcx(input)         # Scaled: exp(x²) * erfc(x)
torch.special.erfinv(input)        # Inverse error function
```

---

## 38.2 Gamma Functions

```python
torch.special.gammaln(input)                    # log(|Gamma(x)|)
torch.special.gammainc(input, other)             # Regularized lower incomplete gamma
torch.special.gammaincc(input, other)            # Regularized upper incomplete gamma
torch.special.digamma(input)                     # psi(x) = d/dx ln(Gamma(x))
torch.special.polygamma(n, input)                # nth derivative of digamma
torch.special.multigammaln(input, p)             # Multivariate log-gamma
```

---

## 38.3 Bessel Functions

```python
torch.special.i0(input)              # Modified Bessel of order 0
torch.special.i0e(input)             # Exponentially scaled I0
torch.special.i1(input)              # Modified Bessel of order 1
torch.special.i1e(input)             # Exponentially scaled I1
torch.special.spherical_bessel_j0(input)  # Spherical Bessel j0
```

---

## 38.4 Exponential and Logistic Functions

```python
torch.special.exp2(input)            # 2^x
torch.special.expm1(input)           # e^x - 1 (precise for small x)
torch.special.expit(input)           # 1/(1+e^(-x)) = sigmoid
torch.special.logit(input, eps=None) # log(x/(1-x)) = inverse sigmoid
torch.special.log_softmax(input, dim) # Numerically stable log-softmax
torch.special.softmax(input, dim)     # Numerically stable softmax
torch.special.logsumexp(input, dim, keepdim=False) # log(sum(exp(x)))
```

---

## 38.5 Other Special Functions

```python
torch.special.entr(input)            # Entropy: -x*log(x) for x>0, 0 for x=0
torch.special.ndtr(input)            # Normal CDF: Phi(x)
torch.special.ndtri(input)           # Normal quantile: Phi^(-1)(x)
torch.special.log_ndtr(input)        # log(Phi(x)), numerically stable
torch.special.sinc(input)            # sin(pi*x)/(pi*x)
torch.special.xlogy(input, other)    # x * log(y), returns 0 when x=0
torch.special.xlog1py(input, other)  # x * log(1+y)
torch.special.zeta(input, other)     # Hurwitz zeta function
torch.special.round(input, decimals=0) # Round to given decimals
```

---

## 38.6 Examples

```python
x = torch.tensor([0.0, 0.5, 1.0, 2.0])

torch.special.erf(x)       # tensor([0.0, 0.5205, 0.8427, 0.9953])
torch.special.gammaln(x+1) # tensor([0.0, 0.0, 0.0, 0.6931]) = log(n!)
torch.special.ndtr(x)      # tensor([0.5, 0.6915, 0.8413, 0.9772])
torch.special.expit(x)     # tensor([0.5, 0.6225, 0.7311, 0.8808])
```
