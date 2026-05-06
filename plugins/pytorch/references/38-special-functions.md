# PyTorch Special Math Functions - Comprehensive Reference

This chapter covers all special mathematical functions in `torch.special`, including error functions, gamma functions, Bessel functions, logistic functions, softmax, log-sum-exp, and other special functions used in scientific computing and machine learning.

---

## 1. Error Functions

### erf

Computes the error function: erf(x) = (2/sqrt(pi)) * integral_0^x exp(-t^2) dt

```python
torch.special.erf(input, *, out=None)
```

```python
import torch

x = torch.tensor([0.0, 0.5, 1.0, 2.0, -1.0])
result = torch.special.erf(x)
print(result)
# tensor([0.0000, 0.5205, 0.8427, 0.9953, -0.8427])

# Properties:
# erf(0) = 0
# erf(inf) = 1
# erf(-inf) = -1
# erf is an odd function: erf(-x) = -erf(x)

# Used in Gaussian CDF computation
def gaussian_cdf(x, mu=0.0, sigma=1.0):
    return 0.5 * (1 + torch.special.erf((x - mu) / (sigma * torch.sqrt(torch.tensor(2.0)))))

x = torch.linspace(-3, 3, 7)
cdf = gaussian_cdf(x)
print(cdf)  # Approx: [0.0013, 0.0228, 0.1587, 0.5000, 0.8413, 0.9772, 0.9987]
```

### erfc

Computes the complementary error function: erfc(x) = 1 - erf(x)

```python
torch.special.erfc(input, *, out=None)
```

```python
x = torch.tensor([0.0, 1.0, 2.0, 10.0])
result = torch.special.erfc(x)
print(result)
# tensor([1.0000, 0.1573, 0.0047, 0.0000])

# More numerically stable than 1 - erf(x) for large x
# erfc(x) = 1 - erf(x) = (2/sqrt(pi)) * integral_x^inf exp(-t^2) dt
```

### erfcx

Computes the scaled complementary error function: erfcx(x) = exp(x^2) * erfc(x)

```python
torch.special.erfcx(input, *, out=None)
```

```python
x = torch.tensor([0.0, 1.0, 5.0, 10.0, 20.0])
result = torch.special.erfcx(x)
print(result)
# tensor([1.0000, 0.4276, 0.1128, 0.0562, 0.0281])

# Useful for avoiding overflow: exp(x^2) * erfc(x) remains bounded
# Used in Faddeeva function computations and certain probability calculations
```

### erfinv

Computes the inverse error function.

```python
torch.special.erfinv(input, *, out=None)
```

```python
x = torch.tensor([0.0, 0.5, 0.8427, 0.9953])
result = torch.special.erfinv(x)
print(result)
# tensor([0.0000, 0.4769, 1.0000, 2.0000])

# Round-trip verification
x_orig = torch.tensor([0.5, 1.0, 1.5])
recovered = torch.special.erfinv(torch.special.erf(x_orig))
print(torch.allclose(x_orig, recovered))  # True

# Used for inverse CDF computation
def gaussian_ppf(p, mu=0.0, sigma=1.0):
    """Percent point function (inverse of CDF) for Gaussian."""
    return mu + sigma * torch.sqrt(torch.tensor(2.0)) * torch.special.erfinv(2 * p - 1)

# Get the value at which CDF = 0.95
x_95 = gaussian_ppf(torch.tensor(0.95))
print(x_95)  # Approximately 1.6449
```

---

## 2. Exponential Functions

### exp2

Computes 2^x (base-2 exponential).

```python
torch.special.exp2(input, *, out=None)
```

```python
x = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0])
result = torch.special.exp2(x)
print(result)
# tensor([1., 2., 4., 8., 1024.])

# Useful in information theory and signal processing
```

### expit (Logistic/Sigmoid)

Computes the logistic sigmoid function: expit(x) = 1 / (1 + exp(-x))

```python
torch.special.expit(input, *, out=None)
```

```python
x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
result = torch.special.expit(x)
print(result)
# tensor([0.0067, 0.2689, 0.5000, 0.7311, 0.9933])

# Properties:
# expit(0) = 0.5
# expit(x) + expit(-x) = 1
# expit(x) = sigmoid(x) = sigma(x)

# Used in logistic regression and neural network activations
# Equivalent to torch.sigmoid()
```

### expm1

Computes exp(x) - 1 with better numerical precision for small x.

```python
torch.special.expm1(input, *, out=None)
```

```python
# For small x, expm1(x) is more accurate than exp(x) - 1
x = torch.tensor([1e-15, 1e-10, 1e-5, 0.1])

inaccurate = torch.exp(x) - 1
accurate = torch.special.expm1(x)

print(f"Inaccurate: {inaccurate}")
print(f"Accurate:   {accurate}")

# For x = 1e-15:
# exp(x) - 1 -> 0.0 (loss of precision)
# expm1(x)   -> 1e-15 (correct)
```

---

## 3. Gamma Functions

### gammainc

Computes the regularized lower incomplete gamma function: P(a, x) = gamma(a, x) / Gamma(a)

```python
torch.special.gammainc(input, other, *, out=None)
```

```python
a = torch.tensor([1.0, 2.0, 3.0, 5.0])
x = torch.tensor([1.0, 1.0, 2.0, 3.0])
result = torch.special.gammainc(a, x)
print(result)
# tensor([0.6321, 0.2642, 0.3233, 0.1847])

# For a=1, gammainc(1, x) = 1 - exp(-x) (CDF of Exponential(1))
x_test = torch.tensor([0.0, 1.0, 2.0])
gamma_result = torch.special.gammainc(torch.ones_like(x_test), x_test)
exp_cdf = 1 - torch.exp(-x_test)
print(torch.allclose(gamma_result, exp_cdf))  # True
```

### gammaincc

Computes the regularized upper incomplete gamma function: Q(a, x) = 1 - P(a, x)

```python
torch.special.gammaincc(input, other, *, out=None)
```

```python
a = torch.tensor([1.0, 2.0, 3.0])
x = torch.tensor([1.0, 2.0, 3.0])
result = torch.special.gammaincc(a, x)
print(result)
# tensor([0.3679, 0.5940, 0.5768])

# gammainc + gammaincc = 1
lower = torch.special.gammainc(a, x)
upper = torch.special.gammaincc(a, x)
print(torch.allclose(lower + upper, torch.ones_like(a)))  # True

# Used in chi-squared test p-value computation
def chi2_sf(x, df):
    """Survival function of chi-squared distribution."""
    return torch.special.gammaincc(torch.tensor(df / 2.0), x / 2.0)
```

### gammaln (lgamma)

Computes the log of the absolute value of the Gamma function: lgamma(x) = log(|Gamma(x)|)

```python
torch.special.gammaln(input, *, out=None)
```

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
result = torch.special.gammaln(x)
print(result)
# tensor([0.0000, 0.0000, 0.6931, 1.7918, 3.1781])

# gammaln(n+1) = log(n!) for positive integers
# gammaln(1) = log(1!) = log(1) = 0
# gammaln(2) = log(2!) = log(2) = 0.6931
# gammaln(3) = log(3!) = log(6) = 1.7918
# gammaln(4) = log(4!) = log(24) = 3.1781

# Useful for computing log-probabilities without overflow
def log_factorial(n):
    """Compute log(n!) without overflow."""
    return torch.special.gammaln(n + 1)

def log_binomial_coefficient(n, k):
    """Compute log(C(n,k)) = log(n! / (k! * (n-k)!))"""
    return (torch.special.gammaln(n + 1)
            - torch.special.gammaln(k + 1)
            - torch.special.gammaln(n - k + 1))
```

### multigammaln

Computes the log multivariate Gamma function.

```python
torch.special.multigammaln(input, p, *, out=None)
```

```python
a = torch.tensor([3.0, 5.0, 10.0])
p = 3  # dimension
result = torch.special.multigammaln(a, p)
print(result)

# Used in multivariate statistics (e.g., Wishart distribution normalization)
```

---

## 4. Digamma and Polygamma

### digamma

Computes the digamma function: psi(x) = d/dx ln(Gamma(x)) = Gamma'(x) / Gamma(x)

```python
torch.special.digamma(input, *, out=None)
```

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
result = torch.special.digamma(x)
print(result)
# tensor([-0.5772,  0.4228,  0.9228,  1.2561,  1.5061])

# digamma(1) = -gamma (Euler-Mascheroni constant = -0.5772...)
# digamma(n) = H_{n-1} - gamma for positive integers
# where H_n is the n-th harmonic number

# Used in gradient computation of gammaln (for optimization)
# d/dx gammaln(x) = digamma(x)
```

### polygamma

Computes the n-th derivative of the digamma function.

```python
torch.special.polygamma(n, input, *, out=None)
```

```python
x = torch.tensor([1.0, 2.0, 3.0])

# n=0: digamma
psi_0 = torch.special.polygamma(0, x)
print(psi_0)  # Same as digamma(x)

# n=1: trigamma (second derivative of log Gamma)
psi_1 = torch.special.polygamma(1, x)
print(psi_1)

# n=2: tetragamma
psi_2 = torch.special.polygamma(2, x)
print(psi_2)

# n=3: pentagamma
psi_3 = torch.special.polygamma(3, x)
print(psi_3)
```

---

## 5. Bessel Functions

### i0

Modified Bessel function of the first kind, order 0.

```python
torch.special.i0(input, *, out=None)
```

```python
x = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0])
result = torch.special.i0(x)
print(result)
# tensor([1.0000, 1.0635, 1.2661, 2.2796, 27.2399])

# i0(0) = 1
# i0(x) grows exponentially for large x
# Used in von Mises distribution normalization
```

### i0e

Exponentially scaled modified Bessel function of the first kind, order 0: i0e(x) = exp(-|x|) * i0(x)

```python
torch.special.i0e(input, *, out=None)
```

```python
x = torch.tensor([0.0, 5.0, 10.0, 50.0, 100.0])
result = torch.special.i0e(x)
print(result)
# More numerically stable than i0 for large x
# Bounded behavior instead of overflow
```

### i1

Modified Bessel function of the first kind, order 1.

```python
torch.special.i1(input, *, out=None)
```

```python
x = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0])
result = torch.special.i1(x)
print(result)
# tensor([0.0000, 0.2579, 0.5652, 1.5906, 24.3356])

# i1(0) = 0
```

### i1e

Exponentially scaled modified Bessel function of the first kind, order 1: i1e(x) = exp(-|x|) * i1(x)

```python
torch.special.i1e(input, *, out=None)
```

```python
x = torch.tensor([0.0, 1.0, 10.0, 100.0])
result = torch.special.i1e(x)
print(result)
# Numerically stable for large x
```

---

## 6. Logarithmic and Normalization Functions

### log_softmax

Computes log(softmax(x)) in a numerically stable way.

```python
torch.special.log_softmax(input, dim=None, _stacklevel=3, dtype=None)
```

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0])

# Numerically stable log-softmax
result = torch.special.log_softmax(x, dim=0)
print(result)
# tensor([-3.1429, -2.1429, -1.1429, -0.1429])

# Equivalent but less stable: torch.log(torch.softmax(x, dim=0))

# Essential for cross-entropy loss computation
def cross_entropy_loss(logits, targets):
    """Compute cross-entropy using log_softmax."""
    log_probs = torch.special.log_softmax(logits, dim=-1)
    return -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()

# 2D case (batch x classes)
logits = torch.randn(32, 10)  # 32 samples, 10 classes
log_probs = torch.special.log_softmax(logits, dim=-1)
print(log_probs.shape)  # torch.Size([32, 10])
print(torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(32)))  # True
```

### logit

Computes the logit function: logit(x) = log(x / (1 - x)). Inverse of sigmoid.

```python
torch.special.logit(input, *, out=None)
```

```python
p = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
result = torch.special.logit(p)
print(result)
# tensor([-2.1972, -1.0986,  0.0000,  1.0986,  2.1972])

# logit is the inverse of sigmoid/expit
p_orig = torch.tensor([0.2, 0.5, 0.8])
recovered = torch.special.expit(torch.special.logit(p_orig))
print(torch.allclose(p_orig, recovered))  # True

# Properties:
# logit(0.5) = 0
# logit(p) = -logit(1-p)
# Used in logistic regression and beta distribution parameterization
```

### logsumexp

Computes log(sum(exp(x))) in a numerically stable way.

```python
torch.special.logsumexp(input, dim, keepdim=False, *, out=None)
```

```python
x = torch.tensor([1000.0, 1001.0, 1002.0])

# Naive (overflow):
# log(sum(exp(x))) -> inf

# Stable:
result = torch.special.logsumexp(x, dim=0)
print(result)  # tensor(1002.4076)

# Along specific dimension
x = torch.randn(3, 5)
row_logsumexp = torch.special.logsumexp(x, dim=1)  # torch.Size([3])
col_logsumexp = torch.special.logsumexp(x, dim=0)  # torch.Size([5])

# With keepdim
result = torch.special.logsumexp(x, dim=1, keepdim=True)
print(result.shape)  # torch.Size([3, 1])

# Used in:
# - Log-space arithmetic (multiply by adding logs)
# - Mixture model log-likelihoods
# - Attention mechanism normalization
# - Cross-entropy loss
```

### log_ndtr

Computes log of the normal cumulative distribution function.

```python
torch.special.log_ndtr(input, *, out=None)
```

```python
x = torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0])
result = torch.special.log_ndtr(x)
print(result)
# tensor([-1.4069e+01, -7.5757e-01, -6.9315e-01, -2.2700e-03, -5.7330e-07])

# More numerically stable than log(ndtr(x)) for large negative x
# Used in probit regression and Gaussian process classification
```

---

## 7. Normal Distribution Related

### ndtr

Computes the standard normal CDF: ndtr(x) = 0.5 * (1 + erf(x / sqrt(2)))

```python
torch.special.ndtr(input, *, out=None)
```

```python
x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
result = torch.special.ndtr(x)
print(result)
# tensor([0.0013, 0.1587, 0.5000, 0.8413, 0.9987])

# ndtr(0) = 0.5
# ndtr(x) + ndtr(-x) = 1
```

### ndtri

Computes the inverse of ndtr (standard normal quantile function).

```python
torch.special.ndtri(input, *, out=None)
```

```python
p = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
result = torch.special.ndtri(p)
print(result)
# tensor([-2.3263, -1.6449, -0.6745,  0.0000,  0.6745,  1.6449,  2.3263])

# Round-trip
x_orig = torch.tensor([0.5, 1.0, 1.5])
recovered = torch.special.ndtri(torch.special.ndtr(x_orig))
print(torch.allclose(x_orig, recovered, atol=1e-5))  # True

# Also known as the probit function or the quantile function of N(0,1)
```

---

## 8. Other Special Functions

### sinc

Computes the sinc function: sinc(x) = sin(pi*x) / (pi*x) for x != 0, and sinc(0) = 1.

```python
torch.special.sinc(input, *, out=None)
```

```python
x = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0])
result = torch.special.sinc(x)
print(result)
# tensor([ 0.0000,  0.0000,  1.0000,  0.6366,  0.0000, -0.2122,  0.0000])

# sinc(n) = 0 for all non-zero integers n
# sinc(0) = 1
# Used in signal processing (ideal low-pass filter)
```

### softmax

Computes the softmax function: softmax(x_i) = exp(x_i) / sum_j(exp(x_j))

```python
torch.special.softmax(input, dim=None, _stacklevel=3, dtype=None)
```

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
result = torch.special.softmax(x, dim=0)
print(result)
# tensor([0.0321, 0.0871, 0.2369, 0.6439])
print(result.sum())  # tensor(1.0)

# For 2D: typically applied along class dimension
logits = torch.randn(4, 10)
probs = torch.special.softmax(logits, dim=-1)
print(probs.sum(dim=-1))  # tensor([1., 1., 1., 1.])

# Temperature scaling (used in knowledge distillation)
temperature = 2.0
soft_probs = torch.special.softmax(logits / temperature, dim=-1)
```

### xlogy

Computes x * log(y) safely: returns 0 when x == 0.

```python
torch.special.xlogy(input, other, *, out=None)
```

```python
x = torch.tensor([0.0, 1.0, 2.0, 3.0])
y = torch.tensor([0.5, 0.5, 0.5, 0.5])

result = torch.special.xlogy(x, y)
print(result)
# tensor([ 0.0000, -0.6931, -1.3863, -2.0794])

# For x=0: xlogy(0, y) = 0 (not NaN or -inf)
# For x!=0: xlogy(x, y) = x * log(y)

# Used in KL divergence computation
def kl_divergence_manual(p, q):
    """KL(p || q) using xlogy for numerical stability."""
    return torch.sum(torch.special.xlogy(p, p) - torch.special.xlogy(p, q))
```

### zeta

Computes the Hurwitz zeta function: zeta(x, q) = sum_{n=0}^{inf} 1 / (q + n)^x

```python
torch.special.zeta(input, other, *, out=None)
```

```python
x = torch.tensor([2.0, 3.0, 4.0])
q = torch.tensor([1.0, 1.0, 1.0])
result = torch.special.zeta(x, q)
print(result)
# tensor([1.6449, 1.2021, 1.0823])  # Riemann zeta: pi^2/6, ~1.2021, ~1.0823

# For q=1, zeta(x, 1) is the Riemann zeta function
# zeta(2, 1) = pi^2/6 = 1.6449...
# Used in analytic number theory and certain statistical distributions
```

### entr

Computes the entropy function: entr(x) = -x * log(x) for x > 0, and 0 for x <= 0.

```python
torch.special.entr(input, *, out=None)
```

```python
x = torch.tensor([0.0, 0.1, 0.25, 0.5, 1.0])
result = torch.special.entr(x)
print(result)
# tensor([0.0000, 0.2303, 0.3466, 0.3466, 0.0000])

# entr(0) = 0
# entr(1) = 0
# Maximum at x = 1/e

# Used in Shannon entropy computation
def shannon_entropy(probs):
    """Compute Shannon entropy of a discrete distribution."""
    return torch.sum(torch.special.entr(probs))

# For uniform distribution over n classes
n = 10
uniform = torch.ones(n) / n
entropy = shannon_entropy(uniform)
print(f"Max entropy for {n} classes: {entropy:.4f}")  # log(10) = 2.3026
```

---

## 9. Complete Examples

### Numerically Stable Cross-Entropy Loss

```python
def stable_cross_entropy(logits, targets, label_smoothing=0.0):
    """Numerically stable cross-entropy with label smoothing."""
    num_classes = logits.shape[-1]

    if label_smoothing > 0:
        # Smooth labels
        smooth = torch.full_like(logits, label_smoothing / num_classes)
        smooth.scatter_(-1, targets.unsqueeze(-1),
                        1.0 - label_smoothing + label_smoothing / num_classes)
        log_probs = torch.special.log_softmax(logits, dim=-1)
        loss = -(smooth * log_probs).sum(dim=-1).mean()
    else:
        # Standard cross-entropy using log_softmax
        log_probs = torch.special.log_softmax(logits, dim=-1)
        loss = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()

    return loss
```

### Focal Loss

```python
def focal_loss(logits, targets, gamma=2.0, alpha=None):
    """Focal loss for handling class imbalance."""
    log_probs = torch.special.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    # Gather target class probabilities
    target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    target_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - target_probs) ** gamma

    loss = -focal_weight * target_log_probs

    if alpha is not None:
        alpha_weight = alpha[targets]
        loss = alpha_weight * loss

    return loss.mean()
```

### Beta Distribution Log-Probability

```python
def beta_log_prob(x, alpha, beta_param):
    """Compute log probability of Beta distribution."""
    # Using gammaln and xlogy for numerical stability
    log_B = (torch.special.gammaln(alpha) + torch.special.gammaln(beta_param)
             - torch.special.gammaln(alpha + beta_param))

    log_prob = (torch.special.xlogy(alpha - 1, x)
                + torch.special.xlogy(beta_param - 1, 1 - x)
                - log_B)

    return log_prob
```

### Gaussian Mixture Model Log-Likelihood

```python
def gmm_log_likelihood(data, weights, means, stds):
    """Compute log-likelihood of data under a Gaussian Mixture Model."""
    N = data.shape[0]
    K = weights.shape[0]

    # Compute log-probability of each data point under each component
    data_expanded = data.unsqueeze(-1)  # [N, 1]
    means_expanded = means.unsqueeze(0)  # [1, K]
    stds_expanded = stds.unsqueeze(0)    # [1, K]

    log_component_probs = (
        -0.5 * torch.log(2 * torch.pi * stds_expanded**2)
        - 0.5 * ((data_expanded - means_expanded) / stds_expanded)**2
    )

    # Add mixing weights
    log_weighted = log_component_probs + torch.log(weights).unsqueeze(0)

    # Log-sum-exp over components
    log_likelihood = torch.special.logsumexp(log_weighted, dim=-1)

    return log_likelihood.sum()
```

### Safe Softmax with Temperature

```python
def temperature_softmax(logits, temperature=1.0, dim=-1):
    """Softmax with temperature scaling, using log_softmax for stability."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    log_probs = torch.special.log_softmax(logits / temperature, dim=dim)
    return log_probs.exp()
```
