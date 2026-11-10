# Provenance

## Source

* Repository: <https://github.com/Weizhe-Chen/attentive_kernels>, inspected
  revision `1ccb7255e1a09d215af39da3bb1efe6305cf4dea`.
* File: `demo/main.py` (sha256
  `9411046f00d3119dd1bee2c275d2567646dcfe718eb2cfc75c36684314ecd645`), the
  self-contained author implementation. It was read, not imported or vendored.
* Papers:
  * W. Chen, R. Khardon and L. Liu, "AK: Attentive Kernel for Information
    Gathering", Robotics: Science and Systems (RSS) 2022,
    [doi:10.15607/RSS.2022.XVIII.047](https://doi.org/10.15607/RSS.2022.XVIII.047),
    [arXiv:2205.06426](https://arxiv.org/abs/2205.06426).
  * W. Chen, R. Khardon and L. Liu, "Adaptive Robotic Information Gathering
    via Non-Stationary Gaussian Processes", The International Journal of
    Robotics Research (first online 2023),
    [doi:10.1177/02783649231184498](https://doi.org/10.1177/02783649231184498),
    [arXiv:2306.01263](https://arxiv.org/abs/2306.01263).

## Adapted

`kernel.py` adapts `TwoHiddenLayerTanhNN` (same layers and module names) and
the forward computation of the `AK` class: one shared network, its softmax
output L2-normalized per input, and the membership inner product times the
representation-weighted sum of fixed unit-amplitude RBF components. The
experiment, robot simulator, planner, scalers, `robust_cholesky` and GP
regression class of the demo were not adapted.

## Differences from the author code

* Kernel: a GPyTorch `Kernel` wrapped in `ScaleKernel`; the amplitude is the
  `ScaleKernel` outputscale with GPyTorch's plain softplus, not the author's
  own parameter with `softplus + 1e-6`. `diag=True` returns the row-matched
  cross-covariance `k(x1_i, x2_i)`; the author's `diag(x)` is the
  self-covariance (the amplitude). Squared distances are accumulated per
  input dimension instead of squaring `torch.cdist` (roundoff-level
  difference). The network is created in float64 inside a forked RNG with a
  configured seed, instead of float32 initialization followed by `.double()`
  (different initial weights for the same seed).
* Inputs: a fixed, bounds-based square normalizer (`InputNormalizer2d` with
  side `L = max(x extent, y extent)`, so `u` is in `[0, x extent / L] x
  [0, y extent / L]`) instead of a per-axis `MinMaxScaler` fitted to the
  initial samples and mapping to `[-1, 1]`. The primitive lengthscale bank is
  therefore given in our units: default `linspace(0.02, 0.5, 10)` of `L`.
  The author's `linspace(0.01, 0.5, 10)` in `[-1, 1]` units corresponds to
  0.005 to 0.25 of the extent.
* Targets: a fixed affine transform from the configuration
  (`target_offset`, `target_scale`) instead of a `StandardScaler` fitted to
  the initial samples.
* Mean and noise: a learned constant mean instead of a zero mean; GPyTorch
  `GaussianLikelihood` with a configured lower bound on the noise variance
  instead of `softplus + 1e-6`.
* Optimization: one Adam optimizer with two parameter groups (network;
  amplitude, noise and mean) instead of two Adam optimizers, which gives the
  same per-parameter updates. Default learning rates 3e-3 and 3e-2 instead
  of 1e-3 and 1e-2 (same 10:1 ratio). The loss is GPyTorch's exact marginal
  log likelihood averaged per observation instead of the sum.
* Numerics: no `robust_cholesky` jitter (1e-6 to 1e-4 added on failure) and
  no clamping of predictive variances at the jitter. A failed factorization
  or a non-finite loss or gradient rejects the measurement and restores the
  previous state; only roundoff-negative variances are set to zero.
* Schedule: a warm-up of `min_distinct_locations` distinct locations, then
  `initial_training_steps` Adam steps, then `training_steps_per_event` steps
  after every `optimize_every` accepted measurements. The demo starts with 50
  samples along a predefined path and optimizes for as many iterations as
  new samples after each batch.

## License of the source

```text
MIT License

Copyright (c) 2022 Weizhe (Wesley) Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
