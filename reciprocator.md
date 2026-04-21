# Reciprocator: Architecture and Mathematics

This document specifies the Reciprocator as a causal complex tensor state-space
architecture. Its central object is a rank-`r` complex tensor memory that
evolves online while interacting with a complex hidden stream. Where the
mathematical form is an abstraction whose instantiation is numerical rather
than closed-form, that is stated explicitly.

## 1. Core Objects

Given a token sequence `x_{1:T}`, the model evolves:

- a complex hidden stream
  `h_t^{(ell)} in C^D`
- a bank of complex tensor memory states
  `S_{k,t}^{(ell)} in C^{M_1 x ... x M_r}`

Indices:

- `t = 1, ..., T` is token position
- `ell = 1, ..., L` is layer index
- `k = 1, ..., K` is engine index inside a layer
- `r` is tensor rank (number of modes)
- `(M_1, ..., M_r)` are mode sizes, with `M_state = prod_m M_m`
- `a^{(ell)} = (a_1^{(ell)}, ..., a_r^{(ell)}) <= (M_1, ..., M_r)` is the
  active support; slices outside `a^{(ell)}` are masked to zero at every step

The model is causal: for every layer `ell` and time `t`, `h_t^{(ell)}` depends
only on `x_{1:t}`.

## 2. Token Lifting

Tokens enter the model as a learned amplitude modulation of an all-ones carrier;
positions enter as a fixed-frequency rotary phase. For token `x_t` at position
`t`:

```math
rho_tok(x_t)[d] = exp( 0.1 · tanh( a_tok(x_t)[d] ) )          (real, in (e^{-0.1}, e^{0.1}))
rho_pos(t)[d]   = exp( i · t · omega_d )                        (unit complex rotation)
h_t^{(0)}       = rho_tok(x_t) odot rho_pos(t)
```

where `a_tok` is a learned embedding and `omega_d = base^{-d/D}` with
`base = 10000`. The `0.1 · tanh` squashing bounds the log-scale to `[-0.1, 0.1]`
(sub-10% magnitude variation per feature). Positional phase uses RoPE frequencies
and is not learned. The token contribution is purely amplitude; the position
contribution is purely phase.

## 3. Layer Architecture

Each layer is a residual block with a complex causal tensor-state mixer
followed by a complex feedforward map:

```math
u_t^{(ell)}      = cLN( h_t^{(ell-1)} )
Delta_t^{(ell)}  = M^{(ell)}( u_{1:t}^{(ell)} )
hat{h}_t^{(ell)} = h_t^{(ell-1)} + Delta_t^{(ell)}
h_t^{(ell)}      = hat{h}_t^{(ell)} + cFFN^{(ell)}( cLN( hat{h}_t^{(ell)} ) )
```

`cLN` is complex layer normalization (mean-subtract then rescale by magnitude
variance). `cFFN` is a complex feedforward network using a modReLU nonlinearity
that thresholds magnitude while preserving phase. `M^{(ell)}` is the only
cross-position transport mechanism in the Reciprocator-only architecture.

## 4. Tensor Signal

The normalized hidden state is lifted into a complex tensor signal via a polar
parameterization using two real linear maps:

```math
m_t^{(ell)}   = softplus( W_mag^{(ell)} · u_t^{(ell)} )       in R^{M_state}_{>0}
phi_t^{(ell)} = pi · tanh( W_phi^{(ell)} · u_t^{(ell)} )      in (-pi, pi)^{M_state}
Y_t^{(ell)}   = m_t^{(ell)} odot exp( i · phi_t^{(ell)} )      in C^{M_state}
X_t^{(ell)}   = reshape( Y_t^{(ell)}, (M_1, ..., M_r) )        in C^{M_1 x ... x M_r}
```

`W_mag` and `W_phi` are real linear maps. The softplus constraint enforces
strictly positive amplitude. The `pi · tanh` constraint confines the per-feature
phase to `(-pi, pi)`, covering the full angular range without wrap-around
discontinuity. The polar form keeps amplitude and phase as structurally separate
learning targets with explicit geometric meaning.

Before entering the memory dynamics, `X_t^{(ell)}` is masked to the active
support `a^{(ell)}` and passed through the chosen normalization operator to
yield the normalized signal `s_t^{(ell)}`.

The learned `W_mag` and `W_phi` together absorb the role that a
Tikhonov-regularized Gram inverse `(E^* E + epsilon I)^{-1}` would play in a
fixed-dictionary formulation; there is no explicit Gram correction in the
architecture (see §14).

## 5. Normalization

Two normalization families are supported, with an optional learned relaxation.

### Frobenius normalization

```math
N_F(S) = S / ||S||_F
```

### Per-mode normalization

The mathematical object is the fixed point of alternating mode projections:

```math
N_PM(S) = lim_{n -> infty} ( P_r circ ... circ P_1 )^n (S)
```

where `P_m` rescales each nonzero mode-`m` fiber by its `l_2` norm raised to
an exponent `q_m`. The canonical choice `q_m = 1` recovers unit-fiber
normalization; learned `q_m in R^+` allows each mode to control how aggressively
it normalizes. In practice the iteration is unrolled to a fixed depth with a
bounded budget; on well-separated states it converges in a handful of sweeps.

### Learned normalization blend

An optional learned interpolation between Frobenius and per-mode normalization
is supported. A predictor reads the current proposal tensor and produces a
per-entry blend weight `beta in [0, 1]`. The normalized output is:

```math
N_blend(S) = (1 - beta) · N_F(S) + beta · N_PM(S)
```

When the gate saturates (`beta` near 0 or 1), the blend recovers one of the two
hard families exactly. The predictor is trained end-to-end; its output adapts
the normalization geometry to the local signal. Per-mode step sizes `q_m` are
jointly active with the blend: the effective per-mode family uses the same
learned exponents regardless of how much weight the gate assigns it.

## 6. Engine Bank Dynamics

Each layer contains a bank of `K` tensor engines. Engine `k` at time `t`
receives:

- the current normalized signal `s_t^{(ell)}`
- its own previous state `S_{k,t-1}^{(ell)}`
- a same-time carry `C_{k-1,t}^{(ell)}` from the previous engine, with
  `C_{0,t}^{(ell)} = 0`
- a per-entry magnitude accumulator `acc_{k,t-1}^{(ell)}` tracking an
  exponential moving average of the previous state magnitude

### Base gain fields

Per-engine static parameter tensors of shape `(M_1, ..., M_r)` under bounded
parameterizations:

```math
D_k^{(ell)}     = sigmoid( D_k_raw^{(ell)} )      # decay,      in (0, 1)
A_k^{(ell)}     = sigmoid( A_k_raw^{(ell)} )      # input gain, in (0, 1)
B_k^{(ell)}     = tanh   ( B_k_raw^{(ell)} )      # recurrent,  in (-1, 1)
Gamma_k^{(ell)} = tanh   ( Gamma_k_raw^{(ell)} )  # carry gain, in (-1, 1)
```

### Input-dependent gain biases

An optional predictor — enabled by default — computes an additive bias
`Delta(s_t, Z_{k,t})` for each gain logit from two paths:

1. **Signal path.** A small elementwise MLP over `[Re(s_t), Im(s_t), |s_t|]`
   (three real features per tensor entry) produces a 4-channel delta of shape
   `(4, M_1, ..., M_r)`.

2. **Context path.** Summary statistics of the relational product
   `Z_{k,t} = s_t odot S_{k,t-1}` are extracted: the circular-mean
   unit-phase components `(bar{u}_{Re}, bar{u}_{Im})`, the mean magnitude
   `bar{|Z|}`, and the RMS fiber energy along each active mode. These
   `(r + 3)` scalar statistics are mapped linearly to a delta of shape
   `(4, M_1, ..., M_r)`.

The two deltas add to form `[Delta_D, Delta_A, Delta_B, Delta_Gamma]`, masked
to the active support. An optional learned scalar selection gate (derived from
the same signal and context features) can further attenuate the full delta,
allowing the predictor to suppress itself on timesteps where the relational
signal is weak.

The effective gain logits are:

```math
D_logit   = D_k_raw   + Delta_D(s_t, Z_{k,t})
A_logit   = A_k_raw   + Delta_A(s_t, Z_{k,t})
B_logit   = B_k_raw   + Delta_B(s_t, Z_{k,t})
Gamma_logit = Gamma_k_raw + Delta_Gamma(s_t, Z_{k,t})
```

Base gains are then:

```math
D_k^{(ell)}     = sigmoid( D_logit )
A_k^{(ell)}     = sigmoid( A_logit )
B_k^{(ell)}     = tanh   ( B_logit )
Gamma_k^{(ell)} = tanh   ( Gamma_logit )
```

### Accumulator modulation

The magnitude accumulator and effective gains are:

```math
acc_{k,t}^{(ell)}  = lambda_acc · acc_{k,t-1}^{(ell)}
                   + (1 - lambda_acc) · | S_{k,t-1}^{(ell)} |

A_eff^{(ell)}     = A_k^{(ell)}     odot (1 + softplus(alpha_A^{(ell)})     · acc_{k,t-1}^{(ell)})
B_eff^{(ell)}     = B_k^{(ell)}     odot (1 + softplus(alpha_B^{(ell)})     · acc_{k,t-1}^{(ell)})
Gamma_eff^{(ell)} = Gamma_k^{(ell)} odot (1 + softplus(alpha_Gamma^{(ell)}) · acc_{k,t-1}^{(ell)})
```

where `lambda_acc = 0.9` is fixed and `alpha_A, alpha_B, alpha_Gamma` are
learned per-engine scalars. Decay `D_k` is not accumulator-modulated.

### Coupling drive

The coupling input is scaled by the accumulator before entering `Cpl`:

```math
Z_{k,t}^{(ell)} = s_t^{(ell)} odot S_{k,t-1}^{(ell)}
                  · (1 + softplus(alpha_cpl^{(ell)}) · acc_{k,t-1}^{(ell)})
```

A learned complex linear prediction map `P_k^{(ell)}: C^{M_state} -> C^{M_state}`
(initialized to zero) forms a base forecast of the incoming signal from the
previous state. Two optional accumulator-modulated terms extend the predictor: a
**state-continuation** path that extrapolates the state directly, and a
**coupling-continuation** path that applies the coupling operator to the current
state as a one-step forward projection:

```math
hat{s}_{k,t}^{(ell)}
  = P_k^{(ell)}( S_{k,t-1}^{(ell)} )
  + ( 1 + softplus( alpha_pred^{(ell)} ) · acc_{k,t-1}^{(ell)} )
    · ( tanh( gamma_S^{(ell)} ) · S_{k,t-1}^{(ell)}
      + tanh( gamma_R^{(ell)} ) · Cpl_{k,t-1}^{(ell)}( S_{k,t-1}^{(ell)} ) )

e_{k,t}^{(ell)} = s_t^{(ell)} - hat{s}_{k,t}^{(ell)}
```

where `gamma_S` and `gamma_R` are learnable scalars and `alpha_pred` is a
learnable accumulator modulation scale. All three are initialized to zero, so
the extended terms are inert at initialization and activate only if the optimizer
finds them beneficial. The coupling-continuation term applies the coupling
operator to the state meeting itself — `Cpl_{k,t-1}(S_{k,t-1})` — projecting
the state's own relational structure forward one step. The prediction error is an
additive drive on the update with rate `eta`.

The full update is:

```math
R_{k,t}^{(ell)} = Cpl_{k,t}^{(ell)}( Z_{k,t}^{(ell)} )

widetilde{S}_{k,t}^{(ell)}
  = D_k^{(ell)}       odot S_{k,t-1}^{(ell)}
  + A_eff^{(ell)}     odot s_t^{(ell)}
  + B_eff^{(ell)}     odot R_{k,t}^{(ell)}
  + Gamma_eff^{(ell)} odot C_{k-1,t}^{(ell)}
  + eta · e_{k,t}^{(ell)}

S_{k,t}^{(ell)} = N( widetilde{S}_{k,t}^{(ell)} )
C_{k,t}^{(ell)} = S_{k,t}^{(ell)}
```

Three recurrence structures emerge:

- temporal recurrence inside each engine:
  `S_{k,t-1}^{(ell)} -> S_{k,t}^{(ell)}`
- bank recurrence across engines at fixed time:
  `C_{k-1,t}^{(ell)} -> C_{k,t}^{(ell)}`
- accumulator recurrence tracking retained state magnitude:
  `acc_{k,t-1}^{(ell)} -> acc_{k,t}^{(ell)}`

## 7. State-Conditioned Mode Coupling

`Cpl_{k,t}^{(ell)}` is the only non-diagonal operator in the update. It is
constructed from the accumulator-scaled local tensor `Z_{k,t}^{(ell)}` (§6)
as a sequential composition of complex per-mode mixing matrices.

### Per-mode mixing matrices

Fix a mode `m`. Let `Z^{(m-1)}` be the tensor after modes `1, ..., m-1`
have been applied (with `Z^{(0)} = Z_{k,t}^{(ell)}`). Flatten `Z^{(m-1)}`
so that mode `m` indexes the rows and the remaining modes index a single
column axis, giving a complex matrix
`Z_m in C^{M_m x (M_state / M_m)}`. A learned complex square
`W_m^{(ell,k)} in C^{M_m x M_m}` produces a state-conditioned score matrix:

```math
Score_m = ( W_m · Z_m ) · Z_m^T                             in C^{M_m x M_m}
```

The coupling matrix is complex-valued. Its magnitude is row-stochastic
(each row's magnitude sums to one), and its phase encodes the routing
direction from the score. Let `c_m = sqrt( M_state / M_m )`:

```math
T_{k,t}^{(ell, m)}
  = softmax_{last axis}( |Score_m| / (tau · c_m) )
    odot ( Score_m / |Score_m| )                             in C^{M_m x M_m}
```

where `tau > 0` is a learned coupling temperature scalar, `|Score_m|`
denotes entry-wise modulus, and the softmax operates on real magnitudes only.
Phase participates in routing through `Score_m`; the softmax selects
destination weights by magnitude, and the unit-phasor factor preserves the
directional rotation from the bilinear score.

### Sequential (expressive) composition

Mode couplings are applied in sequence, each computed from the partially-mixed
tensor after earlier modes have been contracted. The source tensor for mode `m`
is `Z^{(m-1)}`, not the original `Z^{(0)}`:

```math
Z^{(0)} = Z_{k,t}^{(ell)}
Z^{(m)} = T_{k,t}^{(ell, m)} ·_m Z^{(m-1)},  m = 1, ..., r
R_{k,t}^{(ell)} = Z^{(r)}
```

where `T ·_m Z` denotes the mode-`m` matrix product. Because `T_{k,t}^{(ell,m)}`
is computed from `Z^{(m-1)}` (not `Z^{(0)}`), later-mode routing adapts to
what earlier modes have already mixed. This makes the composition strictly
expressive: mode coupling matrices are not independent.

## 8. Hidden-Space Return Map

Each engine state is summarized by a real feature map concatenating its real
part, imaginary part, magnitude, and magnitude accumulator:

```math
psi(S, acc) = [ Re(S), Im(S), |S|, acc ] in R^{4 M_state}
```

The layer-level hidden correction is formed by per-engine complex linear
returns (lifted from real features to complex outputs), a complex fusion, and
a real sigmoid gate:

```math
d_{k,t}^{(ell)} = W_ret_k^{(ell)} · psi( S_{k,t}^{(ell)}, acc_{k,t}^{(ell)} )  in C^D
d_t^{(ell)}     = W_fus^{(ell)} · [ d_{1,t}^{(ell)} ; ... ; d_{K,t}^{(ell)} ]
g_t^{(ell)}     = sigmoid(
                    W_gate^{(ell)} ·
                    [ Re(u_t^{(ell)}), Im(u_t^{(ell)}),
                      Re(d_t^{(ell)}), Im(d_t^{(ell)}) ]
                  )                                                in R^D
Delta_t^{(ell)} = g_t^{(ell)} odot d_t^{(ell)}
```

The gate is real and is broadcast elementwise over both real and imaginary
components of `d_t^{(ell)}`. `W_ret_k` maps real features to complex outputs
(implemented as a complex linear map that treats its real input as a
zero-imaginary complex tensor).

## 9. Readout Geometry

After the final layer the hidden state remains complex. Two real-valued
readouts are supported.

### Magnitude readout

```math
R_mag(h_t) = | h_t |                                              in R^D
```

### Phase-aware readout (U(1)-invariant bilinear)

Let `bar{h}_t = (1/D) sum_i h_t[i]` be the feature-mean reference. Because
`bar{h}_t` rotates under the same global U(1) phase as `h_t`, the bilinear
cross-product is globally phase-invariant:

```math
c_t = h_t odot overline{bar{h}_t}                                 in C^D
R_phase(h_t) = [ Re( c_t ), Im( c_t ), | h_t | ]                  in R^{3D}
```

Phase is preserved as relative-phase features while the global gauge is
removed continuously.

Vocabulary logits are produced by a real output map:

```math
z_t = W_out · R( h_t^{(L)} )
```

The magnitude readout discards phase. The phase-aware readout preserves
relative phase while projecting out the global U(1) action.

## 10. Capacity Allocation

All tensor state is defined over a maximum support `(M_1, ..., M_r)` with an
active support `a^{(ell)} <= (M_1, ..., M_r)` that selects live slices;
inactive slices are masked to zero at every step and contribute no signal to
any downstream linear map.

Two distinct growth axes are supported:

- **Mode-size growth.** Within a fixed rank `r`, the active size in the
  most-novel mode increments by one.
- **Rank growth.** When the current active rank `r_active < r_max`, sustained
  novelty can promote a previously inactive mode to active, incrementing the
  rank and initializing the new mode's slice from the mean residual of the
  active region.

Both axes are governed by two geometric statistics:

- **novelty**, a per-mode ratio of prediction-error energy
  `|e_{k,t}|^2 = |s_t - P_k(S_{k,t-1})|^2` to signal energy `|s_t|^2`
- **usage**, an exponential moving average of per-coordinate state magnitude

On an eligible step with novelty above a threshold, growth (mode-size or rank)
is triggered; the newly activated slice is seeded with the mean of the residual
along the growing axis. Sustained underuse (running magnitude below a floor for
at least a fixed horizon) decrements the active size in that mode.

Within any single forward pass the active support is held constant, so the
forward map is symmetric across training and evaluation regimes. Capacity
adaptation is a separable procedure that may be run in a dedicated phase,
operating on the same state geometry that the forward map defines.

## 11. Streaming Form

The serial mixer only requires:

- the current hidden state
- the current engine-bank carry
- the previous tensor states

so the model admits a true streaming regime. Once a prefix has built the
state

```math
{ S_{k,t}^{(ell)} }_{ell, k}
```

advancing by one new token does not require replaying the consumed prefix.

## 12. Parallel Temporal Form

A sequence-parallel variant separates the update into a linear recurrence and
a nonlinear correction. For each engine, the linear drive absorbs the input
and carry terms (the bank carry is fixed at the last state of the preceding
engine, with `C_0 = 0`):

```math
u_{k,t}^{(ell)} = A_k^{(ell)} odot s_t^{(ell)} + Gamma_k^{(ell)} odot C_{k-1}^{(ell)}
L_{k,t}^{(ell)} = D_k^{(ell)} odot L_{k,t-1}^{(ell)} + u_{k,t}^{(ell)}
```

The linear part is evaluated for all `t` by an inclusive prefix scan over the
semiring `(a_2, b_2) * (a_1, b_1) = (a_2 a_1, a_2 b_1 + b_2)`.

**Provisional accumulator.** Accumulator-modulated gains require a causal
estimate of `acc_{k,t-1}` before the true nonlinear state is known. The
parallel form resolves this with a two-pass approach: a first prefix scan
yields a provisional linear state `L^{prov}_{k,t}`, whose magnitude is used
to compute a provisional accumulator `acc^{prov}_{k,t-1}`. Gains are then
modulated by this provisional accumulator before the final scan.

**Nonlinear correction.** The coupling is instantiated from the provisional
linear state (using magnitude-marginal partial traces rather than the
expressive bilinear of the serial form), and the prediction error is computed
against `L^{prov}_{k,t-1}`:

```math
widetilde{S}_{k,t}^{(ell)}
  = L_{k,t}^{(ell)}
  + B_k^{(ell)} odot Cpl_{k,t}^{(ell)}( s_t^{(ell)} odot L_{k,t-1}^{prov(ell)} )
  + eta · e_{k,t}^{(ell)}
```

followed by normalization.

The parallel form preserves the causal state-space geometry but trades three
properties of the serial form for temporal parallelism:

- the inter-engine bank carry becomes a single end-of-engine hand-off rather
  than a per-time chain
- the coupling operator uses magnitude-marginal partial traces on the
  provisional linear state rather than the expressive sequential bilinear
- the accumulator is estimated from the provisional state rather than the
  true normalized state

It is therefore an asymptotic, not bit-exact, match to the serial mixer.

## 13. Architectural Parameters

- hidden width `D`
- layer depth `L`
- tensor rank `r` (initial active rank)
- maximum tensor rank `r_max >= r` (dynamic rank ceiling)
- maximum mode sizes `(M_1, ..., M_r_max)`
- initial active support `a <= (M_1, ..., M_r)`
- number of engines per layer `K`
- normalization choice (Frobenius, per-mode, or learned blend)
- learned per-mode normalization step sizes `q in R^r` (optional; default: ones)
- learned normalization blend enabled/disabled; when enabled, a predictor gate
  interpolates between Frobenius and per-mode normalization per-entry
- phase scale `pi_phi in R^+` (multiplier on tanh-bounded phase; set to `pi`
  to cover the full circle without wrap-around discontinuity)
- static gain fields `(D_k_raw, A_k_raw, B_k_raw, Gamma_k_raw)` with
  sigmoid/tanh squashings
- accumulator decay rate `lambda_acc` (fixed at 0.9) and per-engine
  accumulator modulation scales `(alpha_A, alpha_B, alpha_Gamma, alpha_cpl)`
- input-dependent gains enabled/disabled; when enabled, gain-predictor
  MLP hidden dim and optional selective gate
- prediction error rate `eta` (fixed or learned)
- optional extended predictor: state-continuation scale `gamma_S`,
  coupling-continuation scale `gamma_R`, and accumulator modulation
  scale `alpha_pred` (all learnable scalars; zero-initialized)
- coupling temperature `tau`
- readout choice (magnitude or phase-aware)
- growth threshold, growth interval, prune floor, prune horizon
- dynamic rank enabled/disabled
- mixer temporal form (serial or parallel)

## 14. Intentional Non-Features

These omissions are design choices, not oversights.

- **No attention.** Cross-position transport is carried entirely by the
  tensor-state mixer. The complex backbone has no self-attention layers.
- **No hard input-gating.** The input-dependent gain predictor (§6) adds
  additive biases to gain logits; it does not saturate or zero-out gains the
  way selective SSMs (Mamba/S6, GLA, DeltaNet) do. The parameterization
  continues to constrain gains through the sigmoid/tanh squashings; the
  predictor shifts the operating point without bypassing the bounds.
- **No explicit Gram inverse.** The role of `(E^* E + epsilon I)^{-1}` from a
  fixed-dictionary formulation is absorbed into the learned signal-lifting maps
  (`W_mag`, `W_phi`) and downstream weights. There is no analytic overlap
  correction in the forward.
- **No phase-equivariant readout.** The phase-aware readout removes the global
  U(1) gauge continuously via a bilinear invariant, then feeds real features
  through a real linear layer. It preserves relative-phase information as
  features but does not impose equivariance.
