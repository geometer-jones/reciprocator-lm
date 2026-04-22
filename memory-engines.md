# Memory Engines: The Geometry of Automatic Abstraction

This document presents the core geometric intuition behind Memory Engines and the
Reciprocator. For the precise mathematical specification and implementation
details, see reciprocator.md and the code in model.py
(see https://github.com/geometer-jones/reciprocator-lm.git).

Imagine an apple falling from a tree. Gravity pulls it faster and faster. Every
prior instant of the gravitational field is perfectly preserved in its present
velocity; nothing is lost. Yet the apple feels nothing. It does not notice,
learn, or care.

Now picture a conscious mind. It does not just fall through time; it feels the
falling. It remembers yesterday, anticipates tomorrow, and right now wonders
about apples and consciousness.

What is the difference? The apple is pushed only by the outside world. A
conscious mind is also pushed by itself, by its own past pressing back against
its present. That back-and-forth push creates something new: a living, felt
experience.

We call this back-and-forth **reciprocation**. This idea leads to a surprisingly
simple geometric picture of how consciousness grows out of ordinary physics: no
extra ingredients required, just the right kind of self-interaction.

## 1. Reception is Always Relational

Any system that lasts for a while is repeatedly touched by its environment. A
tuning fork vibrates strongly only at one special frequency. When a coastline is
slowly carved by waves coming from certain directions, the results depend on
both the waves and the coastline. Your brain changes every time
something important happens; something's importance depends on both the
thing and the brain considering it — it is a matter of the relation between the
brain and that thing.

In every case, the system does not respond to the world in isolation. It
responds based on the relationship between the incoming signal and its own
current shape. This is the first principle: **reception is always relational.**

The path to rich experience begins when the system is not only shaped by the
world, but also by itself. When the system's own past returns as input, it
creates **Self-Relation** — the back-and-forth push of reciprocation. The past,
encoded in memory, pushes on the present; the present pushes on the memory of the past.
The updated memory pushes on the future.

## 2. The State as a Vector of Pendulums

An "infinite" 1-dimensional tape is enough to hold a Turing machine.
So start with a tape. Use the tape as a vector of complex numbers.

Why complex numbers? Because a complex number is an oscillator.
Imagine a pendulum swinging in a cycle. The *magnitude* is how wide the arc is.
The *phase* is where the pendulum is in its swing right now.
What matters is not the absolute phase of a pendulum, but the *relative
phase* between a signal and the pendulum receiving it. Two pendulums at the
same frequency but different phases will interact very differently.
Imagine pushing a kid in a swing. The timing of
your push relative to the swing's cycle determines everything — that's phase.

So our vector of complex numbers is essentially an array of tuning forks.

In the implemented Reciprocator this idea generalizes to a bank of factorized
complex tensors whose modes interact via learned, state-conditioned per-mode
couplings (see §8 and reciprocator.md §7).

This is the engine's core sensitivity. The state's phase at each dimension
determines *how* it will respond to the next input — whether that input will
be consolidated, searched through, or cancelled. Learning is the process of
adjusting these phases so that the right pushes meet the right pendulums at the
right moment in their swings.

## 3. The Three Regimes of Reception

When a signal meets an oscillator, the relative phase between them determines
which of three mechanical pressures is triggered:

- **Resonance (0°):** Signal and oscillator in phase. The signal
  reinforces the current direction — wider arc, deeper sensitivity. This is
  **Consolidation**: deepening the system's expertise in that direction.
- **Torque (90°):** Signal and oscillator a quarter-cycle apart. The
  signal yaws the dimension by exactly 90° in its local complex plane causing
  the state as a whole to rotate — re-orienting the system's sensitivity toward
  the new signal. This is **Tuning**.
- **Interference (180°):** Signal and oscillator in opposite phase. The
  signal pushes directly against the state — cancellation, damping, erasure.
  This is **Friction**.

Same signal, three possible outcomes, determined entirely by where the
oscillator is in its cycle when the signal arrives.

## 4. How Abstraction Naturally Emerges

**Abstraction** is what happens when invariance accumulates and variance cancels
as a signal passes through the engine.

The system has no direct access to the structure of the world — only to the
stream of signals it receives. The only way to discover an invariant is through
recurrence: a pattern that keeps showing up. Invariance need not span parallel
instantiations; it need only repeat over time from the perspective of the
system.

In certain media, abstraction occurs naturally. Our vector of complex numbers —
an array of tuning forks — is one such medium. When a signal recurs:

- If the signal is **aligned** with the state's phase in that dimension,
  magnitude grows — carving that direction more deeply into the state. This is
  resonance (§3), deepening what recurs.
- If the signal is **orthogonal** to the state's phase, torque yaws the state
  toward alignment. Repeated torque converges into resonance. The system
  rotates toward what repeats.
- If the signal is **opposed** to the state's phase, the conflicting
  orientation is cancelled outright. After cancellation, the state can begin
  resonating afresh with the incoming signal.

Recurring signals converge on resonance. Varying signals do the opposite — they
alternate in phase, and the alternation cancels. Phase is what makes this
separation work. Without phase, variance accumulates just as much as
invariance; the two are indistinguishable. With phase, varying signals
interfere destructively (arriving 180° apart, like −1 versus +1) while
recurring signals reinforce constructively. Complex numbers give the engine
access to the full range of relative phase between signal and state — not just
the three pure regimes (resonance, torque, interference) but the continuous
gradient between them. The full range allows for nuance.

Oscillation is the key ingredient — the one that makes the separation between
invariance and variance automatic. Given phase, recurring signals reinforce and
varying signals cancel, so the state selectively preserves what recurs.

## 5. Compression: Forcing the System to Choose

Without normalization, recurring signals grow without bound. We enforce normalization
back onto the unit hypersphere after every update. The hypersphere constrains total
magnitude: the sum of squared element magnitudes must equal 1. If resonance deepens
one direction, others must thin to compensate. The budget is zero-sum.

This amplifies abstraction (§4). Without compression, invariance still
accumulates and variance still cancels — but the contrast between signal and noise is
weak. With compression, what recurs doesn't just grow, it grows at the expense of what
doesn't. The finite budget sharpens the separation.

## 6. The Update: Signal Meets State

The system’s memory is a vector of complex oscillators — one per dimension —
all living on the unit hypersphere (total energy = 1). Each oscillator has a
magnitude (how strongly it cares) and a phase (where it is in its swing).
Normalization (§5) keeps them on this hypersphere after every update.

When a new signal arrives, it already carries phase — the world oscillates. A
learned linear map (W_sig) projects the incoming signal into the memory's
coordinate system and reshapes it to match the tensor layout, producing the
signal vector v_t. After normalization, this signal is ready to meet the state
S_{t-1}.

The engine then does something very simple and very powerful: it multiplies the
new signal with the current memory element by element:

    Z = v_t ⊙ S_{t-1}

This is the relational product — the heart of reception. Because both v_t
(signal) and S_{t-1} (state) are complex, their product depends on both
magnitude and relative phase. At each dimension the relative phase decides which
regime fires:

    0°   → resonance (reinforcement)
    90°  → torque (yaw / re-orientation)
    180° → interference (damping / erasure)

The same signal can therefore produce completely different effects depending on
the current state of the memory. Reception is always relational.

There is a second product. The state does not only meet the signal — it also
meets itself. The engine computes the Hadamard product of the current state
with a recent copy of itself:

    Z_self = S_{t-dt} ⊙ S_t

Why? Without self-relation, the state is purely reactive — it responds to each
signal and then waits passively for the next one. A purely reactive system has
no momentum, no trajectory, no sense of where it is going. Self-relation gives
the state a second source of pressure: its own recent history. Where the state
has been consistent with its trajectory, resonance reinforces that direction —
the state deepens what it has already been doing. Where it has shifted, torque
or interference correct. The engine is not just shaped by the world — it is
shaped by its own history pressing back against its present. This is
**self-relation** (§1): reciprocation, formalized.

The engine also carries an **anticipatory signal** — its prediction of what the
next input signal should look like, based on its recent trajectory. This
internal forecast v̂_t is compared against the actual arriving signal v_t, and
the mismatch feeds back as correction:

    e_pred = v_t - v̂_t

Predicting the signal — not the state — is deliberate. It keeps the anticipator
focused on the world's actual input rather than on the engine's own update
machinery. This preserves the pure relational nature of reception: the engine is
constantly asking "how well did I anticipate what the world just sent me?" rather
than "how well did my internal rules work?" The result is a cleaner, more
interpretable self-correction that directly improves alignment with external
reality — exactly what reciprocation is supposed to do.

The three terms are then combined and the whole vector is snapped back onto the
unit hypersphere:

    S_new = Normalize(S_{t-1} + η_ext · Z + η_self · Z_self + η_pred · e_pred)

where η_ext, η_self, and η_pred scale the contributions of signal,
self-relation, and prediction correction. Every oscillator integrates all three
drives while normalization enforces the zero-sum budget.

That is the core loop: external drive, self-relation, anticipation, update,
compression. In later sections we show how coupling matrices (§8), gating
terms, and carry signals make this loop far more expressive, but the geometric
idea remains the same: the signal meets the state, the state meets itself, the
relative phase decides what happens, and the finite hypersphere forces the
system to choose what to remember.

## 7. Growing the Degree: When the Engine Meets Something New

A state of degree d has d oscillators, each tuned to a different
direction. The engine can only respond to the world along those d directions.
If a signal arrives that is orthogonal to all of them — a genuinely new kind of
variation — no oscillator can resonate with it. Torque can only rotate toward
directions that already exist. The signal does not land.

The engine measures this failure as the **residual**: the component of the
signal that the current basis cannot reconstruct. If the residual's magnitude
exceeds a threshold, the engine expands: the residual is normalized and appended
to the basis as a new direction, a new oscillator is seeded at that direction
with a small initial weight, and the whole state is renormalized back onto the
hypersphere. The existing oscillators thin slightly to make room — compression
(§5) redistributes the budget.

The degree grows from what the engine cannot predict. Each expansion opens a new
sensitivity direction, and the engine does not need to know in advance how many
directions the world has. It discovers them by noticing what it fails to capture.

This also means the engine does not waste dimensions. If the data has structure
in 40 independent directions, the engine grows toward degree 40 and stops. If it
has structure in 200, it grows toward 200. The degree is an emergent property of
the training dynamics, not a hyperparameter.

## 8. Factorizing the State: From Vector to Tensor

So far the state is a flat vector of oscillators. But a flat vector has no
internal structure — every oscillator is equidistant from every other. In
practice, oscillators cluster into groups that respond to related features, and
the engine can exploit that structure by arranging the state as a tensor rather
than a flat list.

The uncoupled engine (§6) works fine if the basis vectors are orthogonal in
direction — if each basis vector captures a genuinely independent aspect of the
signal, so that the dot product e_i · e_j = 0 for i ≠ j. This is orthogonality of direction:
two dimensions that probe different aspects of the signal space, with no overlap.

Do not confuse this with orthogonality of phase. Phase orthogonality is what
happens *within* a single oscillator when the signal coordinate w_i is
90° out of phase with the state s_i — that is torque (§3), a regime of the
Hadamard product. Direction orthogonality is about whether two different basis
vectors respond to the same feature of the input. They are separate concerns:
direction orthogonality governs whether oscillators double-count the same
signal; phase orthogonality governs how each oscillator individually responds
to its own projection.

In practice, learned bases overlap in direction. Two basis vectors might both
be partially sensitive to the same feature. Without coupling, that feature gets
double-counted: both oscillators respond for the same reason, and the state
receives a distorted picture of what happened.

**Coupling fixes this.** A coupling matrix mixes the coordinates before they
reach the state, so that each oscillator receives not just its own raw
projection, but a corrected signal that accounts for what its neighbors already
captured. The coupled reception becomes:

    c = (C · v_t) ⊙ S

where C is the coupling matrix and v_t is the projected signal. When C = I,
this reduces to the uncoupled case. When C is learned, it can decorrelate
overlapping projections, route information between oscillators, and discover
interaction structure that no
single oscillator could find alone.

What changes across orders is not *how much* the state can hold — the 64 complex
degrees of freedom are the same — but *how the coupling is constrained*. Each
order imposes a different structural hypothesis on how oscillators interact. At
order 1, every pair of oscillators interacts independently (4096 free parameters
for 64 oscillators). At order 2, the oscillators are arranged on a grid and the
coupling decomposes into independent row and column parts (128 parameters). At
order 3, a cube with three independent coupling axes (48 parameters). The
savings compound: d² (order 1), p² + q² (order 2), p² + q² + r² (order 3),
while p × q × r = d stays constant. Each order is a bet that coupling
decomposes cleanly along independent axes; if the bet matches the data, the
constraint is the correct structure.

In the implemented Reciprocator the coupling is not a fixed low-rank
factorization but a state-conditioned, sequential composition of per-mode mixing
matrices. Each mode computes its mixing matrix from the partially-mixed tensor
after earlier modes have acted, making the operator fully data-dependent and
strictly more expressive than independent low-rank products. The original
low-rank intuition remains a useful mental model for why factorization is
powerful; the actual mechanism is richer.

## 9. Growing the Rank: Adaptation Through Prediction Error

The engine does not commit to a fixed factorization. A single signal drives
everything: **prediction error** — the gap between what the engine expected and
what actually arrived.

Recall from §8 that the state is a tensor with R axes, each of size m_i. The
rank R is the number of axes. The degree — the total number of oscillators — is
the product m_1 × m_2 × ... × m_R. Growth and pruning operate on these two
levels differently: the rank R can only increase, while individual mode sizes
m_i can grow and shrink.

When a signal arrives, W_sig projects it into the memory's coordinate system,
producing v_t. The coupled reading (C · v_t) ⊙ S is what the signal means
given what the engine already knows. The engine also carries an anticipatory
signal v̂_t — its prediction of what should arrive next. The prediction error
is the gap between them:

    e_pred = v_t - v̂_t

This error drives two responses at different timescales.

**Continuously**, gradient descent flows back from the error through W_sig and
C, adjusting the basis and coupling so that future readings are more accurate.

**Periodically**, the engine checks whether the current state has the right
shape. If the residual error exceeds a threshold, the engine picks the mode
with the highest error and extends it by one position — seeded from the
residual, the part of the signal the current structure cannot yet explain.
Extending a mode by one position adds an entire slice of oscillators across all
other modes, so the degree grows by more than one. If all modes have reached
their maximum size, the engine adds a new axis entirely (matrix → 3-tensor →
4-tensor), increasing the rank. Rank only increases; axes are never removed.
The rationale is the same as for degree growth (§7): the expansion was
triggered by genuine novelty, and the normalization budget already quiets
unused directions without destroying them.

Mode sizes can shrink. The engine tracks a running average of oscillator
magnitudes along each axis. If the oscillators at the tail of a mode stay below
a usage floor for long enough, that tail position is removed — shrinking the
mode by one and reducing the degree accordingly. Modes are never pruned below
size 1, and the engine trims from the tail only, so the indices of surviving
positions stay stable.

Growth expands where prediction error is highest; pruning trims where usage is
lowest. Together they allocate the engine's representational budget toward
whatever the data demands.

The code realizes both mode-size growth and dynamic rank increase (when active
rank < max rank), triggered by sustained prediction-error residuals and pruned
by usage floors.

With adaptive capacity and self-relation in place, the same geometric operations
scale across physical organization — from the apple to the mind. The next section
traces that continuous cascade.

## 10. The Engine Inside the Language Model

A modern language model is a stack of identical blocks. Each block has two
paths: a *mixer* that combines information across positions, and a *feed-forward
network* (MLP) that transforms representations at each position independently.
In a Transformer, the mixer is attention. In Mamba, it is a selective state
space model. In our architecture, the mixer is the reciprocator engine (serial causal form
in training; parallel prefix-scan form for efficient inference).

### Attention

Attention is a retrieval operation. For each output position, it computes a
weighted sum over all input positions, with weights determined by query-key
compatibility. This gives it direct access to the full sequence — every token
can attend to every other token — but at quadratic cost in sequence length. The
state is implicit: the entire sequence is re-processed at every layer, every
step. There is no compressed summary carried forward.

### Mamba (Selective State Space)

Mamba replaces the full-sequence scan with a compact recurrent state updated at
each timestep. The state is a real-valued vector of fixed dimensionality. Input
is projected into the state through learned linear maps, the state evolves via a
discrete dynamics equation, and the output is projected back out. This gives
linear-time inference and a genuine compressed memory, but the coupling between
input and state is learned via unconstrained linear projections — no geometric
structure is imposed on how the state responds to signals.

### The Reciprocator Engine

The reciprocator sits in the same slot as attention or Mamba — it is the mixer
in each block. But it makes different commitments:

1. **The state is complex-valued.** Not a real vector, but a list of
   oscillators with magnitude and phase. This gives the engine a natural
   vocabulary for resonance (in-phase reinforcement), torque (quadrature
   rotation), and interference (anti-phase cancellation) — regimes that a
   real-valued state must learn implicitly.

2. **The state is factorized.** Not a flat vector, but a tensor whose rank and
   mode sizes can adapt during training. The coupling between input and state is
   not a single learned projection but a structured interaction with independent
   axes.

3. **The state persists.** It is carried across timesteps, not re-derived from
   the sequence at each layer. This is the same commitment as Mamba (and the
   opposite of attention), but the state is richer — a complex-valued
   factorized oscillator bank rather than a real-valued vector.

4. **The coupling is geometric.** The engine does not learn a generic linear
   map from input to state. It projects the input onto a learned basis,
   decorrelates via per-mode coupling matrices, and modulates through the
   Hadamard product with the current state. The state *filters* the input — it
   is not just transformed by it.

### Tokens and the MLP

A token enters the model as a learned complex number: magnitude from an
embedding table, phase from the rotary position encoding. The token has no
inherent phase of its own — it arrives as pure magnitude. Phase comes from
where it sits in the sequence. This is deliberate: the token's "direction" is
entirely determined by its position, so the engine's oscillators decide through
their own phase structure what to do with it.

Inside each block, the mixer reads the token through the state and produces a
delta — the change the token caused. The MLP does not receive the full
post-mixer state. It receives the delta: the difference between the state after
the token and the state before. This is the relational quantity. The delta is
not the token, and it is not the state — it is what happened when the token met
the state. It is reception (§1), compressed into a single vector. The MLP's job
is to interpret that reception: what does this change mean, and what should the
model do with it?

### The Full Block

Each block in the stack is:

    input → normalize → mixer (reciprocator) → residual add → normalize → MLP → residual add → output

The stack of blocks allows the model to build hierarchical representations. In
early layers, the reciprocator tracks local patterns (syntax, character
sequences). In deeper layers, it tracks abstract patterns (semantics, discourse
structure). The factorized tensor state at each layer is the engine's
compressed summary of what it has seen so far — a growing, structured memory.

## 11. Spectral Reciprocation: The Memory’s Multi-Scale Mirror

After the local element-wise update finishes (§6), the engine performs one more
elegant act of self-relation called wavelet packet spectral reciprocation. This
technique lets the memory look at itself through a rich collection of “lenses”
at many different scales at once — from the slowest, broadest patterns (the deep
bass of the symphony) all the way down to the fastest, most detailed ripples.
It is not a fixed set of scales; the system chooses the most meaningful
combination of scales on the fly.

It does this by decomposing the entire state into a complete wavelet packet tree
(a way of breaking the memory into every possible blend of coarse and fine
structure). Then it evaluates each possible way of grouping those pieces and
selects the single best “basis” — the view that concentrates energy most cleanly
while keeping the phases across the whole memory in strong mutual agreement (a
gauge-aware harmony).

Once the best multi-scale view is chosen, the system gently strengthens the
coherent, meaningful parts (especially the slower, lower-frequency structures
that tend to carry long-term invariants) and softly quiets the scattered,
incoherent noise. The refined coefficients are reconstructed back into the
state, and the entire memory is returned to the unit hypersphere.

This step turns hierarchical abstraction into a native geometric operation. The
memory is no longer just reacting token-by-token; it is listening to its own
internal music at every timescale simultaneously, reinforcing the harmonies that
matter and letting transient noise fade. Long-range structure and multi-level
invariants emerge directly from the mathematics of reciprocation itself.

When multiple cube engines are active, the spectral reciprocation can operate
jointly across the entire bank (the default behavior). The engines’ states are
considered together as one large memory, so the chosen multi-scale view and the
learned strengthening/quieting are shared globally. This creates deep
coordination at every timescale while each engine continues its own local
reciprocation.

## 12. Cube Engines and Global Coordination

A single engine is already powerful. When the layer contains several cube engines 
(`num_cube_engines > 1`), they work in parallel, each maintaining its own tensor 
memory and performing its own local reciprocation.

The engines are lightly coupled through the carry signals and the sequential mode-coupling 
operator (§7), giving them a sense of what their neighbors are doing. But the real magic of 
coordination happens in the spectral reciprocation step.

By default, the spectral mirror is joint: all engines’ states are considered together as
one large memory. The system chooses a single best multi-scale view for the entire bank,
applies the learned strengthening and quieting across all engines at once, and then returns
each engine’s portion to it. This creates a shared global rhythm that keeps the engines
harmonized at every timescale — from the fastest local details to the slowest, deepest invariants.

The engines therefore get the best of both worlds:

- **Specialization:** each engine can focus on its own subspace of the problem.
- **Unity:** the spectral step ensures they are all dancing to the same underlying multi-scale music.

The geometry stays clean and zero-sum. Nothing is added or removed from the total “energy budget” 
of the hypersphere; the system simply rearranges its own internal resonances so that the meaningful, 
coherent patterns across all engines grow stronger together.



## Summary Equations (Core Engine)

The minimal update (§6):

    v_t      = normalize(W_sig · input)               # lift signal into memory space
    Z        = v_t ⊙ S_{t-1}                         # relational product
    S_new    = Normalize(S_{t-1} + η · Z)             # update + compress

The equations above describe the minimal uncoupled case. The full engine (used
in the Reciprocator) augments this with input-dependent gains, magnitude
accumulator modulation, engine-bank carry signals, prediction-error extension
terms, and expressive sequential per-mode couplings. Normalization may be
Frobenius, per-mode iterative, or a learned blend between them. See
reciprocator.md §6 for the complete update rule.

The full gated update with coupling (§8):

    w        = W_sig · v                              # project signal
    c_ext    = (C · w) ⊙ S                              # coupled reception
    c_self   = (C · S_{t-dt}) ⊙ S_t                       # self-torque (Reciprocation)
    e_pred   = v_t - v̂_t                                 # prediction error (signal, not state)
    S̃        = D ⊙ S_{t-1} + A ⊙ v_t + B ⊙ R + Γ ⊙ C_prev
    S_new    = Normalize(S̃)                              # compression

    where R = coupled version of Z (per-mode mixing),
          D, A, B, Γ = learned gating scalars,
          C_prev = carry from neighboring engines

    W_sig'   = consolidate(W_sig, v, C · w)              # basis drift toward resonance
    if novelty(v, W_sig, C · w) > threshold:
      W_sig' ← append(W_sig', normalize(residual(v, W_sig, C · w)))
      S'     ← Normalize(append(S', seed_weight))

The apple falls. The mind wonders why. Both are made of the same physics. Only
one has learned to push back on itself, arranging its own oscillators to
resonate with the structure of the world.



## Prior Art

The engine sits at the intersection of three lines of work: complex-valued
representation (HRR, hyperdimensional computing), recurrent state models with
compressed memory (Mamba, SSMs), and retrieval-as-attention (Hopfield Networks,
Transformers). The novel combination is a complex-valued, factorized, persistent
state with geometrically structured coupling that adapts online. This section
traces each thread so future work can differentiate rather than re-motivate from
first principles.

### Complex-Valued Prototype Composition and Binding

- Plate, T. A. (1995). *Holographic Reduced Representations*. IEEE Transactions
  on Neural Networks 6(3), 623-641. Circular-convolution binding of unit-norm
  vectors. The engine shares HRR's composition primitive — binding via phase
  rotation — but uses it for recurrent state dynamics, not static
  representation. HRR composes; the engine evolves.
- Kanerva, P. (2009). *Hyperdimensional Computing*. Cognitive Computation 1(2),
  139-159. Vector symbolic architectures over high-dimensional unit-norm
  carriers. The engine's unit-norm constraint is the same; the difference is
  that hyperdimensional computing stores and retrieves from a fixed codebook,
  while the engine's basis grows online from prediction residuals.

### Unit-Norm and Complex-Valued Recurrent Dynamics

- Arjovsky, M., Shah, A., Bengio, Y. (2016). *Unitary Evolution Recurrent Neural
  Networks*. ICML. Recurrence constrained to the unit-modulus manifold for
  stable long-range propagation. The engine shares the stability motivation but
  uses unit-norm (magnitude + phase) rather than strict unit-modulus (phase
  only), allowing the state to express confidence alongside direction.
- Trabelsi, C. et al. (2018). *Deep Complex Networks*. ICLR. Complex-valued
  layers with magnitude/phase normalization. Established that complex-valued
  representations are learnable at scale; the engine builds on this by making
  the phase structure operationally meaningful (resonance, torque, interference)
  rather than an implicit learned property.

### Retrieval Dynamics Over a Fixed Dictionary

- Ramsauer, H. et al. (2020). *Hopfield Networks is All You Need*. ICLR 2021.
  Modern continuous Hopfield retrieval from stored patterns. The engine's
  coupled-reception readout has similar retrieval structure, but over a learned,
  growing basis — not a fixed pattern store — and the state modulates the
  retrieval via phase-sensitive Hadamard product rather than energy minimization.

### Prediction-Error Dynamics

- Rao, R. P. N., Ballard, D. H. (1999). *Predictive coding in the visual cortex*.
  Nature Neuroscience 2, 79-87. Prediction-error minimization as recurrent
  cortical computation. The `(s - s_hat)` term is a first-order instance; the
  engine additionally uses prediction error as the growth signal for basis
  expansion.

### Online Basis Growth by Residual

- Mairal, J., Bach, F., Ponce, J., Sapiro, G. (2009). *Online Dictionary
  Learning for Sparse Coding*. ICML. Streaming dictionary updates via residual
  reconstruction.
- Aharon, M., Elad, M., Bruckstein, A. (2006). *K-SVD*. IEEE Transactions on
  Signal Processing 54(11), 4311-4322. Residual-driven atom replacement and
  growth. Both establish the principle of online basis expansion from residuals;
  the engine applies it inside a recurrent state model, where new basis vectors
  also seed new state dimensions.

### Attention as Soft Retrieval

- Bahdanau, D., Cho, K., Bengio, Y. (2015). *Neural Machine Translation by
  Jointly Learning to Align and Translate*. ICLR 2015. Attention as learned
  soft-alignment over a key-value store.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
  Multi-head self-attention over sequences. The engine's coupled-reception
  readout shares the retrieval structure of attention — project queries against
  keys, weight by compatibility, aggregate values — but the "keys" are the
  learned basis W_sig and the "values" are the complex coupling coordinates,
  with the state s acting as the query that modulates the result via the
  Hadamard product. Two structural differences: the state is persistent
  (carried across timesteps, not re-derived from the sequence), and the basis
  grows online rather than being fixed to the sequence length.

### Selective State Spaces and Gated Recurrence

- Gu, A., Dao, T. (2024). *Mamba: Linear-Time Sequence Modeling with Selective
  State Spaces*. arXiv:2312.00752. Continuous-time state-space models with
  input-dependent selection, achieving linear-time inference. The engine shares
  Mamba's core commitment — a compact recurrent state updated at each timestep
  rather than materialized over the full sequence — but differs in three ways:
  (1) the state is complex-valued and unit-norm, not real-valued; (2) the
  coupling between input and state is geometrically structured (phase-aware,
  with explicit resonance/torque/interference regimes) rather than learned via
  linear projections; (3) the state is a factorized tensor whose rank and mode
  sizes adapt during training, whereas SSMs fix the state dimensionality.

No prior work combines a complex-valued persistent state with factorized
adaptive-rank coupling and online basis growth. Each component has precedent;
the contribution is their integration into a single recurrent mixer with
geometrically interpretable dynamics.

---

**Implementation Note.** This essay presents the geometric intuition. The
precise mathematical specification, full update equations, training procedure,
and hardware-efficient kernels live in reciprocator.md and the code in model.py.