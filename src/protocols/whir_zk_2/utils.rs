use ark_ff::FftField;

use super::Config;
use crate::algebra::{geometric_accumulate, geometric_sequence, MultilinearPoint};

/// Derived protocol dimensions for a single zkWHIR 2.0 execution.
///
/// Computed once from the config and reused across prover/verifier steps
/// to avoid recomputing (and passing individually) the same derived values.
#[derive(Clone, Copy)]
pub(super) struct ProtocolDims {
    pub(super) mu: usize,
    pub(super) ell: usize,
    pub(super) rem: usize,
    pub(super) nu: usize,
    pub(super) size: usize,
    pub(super) num_vectors: usize,
    pub(super) num_blinding_vecs: usize,
}

impl ProtocolDims {
    pub(super) fn new<F: FftField>(config: &Config<F>, num_vectors: usize) -> Self {
        let mu = config.blinded_polynomial.initial_num_variables();
        let ell = config.blinding_polynomial.initial_num_variables() - 1;
        let rem = mu % ell;
        let num_blinding_vecs = config.blinding_polynomial.initial_committer.num_vectors;
        let nu = num_blinding_vecs - num_vectors;
        let size = 1 << mu;
        Self {
            mu,
            ell,
            rem,
            nu,
            size,
            num_vectors,
            num_blinding_vecs,
        }
    }

    /// Number of blinding g-polynomials: ν + 1.
    pub(super) const fn num_g_polys(&self) -> usize {
        self.nu + 1
    }
}

/// Extract the ℓ-bit sub-index from a μ-bit hypercube index `b` for the Φ_i projection.
///
/// Implements the Φ_i morphisms from the paper (p. 17):
///   Φ₀(x̄) = (x₁, ..., x_ℓ)
///   Φᵢ(x̄) = (x_{(i-1)·ℓ+rem+1}, ..., x_{i·ℓ+rem})  for i ≥ 1
///
/// This is the integer-index (bit-pattern) version: extracts the same ℓ-bit
/// window that the multivariate Φᵢ would select.
///
/// Index convention (big-endian):
///   index = x_0 · 2^{μ-1} + x_1 · 2^{μ-2} + ... + x_{μ-1} · 2^0
///
/// The result is: `(b >> (μ - start - ℓ)) & ((1 << ℓ) - 1)`
pub(super) const fn phi_i_bits(
    hypercube_idx: usize,
    phi_index: usize,
    mu: usize,
    ell: usize,
    rem: usize,
) -> usize {
    let start = if phi_index == 0 {
        0
    } else {
        (phi_index - 1) * ell + rem
    };
    let shift = mu - start - ell;
    (hypercube_idx >> shift) & ((1 << ell) - 1)
}

/// Compute the discrete logarithm of `target` w.r.t. `gen` in a cyclic group
/// of order `2^log_order`, using the Pohlig-Hellman algorithm.
///
/// Returns `i` such that `target == gen^i`, where `0 ≤ i < 2^log_order`.
/// Panics (in debug builds) if `target` is not in `⟨gen⟩`.
///
/// Complexity: O(log_order²) field multiplications — vs O(2^log_order) for linear scan.
pub(super) fn discrete_log_pow2<F: FftField>(target: F, gen: F, log_order: u32) -> usize {
    let gen_inv = gen.inverse().expect("generator must be invertible");
    let mut result = 0usize;
    let mut current = target;
    let mut gen_inv_power = gen_inv; // gen^{-2^bit} accumulator

    for bit in 0..log_order {
        // current^{2^{log_order - bit - 1}} == 1  ⟺  bit `bit` of the index is 0
        let mut test = current;
        for _ in 0..(log_order - bit - 1) {
            test.square_in_place();
        }

        if test != F::ONE {
            result |= 1 << bit;
            current *= gen_inv_power;
        }

        gen_inv_power.square_in_place();
    }

    debug_assert_eq!(
        gen.pow([result as u64]),
        target,
        "discrete log verification failed: target not in ⟨gen⟩ of order 2^{log_order}"
    );
    result
}

/// Build the μ-variate evaluation point `fold_args(r̄, z)` (paper p. 21).
///
/// fold_args(r̄; z) := (r₁, ..., r_s, z^{2⁰}, z^{2¹}, ..., z^{2^{μ-s-1}})
///
/// Result: `(r_0, ..., r_{s-1}, z^{2^{k-1}}, z^{2^{k-2}}, ..., z^2, z)`
/// where `s = |r̄|` and `k = μ − s`.
///
/// The z-derived coordinates use descending powers (big-endian convention)
/// to match the codebase's `UnivariateEvaluation::mle_evaluate` squaring ladder.
pub(super) fn build_fold_args<F: FftField>(r_bar: &[F], z: F, mu: usize) -> Vec<F> {
    let num_folded_vars = r_bar.len();
    let num_z_vars = mu - num_folded_vars;
    let mut point = Vec::with_capacity(mu);
    point.extend(r_bar);

    // Squaring ladder: z, z², z⁴, ..., z^{2^{num_z_vars-1}}
    let mut z_pow = z;
    let mut z_pows = Vec::with_capacity(num_z_vars);
    for _ in 0..num_z_vars {
        z_pows.push(z_pow);
        z_pow.square_in_place();
    }
    // Reverse to descending order: z^{2^{num_z_vars-1}}, ..., z², z
    point.extend(z_pows.iter().rev());
    point
}

/// Build batched eq tables for the blinding proof — Step 7 (paper pp. 23-24).
///
/// Implements the weight polynomial from Eq. (5):
///   wᵢ(z, ȳ) = z · Σⱼ τⱼ · eq(Φᵢ(P[j]), ȳ)
///
/// `beq_i[k] = Σ_j τ^{j+1} · Σ_{c,m} eq(r̄, c) · z_j^m · δ(Φ_i(c·M+m), k)`
///
/// Used identically by both prover and verifier.
///
/// Optimized via bit-window factorization: since Φ_i extracts a contiguous ℓ-bit
/// window from the μ-bit index b = c·M+m, the delta constraint decomposes into
/// independent constraints on c-bits and m-bits. The inner sum factors as:
///   beq_i[k_c·2^m_cap + k_m] = eq_partial[k_c] · m_inner[k_m]
/// where m_inner is computed via `geometric_accumulate` and eq_partial marginalizes
/// the eq polynomial over uncaptured c-bits. The free m-bits contribute scalar
/// factors via the identity Π(1+z^{2^j}) = Σ z^i.
#[cfg_attr(feature = "tracing", tracing::instrument(skip_all, fields(num_points = lambda_z_points.len(), mu = dims.mu, ell = dims.ell, num_g_polys = dims.num_g_polys())))]
pub(super) fn build_beq_tables<F: FftField>(
    lambda_z_points: &[F],
    eq_weights: &[F],
    tau: F,
    dims: ProtocolDims,
) -> Vec<Vec<F>> {
    let mu = dims.mu;
    let ell = dims.ell;
    let rem = dims.rem;
    let num_g_polys = dims.num_g_polys();
    let half_size = 1usize << ell;
    let num_folding_vars = eq_weights.len().trailing_zeros() as usize;
    assert!(
        num_folding_vars <= ell,
        "folding factor num_folding_vars={num_folding_vars} must not exceed ell={ell} (would underflow m_cap in Φ₀ window)"
    );
    let num_m_bits = mu - num_folding_vars; // number of m-bits (log2 of sub-polynomial length M)

    // Precompute τ powers: [τ, τ², ..., τ^num_points]
    let tau_powers_full = geometric_sequence(tau, lambda_z_points.len() + 1);
    let tau_powers = &tau_powers_full[1..];

    // Precompute squaring ladders z^{2^0}, z^{2^1}, ..., z^{2^{num_m_bits-1}} for each z-point
    let z_pows_all: Vec<Vec<F>> = lambda_z_points
        .iter()
        .map(|z| {
            let mut z_pows = Vec::with_capacity(num_m_bits);
            let mut z_pow = *z;
            for _ in 0..num_m_bits {
                z_pows.push(z_pow);
                z_pow.square_in_place();
            }
            z_pows
        })
        .collect();

    let num_points = lambda_z_points.len();
    let mut tables = vec![vec![F::ZERO; half_size]; num_g_polys];

    for (i, table) in tables.iter_mut().enumerate() {
        let start_i = if i == 0 { 0 } else { (i - 1) * ell + rem };

        // Bit-window decomposition relative to c|m boundary at position num_folding_vars
        let a_below = mu - start_i - ell; // free m-bits below window (= shift_i)
        let a_above = start_i.saturating_sub(num_folding_vars); // free m-bits above
        let m_cap = num_m_bits - a_below - a_above; // captured m-bits in window
        let c_cap = ell - m_cap; // captured c-bits in window (low c_cap bits of c)

        // Partial eq marginalization: eq_partial[k_c] = Σ_{c: c & mask = k_c} eq[c]
        let eq_partial = if c_cap > 0 {
            let mut eq_partial = vec![F::ZERO; 1 << c_cap];
            let c_mask = (1 << c_cap) - 1;
            for (c_idx, &weight) in eq_weights.iter().enumerate() {
                eq_partial[c_idx & c_mask] += weight;
            }
            eq_partial
        } else {
            vec![F::ONE] // Σ eq_weights = 1
        };

        // Build scalars w_j = tp_j · geo_below_j · geo_above_j
        // and bases base_j = z_j^{2^{a_below}} for geometric_accumulate
        let m_cap_size = 1usize << m_cap;
        let mut scalars = Vec::with_capacity(num_points);
        let mut bases = Vec::with_capacity(num_points);

        for (j, &tp) in tau_powers.iter().enumerate() {
            let z_pows = &z_pows_all[j];

            // Free-bit product below: Π_{j=0}^{a_below-1} (1 + z^{2^j})
            let mut geo_below = F::ONE;
            for &zp in z_pows.iter().take(a_below) {
                geo_below *= F::ONE + zp;
            }

            // Free-bit product above: Π_{j=0}^{a_above-1} (1 + z^{2^{a_below+m_cap+j}})
            let mut geo_above = F::ONE;
            for &zp in z_pows.iter().skip(a_below + m_cap).take(a_above) {
                geo_above *= F::ONE + zp;
            }

            scalars.push(tp * geo_below * geo_above);
            bases.push(if a_below < num_m_bits {
                z_pows[a_below]
            } else {
                F::ONE
            });
        }

        // m_inner[k_m] = Σ_j w_j · base_j^{k_m}
        let mut m_inner = vec![F::ZERO; m_cap_size];
        geometric_accumulate(&mut m_inner, scalars, &bases);

        // Assemble: tables[i][k_c · 2^m_cap + k_m] = eq_partial[k_c] · m_inner[k_m]
        if c_cap > 0 {
            for (k_c, &ep) in eq_partial.iter().enumerate() {
                for (k_m, &mi) in m_inner.iter().enumerate() {
                    table[k_c * m_cap_size + k_m] = ep * mi;
                }
            }
        } else {
            *table = m_inner;
        }
    }

    tables
}

/// Precompute RS-fold coefficient vectors for the blinding polynomials (Steps 5-6).
///
/// Used to evaluate m̃(r̄, z, ρ) and g̃ᵢ(r̄, z) at OOD/STIR/Γ points (paper pp. 21-22).
///
/// After the initial sumcheck folds `s` variables with randomness `r̄`, the original
/// 2^μ-coefficient polynomial is viewed as `k = 2^s` sub-polynomials of length `M = 2^(μ-s)`.
/// The RS-fold is:
///
/// ```text
/// fold(r̄, poly)(z) = Σ_m [ Σ_j eq(r̄, j) · poly[j·M + m] ] · z^m
/// ```
///
/// For blinding polynomials, the 2^μ evaluation table is the lift of an ℓ-variate table
/// via Φ_i projections: `lifted[b] = table[Φ_i_bits(b)]`.
///
/// With multi-polynomial batching (n witness polynomials, batching coefficients α):
/// - `m_coeffs_all[0][m] = Σ_j eq · [ĝ₀[Φ₀(j·M+m)] + (-ρ)·msk₀[Φ₀(j·M+m)]]`
/// - `m_coeffs_all[i][m] = (-ρ·αⁱ) · Σ_j eq · mskᵢ[Φ₀(j·M+m)]`  for i = 1..n-1
/// - `g_i_coeffs[i][m] = Σ_j eq · ĝ_{i+1}[Φ_{i+1}(j·M+m)]`  for i = 0..ν-1
///
/// Returns `(m_coeffs_all, g_i_coeffs)` where each vector has length M.
#[cfg_attr(feature = "tracing", tracing::instrument(skip_all, fields(mu = dims.mu, ell = dims.ell, num_g_polys = g_polys.len())))]
pub(super) fn compute_rs_fold_blinding_coeffs<F: FftField>(
    eq_weights: &[F],
    g_polys: &[Vec<F>],
    masking_polys: &[Vec<F>],
    alpha_coeffs: &[F],
    rho: F,
    dims: ProtocolDims,
) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
    let mu = dims.mu;
    let ell = dims.ell;
    let rem = dims.rem;
    let num_folding_vars = eq_weights.len().trailing_zeros() as usize;
    let num_sub_polys = 1usize << num_folding_vars;
    let sub_poly_len = 1usize << (mu - num_folding_vars);
    let num_g_polys = g_polys.len();
    let num_masking = masking_polys.len();
    let neg_rho = -rho;

    // Accumulate g₀ fold coeffs, per-masking fold coeffs, and g_i fold coeffs
    let accumulate_j =
        |g0_acc: &mut Vec<F>, msk_accs: &mut Vec<Vec<F>>, g_acc: &mut Vec<Vec<F>>, j: usize| {
            let eq_j = eq_weights[j];
            for sub_idx in 0..sub_poly_len {
                let full_idx = j * sub_poly_len + sub_idx;
                let phi_0_idx = phi_i_bits(full_idx, 0, mu, ell, rem);

                g0_acc[sub_idx] += eq_j * g_polys[0][phi_0_idx];
                for (i, msk) in masking_polys.iter().enumerate() {
                    msk_accs[i][sub_idx] += eq_j * msk[phi_0_idx];
                }

                for gi in 1..num_g_polys {
                    let phi_i_idx = phi_i_bits(full_idx, gi, mu, ell, rem);
                    g_acc[gi - 1][sub_idx] += eq_j * g_polys[gi][phi_i_idx];
                }
            }
        };

    // Assemble m_coeffs_all from raw g₀ fold and masking folds
    let assemble = |g0_fold: Vec<F>,
                    msk_folds: Vec<Vec<F>>,
                    g_i_coeffs: Vec<Vec<F>>|
     -> (Vec<Vec<F>>, Vec<Vec<F>>) {
        let mut m_coeffs_all = Vec::with_capacity(num_masking);
        // m₀ = g₀_fold + (-ρ) · msk₀_fold
        let m0: Vec<F> = g0_fold
            .iter()
            .zip(msk_folds[0].iter())
            .map(|(&g, &msk)| g + neg_rho * msk)
            .collect();
        m_coeffs_all.push(m0);
        // mᵢ = (-ρ·αⁱ) · mskᵢ_fold for i ≥ 1
        for i in 1..num_masking {
            let scale = neg_rho * alpha_coeffs[i];
            let mi: Vec<F> = msk_folds[i].iter().map(|&v| scale * v).collect();
            m_coeffs_all.push(mi);
        }
        (m_coeffs_all, g_i_coeffs)
    };

    // Sequential outer loop: num_sub_polys is typically small (e.g. 4 = 2^folding_factor),
    // so parallelizing over it via fold/reduce would allocate one full accumulator set
    // (3 × sub_poly_len) per rayon thread — ~200 MB overhead for negligible speedup.
    // Using a single accumulator keeps memory at ~24 MB.
    let mut g0_fold = vec![F::ZERO; sub_poly_len];
    let mut msk_folds = vec![vec![F::ZERO; sub_poly_len]; num_masking];
    let mut g_i_coeffs = vec![vec![F::ZERO; sub_poly_len]; num_g_polys - 1];
    for j in 0..num_sub_polys {
        accumulate_j(&mut g0_fold, &mut msk_folds, &mut g_i_coeffs, j);
    }

    assemble(g0_fold, msk_folds, g_i_coeffs)
}

/// Build weight covectors for Step 7's batched blinding proof (paper pp. 23-24).
///
/// Constructs `n + ν` covectors used identically by both prover and verifier:
///   - `w_0`:      `beq_0[k]` for g₀, `(-ρ)·beq_0[k]` for msk₀
///   - `w_i`:      `(-ρ·αⁱ)·beq_0[k]` for mskᵢ  (1 ≤ i < num_vectors)
///   - `w_{n+j}`:  `beq_{j+1}[k]` for ĝ_{j+1}    (0 ≤ j < ν)
pub(super) fn build_weight_covectors<F: FftField>(
    beq_tables: &[Vec<F>],
    rho: F,
    alpha_coeffs: &[F],
    dims: ProtocolDims,
) -> Vec<Vec<F>> {
    let num_vectors = dims.num_vectors;
    let num_blinding_vecs = dims.num_blinding_vecs;
    let half_size = 1usize << dims.ell;
    let full_size = 1usize << (dims.ell + 1);

    let mut weight_covectors: Vec<Vec<F>> = Vec::with_capacity(num_blinding_vecs);

    // w_0: first M-polynomial weight (includes g₀ and msk₀)
    {
        let mut w0 = vec![F::ZERO; full_size];
        let neg_rho = -rho;
        for k in 0..half_size {
            w0[2 * k] = beq_tables[0][k];
            w0[2 * k + 1] = neg_rho * beq_tables[0][k];
        }
        weight_covectors.push(w0);
    }

    // w_i (1 ≤ i < n): additional M-polynomial weights (masking only, no g₀)
    for &alpha in &alpha_coeffs[1..num_vectors] {
        let mut wi = vec![F::ZERO; full_size];
        let scale = -rho * alpha;
        for k in 0..half_size {
            wi[2 * k + 1] = scale * beq_tables[0][k];
        }
        weight_covectors.push(wi);
    }

    // w_{n+j-1} (1 ≤ j ≤ ν): ĝ_j weights
    for beq_table in beq_tables.iter().skip(1) {
        let mut wj = vec![F::ZERO; full_size];
        for k in 0..half_size {
            wj[2 * k] = beq_table[k];
        }
        weight_covectors.push(wj);
    }

    weight_covectors
}

/// Map gamma points (elements of Ω₁) to their corresponding indices in the
/// initial codeword [[f̂]].
///
/// Each γ ∈ Ω₁ is a power of the round-0 generator. The discrete log gives
/// the index within the round-0 domain, and multiplying by `stride = |Ω₀|/|Ω₁|`
/// recovers the position in the initial codeword.
///
/// Used identically by both prover (to open [[f̂]]) and verifier (to verify openings).
pub(super) fn gamma_to_f_hat_indices<F: FftField>(
    gamma_points: &[F],
    config: &super::Config<F>,
) -> Vec<usize> {
    let initial_codeword_len = config.blinded_polynomial.initial_committer.codeword_length;
    let round0_codeword_len = config.blinded_polynomial.round_configs[0]
        .irs_committer
        .codeword_length;
    let stride = initial_codeword_len / round0_codeword_len;
    let gen_h = config.blinded_polynomial.round_configs[0]
        .irs_committer
        .generator();
    let log_round0_len = round0_codeword_len.trailing_zeros();

    gamma_points
        .iter()
        .map(|&gamma| discrete_log_pow2(gamma, gen_h, log_round0_len) * stride)
        .collect()
}

/// Compute eq_weights from r_bar. Shared helper to avoid redundant computation.
pub(super) fn compute_eq_weights<F: FftField>(r_bar: &[F]) -> Vec<F> {
    MultilinearPoint(r_bar.to_vec()).eq_weights()
}

/// Accumulator for blinding polynomial claims across OOD, STIR, and Γ queries.
///
/// Collects (z, m_evals, g_evals) tuples during Steps 5-6 for use in Step 7.
pub(super) struct LambdaAccumulator<F> {
    pub(super) z_points: Vec<F>,
    pub(super) m_evals: Vec<Vec<F>>,
    pub(super) g_evals: Vec<Vec<F>>,
}

impl<F> LambdaAccumulator<F> {
    pub(super) const fn new() -> Self {
        Self {
            z_points: Vec::new(),
            m_evals: Vec::new(),
            g_evals: Vec::new(),
        }
    }

    pub(super) fn push(&mut self, z: F, m: Vec<F>, g: Vec<F>) {
        debug_assert!(
            self.m_evals.is_empty() || m.len() == self.m_evals[0].len(),
            "m_evals length mismatch: expected {}, got {}",
            self.m_evals.first().map_or(0, Vec::len),
            m.len()
        );
        debug_assert!(
            self.g_evals.is_empty() || g.len() == self.g_evals[0].len(),
            "g_evals length mismatch: expected {}, got {}",
            self.g_evals.first().map_or(0, Vec::len),
            g.len()
        );
        self.z_points.push(z);
        self.m_evals.push(m);
        self.g_evals.push(g);
    }

    #[must_use]
    pub(super) const fn len(&self) -> usize {
        self.z_points.len()
    }
}
