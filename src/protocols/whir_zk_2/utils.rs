use ark_ff::FftField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::algebra::{geometric_accumulate, geometric_sequence, MultilinearPoint};

/// Extract the ℓ-bit sub-index from a μ-bit hypercube index `b` for the Φ_i variable block.
///
/// This is the integer-index analogue of `calculate_phi_i` — it extracts the same
/// variable block but operates on hypercube indices (bit patterns) rather than field
/// element slices.
///
/// Index convention (big-endian):
///   index = x_0 · 2^{μ-1} + x_1 · 2^{μ-2} + ... + x_{μ-1} · 2^0
///
/// The result is: `(b >> (μ - start - ℓ)) & ((1 << ℓ) - 1)`
pub(super) const fn phi_i_bits(
    b: usize,
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
    (b >> shift) & ((1 << ell) - 1)
}

/// Build the μ-variate evaluation point `fold_args(r̄, z)`.
///
/// Result: `(r_0, ..., r_{s-1}, z^{2^{k-1}}, z^{2^{k-2}}, ..., z^2, z)`
/// where `s = |r̄|` and `k = μ − s`.
///
/// The z-derived coordinates use descending powers (big-endian convention)
/// to match the codebase's `UnivariateEvaluation::mle_evaluate` squaring ladder.
pub(super) fn build_fold_args<F: FftField>(r_bar: &[F], z: F, mu: usize) -> Vec<F> {
    let s = r_bar.len();
    let k = mu - s;
    let mut point = Vec::with_capacity(mu);
    point.extend(r_bar);

    // Squaring ladder: z, z², z⁴, ..., z^{2^{k-1}}
    let mut z_pow = z;
    let mut z_pows = Vec::with_capacity(k);
    for _ in 0..k {
        z_pows.push(z_pow);
        z_pow.square_in_place();
    }
    // Reverse to descending order: z^{2^{k-1}}, ..., z², z
    point.extend(z_pows.iter().rev());
    point
}

/// Build batched eq tables for the blinding proof (Step 7).
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
#[cfg_attr(feature = "tracing", tracing::instrument(skip_all, fields(num_points = lambda_z_points.len(), mu, ell, num_g_polys)))]
pub(super) fn build_beq_tables<F: FftField>(
    lambda_z_points: &[F],
    r_bar: &[F],
    tau: F,
    mu: usize,
    ell: usize,
    rem: usize,
    num_g_polys: usize,
) -> Vec<Vec<F>> {
    let half_size = 1usize << ell;
    let s = r_bar.len();
    let n = mu - s; // number of m-bits (log2 of sub-polynomial length M)
    let eq_weights = MultilinearPoint(r_bar.to_vec()).eq_weights();

    // Precompute τ powers: [τ, τ², ..., τ^num_points]
    let tau_powers_full = geometric_sequence(tau, lambda_z_points.len() + 1);
    let tau_powers = &tau_powers_full[1..];

    // Precompute squaring ladders z^{2^0}, z^{2^1}, ..., z^{2^{n-1}} for each z-point
    let z_pows_all: Vec<Vec<F>> = lambda_z_points
        .iter()
        .map(|z| {
            let mut z_pows = Vec::with_capacity(n);
            let mut zp = *z;
            for _ in 0..n {
                z_pows.push(zp);
                zp.square_in_place();
            }
            z_pows
        })
        .collect();

    let num_points = lambda_z_points.len();
    let mut tables = vec![vec![F::ZERO; half_size]; num_g_polys];

    for (i, table) in tables.iter_mut().enumerate().take(num_g_polys) {
        let start_i = if i == 0 { 0 } else { (i - 1) * ell + rem };

        // Bit-window decomposition relative to c|m boundary at position s
        let a_below = mu - start_i - ell; // free m-bits below window (= shift_i)
        let a_above = start_i.saturating_sub(s); // free m-bits above
        let m_cap = n - a_below - a_above; // captured m-bits in window
        let c_cap = ell - m_cap; // captured c-bits in window (low c_cap bits of c)

        // Partial eq marginalization: eq_partial[k_c] = Σ_{c: c & mask = k_c} eq[c]
        let eq_partial = if c_cap > 0 {
            let mut ep = vec![F::ZERO; 1 << c_cap];
            let c_mask = (1 << c_cap) - 1;
            for (c, &w) in eq_weights.iter().enumerate() {
                ep[c & c_mask] += w;
            }
            ep
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
            bases.push(if a_below < n { z_pows[a_below] } else { F::ONE });
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

/// Precompute RS-fold coefficient vectors for the blinding polynomials.
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
#[cfg_attr(feature = "tracing", tracing::instrument(skip_all, fields(mu, ell, num_g_polys = g_polys.len())))]
pub(super) fn compute_rs_fold_blinding_coeffs<F: FftField>(
    r_bar: &[F],
    g_polys: &[Vec<F>],
    masking_polys: &[Vec<F>],
    alpha_coeffs: &[F],
    rho: F,
    mu: usize,
    ell: usize,
    rem: usize,
) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
    let s = r_bar.len();
    let k = 1usize << s;
    let big_m = 1usize << (mu - s);
    let num_g_polys = g_polys.len();
    let num_masking = masking_polys.len();
    let neg_rho = -rho;

    let eq_weights = MultilinearPoint(r_bar.to_vec()).eq_weights();

    // Accumulate g₀ fold coeffs, per-masking fold coeffs, and g_i fold coeffs
    let accumulate_j =
        |g0_acc: &mut Vec<F>,
         msk_accs: &mut Vec<Vec<F>>,
         g_acc: &mut Vec<Vec<F>>,
         j: usize| {
            let eq_j = eq_weights[j];
            for m in 0..big_m {
                let full_idx = j * big_m + m;
                let phi_0_idx = phi_i_bits(full_idx, 0, mu, ell, rem);

                g0_acc[m] += eq_j * g_polys[0][phi_0_idx];
                for (i, msk) in masking_polys.iter().enumerate() {
                    msk_accs[i][m] += eq_j * msk[phi_0_idx];
                }

                for gi in 1..num_g_polys {
                    let phi_i_idx = phi_i_bits(full_idx, gi, mu, ell, rem);
                    g_acc[gi - 1][m] += eq_j * g_polys[gi][phi_i_idx];
                }
            }
        };

    // Assemble m_coeffs_all from raw g₀ fold and masking folds
    let assemble =
        |g0_fold: Vec<F>,
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

    #[cfg(feature = "parallel")]
    {
        let (g0_fold, msk_folds, g_i_coeffs) = (0..k)
            .into_par_iter()
            .fold(
                || {
                    (
                        vec![F::ZERO; big_m],
                        vec![vec![F::ZERO; big_m]; num_masking],
                        vec![vec![F::ZERO; big_m]; num_g_polys - 1],
                    )
                },
                |(mut g0_acc, mut msk_accs, mut g_acc), j| {
                    accumulate_j(&mut g0_acc, &mut msk_accs, &mut g_acc, j);
                    (g0_acc, msk_accs, g_acc)
                },
            )
            .reduce(
                || {
                    (
                        vec![F::ZERO; big_m],
                        vec![vec![F::ZERO; big_m]; num_masking],
                        vec![vec![F::ZERO; big_m]; num_g_polys - 1],
                    )
                },
                |(mut g0_a, mut msk_a, mut g_a), (g0_b, msk_b, g_b)| {
                    for (a, &b) in g0_a.iter_mut().zip(g0_b.iter()) {
                        *a += b;
                    }
                    for (ma, mb) in msk_a.iter_mut().zip(msk_b.iter()) {
                        for (a, &b) in ma.iter_mut().zip(mb.iter()) {
                            *a += b;
                        }
                    }
                    for (ga, gb) in g_a.iter_mut().zip(g_b.iter()) {
                        for (a, &b) in ga.iter_mut().zip(gb.iter()) {
                            *a += b;
                        }
                    }
                    (g0_a, msk_a, g_a)
                },
            );
        #[allow(clippy::needless_return)]
        return assemble(g0_fold, msk_folds, g_i_coeffs);
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut g0_fold = vec![F::ZERO; big_m];
        let mut msk_folds = vec![vec![F::ZERO; big_m]; num_masking];
        let mut g_i_coeffs = vec![vec![F::ZERO; big_m]; num_g_polys - 1];
        for j in 0..k {
            accumulate_j(&mut g0_fold, &mut msk_folds, &mut g_i_coeffs, j);
        }
        assemble(g0_fold, msk_folds, g_i_coeffs)
    }
}
