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
pub(super) fn phi_i_bits(b: usize, phi_index: usize, mu: usize, ell: usize, rem: usize) -> usize {
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

    for i in 0..num_g_polys {
        let start_i = if i == 0 { 0 } else { (i - 1) * ell + rem };

        // Bit-window decomposition relative to c|m boundary at position s
        let a_below = mu - start_i - ell; // free m-bits below window (= shift_i)
        let a_above = if start_i >= s { start_i - s } else { 0 }; // free m-bits above
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
            for jj in 0..a_below {
                geo_below *= F::ONE + z_pows[jj];
            }

            // Free-bit product above: Π_{j=0}^{a_above-1} (1 + z^{2^{a_below+m_cap+j}})
            let mut geo_above = F::ONE;
            for jj in 0..a_above {
                geo_above *= F::ONE + z_pows[a_below + m_cap + jj];
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
                    tables[i][k_c * m_cap_size + k_m] = ep * mi;
                }
            }
        } else {
            tables[i] = m_inner;
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
/// This function computes:
/// - `m_coeffs[m] = Σ_j eq(r̄, j) · [ĝ₀[Φ₀(j·M+m)] + (-ρ)·msk[Φ₀(j·M+m)]]`
/// - `g_i_coeffs[i][m] = Σ_j eq(r̄, j) · ĝᵢ[Φᵢ(j·M+m)]`  for i = 1..ν
///
/// Returns `(m_coeffs, g_i_coeffs)` where each vector has length M.
/// To evaluate at a point z, use `univariate_evaluate(&coeffs, z)`.
#[cfg_attr(feature = "tracing", tracing::instrument(skip_all, fields(mu, ell, num_g_polys = g_polys.len())))]
pub(super) fn compute_rs_fold_blinding_coeffs<F: FftField>(
    r_bar: &[F],
    g_polys: &[Vec<F>],
    masking_poly: &[F],
    rho: F,
    mu: usize,
    ell: usize,
    rem: usize,
) -> (Vec<F>, Vec<Vec<F>>) {
    let s = r_bar.len();
    let k = 1usize << s; // number of sub-polynomials
    let big_m = 1usize << (mu - s); // length of each sub-polynomial
    let num_g_polys = g_polys.len();
    let neg_rho = -rho;

    // Precompute eq(r̄, j) for all j in 0..k
    let eq_weights = MultilinearPoint(r_bar.to_vec()).eq_weights();

    let accumulate_j =
        |m_acc: &mut Vec<F>, g_acc: &mut Vec<Vec<F>>, j: usize| {
            let eq_j = eq_weights[j];
            for m in 0..big_m {
                let full_idx = j * big_m + m;

                let phi_0_idx = phi_i_bits(full_idx, 0, mu, ell, rem);
                m_acc[m] += eq_j * (g_polys[0][phi_0_idx] + neg_rho * masking_poly[phi_0_idx]);

                for i in 1..num_g_polys {
                    let phi_i_idx = phi_i_bits(full_idx, i, mu, ell, rem);
                    g_acc[i - 1][m] += eq_j * g_polys[i][phi_i_idx];
                }
            }
        };

    #[cfg(feature = "parallel")]
    {
        return (0..k)
            .into_par_iter()
            .fold(
                || {
                    (
                        vec![F::ZERO; big_m],
                        vec![vec![F::ZERO; big_m]; num_g_polys - 1],
                    )
                },
                |(mut m_acc, mut g_acc), j| {
                    accumulate_j(&mut m_acc, &mut g_acc, j);
                    (m_acc, g_acc)
                },
            )
            .reduce(
                || {
                    (
                        vec![F::ZERO; big_m],
                        vec![vec![F::ZERO; big_m]; num_g_polys - 1],
                    )
                },
                |(mut m_a, mut g_a), (m_b, g_b)| {
                    for (a, &b) in m_a.iter_mut().zip(m_b.iter()) {
                        *a += b;
                    }
                    for (ga, gb) in g_a.iter_mut().zip(g_b.iter()) {
                        for (a, &b) in ga.iter_mut().zip(gb.iter()) {
                            *a += b;
                        }
                    }
                    (m_a, g_a)
                },
            );
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut m_coeffs = vec![F::ZERO; big_m];
        let mut g_i_coeffs = vec![vec![F::ZERO; big_m]; num_g_polys - 1];
        for j in 0..k {
            accumulate_j(&mut m_coeffs, &mut g_i_coeffs, j);
        }
        (m_coeffs, g_i_coeffs)
    }
}
