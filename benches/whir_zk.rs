//! Benchmark: non-ZK vs ZK v1 vs ZK v2 WHIR proving.
//!
//! Run with:
//!   cargo bench --bench whir_zk
//!
//! Or filter to a specific group:
//!   cargo bench --bench whir_zk -- non_zk
//!   cargo bench --bench whir_zk -- zk_v1
//!   cargo bench --bench whir_zk -- zk_v2

use std::sync::Arc;

use ark_std::{
    rand::{rngs::StdRng, SeedableRng},
    UniformRand,
};
use divan::{black_box, AllocProfiler, Bencher};
use whir::{
    algebra::{
        fields::{Field64, Field64_2},
        ntt::RSDefault,
        poly_utils::{coeffs::CoefficientList, multilinear::MultilinearPoint},
    },
    hash,
    parameters::{FoldingFactor, MultivariateParameters, ProtocolParameters, SoundnessType},
    transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
    whir::{
        config::WhirConfig,
        statement::{Statement, Weights},
        zk::{ZkParams, ZkPreprocessingPolynomials},
    },
};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

type F = Field64;
type EF = Field64_2;

/// Polynomial sizes to benchmark (log₂ of number of coefficients).
const SIZES: &[usize] = &[16, 18, 20];

// ────────────────────────────────────────────────────────────────────────────
//  Shared setup helpers
// ────────────────────────────────────────────────────────────────────────────

/// Build a deterministic polynomial with `2^num_variables` coefficients.
fn make_polynomial(num_variables: usize) -> CoefficientList<F> {
    CoefficientList::new(vec![F::from(1u64); 1 << num_variables])
}

/// Build a single-evaluation statement for the given polynomial.
fn make_statement(polynomial: &CoefficientList<F>, num_variables: usize) -> Statement<EF> {
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let point = MultilinearPoint::rand(&mut rng, num_variables);
    let eval = polynomial.evaluate_at_extension(&point);
    let mut statement = Statement::new(num_variables);
    statement.add_constraint(Weights::evaluation(point), eval);
    statement
}

/// Standard non-ZK WHIR configuration.
fn non_zk_config(num_variables: usize) -> WhirConfig<EF> {
    let mv = MultivariateParameters::new(num_variables);
    let params = ProtocolParameters {
        initial_statement: true,
        security_level: 32,
        pow_bits: 0,
        folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
        soundness_type: SoundnessType::ConjectureList,
        starting_log_inv_rate: 1,
        batch_size: 1,
        hash_id: hash::SHA2,
    };
    let rs = Arc::new(RSDefault);
    WhirConfig::new(rs.clone(), rs, mv, &params)
}

/// ZK v2 main WHIR configuration (round-0 fold = 2 for small k).
fn zk_main_config(num_variables: usize) -> WhirConfig<EF> {
    let mv = MultivariateParameters::new(num_variables);
    let params = ProtocolParameters {
        initial_statement: true,
        security_level: 32,
        pow_bits: 0,
        folding_factor: FoldingFactor::ConstantFromSecondRound(2, 4),
        soundness_type: SoundnessType::ConjectureList,
        starting_log_inv_rate: 1,
        batch_size: 1,
        hash_id: hash::SHA2,
    };
    let rs = Arc::new(RSDefault);
    WhirConfig::new(rs.clone(), rs, mv, &params)
}

/// ZK v2 helper WHIR configuration, tuned for the given ZK params.
fn zk_helper_config(zk_params: &ZkParams) -> WhirConfig<EF> {
    let helper_vars = zk_params.ell + 1;
    let mv = MultivariateParameters::new(helper_vars);
    let params = ProtocolParameters {
        initial_statement: true,
        security_level: 32,
        pow_bits: 0,
        folding_factor: FoldingFactor::Constant(5),
        soundness_type: SoundnessType::ConjectureList,
        starting_log_inv_rate: 2,
        batch_size: zk_params.mu + 1,
        hash_id: hash::SHA2,
    };
    let rs = Arc::new(RSDefault);
    WhirConfig::new(rs.clone(), rs, mv, &params)
}

/// ZK v1: WHIR config for committing [[f̂, g]] together (batch_size=2, μ+1 variables).
fn zk_v1_commit_config(num_variables: usize) -> WhirConfig<EF> {
    let extended = num_variables + 1;
    let mv = MultivariateParameters::new(extended);
    let params = ProtocolParameters {
        initial_statement: true,
        security_level: 32,
        pow_bits: 0,
        folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
        soundness_type: SoundnessType::ConjectureList,
        starting_log_inv_rate: 1,
        batch_size: 2,
        hash_id: hash::SHA2,
    };
    let rs = Arc::new(RSDefault);
    WhirConfig::new(rs.clone(), rs, mv, &params)
}

/// ZK v1: WHIR config for proving P (batch_size=1, μ+1 variables).
fn zk_v1_prove_config(num_variables: usize) -> WhirConfig<EF> {
    let extended = num_variables + 1;
    let mv = MultivariateParameters::new(extended);
    let params = ProtocolParameters {
        initial_statement: true,
        security_level: 32,
        pow_bits: 0,
        folding_factor: FoldingFactor::ConstantFromSecondRound(4, 4),
        soundness_type: SoundnessType::ConjectureList,
        starting_log_inv_rate: 1,
        batch_size: 1,
        hash_id: hash::SHA2,
    };
    let rs = Arc::new(RSDefault);
    WhirConfig::new(rs.clone(), rs, mv, &params)
}

/// ZK v1 polynomial bundle: f̂(x,y) = f(x) + y·msk(x), random g(x,y), P = ρ·f̂ + g.
struct ZkV1Polys {
    f_hat: CoefficientList<F>,
    g_poly: CoefficientList<F>,
    p_poly: CoefficientList<F>,
}

fn make_zk_v1_polys(num_variables: usize) -> ZkV1Polys {
    let mut rng = StdRng::seed_from_u64(0xCAFE);
    let num_coeffs = 1usize << num_variables;
    let extended_num_coeffs = 1usize << (num_variables + 1);
    let polynomial = make_polynomial(num_variables);

    // f̂(x,y) = f(x) + y·msk(x)
    let mut f_hat_coeffs = vec![F::from(0u64); extended_num_coeffs];
    for (i, &c) in polynomial.coeffs().iter().enumerate() {
        f_hat_coeffs[i] = c;
    }
    for i in 0..num_coeffs {
        f_hat_coeffs[num_coeffs + i] = F::rand(&mut rng);
    }
    let f_hat = CoefficientList::new(f_hat_coeffs);

    // Random g(x,y)
    let g_coeffs: Vec<F> = (0..extended_num_coeffs)
        .map(|_| F::rand(&mut rng))
        .collect();
    let g_poly = CoefficientList::new(g_coeffs);

    // P = ρ·f̂ + g
    let rho = F::rand(&mut rng);
    let p_coeffs: Vec<F> = f_hat
        .coeffs()
        .iter()
        .zip(g_poly.coeffs().iter())
        .map(|(&fh, &gv)| rho * fh + gv)
        .collect();
    let p_poly = CoefficientList::new(p_coeffs);

    ZkV1Polys {
        f_hat,
        g_poly,
        p_poly,
    }
}

/// Build a statement for the (μ+1)-variable polynomial P, evaluated at (ā, 0).
fn make_zk_v1_statement(
    p_poly: &CoefficientList<F>,
    num_variables: usize,
) -> Statement<EF> {
    let extended = num_variables + 1;
    let mut rng = StdRng::seed_from_u64(0xBEEF); // same seed as make_statement → same ā
    let base_point = MultilinearPoint::rand(&mut rng, num_variables);
    let mut coords = base_point.0;
    coords.push(EF::from(0u64)); // y = 0
    let extended_point = MultilinearPoint(coords);
    let eval = p_poly.evaluate_at_extension(&extended_point);
    let mut statement = Statement::new(extended);
    statement.add_constraint(Weights::evaluation(extended_point), eval);
    statement
}

// ────────────────────────────────────────────────────────────────────────────
//  NON-ZK benchmarks
// ────────────────────────────────────────────────────────────────────────────

#[divan::bench(args = SIZES)]
fn non_zk_commit(bencher: Bencher, num_variables: usize) {
    let polynomial = make_polynomial(num_variables);
    let config = non_zk_config(num_variables);
    let ds = DomainSeparator::protocol(&config)
        .session(&format!("bench-non-zk-commit-{num_variables}"))
        .instance(&Empty);

    bencher
        .with_inputs(|| {
            let prover_state = ProverState::new_std(&ds);
            (prover_state, &polynomial, &config)
        })
        .bench_values(|(mut prover_state, poly, cfg)| {
            let _ = black_box(cfg.commit(&mut prover_state, &[poly]));
        });
}

#[divan::bench(args = SIZES)]
fn non_zk_prove(bencher: Bencher, num_variables: usize) {
    let polynomial = make_polynomial(num_variables);
    let config = non_zk_config(num_variables);
    let statement = make_statement(&polynomial, num_variables);
    let ds = DomainSeparator::protocol(&config)
        .session(&format!("bench-non-zk-prove-{num_variables}"))
        .instance(&Empty);

    bencher
        .with_inputs(|| {
            let mut prover_state = ProverState::new_std(&ds);
            let witness = config.commit(&mut prover_state, &[&polynomial]);
            (prover_state, witness)
        })
        .bench_values(|(mut prover_state, witness)| {
            black_box(config.prove(
                &mut prover_state,
                &[&polynomial],
                &[&witness],
                &[&statement],
            ));
        });
}

#[divan::bench(args = SIZES)]
fn non_zk_verify(bencher: Bencher, num_variables: usize) {
    let polynomial = make_polynomial(num_variables);
    let config = non_zk_config(num_variables);
    let statement = make_statement(&polynomial, num_variables);
    let ds = DomainSeparator::protocol(&config)
        .session(&format!("bench-non-zk-verify-{num_variables}"))
        .instance(&Empty);

    // Generate a proof once (outside the benchmark loop).
    let proof = {
        let mut ps = ProverState::new_std(&ds);
        let w = config.commit(&mut ps, &[&polynomial]);
        config.prove(&mut ps, &[&polynomial], &[&w], &[&statement]);
        ps.proof()
    };

    bencher
        .with_inputs(|| {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let commitment = config.receive_commitment(&mut vs).unwrap();
            (vs, commitment)
        })
        .bench_values(|(mut vs, commitment)| {
            black_box(
                config
                    .verify(&mut vs, &[&commitment], &[&statement])
                    .unwrap(),
            );
        });
}

// ────────────────────────────────────────────────────────────────────────────
//  ZK v1 benchmarks  (batch_size=2 commit of [f̂, g], then standard WHIR on P)
// ────────────────────────────────────────────────────────────────────────────

/// Benchmark committing [[f̂, g]] together with batch_size=2 (the ZK blinding commitment).
#[divan::bench(args = SIZES)]
fn zk_v1_commit(bencher: Bencher, num_variables: usize) {
    let polys = make_zk_v1_polys(num_variables);
    let commit_config = zk_v1_commit_config(num_variables);
    let ds = DomainSeparator::protocol(&commit_config)
        .session(&format!("bench-zk-v1-commit-{num_variables}"))
        .instance(&Empty);

    bencher
        .with_inputs(|| ProverState::new_std(&ds))
        .bench_values(|mut prover_state| {
            // Single commit call with 2 polynomials (batch_size=2).
            let _ = black_box(
                commit_config.commit(&mut prover_state, &[&polys.f_hat, &polys.g_poly]),
            );
        });
}

/// Benchmark proving P = ρ·f̂ + g via standard WHIR (μ+1 variables).
/// Setup: commit P. Benchmark: prove P.
#[divan::bench(args = SIZES)]
fn zk_v1_prove(bencher: Bencher, num_variables: usize) {
    let polys = make_zk_v1_polys(num_variables);
    let prove_config = zk_v1_prove_config(num_variables);
    let statement = make_zk_v1_statement(&polys.p_poly, num_variables);
    let ds = DomainSeparator::protocol(&prove_config)
        .session(&format!("bench-zk-v1-prove-{num_variables}"))
        .instance(&Empty);

    bencher
        .with_inputs(|| {
            let mut prover_state = ProverState::new_std(&ds);
            let witness = prove_config.commit(&mut prover_state, &[&polys.p_poly]);
            (prover_state, witness)
        })
        .bench_values(|(mut prover_state, witness)| {
            black_box(prove_config.prove(
                &mut prover_state,
                &[&polys.p_poly],
                &[&witness],
                &[&statement],
            ));
        });
}

/// Benchmark verifying P via standard WHIR.
#[divan::bench(args = SIZES)]
fn zk_v1_verify(bencher: Bencher, num_variables: usize) {
    let polys = make_zk_v1_polys(num_variables);
    let prove_config = zk_v1_prove_config(num_variables);
    let statement = make_zk_v1_statement(&polys.p_poly, num_variables);
    let ds = DomainSeparator::protocol(&prove_config)
        .session(&format!("bench-zk-v1-verify-{num_variables}"))
        .instance(&Empty);

    // Generate a proof once.
    let proof = {
        let mut ps = ProverState::new_std(&ds);
        let w = prove_config.commit(&mut ps, &[&polys.p_poly]);
        prove_config.prove(&mut ps, &[&polys.p_poly], &[&w], &[&statement]);
        ps.proof()
    };

    bencher
        .with_inputs(|| {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let commitment = prove_config.receive_commitment(&mut vs).unwrap();
            (vs, commitment)
        })
        .bench_values(|(mut vs, commitment)| {
            black_box(
                prove_config
                    .verify(&mut vs, &[&commitment], &[&statement])
                    .unwrap(),
            );
        });
}

// ────────────────────────────────────────────────────────────────────────────
//  ZK v2 benchmarks
// ────────────────────────────────────────────────────────────────────────────

#[divan::bench(args = SIZES)]
fn zk_v2_commit(bencher: Bencher, num_variables: usize) {
    let polynomial = make_polynomial(num_variables);
    let config = zk_main_config(num_variables);
    let zk_params = ZkParams::from_whir_params(&config);
    let helper_config = zk_helper_config(&zk_params);
    let mut rng = StdRng::seed_from_u64(42);
    let preprocessing = Arc::new(ZkPreprocessingPolynomials::<EF>::sample(&mut rng, zk_params));

    let ds = DomainSeparator::protocol(&config)
        .session(&format!("bench-zk-v2-commit-{num_variables}"))
        .instance(&Empty);

    bencher
        .with_inputs(|| {
            let prover_state = ProverState::new_std(&ds);
            (prover_state, &polynomial, &config, &helper_config, Arc::clone(&preprocessing))
        })
        .bench_values(|(mut prover_state, poly, cfg, helper_cfg, preproc)| {
            black_box(cfg.commit_zk(&mut prover_state, poly, helper_cfg, preproc));
        });
}

#[divan::bench(args = SIZES)]
fn zk_v2_prove(bencher: Bencher, num_variables: usize) {
    let polynomial = make_polynomial(num_variables);
    let config = zk_main_config(num_variables);
    let zk_params = ZkParams::from_whir_params(&config);
    let helper_config = zk_helper_config(&zk_params);
    let mut rng = StdRng::seed_from_u64(42);
    let preprocessing = Arc::new(ZkPreprocessingPolynomials::<EF>::sample(&mut rng, zk_params));
    let statement = make_statement(&polynomial, num_variables);

    let ds = DomainSeparator::protocol(&config)
        .session(&format!("bench-zk-v2-prove-{num_variables}"))
        .instance(&Empty);

    bencher
        .with_inputs(|| {
            let mut prover_state = ProverState::new_std(&ds);
            let zk_witness =
                config.commit_zk(&mut prover_state, &polynomial, &helper_config, Arc::clone(&preprocessing));
            (prover_state, zk_witness)
        })
        .bench_values(|(mut prover_state, zk_witness)| {
            black_box(config.prove_zk(
                &mut prover_state,
                &polynomial,
                &zk_witness,
                &helper_config,
                &statement,
            ));
        });
}

#[divan::bench(args = SIZES)]
fn zk_v2_verify(bencher: Bencher, num_variables: usize) {
    let polynomial = make_polynomial(num_variables);
    let config = zk_main_config(num_variables);
    let zk_params = ZkParams::from_whir_params(&config);
    let helper_config = zk_helper_config(&zk_params);
    let mut rng = StdRng::seed_from_u64(42);
    let preprocessing = Arc::new(ZkPreprocessingPolynomials::<EF>::sample(&mut rng, zk_params.clone()));
    let statement = make_statement(&polynomial, num_variables);

    let ds = DomainSeparator::protocol(&config)
        .session(&format!("bench-zk-v2-verify-{num_variables}"))
        .instance(&Empty);

    // Generate a proof once (outside the benchmark loop).
    let proof = {
        let mut ps = ProverState::new_std(&ds);
        let zk_witness =
            config.commit_zk(&mut ps, &polynomial, &helper_config, Arc::clone(&preprocessing));
        config.prove_zk(&mut ps, &polynomial, &zk_witness, &helper_config, &statement);
        ps.proof()
    };

    bencher
        .with_inputs(|| {
            let mut vs = VerifierState::new_std(&ds, &proof);
            let f_hat_commitment = config.receive_commitment(&mut vs).unwrap();
            let helper_commitment = helper_config.receive_commitment(&mut vs).unwrap();
            (vs, f_hat_commitment, helper_commitment)
        })
        .bench_values(|(mut vs, f_hat_commitment, helper_commitment)| {
            black_box(
                config
                    .verify_zk(
                        &mut vs,
                        &f_hat_commitment,
                        &helper_commitment,
                        &helper_config,
                        &zk_params,
                        &statement,
                    )
                    .unwrap(),
            );
        });
}

fn main() {
    divan::main();
}
