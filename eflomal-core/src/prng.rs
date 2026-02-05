// PCG32 as simple, fast, deterministic PRNG
#[derive(Clone)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}
impl Pcg32 {
    pub fn new(seed: u64, seq: u64) -> Self {
        let mut pcg = Pcg32 {
            state: 0,
            inc: (seq << 1) | 1,
        };
        pcg.next_u32();
        pcg.state = pcg.state.wrapping_add(seed);
        pcg.next_u32();
        pcg
    }

    // Create an independent child RNG while advancing this stream.
    pub fn split(&mut self) -> Self {
        // Derive a fresh seed and sequence from the current stream.
        let seed_hi = self.next_u32() as u64;
        let seed_lo = self.next_u32() as u64;
        let seq_hi = self.next_u32() as u64;
        let seq_lo = self.next_u32() as u64;
        let seed = (seed_hi << 32) | seed_lo;
        let seq = (seq_hi << 32) | seq_lo;
        Pcg32::new(seed, seq)
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let oldstate = self.state;
        self.state = oldstate
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc);
        let xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) as u32;
        let rot = (oldstate >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        let u = self.next_u32() as f64 / (u32::MAX as f64 + 1.0);
        u as f32
    }
    #[inline]
    pub fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        // unbiased
        let mut x;
        let m = (u32::MAX as u64 + 1) / (n as u64);
        let t = m * (n as u64);
        loop {
            x = self.next_u32() as u64;
            if x < t {
                return (x / m) as usize;
            }
        }
    }
}

// Gamma sampling for Dirichlet
//
// Two-way branching matching the original C eflomal:
//   alpha < 0.6  → log-gamma-small (Liu, Martin & Syring)
//   alpha >= 0.6 → Cheng's method (1977)
pub fn sample_gamma<R: FnMut() -> f64>(shape: f64, rng: &mut R) -> f64 {
    if shape <= 0.0 {
        return 0.0;
    }
    if shape < 0.6 {
        // Log-Gamma approximation (Liu, Martin & Syring)
        // Matches the C code's `random_log_gamma_small64`
        let lambda = (1.0 / shape) - 1.0;
        let w = shape / (std::f64::consts::E * (1.0 - shape));
        let r = 1.0 / (1.0 + w);

        loop {
            let u = rng();
            let z = if u <= r {
                -(u / r).ln()
            } else {
                rng().ln() / lambda
            };
            let h = (-z - (-z / shape).exp()).exp();
            let eta = if z >= 0.0 { (-z).exp() } else { w * lambda * (lambda * z).exp() };
            if h > eta * rng() {
                return (-z / shape).exp();
            }
        }
    }
    // Cheng's method for shape >= 0.6
    // Matches the C code's `random_gamma64`
    // R. C. H. Cheng (1977), "The Generation of Gamma Variables
    // with Non-Integral Shape Parameter"
    let a = 1.0 / (2.0 * shape - 1.0).sqrt();
    let b = shape - (4.0_f64).ln();
    let c = shape + 1.0 / a;
    loop {
        let u1 = rng();
        let u2 = rng();
        let v = a * (u1 / (1.0 - u1)).ln();
        let x = shape * v.exp();
        if b + c * v - x >= (u1 * u1 * u2).ln() {
            return x;
        }
    }
}

// Fast approximate math matching the C eflomal's simd_math_prims.h
// These are used in the C code via `#define expf expapprox` and `#define logf logapprox`

/// Approximate exp(x) with ~1e-5 relative error for normalized outputs.
/// Ported from simd_math_prims.h `expapprox`.
#[inline]
pub fn expapprox(val: f32) -> f32 {
    let exp_cst1: f32 = 2139095040.0;
    let exp_cst2: f32 = 0.0;

    let val2 = 12102203.1615614_f32 * val + 1065353216.0_f32;
    let val3 = if val2 < exp_cst1 { val2 } else { exp_cst1 };
    let val4 = if val3 > exp_cst2 { val3 } else { exp_cst2 };
    let val4i = val4 as i32;
    let xu = f32::from_bits((val4i as u32) & 0x7F800000);
    let b = f32::from_bits(((val4i as u32) & 0x7FFFFF) | 0x3F800000);

    xu * (0.510397365625862338668154
        + b * (0.310670891004095530771135
            + b * (0.168143436463395944830000
                + b * (-2.88093587581985443087955e-3
                    + b * 1.3671023382430374383648148e-2))))
}

/// Approximate ln(x) with ~1e-6 absolute error for normalized inputs.
/// Ported from simd_math_prims.h `logapprox`.
#[inline]
pub fn logapprox(val: f32) -> f32 {
    let valu_i = val.to_bits() as i32;
    let exp = (valu_i >> 23) as f32;
    let addcst = if val > 0.0 { -89.970756366_f32 } else { f32::NEG_INFINITY };
    let x = f32::from_bits(((valu_i as u32) & 0x7FFFFF) | 0x3F800000);

    x * (3.529304993
        + x * (-2.461222105
            + x * (1.130626167
                + x * (-0.288739945
                    + x * 3.110401639e-2))))
        + (addcst + 0.69314718055995 * exp)
}

// Dirichlet (unnormalized sample)
pub fn dirichlet_unnormalized(alpha: &[f64], out: &mut [f64], rng: &mut Pcg32) {
    assert_eq!(alpha.len(), out.len());
    let mut sum = 0.0;
    for (i, a) in alpha.iter().enumerate() {
        let v = sample_gamma(*a as f64, &mut || {
            rng.next_u32() as f64 / (u32::MAX as f64 + 1.0)
        });
        out[i] = v;
        sum += v;
    }
    if sum <= 0.0 {
        // fallback: uniform
        let v = 1.0 / (out.len() as f64);
        for o in out.iter_mut() {
            *o = v;
        }
    }
}
