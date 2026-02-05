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
pub fn sample_gamma<R: FnMut() -> f64>(shape: f64, rng: &mut R) -> f64 {
    if shape <= 0.0 {
        return 0.0;
    }
    if shape < 1.0 {
        // Johnk's transformation
        let u = rng();
        return sample_gamma(shape + 1.0, rng) * u.powf(1.0 / shape);
    }
    // Marsaglia and Tsang
    let d = shape - 1.0 / 3.0;
    let c = (1.0 / 3.0) / d.sqrt();
    loop {
        let mut x: f64;
        let mut v: f64;
        loop {
            // standard normal via Box-Muller
            let u1 = rng().max(1e-12);
            let u2 = rng();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos();
            x = z;
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            }
        }
        v = v * v * v;
        let u = rng();
        if u < 1.0 - 0.0331 * (x * x) * (x * x) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
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
