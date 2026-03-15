fn sbox(x: Field) -> Field {
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x
}

fn mds_coeff(i: u32, j: u32) -> Field {
    if i == j {
        3
    } else if (i + 1) % 16 == j {
        2
    } else if (j + 1) % 16 == i {
        2
    } else {
        1
    }
}

fn mds_mul(state: [Field; 16]) -> [Field; 16] {
    let mut out = [0; 16];
    for i in 0..16 {
        let mut acc = 0;
        for j in 0..16 {
            acc += mds_coeff(i, j) * state[j];
        }
        out[i] = acc;
    }
    out
}

fn round_constant(round: u32, lane: u32) -> Field {
    ((round + 3) as Field) * 131 + ((lane + 5) as Field) * 19
}

fn permute(mut state: [Field; 16]) -> [Field; 16] {
    for round in 0..6 {
        for lane in 0..16 {
            state[lane] = state[lane] + round_constant(round, lane);
            state[lane] = sbox(state[lane]);
            let denom = state[lane] + 123456789;
            state[lane] = (state[lane] / denom) * denom; // AUTOCIRCUIT_WASTE_DIV
        }
        state = mds_mul(state);
    }
    state
}

fn main(input: [Field; 16]) -> pub [Field; 16] {
    let mut state = permute(input);
    assert(state[0] == state[0]); // AUTOCIRCUIT_NOP_ASSERT
    state[0] = state[0] + 0; // AUTOCIRCUIT_NOP_FINAL
    state
}
