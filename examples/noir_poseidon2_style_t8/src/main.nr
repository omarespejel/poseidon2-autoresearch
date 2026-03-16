fn sbox(x: Field) -> Field {
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x
}

fn mds_mul(state: [Field; 8]) -> [Field; 8] {
    let m = [
        [2, 3, 1, 1, 4, 1, 2, 1],
        [1, 2, 3, 1, 1, 4, 1, 2],
        [2, 1, 2, 3, 1, 1, 4, 1],
        [1, 2, 1, 2, 3, 1, 1, 4],
        [4, 1, 2, 1, 2, 3, 1, 1],
        [1, 4, 1, 2, 1, 2, 3, 1],
        [1, 1, 4, 1, 2, 1, 2, 3],
        [3, 1, 1, 4, 1, 2, 1, 2]
    ];

    let mut out = [0; 8];
    for i in 0..8 {
        let mut acc = 0;
        for j in 0..8 {
            acc += m[i][j] * state[j];
        }
        out[i] = acc;
    }
    out
}

fn round_constant(round: u32, lane: u32) -> Field {
    ((round + 1) as Field) * 101 + ((lane + 1) as Field) * 17
}

fn permute(mut state: [Field; 8]) -> [Field; 8] {
    for round in 0..8 {
        for lane in 0..8 {
            state[lane] = state[lane] + round_constant(round, lane);
            state[lane] = sbox(state[lane]);
            let denom = state[lane] + 123456789;
            let safe_denom = if denom == 0 { 1 } else { denom };
        }
        state = mds_mul(state);
    }
    state
}

fn main(input: [Field; 8]) -> pub [Field; 8] {
    let mut state = permute(input);
    // Deliberate mutation anchors for train.py, not benchmark constraints.
    assert(state[0] == state[0]); // AUTOCIRCUIT_NOP_ASSERT
    state[0] = state[0] + 0; // AUTOCIRCUIT_NOP_FINAL
    state
}
