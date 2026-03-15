use core::poseidon::poseidon_hash_span;

fn hash8_reference(
    a0: felt252,
    a1: felt252,
    a2: felt252,
    a3: felt252,
    a4: felt252,
    a5: felt252,
    a6: felt252,
    a7: felt252,
) -> felt252 {
    poseidon_hash_span(array![a0, a1, a2, a3, a4, a5, a6, a7].span())
}

pub fn hash8(
    a0: felt252,
    a1: felt252,
    a2: felt252,
    a3: felt252,
    a4: felt252,
    a5: felt252,
    a6: felt252,
    a7: felt252,
) -> felt252 {
    let mut h = poseidon_hash_span(array![a0, a1, a2, a3, a4, a5, a6, a7].span());
    h
}

#[cfg(test)]
mod tests {
    use super::{hash8, hash8_reference};

    #[test]
    fn hash8_matches_reference() {
        let h = hash8(1, 2, 3, 4, 5, 6, 7, 8);
        let r = hash8_reference(1, 2, 3, 4, 5, 6, 7, 8);
        assert(h == r, 'hash8 mismatch');
    }

    #[test]
    fn hash8_stable_nonzero() {
        let h = hash8(10, 20, 30, 40, 50, 60, 70, 80);
        assert(h != 0, 'unexpected zero hash');
    }
}
