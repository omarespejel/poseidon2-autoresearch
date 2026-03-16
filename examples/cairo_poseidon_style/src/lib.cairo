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
    poseidon_hash_span(array![a0, a1, a2, a3, a4, a5, a6, a7].span())
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
    fn hash8_matches_known_vector() {
        assert_eq!(hash8(1, 2, 3, 4, 5, 6, 7, 8), 142523731258509939608696022271238521916410456401611624853849835202137558864);
    }

    #[test]
    fn hash8_matches_zero_vector() {
        assert_eq!(hash8(0, 0, 0, 0, 0, 0, 0, 0), 2975145535556472711340937403264375774159340593752652611583194611116420827365);
    }
}
