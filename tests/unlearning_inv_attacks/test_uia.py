from attacks.reconstruction.unlearning_inversion_attacks.uia import uia


def test_uia():
    assert uia(-1) == 0
