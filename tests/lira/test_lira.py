from attacks.membership.lira.lira import lira_attack


def test_lira():
    assert lira_attack(2) == 3
