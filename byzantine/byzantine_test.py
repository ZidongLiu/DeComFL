import copy
import random

import pytest
import torch

from byzantine import aggregation as byz_agg
from byzantine import attack as byz_attack

# Shape convention used by the server: one entry per sampled client, each entry is a
# list (one tensor per local update step) of grad scalars of shape (num_pert,).


def make_scalars(values: list[list[list[float]]]) -> list[list[torch.Tensor]]:
    return [[torch.tensor(step) for step in client] for client in values]


def honest_cluster_with_outlier() -> list[list[torch.Tensor]]:
    # 4 honest clients close to each other + 1 far outlier (index 4), 2 local steps.
    return make_scalars(
        [
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            [[1.1, 0.9, 1.0], [2.1, 1.9, 2.0]],
            [[0.9, 1.1, 1.0], [1.9, 2.1, 2.0]],
            [[1.0, 1.0, 1.1], [2.0, 2.0, 2.1]],
            [[100.0, -100.0, 100.0], [-100.0, 100.0, -100.0]],
        ]
    )


@pytest.fixture(autouse=True)
def fix_seed():
    random.seed(0)
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_mean():
    v = make_scalars([[[1.0, 2.0], [3.0, 4.0]], [[3.0, 4.0], [5.0, 6.0]]])
    result = byz_agg.mean(v)
    assert len(result) == 2
    torch.testing.assert_close(result[0], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(result[1], torch.tensor([4.0, 5.0]))


def test_mean_does_not_modify_inputs():
    v = make_scalars([[[1.0, 2.0]], [[3.0, 4.0]]])
    v_before = copy.deepcopy(v)
    byz_agg.mean(v)
    for client, client_before in zip(v, v_before):
        for t, t_before in zip(client, client_before):
            torch.testing.assert_close(t, t_before)


def test_median():
    v = make_scalars([[[1.0, -5.0]], [[2.0, 0.0]], [[100.0, 7.0]]])
    result = byz_agg.median(v)
    assert len(result) == 1
    torch.testing.assert_close(result[0], torch.tensor([2.0, 0.0]))


def test_median_ignores_single_outlier():
    result = byz_agg.median(honest_cluster_with_outlier())
    for step_result in result:
        assert step_result.abs().max() < 3.0


def test_trim():
    v = make_scalars([[[0.0]], [[1.0]], [[2.0]], [[3.0]], [[100.0]]])
    result = byz_agg.trim(v, f=1)
    torch.testing.assert_close(result[0], torch.tensor([2.0]))


def test_trim_is_coordinate_wise():
    v = make_scalars([[[0.0, 30.0]], [[10.0, 20.0]], [[20.0, 10.0]], [[30.0, 0.0]]])
    result = byz_agg.trim(v, f=1)
    torch.testing.assert_close(result[0], torch.tensor([15.0, 15.0]))


@pytest.mark.parametrize("f", [0, 2, 3])
def test_trim_rejects_invalid_f(f):
    v = make_scalars([[[0.0]], [[1.0]], [[2.0]], [[3.0]]])
    with pytest.raises(AssertionError):
        byz_agg.trim(v, f=f)


def test_krum_selects_honest_client():
    v = honest_cluster_with_outlier()
    result = byz_agg.krum(v, f=1)
    assert len(result) == 2
    honest_rows = [torch.stack([client[step] for client in v[:4]]) for step in range(2)]
    for step, step_result in enumerate(result):
        # Krum returns one of the submitted vectors, never an average.
        assert any(torch.equal(step_result, row) for row in honest_rows[step])


@pytest.mark.parametrize("f", [0, 3])
def test_krum_rejects_invalid_f(f):
    with pytest.raises(AssertionError):
        byz_agg.krum(honest_cluster_with_outlier(), f=f)


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------


def num_changed_clients(before, after) -> int:
    return sum(
        any(not torch.equal(t_b, t_a) for t_b, t_a in zip(client_b, client_a))
        for client_b, client_a in zip(before, after)
    )


def test_no_byz_is_identity():
    v = honest_cluster_with_outlier()
    v_before = copy.deepcopy(v)
    assert num_changed_clients(v_before, byz_attack.no_byz(v)) == 0


@pytest.mark.parametrize("attack", [byz_attack.gaussian_attack, byz_attack.sign_attack])
def test_attack_with_zero_attackers_is_identity(attack):
    v = honest_cluster_with_outlier()
    v_before = copy.deepcopy(v)
    assert num_changed_clients(v_before, attack(v, num_attack=0)) == 0


@pytest.mark.parametrize("num_attack", [1, 2, 3])
def test_gaussian_attack_changes_exactly_num_attack_clients(num_attack):
    v = honest_cluster_with_outlier()
    v_before = copy.deepcopy(v)
    v_after = byz_attack.gaussian_attack(v, num_attack=num_attack)
    assert len(v_after) == len(v_before)
    assert num_changed_clients(v_before, v_after) == num_attack
    for client_b, client_a in zip(v_before, v_after):
        for t_b, t_a in zip(client_b, client_a):
            assert t_a.shape == t_b.shape


@pytest.mark.parametrize("num_attack", [1, 2, 3])
def test_sign_attack_negates_exactly_num_attack_clients(num_attack):
    v = honest_cluster_with_outlier()
    v_before = copy.deepcopy(v)
    v_after = byz_attack.sign_attack(v, num_attack=num_attack)
    negated = 0
    for client_b, client_a in zip(v_before, v_after):
        if all(torch.equal(t_a, -t_b) for t_b, t_a in zip(client_b, client_a)):
            negated += 1
        else:
            assert all(torch.equal(t_a, t_b) for t_b, t_a in zip(client_b, client_a))
    assert negated == num_attack


@pytest.mark.parametrize("num_attack", [0, 1, 2])
def test_trim_attack(num_attack):
    v = make_scalars(
        [
            [[1.0, -1.0, 2.0], [0.5, -0.5, 3.0]],
            [[1.2, -0.8, 2.5], [0.4, -0.7, 2.0]],
            [[0.8, -1.3, 1.5], [0.6, -0.2, 4.0]],
            [[1.1, -1.0, 2.2], [0.3, -0.9, 3.5]],
            [[0.9, -1.1, 1.8], [0.7, -0.4, 2.5]],
        ]
    )
    v_before = copy.deepcopy(v)
    stacked = torch.stack([torch.cat(client) for client in v_before])
    honest_min = stacked.min(dim=0).values
    honest_max = stacked.max(dim=0).values
    direction = torch.sign(stacked.sum(dim=0))

    v_after = byz_attack.trim_attack(v, num_attack=num_attack)

    assert len(v_after) == len(v_before)
    # Attackers are the first `num_attack` clients; the rest are untouched.
    for client_b, client_a in zip(v_before[num_attack:], v_after[num_attack:]):
        assert all(torch.equal(t_b, t_a) for t_b, t_a in zip(client_b, client_a))
    for client_a in v_after[:num_attack]:
        assert [t.shape for t in client_a] == [t.shape for t in v_before[0]]
        flat = torch.cat(client_a)
        # Each malicious coordinate is pushed beyond the honest extreme that
        # opposes the aggregate direction.
        assert torch.all(flat[direction > 0] <= honest_min[direction > 0])
        assert torch.all(flat[direction < 0] >= honest_max[direction < 0])


def test_attack_krum_selects_honest_client():
    # attack.krum works on the concatenated (all local steps) vector per client.
    v = honest_cluster_with_outlier()
    min_idx, selected = byz_attack.krum(v, f=1)
    assert min_idx != 4
    assert selected is v[min_idx]


def test_attack_krum_matches_aggregation_krum_on_single_step():
    v = make_scalars([[[1.0, 1.0]], [[1.2, 0.9]], [[0.8, 1.1]], [[1.1, 1.05]], [[50.0, -50.0]]])
    min_idx, _ = byz_attack.krum(v, f=1)
    expected = byz_agg.krum(copy.deepcopy(v), f=1)[0]
    torch.testing.assert_close(v[min_idx][0], expected)


def test_krum_attack_with_zero_attackers_is_identity():
    v = honest_cluster_with_outlier()
    v_before = copy.deepcopy(v)
    assert num_changed_clients(v_before, byz_attack.krum_attack(v, f=0, lr=1e-3)) == 0


@pytest.mark.parametrize("f", [1, 2])
def test_krum_attack(f):
    num_clients, num_steps, num_pert = 7, 2, 3
    v = [[torch.randn(num_pert) for _ in range(num_steps)] for _ in range(num_clients)]
    v_before = copy.deepcopy(v)
    v_after = byz_attack.krum_attack(v, f=f, lr=1e-3)

    assert len(v_after) == num_clients
    for client_b, client_a in zip(v_before[f:], v_after[f:]):
        assert all(torch.equal(t_b, t_a) for t_b, t_a in zip(client_b, client_a))
    for client_a in v_after[:f]:
        # Malicious clients must keep the same per-step structure so the
        # aggregation functions can consume them.
        assert len(client_a) == num_steps
        assert all(t.shape == (num_pert,) for t in client_a)
    # The aggregator must still be able to consume the attacked updates.
    result = byz_agg.mean(v_after)
    assert len(result) == num_steps
