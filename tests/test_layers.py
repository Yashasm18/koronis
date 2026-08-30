import torch

from koronis.models.layers import RelationalLayer


def test_output_shape_and_gradients():
    layer = RelationalLayer(in_dim=6, out_dim=8, n_relations=2)
    x = torch.randn(10, 6, requires_grad=True)
    edges = [torch.tensor([[0, 1], [2, 3]]), torch.tensor([[4], [5]])]
    out = layer(x, edges)
    assert out.shape == (10, 8)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_isolated_node_still_gets_self_transform():
    layer = RelationalLayer(in_dim=4, out_dim=4, n_relations=1)
    x = torch.randn(5, 4)
    out_no_edges = layer(x, [torch.zeros((2, 0), dtype=torch.long)])
    assert torch.isfinite(out_no_edges).all()
    assert not torch.allclose(out_no_edges, torch.zeros_like(out_no_edges))


def test_is_inductive_over_node_count():
    """Same weights must apply to a graph of any size — no per-node params."""
    layer = RelationalLayer(in_dim=3, out_dim=3, n_relations=1)
    e = torch.tensor([[0], [1]])
    assert layer(torch.randn(2, 3), [e]).shape == (2, 3)
    assert layer(torch.randn(50, 3), [e]).shape == (50, 3)


def test_rejects_wrong_relation_count():
    layer = RelationalLayer(in_dim=3, out_dim=3, n_relations=2)
    try:
        layer(torch.randn(4, 3), [torch.tensor([[0], [1]])])
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched relation count")


def test_neighbour_messages_change_the_output():
    """Aggregation must actually do something — a silent no-op would make the
    whole graph-vs-tabular comparison meaningless."""
    torch.manual_seed(0)
    layer = RelationalLayer(in_dim=4, out_dim=4, n_relations=1)
    x = torch.randn(6, 4)
    empty = layer(x, [torch.zeros((2, 0), dtype=torch.long)])
    linked = layer(x, [torch.tensor([[0, 1, 2], [3, 4, 5]])])
    assert not torch.allclose(empty, linked)
