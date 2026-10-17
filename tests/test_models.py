import pytest
import torch

from dds.models import build_model

NAMES = ["rescnn"]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("name", NAMES)
def test_shapes_and_grads(name):
    kwargs = dict(hidden_size=32)
    if name in ("vit", "sit", "ropesit"):
        kwargs.update(depth=2, num_heads=4)
    net = build_model(name, (4, 4), vocab_size=3, out_channels=5,
                      require_time=True, **kwargs)
    x = torch.randint(0, 3, (6, 16))
    t = torch.rand(6)
    out = net(x, t)
    assert out.shape == (6, 16, 5)
    out.sum().backward()


@pytest.mark.parametrize("name", NAMES)
def test_time_free(name):
    kwargs = dict(hidden_size=32)
    if name in ("vit", "sit", "ropesit"):
        kwargs.update(depth=2, num_heads=4)
    net = build_model(name, (4, 4), vocab_size=4, out_channels=3,
                      require_time=False, **kwargs)
    out = net(torch.randint(0, 4, (5, 16)))
    assert out.shape == (5, 16, 3)


def test_leaps_local_equivariance():
    from dds.methods.leaps.nets import LocEquivNet

    net = LocEquivNet((4, 4), vocab_size=3, channels=8, kernel_sizes=(3, 5))
    x = torch.randint(0, 3, (4, 16))
    t = torch.rand(4)
    R = net(x, t)
    # diagonal is exactly zero
    assert torch.gather(R, 2, x.unsqueeze(-1)).abs().max() < 1e-9
    # antisymmetry: R_d(x^{d<-a}, x^d) = -R_d(x, a)
    for d in [0, 7, 15]:
        for a in range(3):
            y = x.clone()
            y[:, d] = a
            Ry = net(y, t)
            lhs = torch.gather(Ry[:, d], 1, x[:, d : d + 1]).squeeze(1)
            rhs = -R[:, d, a]
            mask = x[:, d] != a
            assert (lhs - rhs)[mask].abs().max() < 1e-5


# ----------------------------------------------------------------------
# rescnn: the periodic residual CNN used as the default backbone
# ----------------------------------------------------------------------
def test_rescnn_is_exactly_translation_equivariant_on_the_torus():
    """Circular padding everywhere means the PBC symmetry of the target is
    a symmetry of the network, exactly and not approximately."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        L, D = 8, 64
        net = build_model("rescnn", (L, L), 3, 2, require_time=True,
                          hidden_size=16, n_blocks=3)
        # a zero-initialised head would make every output identically zero,
        # which is trivially equivariant and tests nothing
        for p in net.head.parameters():
            torch.nn.init.normal_(p, std=0.1)
        x = torch.randint(0, 3, (4, D))
        t = torch.rand(4)
        with torch.no_grad():
            out = net(x, t).view(4, L, L, 2)
            rolled = torch.roll(x.view(4, L, L), (2, 3), dims=(1, 2)).reshape(4, D)
            out_rolled = net(rolled, t).view(4, L, L, 2)
        shifted = torch.roll(out, (2, 3), dims=(1, 2))
        assert (shifted - out_rolled).abs().max() < 1e-12
    finally:
        torch.set_default_dtype(old)


@pytest.mark.parametrize("L,n_blocks", [(16, 7), (32, 7), (50, 7), (16, 2)])
def test_rescnn_receptive_field_covers_the_torus(L, n_blocks):
    """`dilations="auto"` must reach radius L//2 -- the maximum distance on
    the torus -- or the denoiser cannot see across the lattice, which is
    what it has to do at and below the critical point."""
    net = build_model("rescnn", (L, L), 3, 2, require_time=True,
                      hidden_size=16, n_blocks=n_blocks)
    assert min(net.radius) >= L // 2, (net.radius, net.dilations)


def test_rescnn_far_site_actually_influences_the_output():
    """The radius bookkeeping is a claim about the network; check it holds
    by perturbing the site furthest away on the torus."""
    L, D = 16, 256
    net = build_model("rescnn", (L, L), 3, 2, require_time=True,
                      hidden_size=16, n_blocks=7)
    for p in net.head.parameters():
        torch.nn.init.normal_(p, std=0.1)
    x = torch.randint(0, 3, (1, D))
    y = x.clone().view(1, L, L)
    y[0, L // 2, L // 2] = (y[0, L // 2, L // 2] + 1) % 3
    t = torch.rand(1)
    with torch.no_grad():
        d = (net(x, t) - net(y.reshape(1, D), t)).abs().view(L, L, 2).amax(-1)
    assert d.min() > 0, "some site is outside the receptive field"


def test_rescnn_refuses_a_receptive_field_that_misses_the_torus():
    with pytest.raises(AssertionError, match="receptive field"):
        build_model("rescnn", (64, 64), 3, 2, require_time=True,
                    hidden_size=16, n_blocks=3, dilations="none")
    # ...and builds anyway when the caller opts out
    build_model("rescnn", (64, 64), 3, 2, require_time=True, hidden_size=16,
                n_blocks=3, dilations="none", cover_torus=False)


def test_rescnn_zero_init_head_starts_uniform():
    """The fixed-point loop must not begin already committed to a mode."""
    net = build_model("rescnn", (8, 8), 3, 2, require_time=True,
                      hidden_size=16, n_blocks=3)
    with torch.no_grad():
        out = net(torch.randint(0, 3, (4, 64)), torch.rand(4))
    assert out.abs().max() == 0.0
