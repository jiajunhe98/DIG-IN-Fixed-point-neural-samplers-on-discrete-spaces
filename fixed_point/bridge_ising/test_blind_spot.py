"""Numerical proof that the two heads at site d cannot see x^d.

The blind spot is the whole point of this architecture: it is what makes
q^d(. | x^{-d}, t) a conditional the sampler may use as a heat-bath kernel, and what
makes the fixed-point label substitution at coordinate d consistent.  It is also easy
to break silently -- a GroupNorm, or one full-plane convolution after the branch, and
the centre leaks with no other symptom.  So it is tested, not asserted.

    python test_blind_spot.py [cpu|cuda]
"""

import sys

import torch

from network_bs import BlindSpotNet, hollow_check


def main():
    dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
    ok = True
    # reach = 3 + 4*n_blocks must stay strictly below L
    for L, n_blocks in ((9, 1), (17, 3), (25, 5)):
        net = BlindSpotNet(L=L, hidden=32, n_blocks=n_blocks).to(dev)
        # a zero-initialised head would pass trivially; randomise so the test has teeth
        with torch.no_grad():
            for m in net.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.normal_(m.weight, 0.0, 0.15)
                    if m.bias is not None:
                        torch.nn.init.normal_(m.bias, 0.0, 0.05)
        at_site, elsewhere = hollow_check(net, L, dev, n_sites=16)
        blind = at_site < 1e-6
        alive = elsewhere > 1e-3
        ok &= blind and alive
        print(f"L={L:>3} n_blocks={n_blocks}  reach rows {net.reach_rows} cols {net.reach_cols}  "
              f"|change at d| {at_site:.2e}  |change elsewhere| {elsewhere:.3f}  "
              f"{'OK' if blind and alive else 'FAIL'}")

    # the vertical reach must stay strictly below L or the circular padding wraps onto the
    # centre row and the blind spot is silently lost; the constructor has to refuse that
    try:
        BlindSpotNet(L=9, hidden=16, n_blocks=5)
    except AssertionError:
        print("over-deep stack at L=9 correctly refused")
    else:
        print("over-deep stack at L=9 was NOT refused  FAIL")
        ok = False

    # interface: two heads, both [B, D, 2]
    net = BlindSpotNet(L=17, hidden=32, n_blocks=3).to(dev)
    x = (torch.rand(5, 289, device=dev) < 0.5).long()
    q, phi = net(x, torch.rand(5, device=dev))
    shapes = q.shape == (5, 289, 2) and phi.shape == (5, 289, 2)
    print(f"head shapes {tuple(q.shape)} {tuple(phi.shape)}  {'OK' if shapes else 'FAIL'}")
    ok &= shapes
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
