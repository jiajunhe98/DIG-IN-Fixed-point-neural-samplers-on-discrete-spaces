from dds.targets.alloy import AlloyFamily, AlloyPoint
from dds.targets.base import DiscreteTarget
from dds.targets.ising import IsingModel
from dds.targets.potts import PottsModel


def build_target(cfg: dict) -> DiscreteTarget:
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name == "ising":
        return IsingModel(**cfg)
    if name == "potts":
        return PottsModel(**cfg)
    if name == "alloy_family":
        return AlloyFamily(**cfg)
    raise ValueError(f"unknown target '{name}'")


def get_mcmc(target: DiscreteTarget, device="cpu"):
    """Reference MCMC sampler matching the target type."""
    from dds.targets.ising.mcmc import IsingMCMC
    from dds.targets.potts.mcmc import PottsMCMC

    if isinstance(target, IsingModel):
        return IsingMCMC(target, device=device)
    if isinstance(target, PottsModel):
        return PottsMCMC(target, device=device)
    raise ValueError(type(target))


__all__ = [
    "DiscreteTarget", "IsingModel", "PottsModel", "AlloyFamily", "AlloyPoint",
    "build_target", "get_mcmc",
]
