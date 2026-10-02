"""
Unified PointNeXt-offset training entry for StaConGrasp.

"""

from __future__ import annotations

import argparse
import sys


def _parse_stage(argv):
    parser = argparse.ArgumentParser(
        description="Unified StaConGrasp training entry. Use --stage diffusion or --stage stability."
    )
    parser.add_argument(
        "--stage",
        choices=("diffusion", "contact_diffusion", "stability", "contact_stability"),
        required=True,
    )
    args, remaining = parser.parse_known_args(argv)
    return args.stage, remaining


def _run_diffusion(remaining):
    import train_contact_diffusion_base as base
    from network.contact_diffusion_pointnext_offset import (
        ContactDiffusionNet,
        build_contact_diffusion_cfg,
        load_contact_diffusion_net,
        save_contact_diffusion_ckpt,
    )

    base.ContactDiffusionNet = ContactDiffusionNet
    base.build_contact_diffusion_cfg = build_contact_diffusion_cfg
    base.save_contact_diffusion_ckpt = save_contact_diffusion_ckpt
    base.load_contact_diffusion_net = load_contact_diffusion_net
    sys.argv = [sys.argv[0]] + remaining
    base.main()


def _run_stability(remaining):
    import train_contact_stability_base as base
    from network.contact_stability_pointnext_offset import (
        ContactStabilityNet,
        build_contact_stability_cfg,
        init_contact_stability_from_v2,
        load_contact_stability_net,
        save_contact_stability_ckpt,
    )

    base.ContactStabilityNet = ContactStabilityNet
    base.build_contact_stability_cfg = build_contact_stability_cfg
    base.init_contact_stability_from_v2 = init_contact_stability_from_v2
    base.save_contact_stability_ckpt = save_contact_stability_ckpt
    base.load_contact_stability_net = load_contact_stability_net
    sys.argv = [sys.argv[0]] + remaining
    base.main()


def main():
    stage, remaining = _parse_stage(sys.argv[1:])
    if stage in ("diffusion", "contact_diffusion"):
        _run_diffusion(remaining)
    else:
        _run_stability(remaining)


if __name__ == "__main__":
    main()
