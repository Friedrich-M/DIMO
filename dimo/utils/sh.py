"""Spherical-harmonics colour helpers (degree-0 band only is used by DIMO)."""

C0 = 0.28209479177387814


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5
