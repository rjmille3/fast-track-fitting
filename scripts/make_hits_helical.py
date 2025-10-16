#!/usr/bin/env python3
import os
import argparse
import numpy as np

# --------------------------
# Geometry / physics helpers
# --------------------------
def track(phi, d0, phi0, pt, dz, tanl):
    """
    Helical track parameterization (x,y,z) as a function of angle increment phi.
    """
    alpha = 1.0 / 2.0  # constant: 1/(cB)
    q = 1.0
    ptinv = 1.0 / pt
    kappa = q * ptinv
    rho = alpha / kappa

    x = d0 * np.cos(phi0) + rho * (np.cos(phi0) - np.cos(phi0 + phi))
    y = d0 * np.sin(phi0) + rho * (np.sin(phi0) - np.sin(phi0 + phi))
    z = dz - rho * tanl * phi
    return x, y, z


def find_phi_inverse(target_radius_squared, d0, phi0, pt, dz, tanl, eps=1e-12):
    """
    Closed-form inversion for the intersection angle that hits radius^2 = target_radius_squared.
    Returns phi in [0, 2π).
    """
    alpha = 1.0 / 2.0  # 1/cB
    q = 1.0
    kappa = q / pt
    rho = alpha / kappa

    calc_x = d0 * np.cos(phi0) + rho * np.cos(phi0)
    calc_y = d0 * np.sin(phi0) + rho * np.sin(phi0)

    hyp = np.sqrt(calc_x**2 + calc_y**2)

    arccos_input = (calc_x**2 + calc_y**2 + rho**2 - target_radius_squared) / (2.0 * rho * hyp)
    clamped = np.clip(arccos_input, -1 + eps, 1 - eps)
    phi = np.arccos(clamped)
    return phi % (2.0 * np.pi)


def make_hits(params, min_r0, max_r0, nlayers, sigma, rng,
              noise="gaussian", skew_shape=10.0, skew_scale_mult=3.0):
    """
    For a single track parameter set, compute hits at detector layers,
    with noise type selected by `noise`:
      - 'gaussian'  : Normal(0, sigma)
      - 'skewed'    : Gamma(k=skew_shape, theta=skew_scale_mult*sigma)  (positive, skewed)
      - 'noiseless' : 0 noise
    """
    xs, ys, zs = [], [], []
    d0, phi0, pt, dz, tanl = params

    for r0 in np.linspace(min_r0, max_r0, nlayers):
        phi_at_r = find_phi_inverse(r0 * r0, d0, phi0, pt, dz, tanl)
        x0, y0, z0 = track(phi_at_r, d0, phi0, pt, dz, tanl)

        if noise == "gaussian":
            nx = rng.normal(scale=sigma)
            ny = rng.normal(scale=sigma)
            nz = rng.normal(scale=sigma)
        elif noise == "skewed":
            # Positive skew; increase scale via multiplier to roughly match stdev scale if desired
            theta = skew_scale_mult * sigma
            nx = rng.gamma(shape=skew_shape, scale=theta)
            ny = rng.gamma(shape=skew_shape, scale=theta)
            nz = rng.gamma(shape=skew_shape, scale=theta)
        elif noise == "noiseless":
            nx = ny = nz = 0.0
        else:
            raise ValueError(f"Unknown noise type: {noise}")

        xs.append(x0 + nx)
        ys.append(y0 + ny)
        zs.append(z0 + nz)

    return xs, ys, zs


def gen_tracks(n,
               min_r0, max_r0, nlayers, sigma,
               # parameter distributions
               d0_sigma=0.03,
               pt_lognorm_mean=4.0, pt_lognorm_sigma=0.5,
               dz_sigma=0.5, tanl_sigma=0.6,
               progress_every=10000,
               seed=None,
               # noise controls
               noise="gaussian", skew_shape=10.0, skew_scale_mult=3.0):
    """
    Generate n tracks with parameters sampled from specified prior distribution,
    and produce hits at the specified layer radii with selected noise type.
    """
    rng = np.random.default_rng(seed)
    tracks = []
    for i in range(n):
        if i % progress_every == 0:
            print(f"Track {i}/{n}", flush=True)

        d0 = abs(rng.normal(scale=d0_sigma))
        phi = rng.uniform(low=0.0, high=2.0 * np.pi)
        pt = rng.lognormal(mean=pt_lognorm_mean, sigma=pt_lognorm_sigma)
        dz = rng.normal(scale=dz_sigma)
        tanl = rng.normal(scale=tanl_sigma)
        params = (d0, phi, pt, dz, tanl)

        xs, ys, zs = make_hits(params, min_r0, max_r0, nlayers, sigma, rng,
                               noise=noise, skew_shape=skew_shape, skew_scale_mult=skew_scale_mult)
        tracks.append((params, xs, ys, zs))
    return tracks


def write_tracks_txt(tracks, out_path):
    """
    Write tracks to a text file.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for params, xs, ys, zs in tracks:
            f.write(f"{params[0]:1.8f}, {params[1]:1.8f}, {params[2]:1.8f}, {params[3]:1.8f}, {params[4]:1.8f}\n")
            for x, y, z in zip(xs, ys, zs):
                f.write(f"{x:1.8f}, {y:1.8f}, {z:1.8f}\n")
            f.write("EOT\n\n")


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Generate noisy helical track hits at concentric detector layers."
    )
    # dataset geometry / noise
    p.add_argument("--min-r0", type=float, default=1.0, help="Minimum detector layer radius.")
    p.add_argument("--max-r0", type=float, default=10.0, help="Maximum detector layer radius.")
    p.add_argument("--nlayers", type=int, default=10, help="Number of detector layers.")
    p.add_argument("--sigma", type=float, default=0.01, help="Base noise scale (used by gaussian/skewed).")

    # noise type
    p.add_argument("--noise", type=str, default="gaussian",
                   choices=["gaussian", "skewed", "noiseless"],
                   help="Noise model: gaussian (default), skewed (Gamma), or noiseless.")
    p.add_argument("--skew-shape", type=float, default=10.0,
                   help="Gamma shape parameter k for --noise=skewed.")
    p.add_argument("--skew-scale-mult", type=float, default=3.0,
                   help="Gamma scale = skew_scale_mult * sigma for --noise=skewed.")

    # count + output
    p.add_argument("-n", "--num-tracks", type=int, default=1_000_000, help="Number of tracks to generate.")
    p.add_argument("-o", "--out", type=str, default="data/helical/testing/1M.txt",
                   help="Output text file path.")

    # sampling distributions
    p.add_argument("--d0-sigma", type=float, default=0.03, help="Stddev for |Normal(0, d0_sigma)|.")
    p.add_argument("--pt-lognormal-mean", type=float, default=4.0, help="Mean of lognormal for pt.")
    p.add_argument("--pt-lognormal-sigma", type=float, default=0.5, help="Sigma of lognormal for pt.")
    p.add_argument("--dz-sigma", type=float, default=0.5, help="Stddev for dz ~ Normal(0, dz_sigma).")
    p.add_argument("--tanl-sigma", type=float, default=0.6, help="Stddev for tanl ~ Normal(0, tanl_sigma).")

    # misc
    p.add_argument("--progress-every", type=int, default=10000, help="Print progress every K tracks.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return p


def main():
    args = build_arg_parser().parse_args()

    tracks = gen_tracks(
        n=args.num_tracks,
        min_r0=args.min_r0,
        max_r0=args.max_r0,
        nlayers=args.nlayers,
        sigma=args.sigma,
        d0_sigma=args.d0_sigma,
        pt_lognorm_mean=args.pt_lognormal_mean,
        pt_lognorm_sigma=args.pt_lognormal_sigma,
        dz_sigma=args.dz_sigma,
        tanl_sigma=args.tanl_sigma,
        progress_every=args.progress_every,
        seed=args.seed,
        noise=args.noise,
        skew_shape=args.skew_shape,
        skew_scale_mult=args.skew_scale_mult,
    )
    write_tracks_txt(tracks, args.out)
    print(f"Done. Wrote {len(tracks)} tracks to {args.out}")


if __name__ == "__main__":
    main()
