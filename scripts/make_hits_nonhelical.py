#!/usr/bin/env python3
# gen_nonhelical_sin_tracks.py
import argparse
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

# --------------------
# Core
# --------------------
def track(phi, d0, phi0, A, omega, dz, tanl):
    """
    Non-helical track with sinusoidal perturbation in y.
    """
    x = (d0 + phi) * np.cos(phi0)
    y = (d0 + phi) * np.sin(phi0) + A * (np.sin(omega * (phi - phi0)))
    z = dz - tanl * phi
    return x, y, z

def dr(phi, r02, d0, phi0, A, omega, dz, tanl):
    x, y, _ = track(phi, d0, phi0, A, omega, dz, tanl)
    r2 = x * x + y * y
    return np.fabs(r2 - r02)

def find_phi(r0, d0, phi0, A, omega, dz, tanl):
    """
    (Optional) optimizer-based intersection finder.
    """
    res = scipy.optimize.minimize(
        dr, 0.0, method="Nelder-Mead",
        args=(r0, d0, phi0, A, omega, dz, tanl),
    )
    return float(res.x[0])

def find_phi_grid(r0_squared, d0, phi0, A, omega, dz, tanl, phi_max=12.0, ngrid=20_000):
    """
    Grid search intersection angle with desired radius^2 ~= r0_squared.
    """
    t_values = np.linspace(0.0, phi_max, ngrid)
    x, y, _ = track(t_values, d0, phi0, A, omega, dz, tanl)
    r2 = x**2 + y**2
    idx_best = np.argmin(np.abs(r2 - r0_squared))
    return float(t_values[idx_best])

def make_hits(params, min_r0, max_r0, nlayers, sigma, phi_max, ngrid,
              rng,
              noise="gaussian", skew_shape=10.0, skew_scale_mult=3.0):
    """
    Compute hits at concentric detector layers with selectable noise:
      - 'gaussian'  : Normal(0, sigma)
      - 'skewed'    : Gamma(k=skew_shape, theta=skew_scale_mult*sigma)  (positive, skewed)
      - 'noiseless' : 0 noise
    """
    xs, ys, zs = [], [], []
    for r0 in np.linspace(min_r0, max_r0, nlayers):
        phi0 = find_phi_grid(r0 * r0, *params, phi_max=phi_max, ngrid=ngrid)
        x0, y0, z0 = track(phi0, *params)

        if noise == "gaussian":
            nx = rng.normal(scale=sigma)
            ny = rng.normal(scale=sigma)
            nz = rng.normal(scale=sigma)
        elif noise == "skewed":
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

def gen_tracks(
    n,
    min_r0, max_r0, nlayers, sigma,
    A_min, A_max, omega_min, omega_max,
    d0_sigma, dz_sigma, tanl_sigma,
    phi_sin_cut,
    phi_max, ngrid,
    log_every,
    rng,
    noise="gaussian", skew_shape=10.0, skew_scale_mult=3.0
):
    tracks = []
    for i in range(n):
        if log_every and (i % log_every == 0):
            print(f"Track {i}/{n}", flush=True)

        while True:
            d0 = np.fabs(rng.normal(scale=d0_sigma))
            phi = rng.uniform(low=0.0, high=2.0 * np.pi)
            A = rng.uniform(low=A_min, high=A_max)
            omega = rng.uniform(low=omega_min, high=omega_max)
            dz = rng.normal(scale=dz_sigma)
            tanl = rng.normal(scale=tanl_sigma)
            params = (d0, phi, A, omega, dz, tanl)

            if np.abs(np.sin(phi)) > phi_sin_cut:
                continue

            xs, ys, zs = make_hits(
                params, min_r0, max_r0, nlayers, sigma, phi_max, ngrid,
                rng=rng, noise=noise, skew_shape=skew_shape, skew_scale_mult=skew_scale_mult
            )

            # distance guard to avoid near-duplicates
            ok = True
            for j in range(len(xs) - 1):
                dx = xs[j + 1] - xs[j]
                dy = ys[j + 1] - ys[j]
                dzv = zs[j + 1] - zs[j]
                if np.sqrt(dx*dx + dy*dy + dzv*dzv) <= 0.05:
                    ok = False
                    break

            if ok:
                tracks.append((params, xs, ys, zs))
                break
    return tracks

def maybe_plot_first(hits, out_png):
    xs, ys, zs = hits
    plt.figure(figsize=(10, 5))

    # XY
    plt.subplot(1, 2, 1)
    plt.plot(xs, ys, "x", label="Actual Hits")
    plt.xlabel("X"); plt.ylabel("Y"); plt.title("Track 0 in XY")
    plt.legend(); plt.xlim(-10, 10); plt.ylim(-10, 10)

    # XZ
    plt.subplot(1, 2, 2)
    plt.plot(xs, zs, "x", label="Actual Hits")
    plt.xlabel("X"); plt.ylabel("Z"); plt.title("Track 0 in XZ")
    plt.legend(); plt.xlim(-10, 10); plt.ylim(-10, 10)

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png)
    else:
        plt.show()
    plt.clf()

# --------------------
# CLI
# --------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate non-helical (sinusoidal-y) tracks and hits, write to a text file."
    )
    # output & amount
    p.add_argument("--out", required=True,
                   help="Output file path (e.g., /path/to/100k.txt)")
    p.add_argument("--n-tracks", type=int, default=100_000,
                   help="Number of tracks to generate (default: 100000)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed (default: 0)")
    p.add_argument("--log-every", type=int, default=1000,
                   help="Print progress every N tracks (0 to disable)")

    # detector/geometry
    p.add_argument("--min-r0", type=float, default=1.0)
    p.add_argument("--max-r0", type=float, default=10.0)
    p.add_argument("--nlayers", type=int, default=10)

    # noise (IDENTICAL DEFAULTS to helical script)
    p.add_argument("--sigma", type=float, default=0.01,
                   help="Base noise scale (used by gaussian/skewed).")
    p.add_argument("--noise", type=str, default="gaussian",
                   choices=["gaussian", "skewed", "noiseless"],
                   help="Noise model: gaussian (default), skewed (Gamma), or noiseless.")
    p.add_argument("--skew-shape", type=float, default=10.0,
                   help="Gamma shape parameter k for --noise=skewed.")
    p.add_argument("--skew-scale-mult", type=float, default=3.0,
                   help="Gamma scale = skew_scale_mult * sigma for --noise=skewed.")

    # grid intersection
    p.add_argument("--phi-max", type=float, default=12.0,
                   help="Max phi for grid search")
    p.add_argument("--ngrid", type=int, default=20_000,
                   help="Number of grid points for phi search")

    # parameter priors
    p.add_argument("--A-min", type=float, default=0.5)
    p.add_argument("--A-max", type=float, default=1.5)
    p.add_argument("--omega-min", type=float, default=0.5)
    p.add_argument("--omega-max", type=float, default=3.0)
    p.add_argument("--d0-sigma", type=float, default=0.01)
    p.add_argument("--dz-sigma", type=float, default=1.0)
    p.add_argument("--tanl-sigma", type=float, default=0.3)
    p.add_argument("--phi-sin-cut", type=float, default=0.95,
                   help="Reject if |sin(phi)| > cut (default: 0.95)")

    # optional quick viz for first track
    p.add_argument("--plot-first", action="store_true",
                   help="Plot the first generated track (XY & XZ)")
    p.add_argument("--plot-out", type=str, default="",
                   help="If set, save the plot to this PNG instead of showing it")

    return p.parse_args()

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    tracks = gen_tracks(
        n=args.n_tracks,
        min_r0=args.min_r0, max_r0=args.max_r0, nlayers=args.nlayers, sigma=args.sigma,
        A_min=args.A_min, A_max=args.A_max, omega_min=args.omega_min, omega_max=args.omega_max,
        d0_sigma=args.d0_sigma, dz_sigma=args.dz_sigma, tanl_sigma=args.tanl_sigma,
        phi_sin_cut=args.phi_sin_cut,
        phi_max=args.phi_max, ngrid=args.ngrid,
        log_every=args.log_every,
        rng=rng,
        noise=args.noise, skew_shape=args.skew_shape, skew_scale_mult=args.skew_scale_mult
    )

    if args.plot_first and len(tracks) > 0:
        _, xs0, ys0, zs0 = tracks[0]
        maybe_plot_first((xs0, ys0, zs0), args.plot_out if args.plot_out else None)

    with open(args.out, "w") as f:
        for params, xs, ys, zs in tracks:
            f.write("%1.8f, %1.8f, %1.8f, %1.8f, %1.8f, %1.8f\n" % params)
            for i in range(len(xs)):
                f.write("%1.8f, %1.8f, %1.8f\n" % (xs[i], ys[i], zs[i]))
            f.write("EOT\n\n")

if __name__ == "__main__":
    main()
