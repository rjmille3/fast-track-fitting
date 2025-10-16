#!/usr/bin/env python3
"""
Helical least-squares fitting with argparse + multiprocessing.

Example:
  python scripts/gaussian_ls_fit.py \
    --input data/helical/testing/gaussian/raw/test.txt \
    --outdir data/helical/testing/gaussian/ls/realistic \
    -n 100000 --workers 20 --print_every 100 --checkpoint_every 10000 --seed 0
"""

import argparse
import os
import time
import random
import numpy as np
import scipy.optimize
import multiprocessing as mp

from utils import *
#from prediction import *

# -------------------------
# Globals (set by --geom)
# -------------------------
MIN_R0 = 1.0
MAX_R0 = 10.0
NLAYERS = 10


# -------------------------
# I/O helpers
# -------------------------
def writefile(least_squares_params, i, path):
    os.makedirs(path, exist_ok=True)
    outlines = []
    for ls in least_squares_params:
        ls_str = np.around(ls, decimals=4).astype(str)
        outlines.append(",".join(ls_str) + "\n")
    path_str = os.path.join(path, f"{i}.txt")
    with open(path_str, "w") as outfile:
        outfile.writelines(outlines)


# -------------------------
# Physics helpers
# -------------------------
def fast_fit_params(x, y, z):
    """
    Fast (approximate) fitter based on 10.1016/0168-9002(88)90722-X.
    Returns (d0, phi, 1/pt, dz, tanl)
    """
    r = x**2 + y**2
    u = x / r
    v = y / r
    pp, _ = np.polyfit(u, v, 2, cov=True)
    b = 0.5 / pp[2]
    a = -pp[1] * b
    R = np.sqrt(a**2 + b**2)
    e = -pp[0] / (R / b) ** 3  # approx d0
    magnetic_field = 2.0
    pT = magnetic_field * R
    p_rz = np.polyfit(np.sqrt(r), z, 2)
    pp_rz = np.poly1d(p_rz)
    z0 = pp_rz(abs(e))
    r3 = np.sqrt(r + z**2)
    p_zr = np.polyfit(r3, z, 2)
    cos_val = p_zr[0] * z0 + p_zr[1]
    cos_val = np.clip(cos_val, -1.0, 1.0)
    theta = np.arccos(cos_val)
    eta = -np.log(np.tan(theta / 2.0))
    phi = np.arctan2(b, a)
    return e, phi, 1.0 / pT, z0, -eta


def track(phi, d0, phi0, ptinv, dz, tanl):
    """Given helix parameters and sweep angle phi, compute the hit position."""
    alpha = 1 / 2.0  # 1/(cB)
    q = 1
    kappa = q * ptinv
    rho = alpha / kappa
    x = d0 * np.cos(phi0) + rho * (np.cos(phi0) - np.cos(phi0 + phi))
    y = d0 * np.sin(phi0) + rho * (np.sin(phi0) - np.sin(phi0 + phi))
    z = dz - rho * tanl * phi
    return x, y, z


def find_phi_inverse(target_radius_squared, d0, phi0, pt, dz, tanl, eps=1e-12):
    alpha = 1 / 2  # 1/cB
    q = 1
    kappa = q / pt
    rho = alpha / kappa

    calc_x = d0 * np.cos(phi0) + rho * np.cos(phi0)
    calc_y = d0 * np.sin(phi0) + rho * np.sin(phi0)
    hyp = np.sqrt(calc_x**2 + calc_y**2)

    arccos_input = (calc_x**2 + calc_y**2 + rho**2 - target_radius_squared) / (2 * rho * hyp)
    clamped = np.clip(arccos_input, -1 + eps, 1 - eps)
    phi = np.arccos(clamped)
    return phi % (2 * np.pi)


def helix_chisq_vectorized(params, hits, doxy=False, doz=False):
    """
    χ² between measured hits and helix hypothesis using analytic φ solution.

    params: (d0, φ0, 1/pT, dz, tanλ)
    hits:   (n_hits, 5) columns [x,y,z,r,φ_meas]  (we use x,y,z,r)
    """
    d0, phi0, ptinv, dz, tanl = params
    pt = 1.0 / ptinv

    target_r2 = np.linspace(MIN_R0, MAX_R0, NLAYERS) ** 2
    phi_vals = find_phi_inverse(target_r2, d0, phi0, pt, dz, tanl)

    x_pred, y_pred, z_pred = track(phi_vals, d0, phi0, ptinv, dz, tanl)
    pred_xyz = np.stack((x_pred, y_pred, z_pred), axis=1)
    meas_xyz = hits[:, :3]

    if doz:
        resid_z = meas_xyz[:, 2] - z_pred
        return np.sum(resid_z**2)

    if doxy:
        resid_xy = np.linalg.norm(meas_xyz[:, :2] - pred_xyz[:, :2], axis=1)
        return np.sum(resid_xy**2)

    resid = np.linalg.norm(meas_xyz - pred_xyz, axis=1)
    return np.sum(resid**2)


def fit_helix_params(hits, init):
    """
    Minimize χ² residuals to fit (d0, phi0, 1/pt, dz, tanl).
    """
    init = np.array(init, dtype=float)

    SIGMA_D0 = 0.03
    MU_LN_PT = 4.0
    SIGMA_LN_PT = 0.5
    SIGMA_DZ = 0.5
    SIGMA_TANL = 0.6
    K = 5

    b_d0 = K * SIGMA_D0
    b_phi = (0.0, 2.0 * np.pi)

    pt_low = np.exp(MU_LN_PT - K * SIGMA_LN_PT)
    pt_high = np.exp(MU_LN_PT + K * SIGMA_LN_PT)
    b_ptinv_dn = 1.0 / pt_high
    b_ptinv_up = 1.0 / pt_low

    b_z0 = K * SIGMA_DZ
    b_tanl = K * SIGMA_TANL

    if not np.isfinite(init[0]): init[0] = 0.0
    d0_max = max(b_d0, abs(init[0]))

    if not np.isfinite(init[1]): init[1] = np.pi
    if not np.isfinite(init[2]): init[2] = 1.0 / np.exp(MU_LN_PT)
    if not np.isfinite(init[3]): init[3] = 0.0
    dz_max = max(b_z0, abs(init[3]))
    if not np.isfinite(init[4]): init[4] = 0.0
    tanl_max = max(b_tanl, abs(init[4]))

    bounds = (
        (0, d0_max),
        b_phi,
        (b_ptinv_dn, b_ptinv_up),
        (-dz_max, dz_max),
        (-tanl_max, tanl_max),
    )

    res = scipy.optimize.minimize(
        helix_chisq_vectorized,
        init,
        args=(hits,),
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(ftol=1e-12, gtol=1e-8, maxiter=20000, maxfun=20000),
    )
    return res.x


def process_track(task):
    """
    Process one track.
    Input: (index, track) where track is (10,3) xyz
    Returns: (index, [chisq, d0, phi0, 1/pt, dz, tanl])
    """
    index, track = task
    r = np.sqrt(track[:, 0] ** 2 + track[:, 1] ** 2)
    phi_angles = np.arctan2(track[:, 1], track[:, 0])
    hits = np.column_stack((track, r, phi_angles))

    # Initial guess
    hinitg = list(fast_fit_params(hits[:, 0], hits[:, 1], hits[:, 2]))
    if hinitg[1] < 0:
        hinitg[1] = hinitg[1] + 2 * np.pi

    init_chisq_xy = helix_chisq_vectorized(hinitg, hits, doxy=True)
    init_chisq_z = helix_chisq_vectorized(hinitg, hits, doz=True)

    # Coarse scans
    scanranges = [
        np.linspace(0, 1, 100),            # d0 
        np.linspace(0, 2 * np.pi, 100),    # phi0
        np.linspace(1.0 / 200, 1.0 / 25, 100),  # 1/pt
        np.linspace(-5, 5, 100),           # dz
        np.linspace(-1.5, 1.5, 100),       # tanl
    ]

    # scan (phi0, 1/pt) in xy
    for v1 in scanranges[1]:
        for v2 in scanranges[2]:
            try_init = hinitg.copy()
            try_init[1] = v1
            try_init[2] = v2
            try_chisq_xy = helix_chisq_vectorized(try_init, hits, doxy=True)
            if try_chisq_xy < init_chisq_xy:
                init_chisq_xy = try_chisq_xy
                hinitg[1] = v1
                hinitg[2] = v2

    # scan (dz, tanl) in z
    for v1 in scanranges[3]:
        for v2 in scanranges[4]:
            try_init = hinitg.copy()
            try_init[3] = v1
            try_init[4] = v2
            try_chisq_z = helix_chisq_vectorized(try_init, hits, doz=True)
            if try_chisq_z < init_chisq_z:
                init_chisq_z = try_chisq_z
                hinitg[3] = v1
                hinitg[4] = v2

    hres = fit_helix_params(hits, hinitg)
    hfch = helix_chisq_vectorized(hres, hits)

    numFits = 0
    while hfch > 20.0 and numFits < 4:
        numFits += 1
        hinitg = hres
        init_chisq = hfch
        deltas = [
            [-0.02, -0.01, 0.01, 0.02],
            [-0.2, -0.1, 0.1, 0.2],
            [-2*hinitg[2], -1.5*hinitg[2], 0.5*hinitg[2], 1.0*hinitg[2]],
            [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5],
            [0.5, -0.25, -0.1, 0.1, 0.25, 0.5],
        ]
        for ip in range(5):
            for delta in deltas[ip]:
                try_init = list(hinitg)
                try_init[ip] = try_init[ip] + delta * (1.0 + np.random.normal(scale=0.25))
                try_chisq = helix_chisq_vectorized(try_init, hits)
                if try_chisq < init_chisq:
                    init_chisq = try_chisq
                    hinitg = try_init.copy()

                if hinitg[1] > 2 * np.pi:
                    hinitg[1] -= 2 * np.pi
                if hinitg[1] < 0:
                    hinitg[1] += 2 * np.pi

                hres = fit_helix_params(hits, hinitg)
                hfch = helix_chisq_vectorized(hres, hits)

    result = [hres[0], hres[1], hres[2], hres[3], hres[4]]
    return index, [hfch] + result


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Helical least-squares fitting")
    p.add_argument("--input", required=True, help="Path to input test.txt")
    p.add_argument("--outdir", required=True, help="Directory to write outputs")
    p.add_argument("-n", "--num", type=int, default=100000, help="Number of tracks to process")
    p.add_argument("--workers", type=int, default=20, help="Number of parallel workers")
    p.add_argument("--print_every", type=int, default=100, help="Print progress every N tracks")
    p.add_argument("--checkpoint_every", type=int, default=10000, help="Write checkpoint every N tracks")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--geom", type=float, nargs=3, metavar=("MIN_R0", "MAX_R0", "NLAYERS"),
                   default=None, help="Detector geometry override (e.g. 1.0 10.0 10)")
    return p.parse_args()


def main():
    global MIN_R0, MAX_R0, NLAYERS

    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Geometry override if provided
    if args.geom is not None:
        MIN_R0, MAX_R0, NLAYERS = float(args.geom[0]), float(args.geom[1]), int(args.geom[2])

    os.makedirs(args.outdir, exist_ok=True)
    start_time = time.time()
    xval_path = os.path.join(args.outdir, "X_vals.txt")

    # Load data
    print(f"[load] {args.input}", flush=True)
    X_test, y_test = parse_data(args.input)
    y_test = reduce_dim(y_test)

    n = min(args.num, len(X_test))
    X_test_short = X_test[:n]
    reshaped_x = X_test_short.reshape(-1, 10, 3)
    print(f"[info] Using geometry: min_r0={MIN_R0}, max_r0={MAX_R0}, nlayers={NLAYERS}", flush=True)
    print(f"[info] Reshaped X: {reshaped_x.shape}", flush=True)

    # Write X + targets for auditing
    with open(xval_path, "w") as fh:
        for i in range(reshaped_x.shape[0]):
            fh.write(", ".join(y_test[i].astype(str)) + "\n")
            for row in reshaped_x[i]:
                fh.write(", ".join(row.astype(str)) + "\n")
            fh.write("EOT\n")
            if i % 500 == 0:
                print(f"[xvals] wrote {i}", flush=True)

    # Multiprocessing
    tasks = [(i, reshaped_x[i]) for i in range(reshaped_x.shape[0])]
    pool = mp.Pool(args.workers)
    results = []

    for count, task_result in enumerate(pool.imap(process_track, tasks), start=1):
        results.append(task_result)

        if count % args.print_every == 0:
            print(f"[prog] {count} tracks in {time.time() - start_time:.2f}s", flush=True)

        if count % args.checkpoint_every == 0:
            print(f"[ckpt] writing checkpoint at {count}", flush=True)
            sorted_results = sorted(results, key=lambda x: x[0])
            ls_preds_arr = np.array([res for (_, res) in sorted_results])
            writefile(ls_preds_arr, count, args.outdir)

    pool.close()
    pool.join()

    # Final write
    results = sorted(results, key=lambda x: x[0])
    ls_preds_arr = np.array([res for (_, res) in results])
    writefile(ls_preds_arr, n, args.outdir)

    print(f"[done] {ls_preds_arr.shape[0]} tracks in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
