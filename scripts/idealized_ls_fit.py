#!/usr/bin/env python3
import argparse, os, time
import numpy as np
import math
from numpy.linalg import norm
import scipy.optimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import random
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import multiprocessing as mp

#from util import *
#from prediction import *

min_r0 = 0.0
max_r0 = 100.0
nlayers = 10
np.set_printoptions(suppress=True)

def parse_data_simple(filename, invert_pt=False):
    f = open(filename)
    data = f.readlines()
    tgts = []
    datapoints = []
    features = []
    newdata = False
    for line in data:
        line = line.strip()
        vals = line.split(",")
        if newdata and len(vals) >= 3 and len(features) == 10:
            #print(features)
            datapoints.append(features)
            newdata = False
            features = []

        if len(vals) == 5 or len(vals) == 6:
            #print(vals)
            tgts.append(vals)
        if len(vals) == 3:
            features.append(vals)
        else:
            newdata = True


    datapoints.append(features)


    datapoints = np.array(datapoints).astype(float)
    flattened_data = datapoints.reshape(datapoints.shape[0], 30)
    tgts = np.array(tgts).astype(float)

    if invert_pt == True:
        tgts[:,2] = 1.0 / tgts[:,2]

    return flattened_data, tgts

def writefile(least_squares_params, i, path):
    outlines = []
    for ls in least_squares_params:
        ls_str = np.around(ls, decimals=6)
        ls_str = ls_str.astype(str)
        out = ",".join(ls_str) + "\n"
        outlines.append(out)
    path_str = os.path.join(path, f"{i}.txt")
    with open(path_str, "w") as outfile:
        outfile.writelines(outlines)

# ---------- Helix helpers (verbatim behavior) ----------
def find_phi_inverse(target_radius_squared, d0, phi0, pt, dz, tanl, eps=1e-12):
    alpha = 1 / 2  # 1/cB
    q = 1
    kappa = q / pt
    rho = alpha / kappa

    calculated_term_x = d0 * np.cos(phi0) + rho * np.cos(phi0)
    calculated_term_y = d0 * np.sin(phi0) + rho * np.sin(phi0)

    hypotenuse = np.sqrt(calculated_term_x ** 2 + calculated_term_y ** 2)

    arccos_input = (calculated_term_x ** 2 + calculated_term_y ** 2 + rho ** 2 - target_radius_squared) / (2 * rho * hypotenuse)
    clamped = np.clip(arccos_input, -1 + eps, 1 - eps)  # to avoid NaNs
    arccos_term = np.arccos(clamped)
    phi = arccos_term

    return phi % (2 * np.pi)  # wrap angle into [0, 2π)

def fast_fit_params(x, y, z):
    r = x**2 + y**2
    u = x / r
    v = y / r
    pp, _ = np.polyfit(u, v, 2, cov=True)
    b = 0.5 / pp[2]
    a = -pp[1] * b
    R = np.sqrt(a**2 + b**2)
    e = -pp[0] / (R / b) ** 3  # approx equals to d0
    magnetic_field = 2.0
    pT = magnetic_field * R  # in MeV
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
    return e, phi, 1.0/pT, z0, -eta

def track(phi, d0, phi0, ptinv, dz, tanl):
    alpha = 1/2.0  # constant: 1/(cB)
    q = 1
    kappa = q * ptinv
    rho = alpha / kappa
    x = d0 * np.cos(phi0) + rho * (np.cos(phi0) - np.cos(phi0 + phi))
    y = d0 * np.sin(phi0) + rho * (np.sin(phi0) - np.sin(phi0 + phi))
    z = dz - rho * tanl * phi
    return x, y, z

def helix_chisq_vectorized(params, hits, trackid=-1, doxy=False, doz=False):
    d0, phi0, ptinv, dz, tanl = params
    pt = 1.0 / ptinv  

    min_r0  = 1.0
    max_r0  = 10.0
    nlayers = 10
    target_r2 = np.linspace(min_r0, max_r0, nlayers)**2

    phi_base  = find_phi_inverse(target_r2, d0, phi0, pt, dz, tanl)
    phi_vals = phi_base

    x_pred, y_pred, z_pred = track(phi_vals, d0, phi0, ptinv, dz, tanl)
    pred_xyz   = np.stack((x_pred, y_pred, z_pred), axis=1)
    meas_xyz   = hits[:, :3]

    if doz:
        resid_z = meas_xyz[:, 2] - z_pred
        return np.sum(resid_z ** 2)

    if doxy:
        resid_xy = np.linalg.norm(meas_xyz[:, :2] - pred_xyz[:, :2], axis=1)
        return np.sum(resid_xy ** 2)

    resid = np.linalg.norm(meas_xyz - pred_xyz, axis=1)
    return np.sum(resid ** 2)

def helix_chisq_vectorized_xy(xy_params, rz_params,hits):
    return helix_chisq_vectorized(np.concatenate((np.array(xy_params), np.array(rz_params)) ),hits,doxy=True)

def helix_chisq_vectorized_z(rz_params, xy_params,hits):
    return helix_chisq_vectorized(np.concatenate((np.array(xy_params),np.array(rz_params))),hits,doz=True)

def fit_helix_params(hits, init):
    init = np.array(init, dtype=float)

    # same prior numbers
    SIGMA_D0   = 0.03
    MU_LN_PT   = 4.0
    SIGMA_LN_PT= 0.5
    SIGMA_DZ   = 0.5
    SIGMA_TANL = 0.6
    K          = 5

    b_d0   = K * SIGMA_D0
    b_phi  = (0.0, 2.0*np.pi)
    pt_low  = np.exp(MU_LN_PT - K*SIGMA_LN_PT)
    pt_high = np.exp(MU_LN_PT + K*SIGMA_LN_PT)
    b_ptinv_dn = 1.0 / pt_high
    b_ptinv_up = 1.0 / pt_low
    b_z0   = K * SIGMA_DZ
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
        (0,        d0_max),
        b_phi,
        (b_ptinv_dn,     b_ptinv_up),
        (-dz_max,        dz_max),
        (-tanl_max,      tanl_max)
    )

    res = scipy.optimize.minimize(
        helix_chisq_vectorized, init, args=(hits,), method='L-BFGS-B',
        bounds=bounds,
        options={
            'ftol': 1e-12,
            'gtol': 1e-8,
            'maxiter': 20000,
            'maxfun': 20000,
        }
    )
    return res.x

def process_track(task):
    index, track, helix_targets, noise_stds = task
    r = np.sqrt(track[:, 0]**2 + track[:, 1]**2)
    phi_angles = np.arctan2(track[:, 1], track[:, 0])
    hits = np.column_stack((track, r, phi_angles))

    helix_targets = np.array(helix_targets)
    noise = np.random.normal(loc=0.0, scale=noise_stds, size=noise_stds.shape)
    helix_targets = helix_targets + noise

    hinitg = [helix_targets[0], helix_targets[1], helix_targets[2], helix_targets[3], helix_targets[4]]

    init_chisq = helix_chisq_vectorized(hinitg,hits)
    init_chisq_xy = helix_chisq_vectorized(hinitg,hits,doxy=True)
    init_chisq_z = helix_chisq_vectorized(hinitg,hits,doz=True)
    true_chi_xy = init_chisq_xy
    true_chi_z = init_chisq_z

    bestg = hinitg
    #orig
    scales_init = [0.05,0.05,0.005,0.02,0.002]
    scales = scales_init.copy()
    tol = 0.005

    # XY refinement (identical)
    while ( any (np.array( scales[:3]) >  tol * np.array(scales_init[:3] ) )):
        scanranges = []
        for i in range( len(scales ) ):
            scanranges.append( np.linspace(-scales[i],scales[i],5 ) )

        ip0=0; ip1=1; ip2=2
        best_prior = bestg.copy()
        for v0 in scanranges[ip0]:
            for v1 in scanranges[ip1]:
                for v2 in scanranges[ip2]:
                    try_init = bestg.copy()
                    try_init[ip0] = try_init[ip0] + v0
                    try_init[ip1] = try_init[ip1] + v1
                    try_init[ip2] = try_init[ip2] + v2
                    try_chisq_xy = helix_chisq_vectorized(try_init,hits,doxy=True)
                    if (try_chisq_xy < init_chisq_xy):
                        init_chisq_xy = try_chisq_xy
                        bestg[ip0] = try_init[ip0]
                        bestg[ip1] = try_init[ip1]
                        bestg[ip2] = try_init[ip2]
        for i in range(3):
            if (np.isclose(best_prior[i],bestg[i]) ):
                scales[i] = scales[i] / 2.0

    # Z refinement (identical)
    while ( any (np.array( scales[3:5]) >  tol * np.array(scales_init[3:5] ) )):
        scanranges = []
        for i in range( len(scales ) ):
            scanranges.append( np.linspace(-scales[i],scales[i],5 ) )

        ip0=3; ip1=4
        best_prior = bestg.copy()
        for v0 in scanranges[ip0]:
            for v1 in scanranges[ip1]:
                try_init = bestg.copy()
                try_init[ip0] = try_init[ip0] + v0
                try_init[ip1] = try_init[ip1] + v1
                try_chisq_z = helix_chisq_vectorized(try_init,hits,doz=True)
                if (try_chisq_z < init_chisq_z):
                    init_chisq_z = try_chisq_z
                    bestg[ip0] = try_init[ip0]
                    bestg[ip1] = try_init[ip1]
        for i in range(3,5):
            if (np.isclose(best_prior[i],bestg[i]) ):
                scales[i] = scales[i] / 2.0

    # Final fit (identical: from hinitg)
    seed = bestg.copy()
    hres = fit_helix_params(hits, seed)
    hfch_xy = helix_chisq_vectorized(hres, hits, doxy=True)
    hfch_z  = helix_chisq_vectorized(hres, hits, doz=True)


    if (hfch_xy < init_chisq_xy):
        result = hres[:3]
        res_chi_xy = hfch_xy
    else:
        result = seed[:3]
        res_chi_xy = init_chisq_xy

    if (hfch_z < init_chisq_z):
        result = np.concatenate((result,hres[3:5]))
        res_chi_z = hfch_z
    else:
        result = np.concatenate((result,seed[3:5]))
        res_chi_z = init_chisq_z

    final_result = [true_chi_xy, true_chi_z, res_chi_xy, res_chi_z] + result.tolist()
    final_result = np.array(final_result)
    return index, final_result

# ---------- Main (CLI wrapper; behavior preserved) ----------
def main():
    parser = argparse.ArgumentParser(description="CLI wrapper.")
    parser.add_argument("--input", required=True, help="Path to test.txt")
    parser.add_argument("--outdir", required=True, help="Output directory for predictions/checkpoints")
    parser.add_argument("-n", type=int, default=100000, help="Number of trajectories to process (default 100000)")
    parser.add_argument("--noise_stds", nargs=5, type=float, default=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6],  help="Stddevs for Gaussian noise added to initial helix targets [d0, phi0, ptinv, dz, tanl].")
    args = parser.parse_args()

    start_time = time.time()

    X_test, y_test = parse_data_simple(args.input, invert_pt=True)

    n = args.n
    ls_preds_arr = []
    X_test_short = X_test[0:n]
    reshaped_x = X_test_short.reshape(-1, 10, 3)
    print("reshaped x shape was ", reshaped_x.shape, flush=True)

    outpath = args.outdir
    X_val_file_path = os.path.join(outpath, "X_vals.txt")

    # Write X values & targets (identical cadence)
    with open(X_val_file_path, "w") as X_val_file:
        for i in range(reshaped_x.shape[0]):
            tgts_str = ", ".join(y_test[i].astype(str)) + "\n"
            X_val_file.write(tgts_str)
            for row in reshaped_x[i]:
                row_str = ", ".join(row.astype(str)) + "\n"
                X_val_file.write(row_str)
            X_val_file.write("EOT \n")
            if i % 500 == 0:
                print(f"Completed writing X values for {i} trajectories", flush=True)

    noise_stds = np.array(args.noise_stds, dtype=float)
    tasks = [(i, reshaped_x[i], y_test[i], noise_stds) for i in range(reshaped_x.shape[0])]

    pool = mp.Pool(mp.cpu_count())
    results = []
    for count, task_result in enumerate(pool.imap(process_track, tasks), start=1):
        results.append(task_result)
        if count % 10000 == 0:
            print(f"Completed processing {count} tracks in {time.time()-start_time:.2f} seconds", flush=True)
            sorted_results = sorted(results, key=lambda x: x[0])
            ls_preds_arr = np.array([res for (_, res) in sorted_results])
            writefile(ls_preds_arr, count, outpath)
        elif count % 100 == 0:
            print(f"Completed processing {count} tracks in {time.time()-start_time:.2f} seconds", flush=True)
    pool.close()
    pool.join()

    results = sorted(results, key=lambda x: x[0])

    ls_preds_arr = np.array([res for (_, res) in results])

    writefile(ls_preds_arr, n, outpath)
    print("Fitting completed for", ls_preds_arr.shape[0], "tracks in", time.time()-start_time, "seconds")

if __name__ == "__main__":
    main()
