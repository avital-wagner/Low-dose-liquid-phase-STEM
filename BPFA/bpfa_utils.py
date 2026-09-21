import os
import shutil

import numpy as np
import arrayfire as af
import cv2
import torch
import matplotlib.pyplot as plt
import pickle
import math

# from niqe import niqe as niqe_fn
from PIL import Image


# =========================
# DISPLAY
# =========================
def imshow(mat, title: str, scale: float = 1.0, delay: int = 1,
           invert: bool = False, vmin: float = 0, vmax: float = 1):

    if isinstance(mat, af.Array):
        mat = mat.to_ndarray()

    if invert:
        mat = 1 - mat

    if scale != 1.0:
        mat = cv2.resize(mat, None, None, scale, scale, cv2.INTER_NEAREST)

    mat = np.clip(mat, vmin, vmax)
    mat = (mat - vmin) / (vmax - vmin)

    cv2.imshow(title, mat)
    cv2.waitKey(delay)


# =========================
# BPFA STEPS
# =========================
def perform_batch(bpfa, input_array, mask_array, batch_size):

    bpfa.batchX, bpfa.batchM, bpfa.batchXhat, bpfa.batchIdx, \
    bpfa.indexes, bpfa.batch_start, bpfa.epoch = \
        bpfa.next_batch(input_array, mask_array, bpfa.patch_shape,
                        bpfa.order, batch_size,
                        bpfa.batch_start, bpfa.epoch, bpfa.indexes)

    bpfa.D = bpfa.post_batch(bpfa.D, bpfa.batchX, bpfa.batchM,
                             bpfa.ones_element, bpfa.avg_element)

    bpfa.batchW, bpfa.batchF, bpfa.dtd, bpfa.dtx, bpfa.ge_dtd, \
    bpfa.diag_dtd, bpfa.dd_sig_dd, bpfa.dr, bpfa.pi, bpfa.penalty, \
    bpfa.scale, bpfa.delta_e, bpfa.dtx_dd, bpfa.mu, bpfa.mu_sigma, bpfa.S = \
        bpfa.sparse_code(bpfa.D, bpfa.batchW, bpfa.batchX, bpfa.batchM,
                         bpfa.batchF, bpfa.batchIdx,
                         bpfa.dtd, bpfa.dtx, bpfa.ge_dtd,
                         bpfa.diag_dtd, bpfa.dd_sig_dd, bpfa.dr,
                         bpfa.pi, bpfa.penalty, bpfa.scale,
                         bpfa.delta_e, bpfa.dtx_dd,
                         bpfa.mu, bpfa.mu_sigma, bpfa.S,
                         batch_size, bpfa.limit, bpfa.g_e, bpfa.g_w)

    bpfa.batchXhat = bpfa.update_prediction(bpfa.D, bpfa.batchW)
    bpfa.batchR = bpfa.update_residual(bpfa.batchXhat, bpfa.batchM, bpfa.batchX)

    bpfa.D, bpfa.bWW, bpfa.A, bpfa.B, bpfa.vg_d, bpfa.g_dI, bpfa.d2 = \
        bpfa.update_dictionary(bpfa.D, bpfa.batchX, bpfa.batchW,
                               bpfa.batchM, bpfa.batchR,
                               bpfa.bWW, bpfa.A, bpfa.B,
                               bpfa.vg_d, bpfa.g_dI,
                               bpfa.g_e, bpfa.g_d,
                               batch_size,
                               bpfa.ones_element, bpfa.avg_element)

    bpfa.aa, bpfa.bb = bpfa.update_local_parameters(
        bpfa.batchW, bpfa.batchM, bpfa.batchX,
        bpfa.aa, bpfa.bb, bpfa.lr,
        batch_size, bpfa.total_patches,
        bpfa.pi_a, bpfa.pi_b, bpfa.K
    )

    bpfa.pi, bpfa.vg_d, bpfa.g_e, bpfa.g_w = \
        bpfa.update_dictionary_parameters(
            bpfa.batchM, bpfa.batchR, bpfa.batchW,
            bpfa.aa, bpfa.bb, bpfa.d2,
            bpfa.pi, bpfa.vg_d, bpfa.K,
            bpfa.lr, bpfa.g_e, bpfa.g_w
        )

    bpfa.fullW, bpfa.fullF = bpfa.save_batch(
        bpfa.batchW, bpfa.batchIdx,
        bpfa.batchF, bpfa.fullW, bpfa.fullF,
        bpfa.total_patches, bpfa.K
    )

def perform_batch_sparse_only(bpfa, input_array, mask_array, batch_size):
    # Get batch
    bpfa.batchX, bpfa.batchM, bpfa.batchXhat, bpfa.batchIdx, \
    bpfa.indexes, bpfa.batch_start, bpfa.epoch = \
        bpfa.next_batch(input_array, mask_array,
                        bpfa.patch_shape, bpfa.order,
                        batch_size,
                        bpfa.batch_start, bpfa.epoch, bpfa.indexes)

    # (Optional) preprocessing step
    bpfa.D = bpfa.post_batch(bpfa.D, bpfa.batchX, bpfa.batchM,
                             bpfa.ones_element, bpfa.avg_element)

    # ONLY DO SPARSE CODING
    bpfa.batchW, bpfa.batchF, bpfa.dtd, bpfa.dtx, bpfa.ge_dtd, \
    bpfa.diag_dtd, bpfa.dd_sig_dd, bpfa.dr, bpfa.pi, bpfa.penalty, \
    bpfa.scale, bpfa.delta_e, bpfa.dtx_dd, bpfa.mu, bpfa.mu_sigma, bpfa.S = \
        bpfa.sparse_code(bpfa.D, bpfa.batchW, bpfa.batchX, bpfa.batchM,
                         bpfa.batchF, bpfa.batchIdx,
                         bpfa.dtd, bpfa.dtx, bpfa.ge_dtd,
                         bpfa.diag_dtd, bpfa.dd_sig_dd, bpfa.dr,
                         bpfa.pi, bpfa.penalty, bpfa.scale,
                         bpfa.delta_e, bpfa.dtx_dd,
                         bpfa.mu, bpfa.mu_sigma, bpfa.S,
                         batch_size, bpfa.limit, bpfa.g_e, bpfa.g_w)

    # Reconstruction (optional)
    bpfa.batchXhat = bpfa.update_prediction(bpfa.D, bpfa.batchW)
    bpfa.batchR = bpfa.update_residual(bpfa.batchXhat, bpfa.batchM, bpfa.batchX)

    # Save results
    bpfa.fullW, bpfa.fullF = bpfa.save_batch(
        bpfa.batchW, bpfa.batchIdx,
        bpfa.batchF, bpfa.fullW, bpfa.fullF,
        bpfa.total_patches, bpfa.K
    )


# =========================
# METRICS
# =========================

def get_metrics(metrics, reconstruction, input_array, mask_array):
    metric_values = {}

    # the Metric class expects an RGB image
    reconstruction_rgb = cv2.cvtColor(np.array(reconstruction), cv2.COLOR_GRAY2RGB)
    
    # loop over metric objects and compute their corresponding values
    # hyperparameters are individually extracted
    for metric in metrics:
        print("Calculating", metric.name())

        dict_values = metric.compute(reconstruction_rgb)

        # add metric + value to dict
        metric_values.update(dict_values)

    recon_masked = reconstruction * mask_array
    
    # residual done separately for now
    residual = np.mean(np.abs(recon_masked - input_array))

    # temporary hardcoded residual
    metric_values["residual"] = residual
    
    return metric_values, residual


# =========================
# PLOTTING
# =========================
def plot_metrics(metric_vals, return_plot = False):
    metric_names = list(metric_vals.keys())
    n_metrics = len(metric_names)
    
    # create a roughly square grid
    n_cols = math.ceil(math.sqrt(n_metrics))
    n_rows = math.ceil(n_metrics / n_cols)
    
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    ax = ax.ravel() # accounts for only 1 metric
    
    for i, m in enumerate(metric_names):
        ax[i].plot(metric_vals[m])
        ax[i].set_title(m)
        ax[i].grid(True)

    # hide any unused axes
    for j in range(n_metrics, len(ax)):
        ax[j].set_visible(False)
    
    plt.tight_layout()
    
    if return_plot:
        return fig
    else:
        fig.canvas.draw()
    
        img = np.asarray(fig.canvas.buffer_rgba())
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
        cv2.imshow("Metrics", img_bgr)
        cv2.waitKey(1)
    
        plt.close(fig)

def plot_atom_scores(bpfa, return_plot = False):
    all_scores = []
    atom_uses = []
    
    for i in range(bpfa.K):
        atom_weights = bpfa.fullW[i].to_ndarray()
        atom_freq = bpfa.fullF[i].to_ndarray()


        nr_in_use = np.count_nonzero(atom_weights)

        # prevent division by zero
        if nr_in_use > 0:
            final_score = np.sum(atom_freq) / nr_in_use
        else:
            final_score = 0

        all_scores.append(final_score)
        atom_uses.append(nr_in_use)

    all_scores = np.array(all_scores)

    # Normalize to [0, 1]
    min_val = all_scores.min()
    max_val = all_scores.max()

    all_scores_norm = 1 - ( (all_scores - min_val) / (max_val - min_val) )
                        
    # # Plot
    fig = plt.figure("Atom Scores", figsize=(12, 8))
    plt.clf()
    plt.plot(all_scores_norm, 'o')
    plt.xlabel("Atom index")
    plt.ylabel("Normalized score")
    plt.title("Atom Score (normalized 0–1)")
    plt.grid(True)

    if return_plot:
        return fig
    else:
        fig.canvas.draw()
    
        # Get RGBA buffer
        img = np.asarray(fig.canvas.buffer_rgba())
        
        # Convert RGBA → BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        cv2.imshow("Atom Scores", img_bgr)
        cv2.waitKey(1)  
    
        plt.close(fig)

def plot_parameters(tracker):
    keys = list(tracker.keys())
    n = len(keys)

    if n == 0:
        print("Tracker is empty.")
        return

    # Define grid size
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, key in enumerate(keys):
        ax = axes[i]
        values = tracker[key]

        if len(values) == 0:
            ax.set_title(f"{key} (no data)")
            continue

        ax.plot(values)
        ax.set_title(key)
        ax.set_xlabel("Time")
        ax.grid(True)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    
    return fig

## COPIED FROM SENSEAI DOCUMENTATION (same as bpfa.gallery)
def create_gallery(D, K, patch_shape, padding: int = 1):
    D_copy = D.copy()
    elements = []
    for k in range(K):
        elem = np.reshape(D_copy[:,k], (patch_shape[0], patch_shape[1]))
        elem -= np.min(elem)
        elem /= np.max(elem) + 0.001 # avoid div by 0
        elements.append(elem)
    Nx = Ny = math.ceil(math.sqrt(K))
    gallery = np.zeros(shape=(Nx * (padding+patch_shape[0]) + padding, Ny * (padding+patch_shape[1]) + padding))
    for x in range(Nx):
        for y in range(Ny):
            i = x * Nx + y
            if (i < K):
                start_x = x*(padding+patch_shape[0])+padding
                start_y = y*(padding+patch_shape[1])+padding
                gallery[start_x:start_x+patch_shape[0], start_y:start_y+patch_shape[1]] = elements[i]
    return gallery
    
# =========================
# SAVING YOUR DATA
# =========================
def save_all(dir_name, bpfa, metrics, tracker, best_recon, best_dict):
    base_dir = "save_all"
    save_dir = os.path.join(base_dir, dir_name)
    # if dir already exists, remove it
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    save_dict(save_dir, bpfa)
    save_recon(save_dir, bpfa)
    save_metrics(save_dir, bpfa, metrics)
    save_atomscores(save_dir, bpfa)
    save_parameters(save_dir, tracker)
    save_best(save_dir, best_recon, best_dict)

    # confirm save went well
    print("Everything was neatly saved in dir:", save_dir)

def save_dict(dir_name, bpfa):
    curr_bpfa_dict = bpfa.D.to_ndarray() # transform to np array first
    K = bpfa.K

    # save dict as pickle
    f_name_pkl = "dictionary_pickle_K" + str(K) + ".pkl"
    path_pkl = os.path.join(dir_name, f_name_pkl)
    
    with open(path_pkl, "wb") as f:
        pickle.dump(curr_bpfa_dict, f)

    # save dict as gallery
    f_name_jpg = "dictionary_gallery_K" + str(K) + ".jpg"
    path_jpg = os.path.join(dir_name, f_name_jpg)
    dict_gallery = bpfa.gallery(padding=1)
    plt.imshow(dict_gallery, cmap = "grey")
    plt.savefig(path_jpg)

def save_recon(dir_name, bpfa):
    f_name_recon = "reconstruction.tiff"
    path = os.path.join(dir_name, f_name_recon)
    
    reconstruction = bpfa.reconstruct(True, False, "FullW")
    reconstruction_arr = reconstruction.to_ndarray()
    im = Image.fromarray(reconstruction_arr)
    im.save(path)

def save_metrics(dir_name, bpfa, metrics):
    f_name_met = "metrics.jpg"
    path = os.path.join(dir_name, f_name_met)
    
    fig = plot_metrics(metrics, return_plot = True)
    fig.savefig(path)
    plt.close(fig)

def save_atomscores(dir_name, bpfa):
    f_name_as = "atom_scores.jpg"
    path = os.path.join(dir_name, f_name_as)

    fig = plot_atom_scores(bpfa, return_plot = True)
    fig.savefig(path)
    plt.close(fig)

def save_parameters(dir_name, tracker):
    f_name_param = "parameters.jpg"
    path = os.path.join(dir_name, f_name_param)

    fig = plot_parameters(tracker)
    fig.savefig(path)
    plt.close(fig)
    
def save_best(dir_name, best_recon, best_dict):
    if best_recon is None or best_dict is None:
        print("Residual was never calculated, so no best dictionary or reconstruction or saved.")
        return

    # get dict params
    K = best_dict.shape[1]
    patch_shape = (int(np.sqrt(best_dict.shape[0])), int(np.sqrt(best_dict.shape[0])))  # assumes square patches

    # save dict as pkl
    f_name_pkl = "best_dictionary_pickle_K" + str(K) + ".pkl"
    path_pkl = os.path.join(dir_name, f_name_pkl)
    
    with open(path_pkl, "wb") as f:
        pickle.dump(best_dict, f)

    # save dict as gallery
    f_name_jpg = "best_dictionary_gallery_K" + str(K) + ".jpg"
    path_jpg = os.path.join(dir_name, f_name_jpg)
    dict_gallery = create_gallery(best_dict, K, patch_shape)
    plt.imshow(dict_gallery, cmap = "grey")
    plt.savefig(path_jpg)

    # save the best reconstruction
    f_name_recon = "best_reconstruction.tiff"
    path = os.path.join(dir_name, f_name_recon)
    
    reconstruction_arr = best_recon.to_ndarray()
    im = Image.fromarray(reconstruction_arr)
    im.save(path)
    

# =========================
# FULL ALGORITHM
# =========================
def perform_reconstruction(bpfa, input_array, mask_array, controls, metrics = [], scale_dict=10, scale_recon=1, only_sparse = False):
    metric_vals = {name: [] for metric in metrics for name in metric.return_names()} # start with dict with empty lists
    metric_vals["residual"] = [] # hardcode for now
    steps = 0

    best_dict = None
    best_recon = None
    best_residual = np.inf
    
    while True:
        try:
            # stops if stop button is pressed
            if controls.stop_requested:
                print("Stopping gracefully...")
                break

            # saves if save button is pressed
            if controls.save_requested:
                try:
                    save_all(controls.f_name.value, bpfa, metric_vals, controls.tracker, best_recon, best_dict)
                except Exception as e:
                    print("An error occurred while saving:", e)
                
                controls.save_requested = False
        
            # update params from UI
            batch_size = controls.update_bpfa_parameters()
        
            if not controls.pause.value:
                if not only_sparse:
                    perform_batch(bpfa, input_array, mask_array, batch_size)
                else:
                    perform_batch_sparse_only(bpfa, input_array, mask_array, batch_size)
        
            # show dictionary
            gallery = bpfa.gallery(padding=1)
            imshow(gallery, "Dictionary", scale=scale_dict)
            controls.attach_dictionary_click("Dictionary")
        
            if not controls.pause.value:

                # Save all current parameters
                if steps % controls.plot_iter.value == 0:
                    controls.update_tracker()
                   
                # Atom score plotting
                if not controls.plot_scores.value: # if checkbox not clicked, try to destroy the windows
                    try:
                        cv2.destroyWindow("Atom Scores")
                    except cv2.error:
                        pass

                elif controls.plot_scores.value and steps % controls.plot_iter.value == 0:
                    plot_atom_scores(bpfa)
                
                # if not reconstructing, close window and do not calc a new residual
                if not controls.show_recon.value:
                    try:
                        if steps % controls.plot_iter.value == 0:
                            for metric in metric_vals.keys():
                                metric_vals[metric].append(metric_vals[metric][-1]) # append last metric value to keep time-consistency intact
                        
                        cv2.destroyWindow("Reconstruction")
                    except:
                        pass
                        
                # Reconstruction and showing it
                else:
                    reconstruction = bpfa.reconstruct(True, False, "FullW")
                    imshow(reconstruction, "Reconstruction", scale=scale_recon, invert=controls.invert_checkbox.value, \
                           vmin=controls.vmin_slider.value, vmax=controls.vmax_slider.value)

                    #  Metric calculation and parameter saving
                    if steps % controls.plot_iter.value == 0:
                        # separately receive res_value to use for best recon/dict
                        metric_dict, res_value = get_metrics(metrics, reconstruction, input_array, mask_array)
                        
                        # append the new metrics to the correct lists
                        for metric, value in metric_dict.items():
                            metric_vals[metric].append(value)

                        # also keep track of best recon and dict until now
                        if res_value < best_residual:
                            best_dict = bpfa.D.copy().to_ndarray()
                            best_recon = reconstruction.copy()
                            best_residual = res_value
    
                    # Metric plotting
                    if not controls.plot_metrics.value: # if checkbox not clicked, try to destroy the windows
                        try:
                            cv2.destroyWindow("Metrics")
                        except cv2.error:
                            pass
                            
                    elif controls.plot_metrics.value and steps % controls.plot_iter.value == 0:
                        plot_metrics(metric_vals)

            steps += 1
        
        except Exception as e:
            print("Error:", e)