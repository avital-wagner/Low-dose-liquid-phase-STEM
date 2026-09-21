import numpy as np
import torch
import torch.nn.functional as F
import piq

from pathlib import Path
from scipy.ndimage import gaussian_filter, median_filter

## own files
from PIQ_utils.niqe import niqe as niqe_fn          # expects gray uint8 (or BGR uint8; converts internally)
from PIQ_utils.piqe import piqe as piqe_fn          # expects gray uint8 (or BGR uint8; converts internally)
from PIQ_utils.dom import DOM

# MAIN METRIC CLASS
class Metric():

    def __init__(self, **kwargs):
        self.need_ref_img = False # standard for IQA

    # abstract func
    def name(self):
        return "Non-specific metric."

    # return dict of params used
    def get_params(self):
        return {}

    # returns list of all metrics it returns, generally only self.name(), otherwise overwrite
    def return_names(self):
        return [self.name()]

    # abstract func (EXPECTS AN RGB IMG - e.g. H x W x 3 - all other conversion done internally)
    def compute(self, img, **kwargs):
        return 0

    def rgb_to_luma(self, rgb_float):
        """Rec.601-ish luma. Input HxWx3 in [0,1]. Output HxW in [0,1]."""
        r = rgb_float[:, :, 0]
        g = rgb_float[:, :, 1]
        b = rgb_float[:, :, 2]
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    
    def to_uint8_gray(self, luma_float):
        """Convert luma float [0,1] -> uint8 grayscale for typical NIQE implementations."""
        x = np.clip(luma_float, 0.0, 1.0)
        return (x * 255.0 + 0.5).astype(np.uint8)

# UNIQUE METRICS (HERE YOU CAN ADD YOUR OWN, FOLLOW A SIMILAR STRUCTURE

"""
Natural Image Quality Evaluator
"""
class NiqeMetric(Metric):
    
    def name(self):
        return "NIQE"

    def compute(self, img):
        luma = self.rgb_to_luma(img) 
        value = float(niqe_fn(self.to_uint8_gray(luma))) #convert to grayscale first
        return {self.name() : value}


"""
Blind/Referenceless Image Spatial Quality Evaluator
"""
class BrisqueMetric(Metric):
    
    def name(self):
        return "BRISQUE"

    def compute(self, img):
        try:
            with torch.no_grad():
                x_rgb = torch.from_numpy(np.clip(img, 0.0, 1.0)).permute(2, 0, 1).unsqueeze(0)
                value = float(piq.brisque(x_rgb, data_range=1.0, reduction="mean").item())
        except Exception as e:
            print("Issue with BRISQUE:", e)
            value = float("nan")
        
        return {self.name() : value}


"""
Perception based Imaged Quality Estimator
"""
class PiqeMetric(Metric):
    
    def name(self):
        return "PIQE"

    def compute(self, img):
        luma = self.rgb_to_luma(img) 
        value = float(piqe_fn(self.to_uint8_gray(luma))) #convert to grayscale first
        return {self.name() : value}


"""
Total Variation
"""
class TotalVarMetric(Metric):

    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device
    
    def name(self):
        return "Total Variation"

    def get_params(self):
        return {"device" : self.device}

    def total_variation_l1(self, img_luma: torch.Tensor) -> torch.Tensor:
        """
        img_luma: (1,1,H,W) float in [0,1]
        Returns scalar TV (mean L1 of finite differences).
        """
        dy = torch.abs(img_luma[:, :, 1:, :] - img_luma[:, :, :-1, :])
        dx = torch.abs(img_luma[:, :, :, 1:] - img_luma[:, :, :, :-1])
        return dx.mean() + dy.mean()
    
    def compute(self, img):
        luma = self.rgb_to_luma(img) 
        luma_t = torch.from_numpy(luma).unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=torch.float32)
    
        with torch.no_grad():
            value = float(self.total_variation_l1(luma_t).item())
            
        return {self.name() : value}


"""
Differences of Medians (edge sharpness)
"""
class DomMetric(Metric):
    
    def name(self):
        return "DOM"

    def compute(self, img):
        luma = self.rgb_to_luma(img) 
        gray_u8 = self.to_uint8_gray(luma)
        value = float(DOM().get_sharpness(gray_u8))
        return {self.name() : value}


"""
Roughness Map (generally P90)
"""
class RoughnessMetric(Metric):

    def __init__(self, roughness_sigma, roughness_perc, **kwargs):
        super().__init__(**kwargs) 
        self.roughness_perc = roughness_perc
        self.roughness_sigma = roughness_sigma
    
    def name(self):
        return "Roughness " + str(self.roughness_perc)

    def get_params(self):
        return {"percentile" : self.roughness_perc,
                "sigma" : self.roughness_sigma}

    def background_roughness_metrics(self, img):
        img = np.asarray(img, dtype=np.float32)
        gray = np.clip(img, 0.0, 1.0)
    
        smooth = gaussian_filter(gray, sigma=self.roughness_sigma)
    
        hf = gray - smooth
        ahf = np.abs(hf)

        return float(np.percentile(ahf, self.roughness_perc))

    def compute(self, img):
        luma = self.rgb_to_luma(img) 
        value = self.background_roughness_metrics(luma)
        return {self.name() : value}


"""
Salt & Pepper Noise (returns fraction)
"""
class SaltPepperMetric(Metric):

    def __init__(self, SP_window, SP_threshold, **kwargs):
        super().__init__(**kwargs) 
        self.SP_window = SP_window
        self.SP_threshold = SP_threshold
    
    def name(self):
        return "Salt & Pepper Noise"

    def get_params(self):
        return {"window" : self.SP_window,
                "threshold" : self.SP_threshold}

    def salt_pepper_metrics(self, img, window = 3, extreme_threshold=0.20):
        img = np.asarray(img, dtype=np.float32)
        gray = np.clip(img, 0.0, 1.0)
    
        med = median_filter(gray, size=window, mode="reflect")
        residual = gray - med
    
        # Salt/pepper pixels are very different from their local median
        sp_mask = np.abs(residual) > extreme_threshold
        sp_fraction = float(np.mean(sp_mask))
    
        return sp_fraction

    def compute(self, img):
        luma = self.rgb_to_luma(img) 
        value = self.salt_pepper_metrics(luma, window=self.SP_window, extreme_threshold=self.SP_threshold)
        return {self.name() : value}


"""
Resolution from Fast Fourier Transform
"""
class FFTMetric(Metric):

    def __init__(self, nbins, tail_frac, exclude_zero_fft_pixels, **kwargs):
        super().__init__() 
        self.nbins = nbins
        self.tail_frac = tail_frac
        self.exclude_zero_fft_pixels = exclude_zero_fft_pixels
    
    def name(self):
        return "Resolution from FFT"

    # FFT returns two metric values, so overwrite func
    def return_names(self):
        return ["res_ampl_px", "res_pwr_px"]

    def get_params(self):
        return {"nbins" : self.nbins,
                "tail_frac" : self.tail_frac,
                "exclude_zero_fft_pixels" : self.exclude_zero_fft_pixels}

    def radial_profile_rfft(self, spectrum, valid_fft_mask, nbins):
        """
        spectrum: H x (W//2+1) float (e.g., |F| or |F|^2)
        valid_fft_mask: same shape bool; False = masked/excluded pixels
        nbins: number of radial bins
    
        Returns:
          bin_centers (cycles/pixel),
          bin_means,
          bin_counts (# contributing pixels)
        """
        H, Wr = spectrum.shape
    
        fy = np.fft.fftfreq(H)              # cycles/pixel
        fx = np.fft.rfftfreq((Wr - 1) * 2)  # recover original W from rfft width
    
        FX, FY = np.meshgrid(fx, fy)
        R = np.sqrt(FX**2 + FY**2)
    
        edges = np.linspace(0.0, float(R.max()), nbins + 1, dtype=np.float32)
    
        r_flat = R.ravel()
        s_flat = spectrum.ravel()
        m_flat = valid_fft_mask.ravel()
    
        bin_idx = np.searchsorted(edges, r_flat, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, nbins - 1)
    
        sums = np.zeros(nbins, dtype=np.float64)
        counts = np.zeros(nbins, dtype=np.int64)
    
        valid = m_flat
        np.add.at(sums, bin_idx[valid], s_flat[valid])
        np.add.at(counts, bin_idx[valid], 1)
    
        means = np.full(nbins, np.nan, dtype=np.float64)
        nz = counts > 0
        means[nz] = sums[nz] / counts[nz]
    
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers.astype(np.float32), means.astype(np.float64), counts
    
    def _tail_noise(self, profile, counts, start):
        """Mean value of the tail of a radial profile."""
        tail = slice(start, None)
    
        valid = (
            (counts[tail] > 0)
            & np.isfinite(profile[tail])
        )
    
        if not np.any(valid):
            return np.nan
    
        return float(profile[tail][valid].mean())


    def _resolution_from_profile(self, profile, counts, radii, noise, sigma=3.0):
        """Estimate cutoff frequency and spatial resolution."""
        if not np.isfinite(noise):
            return np.nan, np.nan
    
        threshold = sigma * noise
    
        valid = (counts > 0) & np.isfinite(profile)
        above = np.flatnonzero(valid & (profile >= threshold))
    
        if above.size == 0:
            return np.nan, np.nan
    
        cutoff = float(radii[above[-1]])
    
        resolution = np.inf if cutoff <= 0 else 1.0 / cutoff
    
        return cutoff, float(resolution)
    
    
    def fft_metrics_and_resolution(self, img_luma, nbins=500, tail_frac=0.30, exclude_zero_fft_pixels=True):
        """
        Estimate spatial resolution from the FFT radial profile.
    
        If exclude_zero_fft_pixels=True, FFT pixels with mag==0 are excluded from stats.
        """
    
        x = np.asarray(img_luma, dtype=np.float32)
    
        F = np.fft.rfft2(x)
    
        ampl = np.abs(F).astype(np.float64)
        pwr = ampl * ampl
    
        valid_fft = ampl > 0 if exclude_zero_fft_pixels else np.ones_like(ampl, dtype=bool)
        
        r, prof_ampl, cnt_ampl = self.radial_profile_rfft(ampl, valid_fft, nbins)
        _, prof_pwr, cnt_pwr = self.radial_profile_rfft(pwr, valid_fft, nbins)
    
        tail_start = min(max(int((1.0 - tail_frac) * nbins), 0), nbins - 1)
    
        noise_ampl = self._tail_noise(prof_ampl, cnt_ampl, tail_start)
        noise_pwr = self._tail_noise(prof_pwr, cnt_pwr, tail_start)
    
        _, res_ampl_px = self._resolution_from_profile(prof_ampl, cnt_ampl, r, noise_ampl)
        _, res_pwr_px = self._resolution_from_profile(prof_pwr, cnt_pwr, r, noise_pwr)
    
        return res_ampl_px, res_pwr_px

    def compute(self, img):
        luma = self.rgb_to_luma(img) 
        res_ampl_px, res_pwr_px = self.fft_metrics_and_resolution(luma, self.nbins, self.tail_frac, self.exclude_zero_fft_pixels)
        return {"res_ampl_px" : res_ampl_px, "res_pwr_px" : res_pwr_px}


"""
Structural Similarity Index Measure with (masked) Reference Image
"""
class SSIMMetric(Metric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.need_ref_img = True # override the main Metric() class
    
    def set_ref(self, ref_img):
        self.ref_img = ref_img
    
    def name(self):
        return "SSIM"
    
    def masked_ssim_from_images(self, rec_img, threshold=0, data_range=None, window_size=11, sigma=1.5, k1=0.01, k2=0.03, device="cpu", eps=1e-8):
        """
        Compute SSIM between a sub-sampled image and a reconstruction, using ONLY
        the observed (non-black) pixels in the sub-sampled image.
    
        Parameters
        ----------
        sub_img, rec_img
            Images as numpy arrays. Shapes: HxW or HxWx3/4. Must match.
            Missing pixels in sub_img should be black (0).
        threshold
            Pixels > threshold are treated as observed. Use 0 if missing pixels are exactly 0.
            If you pass uint8/uint16 images, threshold is in that dtype units.
            If you pass float images in [0,1], threshold is in [0,1].
        data_range
            SSIM data range. If None, inferred:
              - integer images: max dtype value
              - float images: 1.0 if values look like [0,1]
        window_size, sigma, k1, k2
            Standard SSIM parameters (Gaussian window).
        device
            "cpu" or "cuda"
        Returns
        -------
        float
            Masked SSIM score (NaN if no observed pixels).
        """
        sub_img = self.ref_img
        
        if sub_img.shape != rec_img.shape:
            raise ValueError(f"Shape mismatch: {sub_img.shape} vs {rec_img.shape}")
    
        def to_gray_float01(x: np.ndarray) -> np.ndarray:
            # HxW
            if x.ndim == 2:
                g = x.astype(np.float32)
                if np.issubdtype(x.dtype, np.integer):
                    g /= float(np.iinfo(x.dtype).max)
                return np.clip(g, 0.0, 1.0)
    
            # HxWxC
            if x.ndim == 3 and x.shape[2] in (3, 4):
                rgb = x[..., :3].astype(np.float32)
                if np.issubdtype(x.dtype, np.integer):
                    rgb /= float(np.iinfo(x.dtype).max)
                g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
                return np.clip(g, 0.0, 1.0)
    
            raise ValueError("Expected HxW or HxWx3/4 image.")
    
        # Convert to grayscale float in [0,1] for stable SSIM
        sub = to_gray_float01(sub_img)
        rec = to_gray_float01(rec_img)
    
        # Determine data_range for SSIM formula
        if data_range is None:
            # We are working in [0,1] floats here
            data_range = 1.0
    
        # Build mask from sub-sampled image (observed pixels)
        # Convert threshold to [0,1] if user gave integer threshold for integer images.
        thr = float(threshold)
        if np.issubdtype(sub_img.dtype, np.integer):
            thr = thr / float(np.iinfo(sub_img.dtype).max)
        mask_np = (sub > thr).astype(np.float32)
    
        if mask_np.sum() == 0:
            return float("nan")
    
        # Torch tensors: (1,1,H,W)
        sub_t = torch.from_numpy(sub).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
        rec_t = torch.from_numpy(rec).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    
        # Gaussian window
        coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # 1x1xKxK
    
        pad = window_size // 2
    
        # Normalize mask coverage per window
        wsum = F.conv2d(mask_t, window, padding=pad)
        wsum = torch.clamp(wsum, min=eps)
    
        def wmean(x):
            return F.conv2d(x * mask_t, window, padding=pad) / wsum
    
        mu_x = wmean(sub_t)
        mu_y = wmean(rec_t)
    
        sigma_x = wmean(sub_t * sub_t) - mu_x * mu_x
        sigma_y = wmean(rec_t * rec_t) - mu_y * mu_y
        sigma_xy = wmean(sub_t * rec_t) - mu_x * mu_y
    
        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2
    
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        )
    
        # Weighted mean SSIM over valid regions only
        ssim_value = (ssim_map * wsum).sum() / wsum.sum()
        return float(ssim_value.item())

    def compute(self, img):
        value = self.masked_ssim_from_images(img)
        return {self.name() : value}