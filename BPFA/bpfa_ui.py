import numpy as np
import arrayfire as af
import cv2
import ipywidgets as widgets
from IPython.display import display


class BPFAControls:
    def __init__(self, bpfa, batch_size):
        self.bpfa = bpfa
        self.batch_size = batch_size
        self.stop_requested = False
        self.save_requested = False

        self._create_controls()
        self._bind_events()

        # track the parameters for plotting
        self.tracker = {
            "lr" : [],
            "batch_size" : [],
            "K" : [],
            "limit" : [],
            "ones_element" : [],
            "avg_element" :[]
        }

    # =========================
    # UI CREATION
    # =========================
    def _create_controls(self):
        self.lr = widgets.FloatSlider(value=self.bpfa.lr, min=0.0, max=1.0, step=0.01, description='LR:')
        self.bs = widgets.IntSlider(value=self.batch_size, min=0, max=50000, step=1000, description='BS:')
        self.K = widgets.IntSlider(value=self.bpfa.K, min=0, max=128, step=1, description='K:')
        self.limit = widgets.IntSlider(value=self.bpfa.limit, min=0, max=50, step=1, description='Limit:')

        self.ones = widgets.Checkbox(value=self.bpfa.ones_element, description='Ones atom')
        self.avg = widgets.Checkbox(value=self.bpfa.avg_element, description='Avg atom')
        self.reset_atom = widgets.Checkbox(value=False, description='Reset atom on click')

        self.plot_metrics = widgets.Checkbox(value=False, description='Plot metrics')
        self.plot_scores = widgets.Checkbox(value=False, description='Plot atom scores')
        self.plot_iter = widgets.IntSlider(value=10, min=1, max=100, step=10, description='Save every')

        self.show_recon = widgets.Checkbox(value=True, description='Show Reconstruction')
        
        self.invert_checkbox = widgets.Checkbox(value=False, description='Invert Greyscale')
        self.vmin_slider = widgets.FloatSlider(value=0, min=0.0, max=1.0, step=0.01, description='VMIN:',continuous_update=True)        
        self.vmax_slider = widgets.FloatSlider(value=1.0, min=0.0, max=1.0, step=0.01, description='VMAX:', continuous_update=True)
        
        self.f_name = widgets.Text(value='bpfa_save_all', placeholder='...', description='File name:', disabled=False)
        self.save_all = widgets.Button(description="Save Everything")

        self.pause = widgets.Checkbox(value=False, description='Pause')
        self.stop = widgets.Button(description="Stop")

        display(
            self.lr, self.bs, self.K, self.limit,
            self.ones, self.avg, self.reset_atom,
            self.plot_metrics, self.plot_scores, self.plot_iter,
            self.show_recon,
            self.invert_checkbox, self.vmin_slider, self.vmax_slider,
            self.f_name, self.save_all,
            self.pause, self.stop
        )

    # =========================
    # UPDATE THE PARAMETER TRACKER
    # =========================
    def update_tracker(self):
        self.tracker["lr"].append(self.lr.value)
        self.tracker["batch_size"].append(self.bs.value)
        self.tracker["K"].append(self.K.value)
        self.tracker["limit"].append(self.limit.value)
        self.tracker["ones_element"].append(self.ones.value)
        self.tracker["avg_element"].append(self.avg.value)

    # =========================
    # EVENT BINDING
    # =========================
    def _bind_events(self):
        self.stop.on_click(self._on_stop)
        self.save_all.on_click(self._on_save)

    def _on_stop(self, _):
        self.stop_requested = True

    def _on_save(self, _):
        self.save_requested = True

    # =========================
    # APPLY UI -> BPFA
    # =========================
    def update_bpfa_parameters(self):
        bpfa = self.bpfa

        bpfa.lr = self.lr.value
        bpfa.limit = self.limit.value
        bpfa.ones_element = self.ones.value
        bpfa.avg_element = self.avg.value

        new_K = self.K.value

        if new_K < bpfa.K:
            bpfa.D = bpfa.D[:, :new_K]

        elif new_K > bpfa.K:
            new_atoms = bpfa.generate_dictionary(
                bpfa.patch_shape,
                (new_K - bpfa.K),
                "Random"
            ).to_ndarray()

            D_np = bpfa.D.to_ndarray()
            D_np = np.concatenate((D_np, new_atoms), axis=1)
            bpfa.D = af.interop.from_ndarray(D_np)

        bpfa.K = new_K

        return self.bs.value  # return batch size

    # =========================
    # CLICK HANDLER
    # =========================
    def attach_dictionary_click(self, window_name="Dictionary", scale=10, padding=1):
        cv2.setMouseCallback(window_name, self._click_event)
        self._click_scale = scale
        self._click_padding = padding

    def _click_event(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        bpfa = self.bpfa

        ph, pw, _, _ = bpfa.patch_shape
        grid_cols = int(np.ceil(np.sqrt(bpfa.K)))

        # compute index
        col = x // ((pw + self._click_padding) * self._click_scale)
        row = y // ((ph + self._click_padding) * self._click_scale)
        idx = row * grid_cols + col

        if idx >= bpfa.K:
            print("Click outside valid atoms")
            return

        D_np = bpfa.D.to_ndarray()

        if not self.reset_atom.value:
            print(f"Removing atom {idx}")
            D_np = np.delete(D_np, idx, axis=1)
            bpfa.K -= 1
            self.K.value -= 1
        else:
            print(f"Resetting atom {idx}")
            new_atom = bpfa.generate_dictionary(bpfa.patch_shape, 1, "Random")
            D_np[:, idx] = new_atom.to_ndarray().reshape(-1)
        
        bpfa.D = af.interop.from_ndarray(D_np)