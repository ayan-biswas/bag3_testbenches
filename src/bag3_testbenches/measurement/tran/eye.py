from typing import Optional
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from dataclasses import dataclass, field

from bag.math.interpolate import LinearInterpolator


@dataclass
class EyeResults:
    width: float
    height: float
    pos_w0: int = field(repr=False)     # time index of eye left
    pos_w1: int = field(repr=False)     # time index of eye right
    pos_h: int = field(repr=False)      # time index of eye max opening
    val_mid: float = field(repr=False)  # eye common mode
    eye_vals: np.ndarray = field(repr=False)    # collapsed waveforms for plotting eye
    time_eye: np.ndarray = field(repr=False)    # time axis for plotting eye

    def plot(self, ax: Optional[Axes] = None) -> None:
        if ax:
            ax.plot(self.time_eye, self.eye_vals.T, 'b')
            ax.hlines(self.val_mid, self.time_eye[self.pos_w0], self.time_eye[self.pos_w1],
                      label=f'width={self.width:0.3g} s', linestyle="--", colors=['r'])
            ax.vlines(self.time_eye[self.pos_h], self.val_mid - self.height / 2, self.val_mid + self.height / 2,
                      label=f'height={self.height:0.3g} V', linestyle=":", colors=['g'])
            ax.set_xlabel('time (s)')
            ax.set_ylabel('amplitude (V)')
            ax.grid()
            ax.legend(loc='upper right')
        else:
            plt.figure()
            plt.plot(self.time_eye, self.eye_vals.T, 'b')
            plt.hlines(self.val_mid, self.time_eye[self.pos_w0], self.time_eye[self.pos_w1],
                       label=f'width={self.width:0.3g} s', linestyle="--", colors=['r'])
            plt.vlines(self.time_eye[self.pos_h], self.val_mid - self.height / 2, self.val_mid + self.height / 2,
                       label=f'height={self.height:0.3g} V', linestyle=":", colors=['g'])
            plt.xlabel('time (s)')
            plt.ylabel('amplitude (V)')
            plt.grid()
            plt.legend(loc='upper right')
            plt.show()
    
    def savefig(self, fname: str, title: str) -> None:
        plt.figure()
        plt.plot(self.time_eye, self.eye_vals.T, 'b')
        plt.hlines(self.val_mid, self.time_eye[self.pos_w0], self.time_eye[self.pos_w1],
                   label=f'width={self.width:0.3g} s', linestyle="--", colors=['r'])
        plt.vlines(self.time_eye[self.pos_h], self.val_mid - self.height / 2, self.val_mid + self.height / 2,
                   label=f'height={self.height:0.3g} V', linestyle=":", colors=['g'])
        plt.xlabel('time (s)')
        plt.ylabel('amplitude (V)')
        plt.grid()
        plt.legend(loc='upper right')
        plt.title(title)
        plt.savefig(fname)


class EyeAnalysis:
    def __init__(self, t_per: float, t_delay: float, strobe: int = 1000) -> None:
        self._t_per: float = t_per
        self._t_delay: float = t_delay
        self._strobe: int = strobe
        self._eye_per: float = 2 * t_per
        self._time_eye = np.linspace(0, self._eye_per, self._strobe, endpoint=False)

    def _compute_width_height(self, eye_vals: np.ndarray, val_mid: float) -> EyeResults:
        eye_0 = eye_vals - val_mid
        # initialize left and right edge of eye
        time0 = 0
        time1 = self._strobe
        time_mid = self._eye_per / 2
        for _wave in eye_0:
            _wave_intp = InterpolatedUnivariateSpline(self._time_eye, _wave)
            for _root in _wave_intp.roots():
                if _root < time_mid:
                    time0 = max(time0, _root)
                else:
                    time1 = min(time1, _root)

        # replace all negative values with max value
        eye_pos = np.where(eye_0 > 0, eye_0, np.max(eye_0))
        # replace all positive values with min value
        eye_neg = np.where(eye_0 < 0, eye_0, np.min(eye_0))

        eye_heights = np.min(eye_pos, 0) - np.max(eye_neg, 0)

        # find positions of time0 and time1 for plotting
        pos0 = np.where(self._time_eye > time0)[0][0]
        pos1 = np.where(self._time_eye < time1)[0][-1]

        eye_h = np.max(eye_heights[pos0:pos1])
        pos_h = int(np.argmax(eye_heights[pos0:pos1]))
        pos_h += pos0
        eye_w = time1 - time0
        return EyeResults(eye_w, eye_h, pos0, pos1, pos_h, val_mid, eye_vals, self._time_eye)

    def analyze_eye(self, time: np.ndarray, sig: np.ndarray) -> EyeResults:
        """
        Parameters
        ----------
        time: np.ndarray
            Time axis
        sig: np.ndarray
            Signal axis

        Returns
        -------
        ans: EyeResults
            the EyeResults object which has attributes 'width' and 'height', and can be plotted using plot()
        """
        # linearly interpolate the signal
        sig_li = LinearInterpolator([time], sig, [self._t_per / self._strobe])

        # starting from t_delay, find the first transition
        val_mid = (min(sig) + max(sig)) / 2
        _t0 = self._t_delay
        while True:
            _t1 = _t0 + self._t_per
            if (sig_li(np.array([_t0])) - val_mid) * (sig_li(np.array([_t1])) - val_mid) < 0:
                # this is the first transition
                break
            _t0 = _t1

        # Find exact zero crossing between _t0 and _t1.
        # This is the extra delay of the output beyond t_delay of the input.
        t_delay1 = bin_search(sig_li - val_mid, _t0, _t0 + self._t_per, self._t_per / self._strobe) - _t0
        # To center the eye, the first transition should be quarter period shifted.
        t_delay2 = t_delay1 - self._eye_per / 4

        # fold the signal to form centered eye
        t_sim = time[-1]
        num_traces = int((t_sim - self._t_delay - t_delay2) / self._eye_per)
        t_fin = self._t_delay + t_delay2 + self._eye_per * num_traces
        time_tot = np.linspace(self._t_delay + t_delay2, t_fin, self._strobe * num_traces, endpoint=False)
        sig_vals = sig_li(time_tot)
        eye_vals = np.empty((num_traces, self._strobe), dtype=float)
        for tridx in range(num_traces):
            eye_vals[tridx] = sig_vals[tridx * self._strobe:(tridx + 1) * self._strobe]

        # compute eye width and height
        return self._compute_width_height(eye_vals, val_mid)


def bin_search(data: LinearInterpolator, t_i: float, t_f: float, tol: float) -> float:
    t_mid = (t_i + t_f) / 2
    if t_f - t_i <= tol:
        return t_mid
    if data(np.array([t_i])) * data(np.array([t_mid])) < 0:
        return bin_search(data, t_i, t_mid, tol)
    return bin_search(data, t_mid, t_f, tol)
