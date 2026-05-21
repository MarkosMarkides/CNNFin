import os

import numpy as np
from PIL import Image
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


CUSTOM_CMAPS = {
    "blue_black_red": LinearSegmentedColormap.from_list(
        "blue_black_red",
        ["#1f5eff", "#101827", "#ff334e"],
    ),
    "original_paper_divergence": LinearSegmentedColormap.from_list(
        "original_paper_divergence",
        ["#0018ff", "#f4f4f4", "#ff1200"],
    )
}


class ImageGenerator:
    def __init__(self, window_size: int = 30, figsize: tuple[int, int] = (8, 6), target_symbol: str = "BTCUSDT"):
        self.window_size = window_size
        self.figsize = figsize
        self.target_symbol = target_symbol
        self._candle_cache = {}
        self._alt_cols_cache = {}

    def _get_window(self, df, start_idx: int):
        if not isinstance(start_idx, (int, np.integer)):
            raise TypeError("start_idx must be an integer positional index.")
        return df.iloc[start_idx : start_idx + self.window_size]

    def _safe_pct_change(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size < 2:
            return np.zeros((0,), dtype=np.float32)
        prev = arr[:-1]
        curr = arr[1:]
        denom = np.where(np.abs(prev) < 1e-8, 1e-8, prev)
        out = (curr - prev) / denom
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _matrix_to_rgb(self, mat: np.ndarray, cmap: str, vmin=None, vmax=None) -> np.ndarray:
        m = np.asarray(mat, dtype=np.float32)
        if m.ndim == 1:
            m = m[None, :]
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

        if vmin is None:
            vmin = float(np.min(m)) if m.size else 0.0
        if vmax is None:
            vmax = float(np.max(m)) if m.size else 1.0
        if vmax - vmin < 1e-8:
            norm = np.zeros_like(m, dtype=np.float32)
        else:
            norm = (m - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0.0, 1.0)

        color_map = CUSTOM_CMAPS[cmap] if cmap in CUSTOM_CMAPS else colormaps[cmap]
        rgba = color_map(norm)
        return (rgba[..., :3] * 255).astype(np.uint8)

    def _resize_rgb(self, rgb: np.ndarray, size_each: int, nearest: bool) -> Image.Image:
        img = Image.fromarray(rgb, mode="RGB")
        if img.size != (size_each, size_each):
            resample = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
            img = img.resize((size_each, size_each), resample=resample)
        return img

    def _get_alt_cols(self, columns, btc_close_col: str, alt_suffix: str):
        key = (tuple(columns), btc_close_col, alt_suffix, self.target_symbol)
        cached = self._alt_cols_cache.get(key)
        if cached is not None:
            return cached
        target_alias = f"{self.target_symbol}{alt_suffix}"
        cols = [c for c in columns if c.endswith(alt_suffix) and c not in {btc_close_col, target_alias}]
        self._alt_cols_cache[key] = cols
        return cols

    def _generate_altcoin_divergence_from_window(
        self,
        window,
        *,
        size_each: int,
        btc_close_col: str,
        alt_suffix: str,
        cmap: str,
    ) -> Image.Image:
        btc_close = window[btc_close_col].to_numpy(dtype=np.float32, copy=False)
        btc_ret = self._safe_pct_change(btc_close)

        alt_cols = self._get_alt_cols(window.columns, btc_close_col, alt_suffix)
        if not alt_cols:
            raise ValueError(
                f"No altcoin columns found with suffix '{alt_suffix}'. "
                "Expected columns like 'ETH_Close', 'SOL_Close', etc."
            )

        alt_mat = window[alt_cols].to_numpy(dtype=np.float32, copy=False)
        prev = alt_mat[:-1, :]
        curr = alt_mat[1:, :]
        denom = np.where(np.abs(prev) < 1e-8, 1e-8, prev)
        alt_ret = np.nan_to_num((curr - prev) / denom, nan=0.0, posinf=0.0, neginf=0.0)

        diffs = alt_ret.T - btc_ret[None, :]
        vmax = float(np.max(np.abs(diffs))) if diffs.size else 1.0
        vmax = max(vmax, 1e-6)
        rgb = self._matrix_to_rgb(diffs, cmap=cmap, vmin=-vmax, vmax=vmax)
        return self._resize_rgb(rgb, size_each=size_each, nearest=True)

    def _get_candle_renderer(self, size_each: int, dpi: int):
        key = (size_each, dpi)
        cached = self._candle_cache.get(key)
        if cached is not None:
            return cached

        fig = Figure(figsize=(size_each / dpi, size_each / dpi), dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        cached = (fig, canvas, ax)
        self._candle_cache[key] = cached
        return cached

    def _generate_candlestick_from_window(
        self,
        window,
        *,
        size_each: int,
        dpi: int,
        col_open: str,
        col_high: str,
        col_low: str,
        col_close: str,
    ) -> Image.Image:
        o = window[col_open].to_numpy(dtype=np.float32, copy=False)
        h = window[col_high].to_numpy(dtype=np.float32, copy=False)
        l = window[col_low].to_numpy(dtype=np.float32, copy=False)
        c = window[col_close].to_numpy(dtype=np.float32, copy=False)

        n = len(window)
        fig, canvas, ax = self._get_candle_renderer(size_each=size_each, dpi=dpi)
        ax.clear()
        fig.patch.set_facecolor("#2c303c")
        ax.set_facecolor("#2c303c")

        ymin = float(np.min(l)) if l.size else 0.0
        ymax = float(np.max(h)) if h.size else 1.0
        yrange = max(ymax - ymin, 1e-6)
        pad = 0.05 * yrange

        ax.set_xlim(-0.9, max(n - 0.1, 0.9))
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.axis("off")

        candle_width = 0.58
        min_body = max(yrange * 0.002, 1e-6)

        for i in range(n):
            up = c[i] >= o[i]
            line_col = "#2EC886" if up else "#FF3A4C"
            fill_col = "#24A06B" if up else "#CC2E3C"

            ax.vlines(i, l[i], h[i], color=line_col, linewidth=1.0)
            body_low = min(o[i], c[i])
            body_h = max(abs(c[i] - o[i]), min_body)
            rect = Rectangle(
                (i - candle_width / 2.0, body_low),
                candle_width,
                body_h,
                facecolor=fill_col,
                edgecolor=line_col,
                linewidth=1.0,
            )
            ax.add_patch(rect)

        canvas.draw()
        rgb = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
        return Image.fromarray(rgb, mode="RGB")

    def _generate_gaf_from_window(self, window, *, size_each: int) -> Image.Image:
        series = window["Close"].to_numpy(dtype=np.float32, copy=False)

        if series.size == 0:
            norm = np.zeros((self.window_size,), dtype=np.float32)
        else:
            smin = float(np.min(series))
            smax = float(np.max(series))
            if smax - smin < 1e-8:
                norm = np.zeros_like(series, dtype=np.float32)
            else:
                norm = 2.0 * (series - smin) / (smax - smin) - 1.0

        norm = np.clip(norm, -1.0, 1.0)
        if len(norm) < self.window_size:
            pad_len = self.window_size - len(norm)
            pad_val = norm[0] if len(norm) else 0.0
            norm = np.pad(norm, (pad_len, 0), mode="constant", constant_values=pad_val)
        else:
            norm = norm[-self.window_size :]

        phi = np.arccos(norm)
        gaf = np.cos(phi[:, None] + phi[None, :])

        rgb = self._matrix_to_rgb(gaf, cmap="viridis", vmin=-1.0, vmax=1.0)
        return self._resize_rgb(rgb, size_each=size_each, nearest=True)

    def _generate_indicator_heatmap_from_window(self, window, *, indicators: list[str], size_each: int) -> Image.Image:
        mat = window[indicators].to_numpy(dtype=np.float32, copy=True)

        for j, col in enumerate(indicators):
            col_data = mat[:, j]
            if col.startswith("CCI"):
                col_data = np.sign(col_data) * np.log1p(np.abs(col_data))
            mn = float(np.min(col_data))
            mx = float(np.max(col_data))
            mat[:, j] = (col_data - mn) / (mx - mn + 1e-8)

        arr = mat.T
        rgb = self._matrix_to_rgb(arr, cmap="viridis", vmin=0.0, vmax=1.0)
        return self._resize_rgb(rgb, size_each=size_each, nearest=True)

    def generate_altcoin_divergence_image(
        self,
        df,
        start_idx: int,
        btc_close_col: str = "Close",
        alt_suffix: str = "_Close",
        cmap: str = "original_paper_divergence",
        size_each: int = 256,
    ):
        """Top-left panel: altcoin-vs-BTC return divergence heatmap."""
        window = self._get_window(df, start_idx)
        return self._generate_altcoin_divergence_from_window(
            window,
            size_each=size_each,
            btc_close_col=btc_close_col,
            alt_suffix=alt_suffix,
            cmap=cmap,
        )

    def generate_candlestick_image(
        self,
        df,
        start_idx: int,
        col_open: str = "Open",
        col_high: str = "High",
        col_low: str = "Low",
        col_close: str = "Close",
        size_each: int = 256,
        dpi: int = 100,
    ):
        """Top-right panel: candlestick chart."""
        window = self._get_window(df, start_idx)
        return self._generate_candlestick_from_window(
            window,
            size_each=size_each,
            dpi=dpi,
            col_open=col_open,
            col_high=col_high,
            col_low=col_low,
            col_close=col_close,
        )

    def generate_gaf_image(self, df, start_idx: int, size_each: int = 256):
        """Bottom-left panel: Gramian Angular Field (GAF)."""
        window = self._get_window(df, start_idx)
        return self._generate_gaf_from_window(window, size_each=size_each)

    def generate_indicator_heatmap_image(self, df, start_idx: int, indicators: list[str], size_each: int = 256):
        """Bottom-right panel: technical indicator heatmap."""
        window = self._get_window(df, start_idx)
        return self._generate_indicator_heatmap_from_window(window, indicators=indicators, size_each=size_each)

    def generate_four_panel_image(
        self,
        df,
        *,
        start_idx: int,
        indicators: list[str],
        size_each: int = 256,
        dpi: int = 100,
        btc_close_col: str = "Close",
        alt_suffix: str = "_Close",
    ):
        """Create one merged image with panel order:
        top-left divergence, top-right candlestick, bottom-left GAF, bottom-right indicator heatmap.
        """
        window = self._get_window(df, start_idx)

        img_div = self._generate_altcoin_divergence_from_window(
            window,
            size_each=size_each,
            btc_close_col=btc_close_col,
            alt_suffix=alt_suffix,
            cmap="original_paper_divergence",
        )
        img_candle = self._generate_candlestick_from_window(
            window,
            size_each=size_each,
            dpi=dpi,
            col_open="Open",
            col_high="High",
            col_low="Low",
            col_close="Close",
        )
        img_gaf = self._generate_gaf_from_window(window, size_each=size_each)
        img_heat = self._generate_indicator_heatmap_from_window(window, indicators=indicators, size_each=size_each)

        quad = Image.new("RGB", (2 * size_each, 2 * size_each), "#101827")
        quad.paste(img_div, (0, 0))
        quad.paste(img_candle, (size_each, 0))
        quad.paste(img_gaf, (0, size_each))
        quad.paste(img_heat, (size_each, size_each))
        return quad

    def save_four_panel_image(
        self,
        df,
        *,
        start_idx: int,
        indicators: list[str],
        filepath: str,
        size_each: int = 256,
        dpi: int = 100,
        btc_close_col: str = "Close",
        alt_suffix: str = "_Close",
    ):
        quad = self.generate_four_panel_image(
            df,
            start_idx=start_idx,
            indicators=indicators,
            size_each=size_each,
            dpi=dpi,
            btc_close_col=btc_close_col,
            alt_suffix=alt_suffix,
        )

        out_dir = os.path.dirname(filepath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        quad.save(filepath, format="PNG")
        return filepath

    # Backward-compatible alias
    def save_quadrant_image(
        self,
        df,
        *,
        start,
        indicators,
        filepath,
        size_each=256,
        dpi=100,
    ):
        return self.save_four_panel_image(
            df,
            start_idx=start,
            indicators=indicators,
            filepath=filepath,
            size_each=size_each,
            dpi=dpi,
        )
