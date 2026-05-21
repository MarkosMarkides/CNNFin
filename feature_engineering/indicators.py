import pandas as pd
import numpy as np

class IndicatorFactory:
    """Collection of lightweight technical-indicator helpers (pandas-native)."""

    # ------------------------------------------------------------------ #
    # 1. Volatility-style indicators
    # ------------------------------------------------------------------ #
    @staticmethod
    def returns(df, close="Close", windows=[1, 5, 10]):
        df = df.copy()
        df["return"] = df[close].pct_change()
        for w in windows:
            df[f"return_{w}"] = df["return"].rolling(w).mean()
        return df

    @staticmethod
    def BollingerBands(
        df,
        close_col="Close",
        high_col="High",
        low_col="Low",
        periods=[20, 50, 100],
        s=2,
    ):
        df = df.copy()
        typical = (df[close_col] + df[high_col] + df[low_col]) / 3
        for p in periods:
            ma = typical.rolling(p).mean()
            sd = typical.rolling(p).std()
            df[f"BB_MA{p}"] = ma
            df[f"BB_UP{p}"] = ma + s * sd
            df[f"BB_LW{p}"] = ma - s * sd
        return df

    @staticmethod
    def ATR(df, high_col="High", low_col="Low", close_col="Close", windows=[7, 14, 28, 56]):
        df = df.copy()
        prev_close = df[close_col].shift()
        tr = pd.concat(
            [
                df[high_col] - df[low_col],
                (df[high_col] - prev_close).abs(),
                (prev_close - df[low_col]).abs(),
            ],
            axis=1,
        ).max(axis=1)
        for w in windows:
            df[f"ATR_{w}"] = tr.rolling(w).mean()
        return df

    @staticmethod
    def Volatility(df, close_col="Close", window=12, annual_factor=365):
        """Rolling σ of pct-change, annualised."""
        df = df.copy()
        ret = df[close_col].pct_change()
        df[f"Volatility_{window}"] = ret.rolling(window).std() * np.sqrt(annual_factor)
        return df

    # ------------------------------------------------------------------ #
    # 2. Volume-based indicators
    # ------------------------------------------------------------------ #
    @staticmethod
    def vwap(
        df,
        high_col="High",
        low_col="Low",
        close_col="Close",
        vol_col="Volume",
        windows=[14, 50],
    ):
        df = df.copy()
        df["Typical"] = (df[high_col] + df[low_col] + df[close_col]) / 3
        df["PV"] = df["Typical"] * df[vol_col]
        for w in windows:
            df[f"VWAP_{w}"] = df["PV"].rolling(w).sum() / df[vol_col].rolling(w).sum()
        return df

    @staticmethod
    def CMFI(
        df,
        high_col="High",
        low_col="Low",
        close_col="Close",
        vol_col="Volume",
        window=20,
    ):
        """
        Chaikin Money Flow Index.
        """
        df = df.copy()
        mfm = (
            ((df[close_col] - df[low_col]) - (df[high_col] - df[close_col]))
            / (df[high_col] - df[low_col]).replace(0, np.nan)
        )
        mfv = mfm * df[vol_col]
        df["CMFI"] = mfv.rolling(window).sum() / df[vol_col].rolling(window).sum()
        return df

    @staticmethod
    def VPC(df, close_col="Close", vol_col="Volume", window=12):
        """
        Volume-Price Correlation (rolling Pearson r).
        """
        df = df.copy()
        price_ch = df[close_col].pct_change()
        vol_ch = df[vol_col].pct_change()
        df["VPC"] = price_ch.rolling(window).corr(vol_ch)
        return df

    # ------------------------------------------------------------------ #
    # 3. Trend indicators
    # ------------------------------------------------------------------ #
    @staticmethod
    def SMA(df, close_col="Close", windows=[20, 50, 100, 200]):
        df = df.copy()
        for w in windows:
            df[f"SMA_{w}"] = df[close_col].rolling(w).mean()
        return df

    @staticmethod
    def EMA(df, close_col="Close", windows=[10, 30, 60]):
        df = df.copy()
        for w in windows:
            df[f"EMA_{w}"] = df[close_col].ewm(span=w, adjust=False).mean()
        return df

    @staticmethod
    def WMA(df, close_col="Close", windows=[10]):
        df = df.copy()
        for w in windows:
            weights = np.arange(1, w + 1)
            df[f"WMA_{w}"] = df[close_col].rolling(w).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        return df

    @staticmethod
    def HMA(df, close_col="Close", windows=[10]):
        """
        Hull Moving Average built from WMAs.
        """
        df = df.copy()
        for w in windows:
            half = int(w / 2)
            wma_half = IndicatorFactory.WMA(df, close_col, [half])[f"WMA_{half}"]
            wma_full = IndicatorFactory.WMA(df, close_col, [w])[f"WMA_{w}"]
            diff = 2 * wma_half - wma_full
            sqrt_w = int(np.sqrt(w))
            weights = np.arange(1, sqrt_w + 1)
            hma = diff.rolling(sqrt_w).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
            df[f"HMA_{w}"] = hma
        return df

    @staticmethod
    def CMA(df, close_col="Close"):
        df = df.copy()
        df["CMA"] = df[close_col].expanding().mean()
        return df

    @staticmethod
    def encode_weekday_cyclical(df, time_col="Open time"):
        df = df.copy()
        dow = pd.to_datetime(df[time_col]).dt.weekday
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        return df

    # ------------------------------------------------------------------ #
    # 4. Momentum / Oscillator indicators
    # ------------------------------------------------------------------ #
    @staticmethod
    def RSI(df, close_col="Close", periods=[14, 30]):
        df = df.copy()
        delta = df[close_col].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        for p in periods:
            rs = gain.ewm(alpha=1 / p, adjust=False).mean() / (
                loss.ewm(alpha=1 / p, adjust=False).mean() + 1e-8
            )
            df[f"RSI_{p}"] = 100 - 100 / (1 + rs)
        return df

    @staticmethod
    def WilliamsR(df, high_col="High", low_col="Low", close_col="Close", periods=[14]):
        df = df.copy()
        for p in periods:
            high_roll = df[high_col].rolling(p).max()
            low_roll = df[low_col].rolling(p).min()
            df[f"WILLIAMS_R_{p}"] = -100 * (high_roll - df[close_col]) / (
                high_roll - low_roll
            )
        return df

    @staticmethod
    def CCI(df, high_col="High", low_col="Low", close_col="Close", periods=[20]):
        df = df.copy()
        tp = (df[high_col] + df[low_col] + df[close_col]) / 3
        for p in periods:
            ma = tp.rolling(p).mean()
            md = tp.rolling(p).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
            df[f"CCI_{p}"] = (tp - ma) / (0.015 * md)
        return df

    @staticmethod
    def MACD(df, close_col="Close", n_fast=12, n_slow=26, n_signal=9):
        df = df.copy()
        ema_fast = df[close_col].ewm(span=n_fast, adjust=False).mean()
        ema_slow = df[close_col].ewm(span=n_slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=n_signal, adjust=False).mean()
        df[f"MACD{n_slow}_{n_fast}"] = macd
        df[f"SIGNAL{n_slow}_{n_fast}"] = signal
        df[f"HIST{n_slow}_{n_fast}"] = macd - signal
        return df

    @staticmethod
    def PPO(df, close_col="Close", n_fast=12, n_slow=26):
        df = df.copy()
        ema_fast = df[close_col].ewm(span=n_fast, adjust=False).mean()
        ema_slow = df[close_col].ewm(span=n_slow, adjust=False).mean()
        df["PPO"] = (ema_fast - ema_slow) / ema_slow * 100
        return df

    @staticmethod
    def ROC(df, close_col="Close", periods=[12]):
        df = df.copy()
        for p in periods:
            df[f"ROC_{p}"] = df[close_col].pct_change(p) * 100
        return df

    @staticmethod
    def PMI(df, close_col="Close", periods=[12]):
        df = df.copy()
        for p in periods:
            momentum = df[close_col].diff(p)
            df[f"PMI_{p}"] = momentum / df[close_col].shift(p) * 100
        return df

    @staticmethod
    def PSI(df, close_col="Close", periods=[12]):
        df = df.copy()
        for p in periods:
            pos = (df[close_col].diff() > 0).astype(int)
            df[f"PSI_{p}"] = pos.rolling(p).sum() / p * 100
        return df



    # ------------------------------------------------------------------ #
    # 5. Bid / Ask spread helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def spread_adder(df, fixed_spread=1, method="fixed_spread", atr_window=14):
        df = df.copy()
        if method == "atr_fraction":
            if f"ATR_{atr_window}" not in df.columns:
                df = IndicatorFactory.ATR(df, windows=[atr_window])
            df["estimated_spread"] = df[f"ATR_{atr_window}"] * 0.001
        else:
            df["estimated_spread"] = fixed_spread

        for col in ["Open", "High", "Low", "Close"]:
            df[f"{col}_Bid"] = df[col] - df["estimated_spread"] / 2
            df[f"{col}_Ask"] = df[col] + df["estimated_spread"] / 2
        return df

    # ------------------------------------------------------------------ #
    # 6. Composite Sentiment Index (CSI)
    # ------------------------------------------------------------------ #
    @staticmethod
    def calculate_csi(df, close_col="Close", vol_col="Volume", base_rsi="RSI_14"):
        df = df.copy()
        if base_rsi not in df.columns:
            df = IndicatorFactory.RSI(df, periods=[14])

        df = IndicatorFactory.PMI(df)
        df = IndicatorFactory.VPC(df)
        df = IndicatorFactory.Volatility(df)

        norm = lambda x: (x - x.mean()) / x.std()

        csi = (
            0.3 * norm(df[base_rsi])
            + 0.25 * norm(df["PMI_12"])
            + 0.25 * norm(df["VPC"])
            + 0.2 * norm(df["Volatility_12"])
        )
        df["CSI"] = csi
        return df
    
    @staticmethod
    def OBV(df, close_col="Close", vol_col="Volume"):
        """
        On-Balance Volume: cumulative volume added on up days and subtracted on down days.
        """
        df = df.copy()
        delta_close = df[close_col].diff()
        direction = np.where(delta_close > 0, 1, np.where(delta_close < 0, -1, 0))
        # Calculate OBV by cumulatively summing volume * direction
        df["OBV"] = (direction * df[vol_col]).cumsum()
        return df

    @staticmethod
    def Stochastic(df, high_col="High", low_col="Low", close_col="Close", k_window=14, d_window=3):
        """
        Stochastic Oscillator: %K and %D.
        %K = 100 * (Close - min_low_{k_window}) / (max_high_{k_window} - min_low_{k_window});
        %D = SMA of %K over d_window.
        """
        df = df.copy()
        low_min = df[low_col].rolling(k_window).min()
        high_max = df[high_col].rolling(k_window).max()
        stoch_k = 100 * (df[close_col] - low_min) / (high_max - low_min + 1e-8)
        df[f"%K_{k_window}"] = stoch_k
        df[f"%D_{k_window}"] = stoch_k.rolling(d_window).mean()
        return df

    @staticmethod
    def Aroon(df, high_col="High", low_col="Low", periods=[25]):
        """
        Aroon Up/Down indicator.
        """
        df = df.copy()
        for p in periods:
            # Rolling window index of highest high and lowest low
            rolling_high_idx = df[high_col].rolling(p).apply(np.argmax, raw=True)
            rolling_low_idx  = df[low_col].rolling(p).apply(np.argmin, raw=True)
            # Aroon Up = percent of time since highest high within window
            df[f"AroonUp_{p}"] = 100 * ((rolling_high_idx + 1) / p)
            # Aroon Down = percent of time since lowest low within window
            df[f"AroonDown_{p}"] = 100 * ((rolling_low_idx + 1) / p)
        return df

    @staticmethod
    def ADL(df, high_col="High", low_col="Low", close_col="Close", vol_col="Volume"):
        """
        Accumulation/Distribution Line: cumulative money flow volume.
        """
        df = df.copy()
        # Money Flow Multiplier = ( (Close - Low) - (High - Close) ) / (High - Low )
        mfm = ((df[close_col] - df[low_col]) - (df[high_col] - df[close_col])) / (df[high_col] - df[low_col]).replace(0, np.nan)
        mf_volume = mfm * df[vol_col]
        df["ADL"] = mf_volume.cumsum()
        return df
    
    # ------------------------------------------------------------------ #
    # 7. Convenience wrapper – apply everything
    # ------------------------------------------------------------------ #
    @staticmethod
    def apply_all(df):
        """
        Run a broad suite of indicators with default parameters,
        in a strictly causal (no-peek) fashion, then dropna.
        """
        out = df.copy()

        # 1) Volatility
        out = IndicatorFactory.returns(out)                  # returns + rolling means
        out = IndicatorFactory.BollingerBands(out)           # BBands
        out = IndicatorFactory.ATR(out)                      # true range & ATR
        out = IndicatorFactory.Volatility(out)               # rolling σ of returns

        # 2) Volume / price–volume
        out = IndicatorFactory.vwap(out)                     # VWAP
        out = IndicatorFactory.CMFI(out)                     # Chaikin MF Index
        out = IndicatorFactory.VPC(out)                      # rolling corr price/vol

        # 3) Trend
        out = IndicatorFactory.SMA(out)                      # simple MAs
        out = IndicatorFactory.EMA(out)                      # EMAs
        out = IndicatorFactory.WMA(out)                      # weighted MAs
        out = IndicatorFactory.HMA(out)                      # Hull MAs
        out = IndicatorFactory.CMA(out)                      # cumulative MA

        # 4) Momentum / 
        out = IndicatorFactory.RSI(out)                      # RSI(s)
        out = IndicatorFactory.WilliamsR(out)                # Williams %R
        out = IndicatorFactory.CCI(out)                      # CCI
        out = IndicatorFactory.MACD(out)                     # MACD + signal + hist
        out = IndicatorFactory.PPO(out)                      # PPO
        out = IndicatorFactory.ROC(out)                      # rate of change
        out = IndicatorFactory.PMI(out)                      # price momentum index
        out = IndicatorFactory.PSI(out)                      # psychological line


        # 5) Calendar
        out = IndicatorFactory.encode_weekday_cyclical(out)  # day-of-week sine/cosine

        # 6) Composite sentiment
        out = IndicatorFactory.calculate_csi(out)            # CSI

        out = IndicatorFactory.OBV(out)                      # OBV
        out = IndicatorFactory.Stochastic(out)               # Stochastic
        out = IndicatorFactory.Aroon(out)                    # Aroon
        out = IndicatorFactory.ADL(out)                      # ADL

        return out.dropna()