"""Synthetic dataset utilities for training NetTOF strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from EchoModel import echo_function, echo_info_max, echo_info_min

_PARAM_NAMES: Tuple[str, ...] = ("fc", "beta", "alpha", "r", "tau", "psi", "phi")
_PARAM_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(_PARAM_NAMES)}


@dataclass(frozen=True)
class ParameterBounds:
    """Closed interval describing the feasible range for a parameter."""

    minimum: float
    maximum: float

    def clip(self, values: np.ndarray) -> np.ndarray:
        return np.clip(values, self.minimum, self.maximum)


_PARAM_BOUNDS: Dict[str, ParameterBounds] = {
    "fc": ParameterBounds(echo_info_min.fc, echo_info_max.fc),
    "beta": ParameterBounds(echo_info_min.beta, echo_info_max.beta),
    "alpha": ParameterBounds(echo_info_min.alpha, echo_info_max.alpha),
    "r": ParameterBounds(echo_info_min.r, echo_info_max.r),
    "tau": ParameterBounds(echo_info_min.tau, echo_info_max.tau),
    "psi": ParameterBounds(max(echo_info_min.psi, 1.0e11), echo_info_max.psi),
    "phi": ParameterBounds(-np.pi, np.pi),
}

_FC_LOG_MIN = float(np.log(_PARAM_BOUNDS["fc"].minimum))
_FC_LOG_MAX = float(np.log(_PARAM_BOUNDS["fc"].maximum))
_ALPHA_LOG_MIN = float(np.log(_PARAM_BOUNDS["alpha"].minimum))
_ALPHA_LOG_MAX = float(np.log(_PARAM_BOUNDS["alpha"].maximum))
_PSI_LOG_MIN = float(np.log(_PARAM_BOUNDS["psi"].minimum))
_PSI_LOG_MAX = float(np.log(_PARAM_BOUNDS["psi"].maximum))

_BETA_MIN = _PARAM_BOUNDS["beta"].minimum
_BETA_MAX = _PARAM_BOUNDS["beta"].maximum
_TAU_MIN = _PARAM_BOUNDS["tau"].minimum
_TAU_MAX = _PARAM_BOUNDS["tau"].maximum

_EPS = 1.0e-8


def _as_numpy(
    params: Union[np.ndarray, torch.Tensor]
) -> Tuple[np.ndarray, bool, Optional[torch.device], Optional[torch.dtype]]:
    if isinstance(params, torch.Tensor):
        tensor = params.detach()
        device = tensor.device
        dtype = tensor.dtype
        array = tensor.cpu().numpy()
        return array, True, device, dtype
    return np.asarray(params), False, None, None


def _to_tensor(
    array: np.ndarray, was_tensor: bool, device: Optional[torch.device], dtype: Optional[torch.dtype]
) -> Union[np.ndarray, torch.Tensor]:
    if not was_tensor:
        return array
    result = torch.from_numpy(array)
    if dtype is not None:
        result = result.to(dtype=dtype)
    if device is not None:
        result = result.to(device=device)
    return result


def normalize_params(params: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Normalize echo parameters for stable learning."""

    array, was_tensor, device, dtype = _as_numpy(params)
    if array.shape[-1] != len(_PARAM_NAMES):
        raise ValueError("Expected last dimension to enumerate the seven echo parameters.")

    normalized = np.empty_like(array, dtype=np.float32)

    fc = _PARAM_BOUNDS["fc"].clip(array[..., _PARAM_INDEX["fc"]])
    fc_log = np.log(fc)
    normalized[..., _PARAM_INDEX["fc"]] = (fc_log - _FC_LOG_MIN) / (_FC_LOG_MAX - _FC_LOG_MIN)

    beta = _PARAM_BOUNDS["beta"].clip(array[..., _PARAM_INDEX["beta"]])
    normalized[..., _PARAM_INDEX["beta"]] = (
        2.0 * (beta - _BETA_MIN) / (_BETA_MAX - _BETA_MIN + _EPS) - 1.0
    )

    alpha = _PARAM_BOUNDS["alpha"].clip(array[..., _PARAM_INDEX["alpha"]])
    alpha_log = np.log(alpha)
    normalized[..., _PARAM_INDEX["alpha"]] = (alpha_log - _ALPHA_LOG_MIN) / (
        _ALPHA_LOG_MAX - _ALPHA_LOG_MIN
    )

    r = np.clip(array[..., _PARAM_INDEX["r"]], -1.0, 1.0)
    normalized[..., _PARAM_INDEX["r"]] = r

    tau = _PARAM_BOUNDS["tau"].clip(array[..., _PARAM_INDEX["tau"]])
    normalized[..., _PARAM_INDEX["tau"]] = (
        2.0 * (tau - _TAU_MIN) / (_TAU_MAX - _TAU_MIN + _EPS) - 1.0
    )

    psi = _PARAM_BOUNDS["psi"].clip(array[..., _PARAM_INDEX["psi"]])
    psi_log = np.log(psi)
    normalized[..., _PARAM_INDEX["psi"]] = (psi_log - _PSI_LOG_MIN) / (
        _PSI_LOG_MAX - _PSI_LOG_MIN
    )

    phi = np.clip(array[..., _PARAM_INDEX["phi"]], -np.pi, np.pi)
    normalized[..., _PARAM_INDEX["phi"]] = phi / np.pi

    return _to_tensor(normalized.astype(np.float32), was_tensor, device, dtype)


def denormalize_params(parameters: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Invert :func:`normalize_params` back to physical echo parameters."""

    array, was_tensor, device, dtype = _as_numpy(parameters)
    if array.shape[-1] != len(_PARAM_NAMES):
        raise ValueError("Expected last dimension to enumerate the seven echo parameters.")

    restored = np.empty_like(array, dtype=np.float32)

    fc_norm = np.clip(array[..., _PARAM_INDEX["fc"]], 0.0, 1.0)
    fc_log = fc_norm * (_FC_LOG_MAX - _FC_LOG_MIN) + _FC_LOG_MIN
    restored[..., _PARAM_INDEX["fc"]] = np.exp(fc_log)

    beta_norm = np.clip(array[..., _PARAM_INDEX["beta"]], -1.0, 1.0)
    restored[..., _PARAM_INDEX["beta"]] = (
        (beta_norm + 1.0) * 0.5 * (_BETA_MAX - _BETA_MIN) + _BETA_MIN
    )

    alpha_norm = np.clip(array[..., _PARAM_INDEX["alpha"]], 0.0, 1.0)
    alpha_log = alpha_norm * (_ALPHA_LOG_MAX - _ALPHA_LOG_MIN) + _ALPHA_LOG_MIN
    restored[..., _PARAM_INDEX["alpha"]] = np.exp(alpha_log)

    r_norm = np.clip(array[..., _PARAM_INDEX["r"]], -1.0, 1.0)
    restored[..., _PARAM_INDEX["r"]] = r_norm

    tau_norm = np.clip(array[..., _PARAM_INDEX["tau"]], -1.0, 1.0)
    restored[..., _PARAM_INDEX["tau"]] = (
        (tau_norm + 1.0) * 0.5 * (_TAU_MAX - _TAU_MIN) + _TAU_MIN
    )

    psi_norm = np.clip(array[..., _PARAM_INDEX["psi"]], 0.0, 1.0)
    psi_log = psi_norm * (_PSI_LOG_MAX - _PSI_LOG_MIN) + _PSI_LOG_MIN
    restored[..., _PARAM_INDEX["psi"]] = np.exp(psi_log)

    phi_norm = np.clip(array[..., _PARAM_INDEX["phi"]], -1.0, 1.0)
    restored[..., _PARAM_INDEX["phi"]] = phi_norm * np.pi

    return _to_tensor(restored.astype(np.float32), was_tensor, device, dtype)


class SyntheticEchoDataset(Dataset):
    """Dataset of synthetic ultrasound waveforms with multi-echo annotations."""

    def __init__(
        self,
        n_samples: int,
        n_iter_outer: int,
        waveform_length: int = 10000,
        time_span: float = 20.0e-6,
        noise_std: float = 0.015,
        noise_bias_std: float = 0.0025,
        seed: Optional[int] = None,
    ) -> None:
        if n_iter_outer < 1:
            raise ValueError("n_iter_outer must be a positive integer.")
        if n_samples < 1:
            raise ValueError("Need at least one sample to build the dataset.")

        self.n_samples = n_samples
        self.n_iter_outer = n_iter_outer
        self.waveform_length = waveform_length
        self.time_span = time_span
        self.noise_std = noise_std
        self.noise_bias_std = noise_bias_std
        self._rng = np.random.default_rng(seed)

        self._time_axis = np.linspace(0.0, time_span, waveform_length, dtype=np.float64)

        waveforms: List[np.ndarray] = []
        params_raw: List[np.ndarray] = []

        for _ in range(n_samples):
            params = self._sample_parameters()
            waveform = self._synthesize_waveform(params)

            waveform = waveform - waveform.mean()
            std = waveform.std() + _EPS
            waveform = waveform / std

            waveforms.append(waveform.astype(np.float32)[np.newaxis, :])
            params_raw.append(params.astype(np.float32))

        self.waveforms = np.stack(waveforms, axis=0)
        self.params_raw = np.stack(params_raw, axis=0)
        self.params_norm = normalize_params(self.params_raw)

    @property
    def time_axis(self) -> np.ndarray:
        return self._time_axis

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        return _PARAM_NAMES

    def __len__(self) -> int:  # type: ignore[override]
        return self.n_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        waveform = torch.from_numpy(self.waveforms[index])
        target = torch.from_numpy(self.params_norm[index])
        target_raw = torch.from_numpy(self.params_raw[index])
        return {
            "waveform": waveform,
            "target": target,
            "target_raw": target_raw,
        }

    def normalize_params(self, params: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return normalize_params(params)

    def denormalize_params(self, params: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return denormalize_params(params)

    def _sample_parameters(self) -> np.ndarray:
        fc = self._rng.uniform(_PARAM_BOUNDS["fc"].minimum, _PARAM_BOUNDS["fc"].maximum, size=self.n_iter_outer)
        beta = self._rng.uniform(_PARAM_BOUNDS["beta"].minimum, _PARAM_BOUNDS["beta"].maximum, size=self.n_iter_outer)
        alpha = self._rng.uniform(_PARAM_BOUNDS["alpha"].minimum, _PARAM_BOUNDS["alpha"].maximum, size=self.n_iter_outer)
        r = self._rng.uniform(-1.0, 1.0, size=self.n_iter_outer)
        tau = self._rng.uniform(_PARAM_BOUNDS["tau"].minimum, _PARAM_BOUNDS["tau"].maximum, size=self.n_iter_outer)
        psi = self._rng.uniform(_PARAM_BOUNDS["psi"].minimum, _PARAM_BOUNDS["psi"].maximum, size=self.n_iter_outer)
        phi = self._rng.uniform(-np.pi, np.pi, size=self.n_iter_outer)

        order = np.argsort(tau)
        fc = fc[order]
        beta = beta[order]
        alpha = alpha[order]
        r = r[order]
        tau = tau[order]
        psi = psi[order]
        phi = phi[order]

        stacked = np.stack([fc, beta, alpha, r, tau, psi, phi], axis=-1)
        return stacked.astype(np.float64)

    def _synthesize_waveform(self, params: np.ndarray) -> np.ndarray:
        waveform = np.zeros(self.waveform_length, dtype=np.float64)
        for echo in params:
            waveform += echo_function(self._time_axis, *echo.tolist())

        if self.noise_bias_std > 0.0:
            waveform += self._rng.normal(0.0, self.noise_bias_std)
        if self.noise_std > 0.0:
            waveform += self._rng.normal(0.0, self.noise_std, size=waveform.shape)

        slope = self._rng.normal(0.0, self.noise_bias_std * 0.25)
        waveform += slope * np.linspace(-0.5, 0.5, self.waveform_length)

        return waveform

    def get_dataloaders(
        self,
        batch_size: int,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        num_workers: int = 0,
        seed: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1).")
        if not 0.0 <= test_ratio < 1.0:
            raise ValueError("test_ratio must be in [0, 1).")
        if val_ratio + test_ratio >= 1.0:
            raise ValueError("Validation and test ratios must leave room for training samples.")

        n_test = max(1, int(round(self.n_samples * test_ratio)))
        n_val = max(1, int(round(self.n_samples * val_ratio)))
        n_train = max(self.n_samples - n_val - n_test, 1)
        total = n_train + n_val + n_test
        if total != self.n_samples:
            diff = self.n_samples - total
            n_train += diff

        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        splits = random_split(self, [n_train, n_val, n_test], generator=generator)
        train_subset, val_subset, test_subset = splits

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, test_loader


__all__ = [
    "SyntheticEchoDataset",
    "normalize_params",
    "denormalize_params",
]
