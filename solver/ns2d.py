"""2D incompressible Navier–Stokes in vorticity form on the periodic box [0, 2π]².

Conventions (used everywhere in the project)
--------------------------------------------
    omega = dv/dx - du/dy = -lap(psi),   u = dpsi/dy,   v = -dpsi/dx
    d(omega)/dt + u . grad(omega) = nu lap(omega) - alpha omega + f

    E = 1/2 <|u|^2>,  Z = 1/2 <omega^2>,  P = 1/2 <|grad omega|^2>   (<.> = spatial mean)
    so that  dE/dt = -2 nu Z - 2 alpha E + <psi f>   and, unforced,  dZ/dt = -2 nu P.

Spectral arrays are unnormalized ``torch.fft.rfft2`` coefficients over the last two axes,
ordered (y, x): shape (..., n, n // 2 + 1). Every state is kept band-limited by the 2/3 rule
(|k_x|, |k_y| <= n // 3), which makes the quadratic nonlinearity alias-free, so the
semi-discrete system is the Galerkin truncation and conserves E and Z exactly when nu = alpha = 0.

Time stepping is integrating-factor (Lawson) RK4: the linear viscous/drag part is integrated
exactly, the nonlinear part with classical RK4.
"""
from __future__ import annotations

import math

import torch


class SpectralGrid:
    """Wavenumbers, dealiasing mask and transforms for an n x n periodic grid on [0, 2π]²."""

    def __init__(self, n: int, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64):
        if n % 2:
            raise ValueError("grid size must be even")
        self.n = n
        self.device = torch.device(device)
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.dx = 2.0 * math.pi / n
        self.kmax = n // 3  # 2/3 rule: 3 * kmax < n
        ky = torch.fft.fftfreq(n, d=1.0 / n, dtype=dtype, device=self.device)
        kx = torch.fft.rfftfreq(n, d=1.0 / n, dtype=dtype, device=self.device)
        self.ky = ky.view(n, 1)
        self.kx = kx.view(1, n // 2 + 1)
        self.k2 = self.kx**2 + self.ky**2
        self.kmag = torch.sqrt(self.k2)
        self.mask = ((self.kx.abs() <= self.kmax) & (self.ky.abs() <= self.kmax)).to(dtype)
        self.inv_k2 = torch.where(self.k2 > 0, 1.0 / torch.clamp(self.k2, min=1.0), torch.zeros_like(self.k2))
        # Parseval weights for rfft coefficients: <f^2> = sum_k w_k |f_k|^2.
        w = torch.full((n // 2 + 1,), 2.0, dtype=dtype, device=self.device)
        w[0] = 1.0
        w[-1] = 1.0  # Nyquist column (n even)
        self.parseval_w = w.view(1, n // 2 + 1) / float(n) ** 4
        coords = torch.arange(n, dtype=dtype, device=self.device) * self.dx
        self.x = coords.view(1, n)
        self.y = coords.view(n, 1)

    # --- transforms -------------------------------------------------------------------------
    def fft(self, f: torch.Tensor) -> torch.Tensor:
        return torch.fft.rfft2(f.to(self.dtype))

    def ifft(self, fh: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfft2(fh, s=(self.n, self.n))

    def mean_square(self, fh: torch.Tensor) -> torch.Tensor:
        """Spatial mean of f^2 from rfft coefficients (exact Parseval)."""
        return ((fh.real**2 + fh.imag**2) * self.parseval_w).sum(dim=(-2, -1))

    # --- kinematics -------------------------------------------------------------------------
    def streamfunction_hat(self, w_hat: torch.Tensor) -> torch.Tensor:
        return w_hat * self.inv_k2

    def velocity_hat(self, w_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        psi = self.streamfunction_hat(w_hat)
        return 1j * self.ky * psi, -1j * self.kx * psi

    def velocity(self, w_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_hat, v_hat = self.velocity_hat(w_hat)
        return self.ifft(u_hat), self.ifft(v_hat)

    def vorticity_hat_from_velocity(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (1j * self.kx * self.fft(v) - 1j * self.ky * self.fft(u)) * self.mask

    def max_speed(self, w_hat: torch.Tensor) -> torch.Tensor:
        """max over the grid of |u| + |v|, per batch element."""
        u, v = self.velocity(w_hat)
        return (u.abs() + v.abs()).amax(dim=(-2, -1))


def resample_hat(w_hat: torch.Tensor, n_in: int, n_out: int) -> torch.Tensor:
    """Move a field between grid sizes by copying shared Fourier modes (|k_i| < min(n)/2).
    Upsampling is exact for band-limited fields; downsampling keeps the shared band (apply the
    target grid's mask afterwards to get its 2/3 band)."""
    h = min(n_in, n_out) // 2
    out = torch.zeros(w_hat.shape[:-2] + (n_out, n_out // 2 + 1), dtype=w_hat.dtype, device=w_hat.device)
    out[..., :h, :h] = w_hat[..., :h, :h]
    out[..., n_out - h + 1 :, :h] = w_hat[..., n_in - h + 1 :, :h]
    return out * (n_out / n_in) ** 2


def nonlinear_hat(grid: SpectralGrid, w_hat: torch.Tensor) -> torch.Tensor:
    """Dealiased -u . grad(omega) in spectral space (input must already be band-limited)."""
    psi = w_hat * grid.inv_k2
    u = grid.ifft(1j * grid.ky * psi)
    v = grid.ifft(-1j * grid.kx * psi)
    wx = grid.ifft(1j * grid.kx * w_hat)
    wy = grid.ifft(1j * grid.ky * w_hat)
    return -grid.fft(u * wx + v * wy) * grid.mask


class NavierStokes2D:
    """Batched IF-RK4 integrator. ``nu`` may be a float or a tensor with one value per trajectory."""

    def __init__(self, grid: SpectralGrid, nu, alpha: float = 0.0, forcing_hat: torch.Tensor | None = None):
        self.grid = grid
        nu_t = torch.as_tensor(nu, dtype=grid.dtype, device=grid.device)
        if nu_t.ndim == 1:
            nu_t = nu_t.view(-1, 1, 1)
        self.nu = nu_t
        self.alpha = float(alpha)
        self.lin = -(nu_t * grid.k2 + self.alpha)
        self.forcing_hat = None if forcing_hat is None else forcing_hat.to(grid.cdtype) * grid.mask

    def rhs(self, w_hat: torch.Tensor) -> torch.Tensor:
        out = nonlinear_hat(self.grid, w_hat)
        if self.forcing_hat is not None:
            out = out + self.forcing_hat
        return out

    def step(self, w_hat: torch.Tensor, dt: float) -> torch.Tensor:
        e1 = torch.exp(self.lin * (0.5 * dt))
        e2 = e1 * e1
        k1 = dt * self.rhs(w_hat)
        k2 = dt * self.rhs(e1 * (w_hat + 0.5 * k1))
        k3 = dt * self.rhs(e1 * w_hat + 0.5 * k2)
        k4 = dt * self.rhs(e2 * w_hat + e1 * k3)
        return e2 * w_hat + (e2 * k1 + 2.0 * e1 * (k2 + k3) + k4) / 6.0

    def advance(self, w_hat: torch.Tensor, t_span: float, cfl: float, dt_max: float | None = None):
        """Integrate exactly ``t_span`` with equal substeps; the step obeys the CFL limit
        cfl * dx / max(|u| + |v|) evaluated over the whole batch at the start of the span."""
        speed = float(self.grid.max_speed(w_hat).max())
        dt = cfl * self.grid.dx / max(speed, 1e-12)
        if dt_max is not None:
            dt = min(dt, dt_max)
        nsub = max(1, math.ceil(t_span / dt - 1e-9))
        h = t_span / nsub
        for _ in range(nsub):
            w_hat = self.step(w_hat, h)
        return w_hat, nsub

    def simulate(self, w_hat0: torch.Tensor, n_frames: int, dt_out: float, cfl: float,
                 spinup: float = 0.0, dt_max: float | None = None):
        """Yield (t, w_hat) for n_frames snapshots at t = spinup + i * dt_out."""
        w_hat = w_hat0 * self.grid.mask
        t = 0.0
        if spinup > 0:
            w_hat, _ = self.advance(w_hat, spinup, cfl, dt_max)
            t = spinup
        for i in range(n_frames):
            if i > 0:
                w_hat, _ = self.advance(w_hat, dt_out, cfl, dt_max)
                t += dt_out
            yield t, w_hat
