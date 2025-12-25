from typing import Tuple, Union, Optional
import numpy as np

class CubicSpline:
    def __init__(self, x: np.ndarray, y: np.ndarray, axis: int = 0, 
                 bc_type: str = 'natural', allow_extrapolation: bool = True,
                 extrapolation_mode: str = 'linear'):

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        if x.ndim != 1:
            raise ValueError("x должен быть одномерным массивом")
        
        if x.shape[0] < 2:
            raise ValueError("Требуется минимум 2 точки для интерполяции")
        
        if not np.all(np.diff(x) > 0):
            raise ValueError("x должен быть строго возрастающим")
        
        if axis != 0:
            y = np.moveaxis(y, axis, 0)
        
        self.x = x.copy()
        self.axis = axis
        self.bc_type = bc_type.lower()
        self.allow_extrapolation = allow_extrapolation
        self.extrapolation_mode = extrapolation_mode.lower()

        self.y_shape = y.shape
        self.n_points = len(x)

        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y.reshape(self.n_points, -1)
        
        self.n_components = y_2d.shape[1]

        self.y_original = y_2d.copy()

        self.coeffs = np.zeros((self.n_points - 1, 4, self.n_components))

        self.derivatives = None
        if allow_extrapolation and extrapolation_mode == 'linear':
            self._compute_derivatives()

        for comp_idx in range(self.n_components):
            y_comp = y_2d[:, comp_idx]
            self.coeffs[:, :, comp_idx] = self._compute_coeffs_efficient(x, y_comp)
    
    def _compute_coeffs_efficient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = len(x)
        h = np.diff(x)
        
        if n == 2:
            a0 = y[0]
            a1 = y[1]
            b = (a1 - a0) / h[0]
            return np.array([[a0, b, 0.0, 0.0]])

        a = np.zeros(n)
        b = np.zeros(n)
        c = np.zeros(n)
        d = np.zeros(n)
        
        for i in range(1, n-1):
            a[i] = h[i-1]
            b[i] = 2 * (h[i-1] + h[i])
            c[i] = h[i]
            d[i] = 3 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
        
        b[0] = 1.0
        b[n-1] = 1.0
        d[0] = 0.0
        d[n-1] = 0.0
        
        m = self._solve_tridiagonal(a, b, c, d)
        
        coeffs = np.zeros((n-1, 4))
        
        for i in range(n-1):
            yi = y[i]
            yi1 = y[i+1]
            mi = m[i]
            mi1 = m[i+1]
            hi = h[i]
            
            a_coeff = yi
            b_coeff = (yi1 - yi) / hi - hi * (2*mi + mi1) / 6
            c_coeff = mi / 2
            d_coeff = (mi1 - mi) / (6 * hi)
            
            coeffs[i] = [a_coeff, b_coeff, c_coeff, d_coeff]
        
        return coeffs
    
    def _solve_tridiagonal(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
        n = len(b)
        x = np.zeros(n)
        
        cp = np.zeros(n-1)
        dp = np.zeros(n)

        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        
        for i in range(1, n-1):
            denom = b[i] - a[i] * cp[i-1]
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i] * dp[i-1]) / denom
        
        denom = b[n-1] - a[n-1] * cp[n-2]
        dp[n-1] = (d[n-1] - a[n-1] * dp[n-2]) / denom
        
        x[n-1] = dp[n-1]
        for i in range(n-2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i+1]
        
        return x
    
    def _compute_derivatives(self):
        self.derivatives = np.zeros((2, self.n_components))

        for comp_idx in range(self.n_components):
            coeffs = self.coeffs[0, :, comp_idx]
            self.derivatives[0, comp_idx] = coeffs[1]
        for comp_idx in range(self.n_components):
            coeffs = self.coeffs[-1, :, comp_idx]
            h = self.x[-1] - self.x[-2]
            self.derivatives[1, comp_idx] = (
                coeffs[1] + 2*coeffs[2]*h + 3*coeffs[3]*h**2
            )
    
    def __call__(self, x_new: Union[float, np.ndarray], nu: int = 0) -> np.ndarray:
        x_new = np.asarray(x_new, dtype=float)
        scalar_input = x_new.ndim == 0
        if scalar_input:
            x_new = np.array([x_new])

        below_min = x_new < self.x[0]
        above_max = x_new > self.x[-1]
        in_range = (~below_min) & (~above_max)
        
        if not self.allow_extrapolation and (np.any(below_min) or np.any(above_max)):
            out_of_range_points = x_new[below_min | above_max]
            raise ValueError(f"Точки вне диапазона [{self.x[0]}, {self.x[-1]}]: {out_of_range_points[:5]}...")
        
        result = np.zeros((x_new.size, self.n_components))

        if np.any(in_range):
            x_in_range = x_new[in_range]
            indices = np.searchsorted(self.x, x_in_range) - 1
            indices = np.clip(indices, 0, self.n_points - 2)
            
            for comp_idx in range(self.n_components):
                comp_coeffs = self.coeffs[:, :, comp_idx]
                
                for i, (x_val, idx) in enumerate(zip(x_in_range, indices)):
                    global_idx = np.where(in_range)[0][i]
                    t = x_val - self.x[idx]
                    a, b, c, d = comp_coeffs[idx]
                    
                    if nu == 0:
                        result[global_idx, comp_idx] = a + b*t + c*t**2 + d*t**3
                    elif nu == 1:
                        result[global_idx, comp_idx] = b + 2*c*t + 3*d*t**2
                    elif nu == 2:
                        result[global_idx, comp_idx] = 2*c + 6*d*t
                    else:
                        raise ValueError("Поддерживаются только производные до 2-го порядка")

        if np.any(below_min) and self.allow_extrapolation:
            x_below = x_new[below_min]
            t_below = x_below - self.x[0]
            
            for comp_idx in range(self.n_components):
                if self.extrapolation_mode == 'linear':
                    if self.derivatives is None:
                        self._compute_derivatives()
                    y0 = self.y_original[0, comp_idx]
                    dy0 = self.derivatives[0, comp_idx]
                    
                    if nu == 0:
                        result[below_min, comp_idx] = y0 + dy0 * t_below
                    elif nu == 1:
                        result[below_min, comp_idx] = dy0 * np.ones_like(t_below)
                    elif nu == 2:
                        result[below_min, comp_idx] = np.zeros_like(t_below)
                elif self.extrapolation_mode == 'constant':
                    y0 = self.y_original[0, comp_idx]
                    if nu == 0:
                        result[below_min, comp_idx] = y0
                    else:
                        result[below_min, comp_idx] = np.zeros_like(t_below)

        if np.any(above_max) and self.allow_extrapolation:
            x_above = x_new[above_max]
            t_above = x_above - self.x[-1]
            
            for comp_idx in range(self.n_components):
                if self.extrapolation_mode == 'linear':
                    if self.derivatives is None:
                        self._compute_derivatives()
                    y_end = self.y_original[-1, comp_idx]
                    dy_end = self.derivatives[1, comp_idx]
                    
                    if nu == 0:
                        result[above_max, comp_idx] = y_end + dy_end * t_above
                    elif nu == 1:
                        result[above_max, comp_idx] = dy_end * np.ones_like(t_above)
                    elif nu == 2:
                        result[above_max, comp_idx] = np.zeros_like(t_above)
                elif self.extrapolation_mode == 'constant':
                    y_end = self.y_original[-1, comp_idx]
                    if nu == 0:
                        result[above_max, comp_idx] = y_end
                    else:
                        result[above_max, comp_idx] = np.zeros_like(t_above)

        if self.n_components == 1:
            result = result.reshape(x_new.shape)
        else:
            result = result.reshape(x_new.shape + (self.n_components,))
        
        if scalar_input:
            if self.n_components == 1:
                return result.item()
            else:
                return result.reshape(self.y_shape[1:])
        
        return result
    
    def derivative(self, nu: int = 1) -> 'CubicSpline':
        if nu < 0:
            raise ValueError("Порядок производной должен быть неотрицательным")
        deriv = CubicSpline(
            self.x, 
            np.zeros((self.n_points, self.n_components)),
            axis=self.axis,
            bc_type=self.bc_type,
            allow_extrapolation=self.allow_extrapolation,
            extrapolation_mode=self.extrapolation_mode
        )
        
        deriv_coeffs = np.zeros_like(self.coeffs)
        
        for i in range(self.n_points - 1):
            for comp_idx in range(self.n_components):
                a, b, c, d = self.coeffs[i, :, comp_idx]
                
                if nu == 1:
                    deriv_coeffs[i, 0, comp_idx] = b
                    deriv_coeffs[i, 1, comp_idx] = 2*c
                    deriv_coeffs[i, 2, comp_idx] = 3*d
                    deriv_coeffs[i, 3, comp_idx] = 0
                elif nu == 2:
                    deriv_coeffs[i, 0, comp_idx] = 2*c
                    deriv_coeffs[i, 1, comp_idx] = 6*d
                    deriv_coeffs[i, 2, comp_idx] = 0
                    deriv_coeffs[i, 3, comp_idx] = 0
                elif nu == 3:
                    deriv_coeffs[i, 0, comp_idx] = 6*d
                    deriv_coeffs[i, 1, comp_idx] = 0
                    deriv_coeffs[i, 2, comp_idx] = 0
                    deriv_coeffs[i, 3, comp_idx] = 0
                elif nu >= 4:
                    deriv_coeffs[i, :, comp_idx] = 0
        
        deriv.coeffs = deriv_coeffs
        deriv.y_original = self.y_original.copy()
        
        if deriv.allow_extrapolation and deriv.extrapolation_mode == 'linear':
            deriv._compute_derivatives()
        
        return deriv


def create_interpolators(time_s: np.ndarray, positions: np.ndarray, velocities: np.ndarray, 
                        allow_extrapolation: bool = True) -> Tuple[CubicSpline, CubicSpline]:
    pos_interp = CubicSpline(
        time_s, positions, axis=0, 
        allow_extrapolation=allow_extrapolation,
        extrapolation_mode='linear'
    )
    vel_interp = CubicSpline(
        time_s, velocities, axis=0,
        allow_extrapolation=allow_extrapolation,
        extrapolation_mode='linear'
    )
    return pos_interp, vel_interp