import numpy as np
from typing import Callable, Tuple, Dict, Optional
from tqdm import tqdm

class ManualLeastSquares:
    
    def __init__(self, residual_func: Callable, 
                 initial_params: np.ndarray,
                 bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.residual_func = residual_func
        self.params = initial_params.copy()
        self.bounds = bounds
        self.n_params = len(initial_params)
        
        self.history = {
            'params': [],
            'cost': [],
            'grad_norm': [],
            'step_norm': []
        }
    
    def compute_jacobian_finite_difference(self, params: np.ndarray, 
                                          epsilon: float = 1e-6) -> np.ndarray:
        r0 = self.residual_func(params)
        n_obs = len(r0)
        J = np.zeros((n_obs, self.n_params))
        
        for j in range(self.n_params):
            params_plus = params.copy()
            params_plus[j] += epsilon
            r_plus = self.residual_func(params_plus)
            J[:, j] = (r_plus - r0) / epsilon
        
        return J
    
    def gauss_newton_step(self, params: np.ndarray, 
                         damping: float = 0.0) -> np.ndarray:
        r = self.residual_func(params)
        J = self.compute_jacobian_finite_difference(params)
        JTJ = J.T @ J
        JTr = J.T @ r
        
        if damping > 0:
            JTJ = JTJ + damping * np.eye(self.n_params)
        
        try:
            delta = np.linalg.solve(JTJ, -JTr)
        except np.linalg.LinAlgError:
            delta = -np.linalg.pinv(JTJ) @ JTr
        
        return delta, J, r
    
    def apply_bounds(self, params: np.ndarray) -> np.ndarray:
        if self.bounds is None:
            return params
        
        lower, upper = self.bounds
        params_clipped = np.clip(params, lower, upper)
        
        self.on_boundary = np.logical_or(params_clipped == lower, 
                                        params_clipped == upper)
        
        return params_clipped
    
    def line_search(self, params: np.ndarray, delta: np.ndarray, 
                   max_backtrack: int = 10, alpha: float = 0.5) -> Tuple[np.ndarray, float]:
        current_cost = np.sum(self.residual_func(params)**2)
        step_size = 1.0
        
        for i in range(max_backtrack):
            new_params = params + step_size * delta
            new_params = self.apply_bounds(new_params)
            new_cost = np.sum(self.residual_func(new_params)**2)
            
            if new_cost < current_cost:
                return new_params, step_size
            
            step_size *= alpha
        
        return params, 0.0
    
    def optimize(self, max_iter: int = 100, cost_tol: float = 1e-6,
                param_tol: float = 1e-8, grad_tol: float = 1e-6,
                initial_damping: float = 0.1, verbose: bool = True) -> Dict:
        current_params = self.params.copy()
        current_cost = np.sum(self.residual_func(current_params)**2)
        damping = initial_damping
        iteration = 0
        if verbose:
            print(f"\n{'Итер':>4} {'Норма град':>12} "
                  f"{'Норма шага':>12}")
        
        for iteration in range(max_iter):
            self.history['params'].append(current_params.copy())
            self.history['cost'].append(current_cost)
            
            delta, J, r = self.gauss_newton_step(current_params, damping)
            
            gradient = 2.0 * J.T @ r
            grad_norm = np.linalg.norm(gradient)
            
            new_params, step_size = self.line_search(current_params, delta)
            
            new_cost = np.sum(self.residual_func(new_params)**2)
            cost_reduction = current_cost - new_cost
            
            if cost_reduction > 0:
                damping = max(damping * 0.5, 1e-10)
                current_params = new_params
                current_cost = new_cost
                success = True
            else:
                damping = min(damping * 2.0, 1e6)
                success = False
            
            if verbose:
                print(f"{iteration:4d}  {grad_norm:12.6e} "
                      f"{np.linalg.norm(delta):12.6e}")
            
            if iteration > 0:
                if abs(cost_reduction) < cost_tol * (1 + abs(current_cost)):
                    break
                
                param_change = np.linalg.norm(delta)
                if param_change < param_tol * (1 + np.linalg.norm(current_params)):
                    break
                
                if grad_norm < grad_tol:
                    break
        
        final_cost = np.sum(self.residual_func(current_params)**2)
        final_residuals = self.residual_func(current_params)
        
        print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
        print(f"Итераций выполнено: {iteration+1}")
        print(f"RMS невязок: {np.sqrt(np.mean(final_residuals**2)):.3f}")
        print(f"Норма градиента: {grad_norm:.6e}")
        
        return {
            'success': True,
            'params': current_params,
            'cost': final_cost,
            'residuals': final_residuals,
            'grad_norm': grad_norm,
            'iterations': iteration + 1,
            'history': self.history
        }
    
    def compute_covariance_matrix(self, params: np.ndarray) -> np.ndarray:
        residuals = self.residual_func(params)
        n_obs = len(residuals)
        
        sigma2 = np.sum(residuals**2) / (n_obs - self.n_params)
        
        J = self.compute_jacobian_finite_difference(params)
        
        try:
            cov = sigma2 * np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            cov = sigma2 * np.linalg.pinv(J.T @ J)
        
        return cov
