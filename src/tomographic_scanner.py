"""
Warp-Pulse Tomographic Scanner Module
===================================

Implements iterative tomographic reconstruction using Algebraic Reconstruction Technique (ART).

Mathematical Foundation:
δn^(k+1) = δn^(k) + λ * (φ - R{δn^(k)}) / ||R_i||²

Where:
- δn: Refractive index perturbation  
- φ: Measured phase shift
- R: Radon transform operator (forward projection)
- λ: Relaxation parameter
- k: Iteration index
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
import logging
import time
import matplotlib.pyplot as plt

@dataclass
class TomographyParams:
    """Parameters for tomographic reconstruction"""
    grid_size: int = 128                    # Reconstruction grid size
    domain_size: float = 10.0               # Physical domain size (m)
    n_angles: int = 180                     # Number of projection angles
    n_detectors: int = 256                  # Number of detectors per angle
    frequency: float = 2.4e12               # Probing frequency (Hz)
    c_s: float = 5e8                        # Subspace wave speed (m/s)
    
    # ART parameters
    n_iterations: int = 20                  # Number of ART iterations
    relaxation_factor: float = 0.1          # λ relaxation parameter
    convergence_threshold: float = 1e-6     # Convergence criterion
    
    # Filtering parameters
    filter_type: str = "ram-lak"            # Filter for FBP
    filter_cutoff: float = 0.8              # Normalized cutoff frequency
    noise_variance: float = 1e-8            # Measurement noise variance

class WarpTomographicImager:
    """
    Tomographic imager for warp field characterization using phase measurements.
    """
    
    def __init__(self, params: TomographyParams):
        """
        Initialize the tomographic imager.
        
        Args:
            params: Tomography parameters
        """
        self.params = params
        self.logger = logging.getLogger(__name__)
        
        # Initialize coordinate grids
        self.x = np.linspace(-params.domain_size/2, params.domain_size/2, params.grid_size)
        self.y = np.linspace(-params.domain_size/2, params.domain_size/2, params.grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Projection angles
        self.angles = np.linspace(0, np.pi, params.n_angles, endpoint=False)
        
        # Detector coordinates
        self.detector_coords = np.linspace(-params.domain_size/2, params.domain_size/2, params.n_detectors)
        
        # Storage for measurements
        self.sinogram = np.zeros((params.n_angles, params.n_detectors))
        self.phi_dict = {}  # Phase measurements by angle
        
        # Reconstruction storage
        self.delta_n = np.zeros((params.grid_size, params.grid_size))
        self.fbp_reconstruction = None
        self.art_reconstruction = None
        
        self.logger.info(f"Initialized tomographic imager with {params.grid_size}x{params.grid_size} grid")
    
    def simulate_phantom(self, phantom_type: str = "warp_bubble") -> np.ndarray:
        """
        Generate a synthetic phantom for testing.
        
        Args:
            phantom_type: Type of phantom to generate
            
        Returns:
            Phantom refractive index perturbation
        """
        if phantom_type == "warp_bubble":
            # Alcubierre-like warp bubble
            r = np.sqrt(self.X**2 + self.Y**2)
            R_s = 2.0  # Bubble radius
            sigma = 0.8  # Transition width
            
            # Warp bubble profile with negative energy density
            delta_n = -0.01 * np.exp(-(r - R_s)**2 / (2*sigma**2))
            # Add positive rim
            delta_n += 0.005 * np.exp(-(r - R_s - sigma)**2 / (2*(sigma/2)**2))
            
        elif phantom_type == "gaussian_cluster":
            # Multiple Gaussian perturbations
            centers = [(-2, -2), (2, 2), (-2, 2), (0, 0)]
            delta_n = np.zeros_like(self.X)
            for i, (cx, cy) in enumerate(centers):
                amp = 0.005 * (1 + 0.5*np.sin(i))
                sigma = 0.8 + 0.3*np.cos(i)
                delta_n += amp * np.exp(-((self.X - cx)**2 + (self.Y - cy)**2)/(2*sigma**2))
                
        elif phantom_type == "shepp_logan":
            # Modified Shepp-Logan phantom
            delta_n = self._generate_shepp_logan()
            
        else:
            # Simple circular phantom
            r = np.sqrt(self.X**2 + self.Y**2)
            delta_n = 0.01 * (r < 3.0).astype(float)
        
        return delta_n
    
    def _generate_shepp_logan(self) -> np.ndarray:
        """Generate a modified Shepp-Logan phantom."""
        delta_n = np.zeros_like(self.X)
        
        # Define ellipses: (center_x, center_y, a, b, angle, amplitude)
        ellipses = [
            (0, 0, 4.6, 3.45, 0, 1.0),      # Main ellipse
            (0, -0.6, 4.14, 3.105, 0, -0.8), # Large inner ellipse
            (1.5, -0.6, 1.61, 0.41, 108, -0.2), # Right ellipse
            (-1.5, -0.6, 1.61, 0.41, 72, -0.2), # Left ellipse
            (0, 1.0, 2.3, 0.46, 0, 0.1),    # Top ellipse
            (0, 1.5, 0.46, 0.23, 0, 0.1),   # Small top ellipse
            (-0.8, -1.8, 0.46, 0.23, 0, 0.1), # Bottom left
            (-0.6, -1.4, 0.23, 0.23, 0, 0.1), # Bottom left small
            (0.6, -1.4, 0.23, 0.115, 0, 0.1), # Bottom right
            (0, -3.8, 0.69, 0.23, 90, 0.1)  # Bottom
        ]
        
        for cx, cy, a, b, angle, amp in ellipses:
            # Rotate coordinates
            theta = np.radians(angle)
            x_rot = (self.X - cx) * np.cos(theta) + (self.Y - cy) * np.sin(theta)
            y_rot = -(self.X - cx) * np.sin(theta) + (self.Y - cy) * np.cos(theta)
            
            # Ellipse equation
            ellipse = (x_rot/a)**2 + (y_rot/b)**2 <= 1
            delta_n += amp * ellipse * 0.01  # Scale to reasonable refractive index change
            
        return delta_n
    
    def forward_project(self, delta_n: np.ndarray, angle: float) -> np.ndarray:
        """
        Compute forward projection (Radon transform) for given angle.
        
        Args:
            delta_n: Refractive index perturbation field
            angle: Projection angle in radians
            
        Returns:
            Projection data (line integrals)
        """
        projection = np.zeros(self.params.n_detectors)
        
        # Rotation matrix
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        for i, s in enumerate(self.detector_coords):
            # Ray equation: parametric line through rotated coordinates
            # x = s*cos(θ) - t*sin(θ)
            # y = s*sin(θ) + t*cos(θ)
            
            # Integrate along ray using trapezoidal rule
            t_vals = np.linspace(-self.params.domain_size, self.params.domain_size, 500)
            x_ray = s * cos_a - t_vals * sin_a
            y_ray = s * sin_a + t_vals * cos_a
            
            # Interpolate delta_n values along ray
            points = np.column_stack([self.X.ravel(), self.Y.ravel()])
            values = delta_n.ravel()
            ray_points = np.column_stack([x_ray, y_ray])
            
            # Only interpolate points within domain
            valid_mask = (np.abs(x_ray) <= self.params.domain_size/2) & \
                        (np.abs(y_ray) <= self.params.domain_size/2)
            
            if np.any(valid_mask):
                ray_values = griddata(points, values, ray_points[valid_mask], 
                                    method='linear', fill_value=0.0)
                projection[i] = np.trapz(ray_values, t_vals[valid_mask])
        
        return projection
    
    def collect_data(self, phantom: Optional[np.ndarray] = None) -> Dict:
        """
        Collect tomographic data (sinogram) from phantom or real measurements.
        
        Args:
            phantom: Optional phantom to use for simulation
            
        Returns:
            Collection results dictionary
        """
        start_time = time.time()
        
        if phantom is None:
            phantom = self.simulate_phantom("warp_bubble")
        
        # Collect projections for all angles
        for i, angle in enumerate(self.angles):
            projection = self.forward_project(phantom, angle)
            
            # Add noise to simulate real measurements
            noise = np.random.normal(0, np.sqrt(self.params.noise_variance), 
                                   projection.shape)
            projection += noise
            
            self.sinogram[i, :] = projection
            
            # Convert to phase measurements
            k = 2 * np.pi * self.params.frequency / self.params.c_s
            self.phi_dict[angle] = {
                'phase_shifts': k * projection,
                'amplitude': np.ones_like(projection),
                'noise_level': np.std(noise)
            }
        
        collection_time = time.time() - start_time
        
        results = {
            'sinogram': self.sinogram,
            'phantom_truth': phantom,
            'collection_time': collection_time,
            'n_projections': len(self.angles),
            'noise_variance': self.params.noise_variance
        }
        
        self.logger.info(f"Data collection completed in {collection_time:.2f}s")
        return results
    
    def filtered_backprojection(self) -> np.ndarray:
        """
        Perform filtered backprojection reconstruction.
        
        Returns:
            Reconstructed image
        """
        # Apply ramp filter in frequency domain
        n_det = self.params.n_detectors
        freq = np.fft.fftfreq(n_det)
        
        # Ramp filter
        if self.params.filter_type == "ram-lak":
            filter_kernel = np.abs(freq)
        elif self.params.filter_type == "shepp-logan":
            filter_kernel = np.abs(freq) * np.sinc(freq / (2 * self.params.filter_cutoff))
        elif self.params.filter_type == "cosine":
            filter_kernel = np.abs(freq) * np.cos(np.pi * freq / (2 * self.params.filter_cutoff))
        else:
            filter_kernel = np.abs(freq)  # Default to ram-lak
        
        # Apply cutoff
        filter_kernel[np.abs(freq) > self.params.filter_cutoff] = 0
        
        # Filter projections
        filtered_sinogram = np.zeros_like(self.sinogram)
        for i in range(self.params.n_angles):
            proj_fft = np.fft.fft(self.sinogram[i, :])
            filtered_proj_fft = proj_fft * filter_kernel
            filtered_sinogram[i, :] = np.real(np.fft.ifft(filtered_proj_fft))
        
        # Backproject
        reconstruction = np.zeros((self.params.grid_size, self.params.grid_size))
        
        for i, angle in enumerate(self.angles):
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Compute detector coordinate for each image pixel
            s_coords = self.X * cos_a + self.Y * sin_a
            
            # Interpolate filtered projection values
            interp_values = np.interp(s_coords, self.detector_coords, 
                                    filtered_sinogram[i, :], left=0, right=0)
            
            reconstruction += interp_values
        
        # Normalize
        reconstruction *= np.pi / (2 * self.params.n_angles)
        
        self.fbp_reconstruction = reconstruction
        return reconstruction
    
    def algebraic_reconstruction_technique(self, initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform iterative ART reconstruction.
        
        Args:
            initial_guess: Optional initial guess for reconstruction
            
        Returns:
            ART reconstructed image
        """
        start_time = time.time()
        
        if initial_guess is None:
            delta_n = np.zeros((self.params.grid_size, self.params.grid_size))
        else:
            delta_n = initial_guess.copy()
        
        # Convergence tracking
        residuals = []
        
        for iteration in range(self.params.n_iterations):
            iter_start = time.time()
            total_residual = 0.0
            
            for i, angle in enumerate(self.angles):
                # Forward project current estimate
                current_proj = self.forward_project(delta_n, angle)
                measured_proj = self.sinogram[i, :]
                
                # Compute residual
                residual = measured_proj - current_proj
                total_residual += np.sum(residual**2)
                
                # Backproject residual with relaxation
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                s_coords = self.X * cos_a + self.Y * sin_a
                
                # Interpolate residual to image grid
                backproj_residual = np.interp(s_coords, self.detector_coords, 
                                            residual, left=0, right=0)
                
                # Normalize by ray length (approximate)
                norm_factor = np.sum(backproj_residual**2) + 1e-12
                
                # Update with relaxation
                delta_n += self.params.relaxation_factor * backproj_residual / norm_factor
            
            # Check convergence
            avg_residual = np.sqrt(total_residual / (self.params.n_angles * self.params.n_detectors))
            residuals.append(avg_residual)
            
            iter_time = time.time() - iter_start
            self.logger.info(f"ART iteration {iteration+1}/{self.params.n_iterations}: "
                           f"residual = {avg_residual:.2e}, time = {iter_time:.2f}s")
            
            if avg_residual < self.params.convergence_threshold:
                self.logger.info(f"ART converged after {iteration+1} iterations")
                break
        
        reconstruction_time = time.time() - start_time
        self.logger.info(f"ART reconstruction completed in {reconstruction_time:.2f}s")
        
        self.art_reconstruction = delta_n
        return delta_n
    
    def reconstruct_slice(self, method: str = "art") -> np.ndarray:
        """
        Reconstruct a 2D slice using specified method.
        
        Args:
            method: Reconstruction method ("art" or "fbp")
            
        Returns:
            Reconstructed image
        """
        if method == "art":
            return self.algebraic_reconstruction_technique()
        elif method == "fbp":
            return self.filtered_backprojection()
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")
    
    def run_diagnostics(self) -> Dict:
        """
        Run comprehensive diagnostics on the tomographic system.
        
        Returns:
            Diagnostics results
        """
        results = {}
        
        # Test with known phantom
        phantom = self.simulate_phantom("shepp_logan")
        collection_results = self.collect_data(phantom)
        
        # Compare reconstruction methods
        fbp_recon = self.filtered_backprojection()
        art_recon = self.algebraic_reconstruction_technique()
        
        # Compute metrics
        fbp_mse = np.mean((fbp_recon - phantom)**2)
        art_mse = np.mean((art_recon - phantom)**2)
        
        fbp_psnr = 10 * np.log10(np.max(phantom)**2 / fbp_mse)
        art_psnr = 10 * np.log10(np.max(phantom)**2 / art_mse)
        
        results = {
            'phantom_max': float(np.max(phantom)),
            'phantom_min': float(np.min(phantom)),
            'fbp_mse': float(fbp_mse),
            'art_mse': float(art_mse),
            'fbp_psnr': float(fbp_psnr),
            'art_psnr': float(art_psnr),
            'collection_time': collection_results['collection_time'],
            'n_projections': self.params.n_angles,
            'grid_size': self.params.grid_size
        }
        
        self.logger.info(f"Diagnostics: FBP PSNR = {fbp_psnr:.1f} dB, ART PSNR = {art_psnr:.1f} dB")
        return results
    
    def visualize_results(self, save_path: Optional[str] = None) -> None:
        """
        Visualize tomographic reconstruction results.
        
        Args:
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original phantom
        phantom = self.simulate_phantom("shepp_logan")
        im1 = axes[0, 0].imshow(phantom, cmap='gray', extent=[-5, 5, -5, 5])
        axes[0, 0].set_title('Original Phantom')
        axes[0, 0].set_xlabel('x (m)')
        axes[0, 0].set_ylabel('y (m)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Sinogram
        im2 = axes[0, 1].imshow(self.sinogram, cmap='gray', aspect='auto',
                               extent=[self.detector_coords[0], self.detector_coords[-1],
                                     np.degrees(self.angles[-1]), np.degrees(self.angles[0])])
        axes[0, 1].set_title('Sinogram')
        axes[0, 1].set_xlabel('Detector Position (m)')
        axes[0, 1].set_ylabel('Angle (degrees)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # FBP reconstruction
        if self.fbp_reconstruction is not None:
            im3 = axes[0, 2].imshow(self.fbp_reconstruction, cmap='gray', extent=[-5, 5, -5, 5])
            axes[0, 2].set_title('FBP Reconstruction')
            axes[0, 2].set_xlabel('x (m)')
            axes[0, 2].set_ylabel('y (m)')
            plt.colorbar(im3, ax=axes[0, 2])
        
        # ART reconstruction
        if self.art_reconstruction is not None:
            im4 = axes[1, 0].imshow(self.art_reconstruction, cmap='gray', extent=[-5, 5, -5, 5])
            axes[1, 0].set_title('ART Reconstruction')
            axes[1, 0].set_xlabel('x (m)')
            axes[1, 0].set_ylabel('y (m)')
            plt.colorbar(im4, ax=axes[1, 0])
        
        # Difference maps
        if self.fbp_reconstruction is not None:
            diff_fbp = self.fbp_reconstruction - phantom
            im5 = axes[1, 1].imshow(diff_fbp, cmap='RdBu_r', extent=[-5, 5, -5, 5])
            axes[1, 1].set_title('FBP Error')
            axes[1, 1].set_xlabel('x (m)')
            axes[1, 1].set_ylabel('y (m)')
            plt.colorbar(im5, ax=axes[1, 1])
        
        if self.art_reconstruction is not None:
            diff_art = self.art_reconstruction - phantom
            im6 = axes[1, 2].imshow(diff_art, cmap='RdBu_r', extent=[-5, 5, -5, 5])
            axes[1, 2].set_title('ART Error')
            axes[1, 2].set_xlabel('x (m)')
            axes[1, 2].set_ylabel('y (m)')
            plt.colorbar(im6, ax=axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
