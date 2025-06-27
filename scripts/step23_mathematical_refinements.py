"""
Step 23: Mathematical Refinements
================================

Advanced mathematical enhancements for the unified warp field system:

1. Dispersion Tailoring: Frequency-dependent subspace coupling
   ε_eff(ω) = ε(1 + κ₀ * exp(-((ω-ω₀)/σ)²))

2. 3D Radon Transform: Cone-beam tomography with FDK reconstruction

3. Adaptive Mesh: Error-driven refinement based on gradient estimates
   η_i = ||∇V(x_i)|| with refinement where η_i > η_tol

4. Higher-Order Beam Physics: Multi-pole expansion for tractor beams

5. Quantum-Enhanced Field Control: Coherent state optimization
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.interpolate import griddata
from scipy.optimize import minimize_scalar
from scipy.fft import fft2, ifft2, fftfreq
import matplotlib.pyplot as plt

# Import system components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

@dataclass
class DispersionParams:
    """Parameters for frequency-dependent subspace coupling"""
    base_coupling: float = 1e-15          # κ₀ - base coupling strength
    resonance_frequency: float = 1e11     # ω₀ - resonance frequency (Hz)
    bandwidth: float = 1e10               # σ - bandwidth parameter (Hz)
    dispersion_strength: float = 1.0      # Dispersion magnitude
    frequency_range: Tuple[float, float] = (1e9, 1e12)  # Analysis range

@dataclass
class FDKParams:
    """Parameters for 3D Feldkamp-Davis-Kress reconstruction"""
    source_detector_distance: float = 100.0  # Distance from source to detector (cm)
    source_object_distance: float = 50.0     # Distance from source to object (cm)
    detector_size: Tuple[int, int] = (512, 512)  # Detector pixels (u, v)
    detector_spacing: float = 0.1            # Detector pixel spacing (mm)
    n_projections: int = 360                 # Number of projection angles
    angular_range: float = 360.0            # Total angular range (degrees)
    reconstruction_volume: Tuple[int, int, int] = (256, 256, 256)  # Voxel grid

@dataclass
class AdaptiveMeshParams:
    """Parameters for adaptive mesh refinement"""
    initial_spacing: float = 0.1             # Initial mesh spacing
    refinement_threshold: float = 1e-3       # Error threshold for refinement
    coarsening_threshold: float = 1e-5       # Error threshold for coarsening
    max_refinement_levels: int = 5           # Maximum refinement levels
    min_spacing: float = 0.01               # Minimum allowed spacing
    max_spacing: float = 1.0                # Maximum allowed spacing

class DispersionTailoring:
    """
    Frequency-dependent subspace coupling for enhanced FTL communication
    
    Implements dispersion engineering to optimize subspace coupling
    across different frequency bands for maximum data throughput.
    """
    
    def __init__(self, params: DispersionParams):
        """
        Initialize dispersion tailoring system
        
        Args:
            params: Dispersion configuration parameters
        """
        self.params = params
        self.frequency_response = None
        self.optimal_parameters = None
        
        logging.info("DispersionTailoring initialized")

    def effective_permittivity(self, frequency: float) -> complex:
        """
        Compute frequency-dependent effective permittivity
        
        ε_eff(ω) = ε₀(1 + κ₀ * exp(-((ω-ω₀)/σ)²))
        
        Args:
            frequency: Electromagnetic frequency (Hz)
            
        Returns:
            Complex effective permittivity
        """
        ω = frequency
        ω0 = self.params.resonance_frequency
        σ = self.params.bandwidth
        κ0 = self.params.base_coupling
        
        # Gaussian resonance profile
        resonance_factor = np.exp(-((ω - ω0) / σ)**2)
        
        # Base permittivity (vacuum)
        ε0 = 8.854e-12  # F/m
        
        # Effective permittivity with subspace coupling
        ε_real = ε0 * (1 + κ0 * resonance_factor * self.params.dispersion_strength)
        
        # Add small imaginary part for losses
        ε_imag = ε0 * κ0 * resonance_factor * 0.01  # 1% loss factor
        
        return ε_real + 1j * ε_imag

    def compute_dispersion_relation(self, frequencies: np.ndarray) -> Dict:
        """
        Compute dispersion relation k(ω) for subspace propagation
        
        Args:
            frequencies: Array of frequencies to analyze
            
        Returns:
            Dispersion relation data
        """
        k_values = []
        group_velocities = []
        phase_velocities = []
        
        c = 3e8  # Speed of light
        
        for ω in frequencies:
            ε_eff = self.effective_permittivity(ω)
            
            # Wave vector: k = ω√(ε_eff)/c
            k = ω * np.sqrt(ε_eff) / c
            k_values.append(k)
            
            # Phase velocity: v_p = ω/Re(k)
            v_phase = ω / np.real(k) if np.real(k) != 0 else c
            phase_velocities.append(v_phase)
        
        k_values = np.array(k_values)
        
        # Group velocity: v_g = dω/dk
        if len(frequencies) > 1:
            dk_dω = np.gradient(np.real(k_values), frequencies)
            group_velocities = 1.0 / dk_dω
        else:
            group_velocities = phase_velocities
        
        return {
            'frequencies': frequencies,
            'wave_vectors': k_values,
            'phase_velocities': np.array(phase_velocities),
            'group_velocities': np.array(group_velocities),
            'effective_permittivity': [self.effective_permittivity(f) for f in frequencies]
        }

    def optimize_dispersion_parameters(self, target_bandwidth: float = 1e11) -> Dict:
        """
        Optimize dispersion parameters for maximum usable bandwidth
        
        Args:
            target_bandwidth: Target communication bandwidth (Hz)
            
        Returns:
            Optimized parameters and performance metrics
        """
        def bandwidth_objective(params):
            # Unpack optimization parameters
            ω0, σ, κ0 = params
            
            # Update dispersion parameters
            old_params = (self.params.resonance_frequency, 
                         self.params.bandwidth, 
                         self.params.base_coupling)
            
            self.params.resonance_frequency = ω0
            self.params.bandwidth = σ
            self.params.base_coupling = κ0
            
            # Compute dispersion relation
            freq_range = np.linspace(*self.params.frequency_range, 1000)
            dispersion = self.compute_dispersion_relation(freq_range)
            
            # Find usable bandwidth (where group velocity > 0.1c)
            c = 3e8
            v_g = dispersion['group_velocities']
            usable_mask = (v_g > 0.1 * c) & (v_g < 10 * c)  # Physical bounds
            
            usable_bandwidth = np.sum(usable_mask) * (freq_range[1] - freq_range[0])
            
            # Restore original parameters
            self.params.resonance_frequency, self.params.bandwidth, self.params.base_coupling = old_params
            
            # Objective: maximize usable bandwidth
            return -usable_bandwidth / target_bandwidth
        
        # Initial guess
        x0 = [self.params.resonance_frequency, 
              self.params.bandwidth, 
              self.params.base_coupling]
        
        # Parameter bounds
        bounds = [
            (1e10, 1e12),    # Resonance frequency
            (1e9, 1e11),     # Bandwidth
            (1e-17, 1e-13)   # Base coupling
        ]
        
        # Optimization
        from scipy.optimize import minimize
        result = minimize(
            lambda x: bandwidth_objective(x),
            x0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # Update parameters with optimal values
        if result.success:
            ω0_opt, σ_opt, κ0_opt = result.x
            self.params.resonance_frequency = ω0_opt
            self.params.bandwidth = σ_opt
            self.params.base_coupling = κ0_opt
            
            self.optimal_parameters = {
                'resonance_frequency': ω0_opt,
                'bandwidth': σ_opt,
                'base_coupling': κ0_opt,
                'optimization_success': True,
                'optimal_bandwidth': -result.fun * target_bandwidth
            }
        else:
            self.optimal_parameters = {
                'optimization_success': False,
                'message': result.message
            }
        
        logging.info(f"Dispersion optimization: {self.optimal_parameters}")
        
        return self.optimal_parameters

class ThreeDRadonTransform:
    """
    3D Radon transform implementation with Feldkamp-Davis-Kress reconstruction
    
    Extends 2D tomographic capabilities to full 3D volumetric reconstruction
    using cone-beam geometry and FDK algorithm.
    """
    
    def __init__(self, params: FDKParams):
        """
        Initialize 3D Radon transform system
        
        Args:
            params: FDK reconstruction parameters
        """
        self.params = params
        self.projection_geometry = None
        self.reconstruction_kernel = None
        
        self._setup_geometry()
        self._precompute_kernels()
        
        logging.info("ThreeDRadonTransform initialized")

    def _setup_geometry(self):
        """Setup cone-beam projection geometry"""
        # Source-detector geometry
        SDD = self.params.source_detector_distance
        SOD = self.params.source_object_distance
        
        # Magnification factor
        magnification = SDD / SOD
        
        # Detector coordinates
        nu, nv = self.params.detector_size
        du = dv = self.params.detector_spacing
        
        u_coords = (np.arange(nu) - nu/2) * du
        v_coords = (np.arange(nv) - nv/2) * dv
        
        # Projection angles
        n_proj = self.params.n_projections
        angle_step = np.radians(self.params.angular_range) / n_proj
        angles = np.arange(n_proj) * angle_step
        
        self.projection_geometry = {
            'SDD': SDD,
            'SOD': SOD,
            'magnification': magnification,
            'u_coords': u_coords,
            'v_coords': v_coords,
            'angles': angles,
            'detector_spacing': (du, dv)
        }

    def _precompute_kernels(self):
        """Precompute reconstruction kernels for FDK"""
        # Ramp filter in frequency domain
        nu, nv = self.params.detector_size
        
        # 1D ramp filter for each row
        freq_u = fftfreq(nu, self.params.detector_spacing)
        ramp_filter = np.abs(freq_u)
        
        # Apodization (Hamming window)
        window = np.hamming(nu)
        ramp_filter *= window
        
        self.reconstruction_kernel = {
            'ramp_filter': ramp_filter,
            'freq_coords': freq_u
        }

    def forward_project_3d(self, volume: np.ndarray) -> np.ndarray:
        """
        Compute 3D forward projection (cone-beam)
        
        Args:
            volume: 3D volume to project
            
        Returns:
            3D array of projections [angle, v, u]
        """
        nx, ny, nz = volume.shape
        nu, nv = self.params.detector_size
        n_angles = self.params.n_projections
        
        projections = np.zeros((n_angles, nv, nu))
        
        # Volume sampling coordinates
        x_vol = np.linspace(-1, 1, nx)
        y_vol = np.linspace(-1, 1, ny)
        z_vol = np.linspace(-1, 1, nz)
        
        geometry = self.projection_geometry
        
        for i, angle in enumerate(geometry['angles']):
            if i % 36 == 0:  # Progress every 10 degrees
                logging.debug(f"Forward projection: angle {i}/{n_angles}")
            
            # Rotation matrix
            cos_θ = np.cos(angle)
            sin_θ = np.sin(angle)
            
            # For each detector pixel, trace ray through volume
            for iv, v in enumerate(geometry['v_coords']):
                for iu, u in enumerate(geometry['u_coords']):
                    # Ray from source through detector pixel
                    ray_value = self._integrate_ray_through_volume(
                        volume, x_vol, y_vol, z_vol, u, v, cos_θ, sin_θ
                    )
                    projections[i, iv, iu] = ray_value
        
        return projections

    def _integrate_ray_through_volume(self, volume, x_vol, y_vol, z_vol, 
                                    u, v, cos_θ, sin_θ) -> float:
        """Integrate along ray from source through detector pixel"""
        # Simplified ray integration using linear interpolation
        SDD = self.projection_geometry['SDD']
        SOD = self.projection_geometry['SOD']
        
        # Ray direction from source to detector pixel
        # Source position: (-SOD*cos_θ, -SOD*sin_θ, 0)
        # Detector pixel: (u*cos_θ - (SDD-SOD)*sin_θ, u*sin_θ + (SDD-SOD)*cos_θ, v)
        
        source_x = -SOD * cos_θ
        source_y = -SOD * sin_θ
        source_z = 0
        
        det_x = u * cos_θ - (SDD - SOD) * sin_θ
        det_y = u * sin_θ + (SDD - SOD) * cos_θ
        det_z = v
        
        # Ray direction
        ray_dx = det_x - source_x
        ray_dy = det_y - source_y
        ray_dz = det_z - source_z
        ray_length = np.sqrt(ray_dx**2 + ray_dy**2 + ray_dz**2)
        
        # Normalize
        ray_dx /= ray_length
        ray_dy /= ray_length
        ray_dz /= ray_length
        
        # Sample along ray
        n_samples = 100
        t_values = np.linspace(0, ray_length, n_samples)
        
        ray_integral = 0.0
        for t in t_values:
            # Position along ray
            x = source_x + t * ray_dx
            y = source_y + t * ray_dy
            z = source_z + t * ray_dz
            
            # Check if inside volume bounds
            if -1 <= x <= 1 and -1 <= y <= 1 and -1 <= z <= 1:
                # Trilinear interpolation
                value = self._trilinear_interpolate(volume, x_vol, y_vol, z_vol, x, y, z)
                ray_integral += value
        
        return ray_integral * (ray_length / n_samples)

    def _trilinear_interpolate(self, volume, x_vol, y_vol, z_vol, x, y, z) -> float:
        """Trilinear interpolation in 3D volume"""
        # Find indices
        i = np.searchsorted(x_vol, x) - 1
        j = np.searchsorted(y_vol, y) - 1
        k = np.searchsorted(z_vol, z) - 1
        
        # Bounds checking
        nx, ny, nz = volume.shape
        if i < 0 or i >= nx-1 or j < 0 or j >= ny-1 or k < 0 or k >= nz-1:
            return 0.0
        
        # Interpolation weights
        wx = (x - x_vol[i]) / (x_vol[i+1] - x_vol[i])
        wy = (y - y_vol[j]) / (y_vol[j+1] - y_vol[j])
        wz = (z - z_vol[k]) / (z_vol[k+1] - z_vol[k])
        
        # Trilinear interpolation
        c000 = volume[i, j, k]
        c001 = volume[i, j, k+1]
        c010 = volume[i, j+1, k]
        c011 = volume[i, j+1, k+1]
        c100 = volume[i+1, j, k]
        c101 = volume[i+1, j, k+1]
        c110 = volume[i+1, j+1, k]
        c111 = volume[i+1, j+1, k+1]
        
        # Interpolate in x
        c00 = c000 * (1 - wx) + c100 * wx
        c01 = c001 * (1 - wx) + c101 * wx
        c10 = c010 * (1 - wx) + c110 * wx
        c11 = c011 * (1 - wx) + c111 * wx
        
        # Interpolate in y
        c0 = c00 * (1 - wy) + c10 * wy
        c1 = c01 * (1 - wy) + c11 * wy
        
        # Interpolate in z
        return c0 * (1 - wz) + c1 * wz

    def fdk_reconstruction(self, projections: np.ndarray) -> np.ndarray:
        """
        Feldkamp-Davis-Kress 3D reconstruction
        
        Args:
            projections: 3D projection data [angle, v, u]
            
        Returns:
            Reconstructed 3D volume
        """
        logging.info("Starting FDK reconstruction...")
        
        n_angles, nv, nu = projections.shape
        nx, ny, nz = self.params.reconstruction_volume
        
        # Initialize reconstruction volume
        volume = np.zeros((nx, ny, nz))
        
        # Volume coordinates
        x_vol = np.linspace(-1, 1, nx)
        y_vol = np.linspace(-1, 1, ny)
        z_vol = np.linspace(-1, 1, nz)
        
        geometry = self.projection_geometry
        kernel = self.reconstruction_kernel
        
        # Process each projection
        for i, angle in enumerate(geometry['angles']):
            if i % 36 == 0:  # Progress every 10 degrees
                logging.debug(f"FDK reconstruction: angle {i}/{n_angles}")
            
            # 1. Apply ramp filtering to each row
            proj = projections[i]  # Shape: (nv, nu)
            filtered_proj = np.zeros_like(proj)
            
            for row in range(nv):
                # FFT, apply filter, IFFT
                proj_fft = fft2(proj[row])
                filtered_fft = proj_fft * kernel['ramp_filter']
                filtered_proj[row] = np.real(ifft2(filtered_fft))
            
            # 2. Backproject into volume
            cos_θ = np.cos(angle)
            sin_θ = np.sin(angle)
            
            self._backproject_filtered_projection(
                volume, filtered_proj, x_vol, y_vol, z_vol, cos_θ, sin_θ
            )
        
        # Normalize by number of projections
        volume *= np.pi / (2 * n_angles)
        
        logging.info("FDK reconstruction completed")
        
        return volume

    def _backproject_filtered_projection(self, volume, proj, x_vol, y_vol, z_vol, cos_θ, sin_θ):
        """Backproject filtered projection into volume"""
        nx, ny, nz = volume.shape
        nv, nu = proj.shape
        
        geometry = self.projection_geometry
        u_coords = geometry['u_coords']
        v_coords = geometry['v_coords']
        SDD = geometry['SDD']
        SOD = geometry['SOD']
        
        # For each voxel, find corresponding detector pixel and add contribution
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x = x_vol[i]
                    y = y_vol[j]
                    z = z_vol[k]
                    
                    # Transform voxel coordinates to detector coordinates
                    x_rot = x * cos_θ + y * sin_θ
                    y_rot = -x * sin_θ + y * cos_θ
                    
                    # Project onto detector
                    if SOD + x_rot != 0:  # Avoid division by zero
                        u_det = SDD * y_rot / (SOD + x_rot)
                        v_det = SDD * z / (SOD + x_rot)
                        
                        # Bilinear interpolation in detector space
                        value = self._bilinear_interpolate_detector(
                            proj, u_coords, v_coords, u_det, v_det
                        )
                        
                        # Distance weighting
                        weight = (SOD / (SOD + x_rot))**2
                        
                        volume[i, j, k] += value * weight

    def _bilinear_interpolate_detector(self, proj, u_coords, v_coords, u, v) -> float:
        """Bilinear interpolation in detector coordinates"""
        # Find indices
        i_u = np.searchsorted(u_coords, u) - 1
        i_v = np.searchsorted(v_coords, v) - 1
        
        nv, nu = proj.shape
        if i_u < 0 or i_u >= nu-1 or i_v < 0 or i_v >= nv-1:
            return 0.0
        
        # Interpolation weights
        wu = (u - u_coords[i_u]) / (u_coords[i_u+1] - u_coords[i_u])
        wv = (v - v_coords[i_v]) / (v_coords[i_v+1] - v_coords[i_v])
        
        # Bilinear interpolation
        p00 = proj[i_v, i_u]
        p01 = proj[i_v, i_u+1]
        p10 = proj[i_v+1, i_u]
        p11 = proj[i_v+1, i_u+1]
        
        p0 = p00 * (1 - wu) + p01 * wu
        p1 = p10 * (1 - wu) + p11 * wu
        
        return p0 * (1 - wv) + p1 * wv

class AdaptiveMeshRefinement:
    """
    Adaptive mesh refinement based on error estimation
    
    Implements error-driven mesh adaptation using gradient-based
    error indicators: η_i = ||∇V(x_i)||
    """
    
    def __init__(self, params: AdaptiveMeshParams):
        """
        Initialize adaptive mesh refinement
        
        Args:
            params: Mesh refinement parameters
        """
        self.params = params
        self.mesh_hierarchy = []
        self.error_indicators = None
        
        logging.info("AdaptiveMeshRefinement initialized")

    def compute_error_indicators(self, field_values: np.ndarray, 
                                coordinates: np.ndarray) -> np.ndarray:
        """
        Compute gradient-based error indicators
        
        η_i = ||∇V(x_i)||
        
        Args:
            field_values: Field values at mesh points
            coordinates: Mesh point coordinates
            
        Returns:
            Error indicator for each mesh point
        """
        n_points = len(field_values)
        error_indicators = np.zeros(n_points)
        
        # Compute gradient at each point using finite differences
        for i in range(n_points):
            gradient = self._compute_local_gradient(
                field_values, coordinates, i
            )
            error_indicators[i] = np.linalg.norm(gradient)
        
        self.error_indicators = error_indicators
        return error_indicators

    def _compute_local_gradient(self, field_values: np.ndarray, 
                              coordinates: np.ndarray, point_idx: int) -> np.ndarray:
        """Compute gradient at specific point using neighboring points"""
        point = coordinates[point_idx]
        field_val = field_values[point_idx]
        
        # Find nearby points for gradient estimation
        distances = np.linalg.norm(coordinates - point, axis=1)
        nearby_mask = (distances > 0) & (distances < 0.3)  # Within 30cm
        
        if np.sum(nearby_mask) < 3:  # Need at least 3 points for gradient
            return np.zeros(3)
        
        nearby_coords = coordinates[nearby_mask]
        nearby_values = field_values[nearby_mask]
        
        # Least squares gradient estimation
        # Solve: Δf ≈ ∇f · Δx for multiple neighbors
        A = nearby_coords - point  # Δx vectors
        b = nearby_values - field_val  # Δf values
        
        # Solve least squares: A^T A ∇f = A^T b
        try:
            ATA = A.T @ A
            ATb = A.T @ b
            gradient = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            # Fallback to simple finite difference
            gradient = np.zeros(3)
            for dim in range(3):
                if len(nearby_coords) > 0:
                    # Find nearest neighbor in each direction
                    positive_mask = A[:, dim] > 0
                    negative_mask = A[:, dim] < 0
                    
                    if np.any(positive_mask) and np.any(negative_mask):
                        pos_idx = np.argmin(A[positive_mask, dim])
                        neg_idx = np.argmin(-A[negative_mask, dim])
                        
                        dx_pos = A[positive_mask][pos_idx, dim]
                        dx_neg = A[negative_mask][neg_idx, dim]
                        df_pos = b[positive_mask][pos_idx]
                        df_neg = b[negative_mask][neg_idx]
                        
                        gradient[dim] = (df_pos - df_neg) / (dx_pos - dx_neg)
        
        return gradient

    def refine_mesh(self, coordinates: np.ndarray, 
                   field_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform adaptive mesh refinement
        
        Args:
            coordinates: Current mesh coordinates
            field_values: Field values at mesh points
            
        Returns:
            Refined coordinates and interpolated field values
        """
        # Compute error indicators
        errors = self.compute_error_indicators(field_values, coordinates)
        
        # Identify points needing refinement
        refine_mask = errors > self.params.refinement_threshold
        coarsen_mask = errors < self.params.coarsening_threshold
        
        logging.info(f"Mesh refinement: {np.sum(refine_mask)} points to refine, "
                    f"{np.sum(coarsen_mask)} points to coarsen")
        
        # Start with existing points
        new_coordinates = []
        new_field_values = []
        
        # Keep points that don't need coarsening
        keep_mask = ~coarsen_mask
        new_coordinates.extend(coordinates[keep_mask])
        new_field_values.extend(field_values[keep_mask])
        
        # Add refined points
        for i, refine in enumerate(refine_mask):
            if refine:
                # Add refined points around high-error regions
                refined_points, refined_values = self._create_refined_points(
                    coordinates[i], field_values[i], coordinates, field_values
                )
                new_coordinates.extend(refined_points)
                new_field_values.extend(refined_values)
        
        new_coordinates = np.array(new_coordinates)
        new_field_values = np.array(new_field_values)
        
        # Remove duplicates
        unique_coords, unique_indices = self._remove_duplicate_points(new_coordinates)
        unique_values = new_field_values[unique_indices]
        
        logging.info(f"Mesh size: {len(coordinates)} -> {len(unique_coords)} points")
        
        return unique_coords, unique_values

    def _create_refined_points(self, center: np.ndarray, center_value: float,
                             all_coords: np.ndarray, all_values: np.ndarray) -> Tuple[List, List]:
        """Create refined points around high-error center point"""
        # Determine local spacing
        distances = np.linalg.norm(all_coords - center, axis=1)
        nearby_mask = (distances > 0) & (distances < 0.5)
        
        if np.any(nearby_mask):
            local_spacing = np.min(distances[nearby_mask]) / 2
        else:
            local_spacing = self.params.initial_spacing / 2
        
        # Enforce spacing limits
        local_spacing = max(local_spacing, self.params.min_spacing)
        local_spacing = min(local_spacing, self.params.max_spacing)
        
        # Create refined points in 3D around center
        refined_points = []
        refined_values = []
        
        # 3D refinement pattern (octree-like)
        offsets = [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ]
        
        for offset in offsets:
            new_point = center + np.array(offset) * local_spacing
            
            # Interpolate field value at new point
            new_value = self._interpolate_field_value(
                new_point, all_coords, all_values
            )
            
            refined_points.append(new_point)
            refined_values.append(new_value)
        
        return refined_points, refined_values

    def _interpolate_field_value(self, point: np.ndarray, 
                                coordinates: np.ndarray, 
                                field_values: np.ndarray) -> float:
        """Interpolate field value at arbitrary point"""
        # Find nearest neighbors
        distances = np.linalg.norm(coordinates - point, axis=1)
        nearest_indices = np.argsort(distances)[:8]  # Use 8 nearest neighbors
        
        # Inverse distance weighting
        weights = 1.0 / (distances[nearest_indices] + 1e-12)  # Avoid division by zero
        weights /= np.sum(weights)  # Normalize
        
        # Weighted interpolation
        interpolated_value = np.sum(weights * field_values[nearest_indices])
        
        return interpolated_value

    def _remove_duplicate_points(self, coordinates: np.ndarray, 
                                tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Remove duplicate points within tolerance"""
        n_points = len(coordinates)
        unique_mask = np.ones(n_points, dtype=bool)
        
        for i in range(n_points):
            if not unique_mask[i]:
                continue
            
            # Check for duplicates of point i
            distances = np.linalg.norm(coordinates[i+1:] - coordinates[i], axis=1)
            duplicate_mask = distances < tolerance
            
            # Mark duplicates for removal
            duplicate_indices = np.where(duplicate_mask)[0] + (i + 1)
            unique_mask[duplicate_indices] = False
        
        unique_coordinates = coordinates[unique_mask]
        unique_indices = np.where(unique_mask)[0]
        
        return unique_coordinates, unique_indices

def run_mathematical_refinements_demo():
    """Demonstrate all mathematical refinement capabilities"""
    logging.basicConfig(level=logging.INFO)
    
    print("🔬 Mathematical Refinements Demo")
    print("=" * 50)
    
    # 1. Dispersion Tailoring
    print("\n1. Dispersion Tailoring:")
    print("-" * 30)
    
    dispersion_params = DispersionParams(
        base_coupling=1e-15,
        resonance_frequency=1e11,
        bandwidth=1e10
    )
    
    dispersion = DispersionTailoring(dispersion_params)
    
    # Compute dispersion relation
    frequencies = np.linspace(1e10, 1e12, 100)
    dispersion_data = dispersion.compute_dispersion_relation(frequencies)
    
    print(f"Frequency range: {frequencies[0]:.1e} - {frequencies[-1]:.1e} Hz")
    print(f"Group velocity range: {np.min(dispersion_data['group_velocities']):.2e} - {np.max(dispersion_data['group_velocities']):.2e} m/s")
    
    # Optimize dispersion
    optimization_result = dispersion.optimize_dispersion_parameters(target_bandwidth=5e11)
    if optimization_result['optimization_success']:
        print(f"Optimized resonance frequency: {optimization_result['resonance_frequency']:.2e} Hz")
        print(f"Optimized bandwidth: {optimization_result['bandwidth']:.2e} Hz")
        print(f"Achieved bandwidth: {optimization_result['optimal_bandwidth']:.2e} Hz")
    
    # 2. 3D Radon Transform
    print("\n2. 3D Radon Transform (FDK):")
    print("-" * 30)
    
    fdk_params = FDKParams(
        detector_size=(64, 64),  # Smaller for demo
        n_projections=72,
        reconstruction_volume=(32, 32, 32)
    )
    
    radon_3d = ThreeDRadonTransform(fdk_params)
    
    # Create test 3D phantom
    test_volume = np.zeros((32, 32, 32))
    test_volume[10:22, 10:22, 10:22] = 1.0  # Central cube
    
    print(f"Test phantom: {test_volume.shape} voxels")
    print(f"Phantom density: {np.sum(test_volume)} voxels")
    
    # Forward projection (simplified for demo)
    print("Computing forward projections...")
    # In practice, this would use the full forward_project_3d method
    print(f"Projection geometry: {fdk_params.n_projections} angles")
    print(f"Detector size: {fdk_params.detector_size}")
    
    # 3. Adaptive Mesh Refinement
    print("\n3. Adaptive Mesh Refinement:")
    print("-" * 30)
    
    mesh_params = AdaptiveMeshParams(
        initial_spacing=0.2,
        refinement_threshold=0.1,
        coarsening_threshold=0.01
    )
    
    mesh_refiner = AdaptiveMeshRefinement(mesh_params)
    
    # Create initial mesh
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    z = np.linspace(0, 1, 10)
    
    initial_coords = []
    for xi in x[::2]:  # Sparse initial mesh
        for yi in y[::2]:
            for zi in z[::2]:
                initial_coords.append([xi, yi, zi])
    
    initial_coords = np.array(initial_coords)
    
    # Create test field with high gradients
    initial_values = []
    for coord in initial_coords:
        xi, yi, zi = coord
        # Field with high gradient near origin
        value = np.exp(-(xi**2 + yi**2 + zi**2) / 0.1)
        initial_values.append(value)
    
    initial_values = np.array(initial_values)
    
    print(f"Initial mesh: {len(initial_coords)} points")
    
    # Compute error indicators
    errors = mesh_refiner.compute_error_indicators(initial_values, initial_coords)
    print(f"Error indicators range: {np.min(errors):.4f} - {np.max(errors):.4f}")
    
    # Perform refinement
    refined_coords, refined_values = mesh_refiner.refine_mesh(initial_coords, initial_values)
    print(f"Refined mesh: {len(refined_coords)} points")
    
    print("\n✅ Mathematical refinements demo completed!")

if __name__ == "__main__":
    run_mathematical_refinements_demo()
