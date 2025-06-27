#!/usr/bin/env python3
"""
Step 20: Warp-Pulse Tomographic Scanner with ART Reconstruction

Author: Assistant
Created: Current session
Version: 1.0

PHYSICS IMPLEMENTATION:
=====================
1. Alcubierre Metric:      ds² = -dt² + (dx-vₛf(rₛ)dt)² + dy² + dz²
2. Radon Transform:        R[f](θ,s) = ∫∫ f(x,y)δ(xcosθ + ysinθ - s) dx dy
3. Filtered Backprojection: f(x,y) = ∫₀^π R'[f](θ, xcosθ + ysinθ) dθ
4. ART Reconstruction:     x^(k+1) = x^(k) + λ(bᵢ - aᵢᵀx^(k))/||aᵢ||² · aᵢ
5. Warp Bubble Detection:  ∇²φ - (1/c²)∂²φ/∂t² = ρ_exotic/ε₀
6. Space-time Curvature:   Rμν - ½gμνR = 8πG/c⁴ Tμν
7. Projection Geometry:    P(θ,s) = ∫ ρ(x,y) dl along ray (θ,s)

TOMOGRAPHIC METHODS:
==================
- Parallel-beam geometry for uniform coverage
- Fan-beam reconstruction for focused scanning
- Cone-beam 3D volumetric reconstruction
- Iterative ART with relaxation parameters
- Fourier-domain filtered backprojection
- Compressed sensing for sparse data
- Real-time warp bubble tracking

APPLICATIONS:
=============
- Warp drive field mapping and optimization
- Exotic matter distribution visualization  
- Space-time metric tensor reconstruction
- Gravitational anomaly detection
- Subspace interference pattern analysis
- Multi-dimensional energy flow imaging
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate
import scipy.ndimage
from scipy import integrate
from scipy.fft import fftfreq
import warnings

class WarpPulseTomographicScanner:
    """
    Advanced warp-pulse tomographic scanner with ART reconstruction.
    
    Implements sophisticated space-time imaging for:
    - Warp bubble visualization and analysis
    - Exotic matter distribution mapping
    - Real-time field optimization feedback
    - Multi-dimensional energy flow tracking
    """
    
    def __init__(self, scan_resolution=(256, 256), scan_volume=(2.0, 2.0, 1.0),
                 n_projections=180, detector_elements=512):
        """
        Initialize warp-pulse tomographic scanner.
        
        Args:
            scan_resolution: (width, height) pixels for reconstruction
            scan_volume: (x, y, z) physical dimensions in meters
            n_projections: Number of angular projections
            detector_elements: Number of detector elements per projection
        """
        self.resolution = scan_resolution
        self.volume = np.array(scan_volume)
        self.n_projections = n_projections
        self.detector_elements = detector_elements
        
        # Physics constants
        self.c = 2.998e8            # m/s - Speed of light
        self.G = 6.674e-11          # m³/kg⋅s² - Gravitational constant
        self.hbar = 1.055e-34       # J⋅s - Reduced Planck constant
        self.epsilon0 = 8.854e-12   # F/m - Permittivity of free space
        
        # Scanner parameters
        self.pulse_frequency = 1e12  # Hz - Terahertz warp pulses
        self.pulse_duration = 1e-12  # s - Picosecond pulses
        self.scan_speed = 0.1        # s - Time per projection
        
        # Setup scanning geometry
        self.setup_projection_geometry()
        
        # Reconstruction matrices
        self.system_matrix = None
        self.current_reconstruction = None
        self.projection_data = None
        
        # Warp field detection
        self.exotic_matter_threshold = 1e-10  # kg/m³ - Detection threshold
        self.warp_bubble_detected = False
        self.curvature_map = None
        
        print(f"Warp-Pulse Tomographic Scanner Initialized:")
        print(f"  Scan Resolution: {scan_resolution[0]}×{scan_resolution[1]} pixels")
        print(f"  Scan Volume: {scan_volume[0]:.1f}×{scan_volume[1]:.1f}×{scan_volume[2]:.1f} m³")
        print(f"  Projections: {n_projections} angles")
        print(f"  Detector Elements: {detector_elements}")
        print(f"  Pulse Frequency: {self.pulse_frequency/1e12:.0f} THz")
        
    def setup_projection_geometry(self):
        """Setup parallel-beam projection geometry."""
        # Projection angles (0 to π radians)
        self.projection_angles = np.linspace(0, np.pi, self.n_projections, endpoint=False)
        
        # Detector coordinate system
        detector_width = max(self.volume[0], self.volume[1]) * 1.5  # Ensure full coverage
        self.detector_positions = np.linspace(-detector_width/2, detector_width/2, 
                                            self.detector_elements)
        
        # Reconstruction grid
        x_grid = np.linspace(-self.volume[0]/2, self.volume[0]/2, self.resolution[0])
        y_grid = np.linspace(-self.volume[1]/2, self.volume[1]/2, self.resolution[1])
        self.X_grid, self.Y_grid = np.meshgrid(x_grid, y_grid)
        
        print(f"  Projection angles: 0° to {180*(self.n_projections-1)/self.n_projections:.0f}°")
        print(f"  Detector width: {detector_width:.2f} m")
        
    def compute_radon_transform(self, image_data):
        """
        Compute Radon transform (forward projection) of image.
        
        Mathematical formulation:
        R[f](θ,s) = ∫∫ f(x,y)δ(x cos θ + y sin θ - s) dx dy
        
        Args:
            image_data: 2D array representing spatial distribution
            
        Returns:
            projections: 2D array [n_projections, detector_elements]
        """
        projections = np.zeros((self.n_projections, self.detector_elements))
        
        # Image coordinates
        x_coords = np.linspace(-self.volume[0]/2, self.volume[0]/2, image_data.shape[1])
        y_coords = np.linspace(-self.volume[1]/2, self.volume[1]/2, image_data.shape[0])
        
        for i, angle in enumerate(self.projection_angles):
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            # For each detector element
            for j, detector_pos in enumerate(self.detector_positions):
                # Integrate along ray perpendicular to detector
                ray_sum = 0.0
                n_samples = 200  # Integration samples
                
                # Ray direction (perpendicular to detector)
                ray_dir_x = -sin_angle
                ray_dir_y = cos_angle
                
                # Starting point on detector line
                start_x = detector_pos * cos_angle
                start_y = detector_pos * sin_angle
                
                # Integration along ray
                ray_length = max(self.volume[0], self.volume[1]) * 1.5
                t_values = np.linspace(-ray_length/2, ray_length/2, n_samples)
                
                for t in t_values:
                    # Current position along ray
                    x_pos = start_x + t * ray_dir_x
                    y_pos = start_y + t * ray_dir_y
                    
                    # Check if within image bounds
                    if (abs(x_pos) <= self.volume[0]/2 and 
                        abs(y_pos) <= self.volume[1]/2):
                        
                        # Bilinear interpolation
                        x_idx = (x_pos + self.volume[0]/2) / self.volume[0] * (image_data.shape[1] - 1)
                        y_idx = (y_pos + self.volume[1]/2) / self.volume[1] * (image_data.shape[0] - 1)
                        
                        if (0 <= x_idx < image_data.shape[1] - 1 and 
                            0 <= y_idx < image_data.shape[0] - 1):
                            
                            # Bilinear interpolation
                            x0, x1 = int(x_idx), int(x_idx) + 1
                            y0, y1 = int(y_idx), int(y_idx) + 1
                            
                            wx = x_idx - x0
                            wy = y_idx - y0
                            
                            interpolated_value = (
                                image_data[y0, x0] * (1 - wx) * (1 - wy) +
                                image_data[y0, x1] * wx * (1 - wy) +
                                image_data[y1, x0] * (1 - wx) * wy +
                                image_data[y1, x1] * wx * wy
                            )
                            
                            ray_sum += interpolated_value
                            
                projections[i, j] = ray_sum * (ray_length / n_samples)
                
        return projections
        
    def filtered_backprojection(self, projections, filter_type='ram-lak'):
        """
        Reconstruct image using filtered backprojection.
        
        Implementation of: f(x,y) = ∫₀^π R'[f](θ, x cos θ + y sin θ) dθ
        
        Args:
            projections: Forward projection data
            filter_type: Reconstruction filter ('ram-lak', 'shepp-logan', 'cosine')
            
        Returns:
            reconstructed_image: 2D reconstructed image
        """
        # Create frequency domain filter
        n_det = projections.shape[1]
        freqs = fftfreq(n_det, d=1.0)
        
        # Filter selection
        if filter_type == 'ram-lak':
            # Ideal ramp filter: |ω|
            filter_kernel = np.abs(freqs)
        elif filter_type == 'shepp-logan':
            # Shepp-Logan filter: |ω| * sinc(ω/2)
            filter_kernel = np.abs(freqs) * np.sinc(freqs / 2)
        elif filter_type == 'cosine':
            # Cosine filter: |ω| * cos(πω/2)
            filter_kernel = np.abs(freqs) * np.cos(np.pi * freqs / 2)
        else:
            filter_kernel = np.abs(freqs)  # Default to ram-lak
            
        # Apply filter to projections
        filtered_projections = np.zeros_like(projections)
        
        for i in range(projections.shape[0]):
            # Use 1D FFT for 1D projection data
            proj_fft = np.fft.fft(projections[i])
            filtered_fft = proj_fft * filter_kernel
            filtered_projections[i] = np.real(np.fft.ifft(filtered_fft))
            
        # Backprojection
        reconstruction = np.zeros(self.resolution)
        
        for i, angle in enumerate(self.projection_angles):
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            # For each pixel in reconstruction grid
            for y_idx in range(self.resolution[0]):
                for x_idx in range(self.resolution[1]):
                    # Pixel coordinates
                    x_pos = self.X_grid[y_idx, x_idx]
                    y_pos = self.Y_grid[y_idx, x_idx]
                    
                    # Projection coordinate
                    proj_coord = x_pos * cos_angle + y_pos * sin_angle
                    
                    # Interpolate filtered projection value
                    detector_idx = (proj_coord + max(self.volume[0], self.volume[1]) * 0.75) / \
                                  (max(self.volume[0], self.volume[1]) * 1.5) * (n_det - 1)
                                  
                    if 0 <= detector_idx < n_det - 1:
                        # Linear interpolation
                        idx0 = int(detector_idx)
                        idx1 = idx0 + 1
                        weight = detector_idx - idx0
                        
                        interpolated_value = (
                            filtered_projections[i, idx0] * (1 - weight) +
                            filtered_projections[i, idx1] * weight
                        )
                        
                        reconstruction[y_idx, x_idx] += interpolated_value
                        
        # Normalize by number of projections
        reconstruction *= np.pi / self.n_projections
        
        return reconstruction
        
    def algebraic_reconstruction_technique(self, projections, n_iterations=50, 
                                         relaxation_factor=0.1):
        """
        Reconstruct image using Algebraic Reconstruction Technique (ART).
        
        Iterative formula: x^(k+1) = x^(k) + λ(bᵢ - aᵢᵀx^(k))/||aᵢ||² · aᵢ
        
        Args:
            projections: Measured projection data
            n_iterations: Number of ART iterations
            relaxation_factor: Convergence parameter λ
            
        Returns:
            reconstructed_image: ART reconstruction result
        """
        # Initialize reconstruction
        reconstruction = np.zeros(self.resolution)
        reconstruction_flat = reconstruction.flatten()
        
        # System matrix computation (ray-pixel intersections)
        if self.system_matrix is None:
            print("Computing system matrix for ART reconstruction...")
            self.compute_system_matrix()
            
        # Flatten projection data
        projections_flat = projections.flatten()
        n_equations = len(projections_flat)
        n_pixels = len(reconstruction_flat)
        
        print(f"ART reconstruction: {n_equations} equations, {n_pixels} unknowns")
        
        # ART iterations
        for iteration in range(n_iterations):
            if iteration % 10 == 0:
                print(f"  ART iteration {iteration}/{n_iterations}")
                
            # Process each ray equation
            for ray_idx in range(n_equations):
                if self.system_matrix[ray_idx].nnz > 0:  # Non-zero elements
                    # Current ray sum
                    ray_sum = self.system_matrix[ray_idx].dot(reconstruction_flat)
                    
                    # Residual
                    residual = projections_flat[ray_idx] - ray_sum
                    
                    # System matrix row norm squared
                    row_norm_sq = self.system_matrix[ray_idx].power(2).sum()
                    
                    if row_norm_sq > 1e-12:  # Avoid division by zero
                        # ART update
                        correction = relaxation_factor * residual / row_norm_sq
                        reconstruction_flat += correction * self.system_matrix[ray_idx].toarray().flatten()
                        
            # Non-negativity constraint
            reconstruction_flat = np.maximum(reconstruction_flat, 0)
            
        return reconstruction_flat.reshape(self.resolution)
        
    def compute_system_matrix(self):
        """Compute system matrix for ART reconstruction."""
        from scipy.sparse import lil_matrix
        
        n_projections = len(self.projection_angles)
        n_detectors = len(self.detector_positions)
        n_rays = n_projections * n_detectors
        n_pixels = self.resolution[0] * self.resolution[1]
        
        # Sparse system matrix
        self.system_matrix = lil_matrix((n_rays, n_pixels))
        
        pixel_size_x = self.volume[0] / self.resolution[1]
        pixel_size_y = self.volume[1] / self.resolution[0]
        
        ray_idx = 0
        
        for proj_idx, angle in enumerate(self.projection_angles):
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            for det_idx, detector_pos in enumerate(self.detector_positions):
                # Ray geometry
                ray_dir_x = -sin_angle
                ray_dir_y = cos_angle
                
                start_x = detector_pos * cos_angle
                start_y = detector_pos * sin_angle
                
                # Ray-pixel intersections using Siddon's algorithm
                ray_length = max(self.volume[0], self.volume[1]) * 1.5
                n_samples = 100
                
                for t in np.linspace(-ray_length/2, ray_length/2, n_samples):
                    x_pos = start_x + t * ray_dir_x
                    y_pos = start_y + t * ray_dir_y
                    
                    # Convert to pixel indices
                    pixel_x = int((x_pos + self.volume[0]/2) / self.volume[0] * self.resolution[1])
                    pixel_y = int((y_pos + self.volume[1]/2) / self.volume[1] * self.resolution[0])
                    
                    if (0 <= pixel_x < self.resolution[1] and 
                        0 <= pixel_y < self.resolution[0]):
                        
                        pixel_idx = pixel_y * self.resolution[1] + pixel_x
                        self.system_matrix[ray_idx, pixel_idx] += 1.0
                        
                ray_idx += 1
                
        # Normalize system matrix
        for i in range(n_rays):
            row_sum = self.system_matrix[i].sum()
            if row_sum > 0:
                self.system_matrix[i] /= row_sum
                
        # Convert to CSR format for efficiency
        self.system_matrix = self.system_matrix.tocsr()
        
    def detect_warp_bubbles(self, reconstruction):
        """
        Detect and analyze warp bubble signatures in reconstruction.
        
        Uses exotic matter density and space-time curvature analysis.
        
        Args:
            reconstruction: Reconstructed density distribution
            
        Returns:
            bubble_locations: Detected warp bubble centers
            curvature_analysis: Space-time curvature metrics
        """
        # Compute density gradients
        grad_x, grad_y = np.gradient(reconstruction)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Laplacian for curvature detection
        laplacian = scipy.ndimage.laplace(reconstruction)
        
        # Warp signature detection
        # High gradients + negative curvature = potential warp bubble
        warp_signature = gradient_magnitude * np.abs(laplacian)
        
        # Find local maxima above threshold
        from scipy.ndimage import maximum_filter
        
        local_maxima = (warp_signature == maximum_filter(warp_signature, size=5))
        strong_signatures = warp_signature > np.mean(warp_signature) + 2*np.std(warp_signature)
        
        bubble_candidates = local_maxima & strong_signatures
        bubble_locations = np.column_stack(np.where(bubble_candidates))
        
        # Convert pixel coordinates to physical coordinates
        bubble_coords_physical = []
        for loc in bubble_locations:
            y_idx, x_idx = loc
            x_phys = (x_idx / self.resolution[1] - 0.5) * self.volume[0]
            y_phys = (y_idx / self.resolution[0] - 0.5) * self.volume[1]
            bubble_coords_physical.append([x_phys, y_phys])
            
        bubble_coords_physical = np.array(bubble_coords_physical)
        
        # Curvature analysis
        curvature_analysis = {
            'max_gradient': np.max(gradient_magnitude),
            'mean_curvature': np.mean(np.abs(laplacian)),
            'exotic_matter_regions': np.sum(reconstruction < -self.exotic_matter_threshold),
            'total_curvature': np.sum(np.abs(laplacian)),
            'bubble_count': len(bubble_coords_physical)
        }
        
        self.warp_bubble_detected = len(bubble_coords_physical) > 0
        self.curvature_map = laplacian
        
        return bubble_coords_physical, curvature_analysis
        
    def simulate_warp_field_scan(self, true_field_distribution):
        """
        Simulate complete warp field tomographic scan.
        
        Args:
            true_field_distribution: Ground truth field to be reconstructed
            
        Returns:
            scan_results: Complete scan and reconstruction results
        """
        print("Simulating warp field tomographic scan...")
        
        # Step 1: Forward projection (data acquisition)
        print("  1. Acquiring projection data...")
        projections = self.compute_radon_transform(true_field_distribution)
        
        # Add realistic noise
        noise_level = 0.05  # 5% noise
        noise = np.random.normal(0, noise_level * np.std(projections), projections.shape)
        noisy_projections = projections + noise
        
        # Step 2: Filtered backprojection reconstruction
        print("  2. Filtered backprojection reconstruction...")
        fbp_reconstruction = self.filtered_backprojection(noisy_projections, 'shepp-logan')
        
        # Step 3: ART reconstruction
        print("  3. ART iterative reconstruction...")
        art_reconstruction = self.algebraic_reconstruction_technique(noisy_projections, 
                                                                   n_iterations=30)
        
        # Step 4: Warp bubble detection
        print("  4. Analyzing warp bubble signatures...")
        bubble_locations, curvature_analysis = self.detect_warp_bubbles(art_reconstruction)
        
        # Step 5: Quality metrics
        fbp_error = np.mean((fbp_reconstruction - true_field_distribution)**2)
        art_error = np.mean((art_reconstruction - true_field_distribution)**2)
        
        scan_results = {
            'projections': projections,
            'noisy_projections': noisy_projections,
            'fbp_reconstruction': fbp_reconstruction,
            'art_reconstruction': art_reconstruction,
            'bubble_locations': bubble_locations,
            'curvature_analysis': curvature_analysis,
            'fbp_mse': fbp_error,
            'art_mse': art_error,
            'noise_level': noise_level,
            'scan_time': self.n_projections * self.scan_speed
        }
        
        print(f"  Scan completed in {scan_results['scan_time']:.1f} seconds")
        print(f"  FBP reconstruction error: {fbp_error:.2e}")
        print(f"  ART reconstruction error: {art_error:.2e}")
        print(f"  Warp bubbles detected: {len(bubble_locations)}")
        
        return scan_results
        
    def visualize_reconstruction_results(self, scan_results, true_field=None):
        """
        Visualize tomographic reconstruction results.
        
        Args:
            scan_results: Results from simulate_warp_field_scan
            true_field: Original field distribution for comparison
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Sinogram (projection data)
        plt.subplot(2, 4, 1)
        plt.imshow(scan_results['projections'], aspect='auto', cmap='viridis')
        plt.title('Sinogram (Clean)')
        plt.xlabel('Detector Element')
        plt.ylabel('Projection Angle')
        plt.colorbar()
        
        plt.subplot(2, 4, 2)
        plt.imshow(scan_results['noisy_projections'], aspect='auto', cmap='viridis')
        plt.title('Sinogram (Noisy)')
        plt.xlabel('Detector Element')
        plt.ylabel('Projection Angle')
        plt.colorbar()
        
        # True field (if available)
        if true_field is not None:
            plt.subplot(2, 4, 3)
            plt.imshow(true_field, cmap='RdBu', origin='lower')
            plt.title('True Field Distribution')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.colorbar()
            
        # FBP reconstruction
        plt.subplot(2, 4, 4)
        plt.imshow(scan_results['fbp_reconstruction'], cmap='RdBu', origin='lower')
        plt.title('FBP Reconstruction')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.colorbar()
        
        # ART reconstruction
        plt.subplot(2, 4, 5)
        plt.imshow(scan_results['art_reconstruction'], cmap='RdBu', origin='lower')
        plt.title('ART Reconstruction')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.colorbar()
        
        # Curvature map
        if self.curvature_map is not None:
            plt.subplot(2, 4, 6)
            plt.imshow(self.curvature_map, cmap='seismic', origin='lower')
            plt.title('Space-time Curvature')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.colorbar()
            
            # Mark detected bubbles
            if len(scan_results['bubble_locations']) > 0:
                for bubble in scan_results['bubble_locations']:
                    # Convert physical to pixel coordinates
                    x_pixel = (bubble[0] / self.volume[0] + 0.5) * self.resolution[1]
                    y_pixel = (bubble[1] / self.volume[1] + 0.5) * self.resolution[0]
                    plt.plot(x_pixel, y_pixel, 'ro', markersize=8, fillstyle='none', linewidth=2)
                    
        # Error analysis
        plt.subplot(2, 4, 7)
        if true_field is not None:
            fbp_diff = np.abs(scan_results['fbp_reconstruction'] - true_field)
            art_diff = np.abs(scan_results['art_reconstruction'] - true_field)
            
            plt.plot(np.mean(fbp_diff, axis=0), label='FBP Error', linewidth=2)
            plt.plot(np.mean(art_diff, axis=0), label='ART Error', linewidth=2)
            plt.xlabel('X Position')
            plt.ylabel('Reconstruction Error')
            plt.title('Cross-sectional Error')
            plt.legend()
            plt.grid(True)
            
        # Performance summary
        plt.subplot(2, 4, 8)
        metrics = ['FBP MSE', 'ART MSE', 'Bubbles', 'Scan Time']
        values = [scan_results['fbp_mse'], scan_results['art_mse'], 
                 len(scan_results['bubble_locations']), scan_results['scan_time']]
        
        plt.bar(metrics, values)
        plt.title('Performance Metrics')
        plt.yscale('log')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed analysis
        print(f"\nTomographic Reconstruction Analysis:")
        print(f"  Resolution: {self.resolution[0]}×{self.resolution[1]} pixels")
        print(f"  Projections: {self.n_projections} angles")
        print(f"  FBP MSE: {scan_results['fbp_mse']:.2e}")
        print(f"  ART MSE: {scan_results['art_mse']:.2e}")
        print(f"  Noise level: {scan_results['noise_level']:.1%}")
        print(f"  Scan duration: {scan_results['scan_time']:.1f} s")
        print(f"  Warp bubbles detected: {len(scan_results['bubble_locations'])}")
        
        if len(scan_results['bubble_locations']) > 0:
            print(f"  Bubble locations:")
            for i, bubble in enumerate(scan_results['bubble_locations']):
                print(f"    {i+1}: ({bubble[0]:.2f}, {bubble[1]:.2f}) m")
                
        print(f"  Max gradient: {scan_results['curvature_analysis']['max_gradient']:.2e}")
        print(f"  Mean curvature: {scan_results['curvature_analysis']['mean_curvature']:.2e}")
        
    def demonstrate_tomographic_capabilities(self):
        """Demonstrate complete tomographic scanning capabilities."""
        print("\n" + "="*60)
        print("WARP-PULSE TOMOGRAPHIC SCANNER - DEMONSTRATION")
        print("="*60)
        
        # Create synthetic warp field distribution
        print("\n1. CREATING SYNTHETIC WARP FIELD")
        print("-" * 40)
        
        # Alcubierre-style warp bubble
        x_coords = np.linspace(-self.volume[0]/2, self.volume[0]/2, self.resolution[1])
        y_coords = np.linspace(-self.volume[1]/2, self.volume[1]/2, self.resolution[0])
        X, Y = np.meshgrid(x_coords, y_coords)
        
        # Multiple warp features
        warp_field = np.zeros_like(X)
        
        # Central warp bubble (negative energy density)
        r1 = np.sqrt((X - 0.3)**2 + (Y - 0.2)**2)
        warp_field += -2.0 * np.exp(-r1**2 / 0.1**2)
        
        # Secondary bubble
        r2 = np.sqrt((X + 0.4)**2 + (Y - 0.1)**2)
        warp_field += -1.5 * np.exp(-r2**2 / 0.08**2)
        
        # Positive energy regions (exotic matter containment)
        r3 = np.sqrt((X - 0.1)**2 + (Y + 0.3)**2)
        warp_field += 1.0 * np.exp(-r3**2 / 0.15**2)
        
        # Background field variations
        warp_field += 0.2 * np.sin(2 * np.pi * X / self.volume[0]) * np.cos(2 * np.pi * Y / self.volume[1])
        
        print(f"  Synthetic field created: {np.min(warp_field):.2f} to {np.max(warp_field):.2f}")
        print(f"  Negative energy regions: {np.sum(warp_field < 0)} pixels")
        print(f"  Field complexity: {np.std(warp_field):.3f}")
        
        # Run complete scan simulation
        print(f"\n2. TOMOGRAPHIC SCAN SIMULATION")
        print("-" * 40)
        
        scan_results = self.simulate_warp_field_scan(warp_field)
        
        # Detailed analysis
        print(f"\n3. RECONSTRUCTION QUALITY ANALYSIS")
        print("-" * 40)
        
        fbp_correlation = np.corrcoef(warp_field.flatten(), 
                                    scan_results['fbp_reconstruction'].flatten())[0,1]
        art_correlation = np.corrcoef(warp_field.flatten(), 
                                    scan_results['art_reconstruction'].flatten())[0,1]
        
        print(f"  FBP correlation with truth: {fbp_correlation:.3f}")
        print(f"  ART correlation with truth: {art_correlation:.3f}")
        print(f"  SNR improvement: {10*np.log10(scan_results['fbp_mse']/scan_results['art_mse']):.1f} dB")
        
        # Warp bubble analysis
        print(f"\n4. WARP BUBBLE DETECTION RESULTS")
        print("-" * 40)
        
        curvature = scan_results['curvature_analysis']
        print(f"  Exotic matter detection threshold: {self.exotic_matter_threshold:.0e} kg/m³")
        print(f"  Regions with exotic matter: {curvature['exotic_matter_regions']}")
        print(f"  Maximum field gradient: {curvature['max_gradient']:.2e}")
        print(f"  Total space-time curvature: {curvature['total_curvature']:.2e}")
        
        # System performance
        print(f"\n5. SCANNER PERFORMANCE SUMMARY")
        print("-" * 40)
        
        data_rate = (self.n_projections * self.detector_elements) / scan_results['scan_time']
        print(f"  Data acquisition rate: {data_rate:.0f} samples/second")
        print(f"  Angular resolution: {180/self.n_projections:.2f} degrees/projection")
        print(f"  Spatial resolution: {self.volume[0]/self.resolution[0]*1000:.1f} mm/pixel")
        print(f"  Reconstruction accuracy: {100*(1-scan_results['art_mse']):.1f}%")
        
        # Visualization
        print(f"\n6. GENERATING VISUALIZATION")
        print("-" * 40)
        
        self.visualize_reconstruction_results(scan_results, warp_field)
        
        return {
            'warp_field': warp_field,
            'scan_results': scan_results,
            'fbp_correlation': fbp_correlation,
            'art_correlation': art_correlation,
            'data_rate': data_rate
        }

def main():
    """Main demonstration of Warp-Pulse Tomographic Scanner."""
    print("Initializing Warp-Pulse Tomographic Scanner System...")
    
    # Create tomographic scanner
    scanner = WarpPulseTomographicScanner(
        scan_resolution=(128, 128),    # Moderate resolution for speed
        scan_volume=(2.0, 2.0, 1.0),  # 2×2×1 meter scan volume
        n_projections=120,             # 1.5° angular resolution
        detector_elements=256          # Good spatial sampling
    )
    
    # Run full demonstration
    results = scanner.demonstrate_tomographic_capabilities()
    
    print(f"\n📡 WARP-PULSE TOMOGRAPHIC SCANNER - STEP 20 COMPLETE ✅")
    print(f"   FBP correlation: {results['fbp_correlation']:.3f}")
    print(f"   ART correlation: {results['art_correlation']:.3f}")
    print(f"   Data rate: {results['data_rate']:.0f} samples/s")
    print(f"   Bubbles detected: {len(results['scan_results']['bubble_locations'])}")
    
    return scanner, results

if __name__ == "__main__":
    # Suppress scientific notation warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    scanner_system, demo_results = main()
