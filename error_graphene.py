import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import json
import os
from typing import List, Dict, Tuple, Optional

class RamanAnalyzer:
    """
    Comprehensive Raman spectrum analyzer with peak detection, fitting, and classification
    """
    
    def __init__(self):
        # Define expected peak ranges for different materials
        self.graphene_ranges = {
            'D': (1200, 1450),    # D band
            'G': (1500, 1650),    # G band
            '2D': (2500, 2900)    # 2D band
        }
        
        # Common materials database (can be expanded)
        self.material_database = {
            'silicon': [520.7],
            'diamond': [1332],
            'quartz': [464, 697, 797, 1160],
            'titanium_dioxide': [144, 197, 399, 513, 519, 639],
            'silicon_carbide': [796, 972],
        }
        
    def load_spectrum(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load Raman spectrum from file with robust encoding handling"""
        print(f"Loading file: {filename}")
        
        # List of encodings to try
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'ascii']
        
        for encoding in encodings:
            try:
                print(f"Trying encoding: {encoding}")
                data = np.loadtxt(filename, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                if "could not convert string to float" in str(e).lower():
                    # Try skipping header rows
                    try:
                        data = np.loadtxt(filename, encoding=encoding, skiprows=1)
                        print(f"Success with encoding {encoding} (skipped 1 header row)")
                        break
                    except:
                        continue
                else:
                    continue
        else:
            # If all encodings fail, try reading as binary and inspect
            try:
                with open(filename, 'rb') as f:
                    first_bytes = f.read(100)
                    print(f"First 100 bytes of file: {first_bytes}")
                    
                # Try pandas as fallback
                try:
                    import pandas as pd
                    df = pd.read_csv(filename, sep=r'\s+', header=None, encoding='latin1')
                    data = df.values
                    print("Successfully loaded using pandas with latin1 encoding")
                except ImportError:
                    raise FileNotFoundError("Could not load file with any encoding. Try installing pandas: pip install pandas")
                except Exception as e:
                    raise FileNotFoundError(f"Could not load file with any method. File might be corrupted or in unsupported format. Error: {e}")
                    
            except Exception as e:
                raise FileNotFoundError(f"Could not read file at all: {e}")
        
        # Validate data shape
        if len(data.shape) == 1:
            raise ValueError(f"Data appears to be 1D. Expected 2 columns, got shape: {data.shape}")
        elif data.shape[1] == 2:
            wavenumber = data[:, 0]
            intensity = data[:, 1]
        elif data.shape[1] > 2:
            print(f"Warning: File has {data.shape[1]} columns, using first 2")
            wavenumber = data[:, 0]
            intensity = data[:, 1]
        else:
            raise ValueError(f"Data file should have at least 2 columns. Got shape: {data.shape}")
        
        # Basic data validation
        if len(wavenumber) == 0:
            raise ValueError("No data points found in file")
        
        print(f"Successfully loaded {len(wavenumber)} data points")
        print(f"Wavenumber range: {wavenumber.min():.1f} to {wavenumber.max():.1f} cm⁻¹")
        print(f"Intensity range: {intensity.min():.1f} to {intensity.max():.1f}")
        
        return wavenumber, intensity
    
    def remove_baseline(self, intensity: np.ndarray, lambda_param: float = 1e5) -> np.ndarray:
        """
        Remove baseline using asymmetric least squares smoothing
        """
        L = len(intensity)
        D = np.diff(np.eye(L), 2, axis=0)
        w = np.ones(L)
        
        for _ in range(10):  # iterations
            W = np.diag(w)
            Z = np.linalg.solve(W + lambda_param * D.T @ D, w * intensity)
            w = np.where(intensity > Z, 0.001, 1)
        
        return intensity - Z
    
    def smooth_spectrum(self, intensity: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing to reduce noise"""
        return gaussian_filter1d(intensity, sigma=sigma)
    
    def gaussian_peak(self, x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
        """Gaussian peak function for fitting"""
        return amplitude * np.exp(-((x - center) / width) ** 2)
    
    def lorentzian_peak(self, x: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
        """Lorentzian peak function for fitting"""
        return amplitude / (1 + ((x - center) / width) ** 2)
    
    def voigt_peak(self, x: np.ndarray, amplitude: float, center: float, gamma: float, sigma: float) -> np.ndarray:
        """Pseudo-Voigt peak (approximation of Voigt profile)"""
        # Simple approximation: weighted sum of Gaussian and Lorentzian
        eta = 1.36603 * (gamma/sigma) - 0.47719 * (gamma/sigma)**2 + 0.11116 * (gamma/sigma)**3
        if eta > 1:
            eta = 1
        elif eta < 0:
            eta = 0
            
        gaussian = np.exp(-((x - center) / sigma) ** 2)
        lorentzian = 1 / (1 + ((x - center) / gamma) ** 2)
        
        return amplitude * (eta * lorentzian + (1 - eta) * gaussian)
    
    def find_seeded_peaks(self, wavenumber: np.ndarray, intensity: np.ndarray, 
                         seed_positions: List[float], search_window: float = 10.0) -> List[Dict]:
        """
        Find peaks around seeded positions with improved fitting
        """
        peaks = []
        
        for seed_pos in seed_positions:
            # Define search window
            mask = (wavenumber >= seed_pos - search_window) & (wavenumber <= seed_pos + search_window)
            
            if not np.any(mask):
                continue
                
            x_window = wavenumber[mask]
            y_window = intensity[mask]
            
            # Find the actual peak maximum in the window
            max_idx = np.argmax(y_window)
            actual_center = x_window[max_idx]
            max_intensity = y_window[max_idx]
            
            # Skip if intensity is too low (noise threshold)
            if max_intensity < np.max(intensity) * 0.01:  # 1% of max intensity
                continue
            
            # Initial parameter estimates
            amplitude_init = max_intensity
            center_init = actual_center
            width_init = 5.0  # Initial guess for width
            
            try:
                # Try Gaussian fit first
                popt_gauss, _ = curve_fit(
                    self.gaussian_peak, 
                    x_window, 
                    y_window,
                    p0=[amplitude_init, center_init, width_init],
                    bounds=([0, seed_pos - search_window, 0.1], 
                           [np.inf, seed_pos + search_window, 50]),
                    maxfev=1000
                )
                
                # Calculate FWHM for Gaussian (FWHM = 2.355 * sigma)
                fwhm = 2.355 * abs(popt_gauss[2])
                
                peak_info = {
                    'center_cm': popt_gauss[1],
                    'fwhm_cm': fwhm,
                    'amplitude': popt_gauss[0],
                    'fit_type': 'gaussian',
                    'seed_position': seed_pos,
                    'classification': 'other'
                }
                
                peaks.append(peak_info)
                
            except Exception as e:
                print(f"Warning: Could not fit peak near {seed_pos:.1f} cm⁻¹: {e}")
                continue
        
        return peaks
    
    def merge_duplicate_peaks(self, peaks: List[Dict], merge_threshold: float = 3.0) -> List[Dict]:
        """
        Merge peaks that are too close together (likely duplicates)
        """
        if not peaks:
            return peaks
        
        # Sort peaks by center position
        sorted_peaks = sorted(peaks, key=lambda x: x['center_cm'])
        merged_peaks = []
        
        i = 0
        while i < len(sorted_peaks):
            current_peak = sorted_peaks[i].copy()
            merge_group = [current_peak]
            
            # Find all peaks within merge threshold
            j = i + 1
            while j < len(sorted_peaks):
                if abs(sorted_peaks[j]['center_cm'] - current_peak['center_cm']) <= merge_threshold:
                    merge_group.append(sorted_peaks[j])
                    j += 1
                else:
                    break
            
            if len(merge_group) > 1:
                # Merge peaks by taking weighted average
                total_amplitude = sum(p['amplitude'] for p in merge_group)
                
                merged_peak = {
                    'center_cm': sum(p['center_cm'] * p['amplitude'] for p in merge_group) / total_amplitude,
                    'fwhm_cm': sum(p['fwhm_cm'] * p['amplitude'] for p in merge_group) / total_amplitude,
                    'amplitude': total_amplitude,
                    'fit_type': merge_group[0]['fit_type'],
                    'seed_position': merge_group[0]['seed_position'],
                    'classification': merge_group[0]['classification'],
                    'merged_from': len(merge_group)
                }
                merged_peaks.append(merged_peak)
                print(f"Merged {len(merge_group)} peaks near {merged_peak['center_cm']:.1f} cm⁻¹")
            else:
                merged_peaks.append(current_peak)
            
            i = j
        
        return merged_peaks
    
    def classify_peaks(self, peaks: List[Dict]) -> List[Dict]:
        """
        Classify peaks based on known material databases
        """
        for peak in peaks:
            center = peak['center_cm']
            
            # Check for graphene peaks
            for band_name, (min_pos, max_pos) in self.graphene_ranges.items():
                if min_pos <= center <= max_pos:
                    peak['classification'] = f'graphene_{band_name}'
                    break
            
            # Check for other materials
            if peak['classification'] == 'other':
                for material, positions in self.material_database.items():
                    for pos in positions:
                        if abs(center - pos) <= 5.0:  # 5 cm⁻¹ tolerance
                            peak['classification'] = f'{material}'
                            break
                    if peak['classification'] != 'other':
                        break
        
        return peaks
    
    def analyze_spectrum(self, filename: str, seed_positions: List[float], 
                        output_dir: str = "output", save_all_peaks: bool = True) -> Dict:
        """
        Complete spectrum analysis pipeline
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/features", exist_ok=True)
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        
        print(f"Analyzing spectrum: {filename}")
        print(f"Seeded peaks: {[f'{pos:.1f} cm⁻¹' for pos in seed_positions]}")
        
        try:
            # Load and preprocess spectrum
            wavenumber, intensity_raw = self.load_spectrum(filename)
            intensity_baseline_removed = self.remove_baseline(intensity_raw)
            intensity_smoothed = self.smooth_spectrum(intensity_baseline_removed)
            
            # Find seeded peaks
            peaks = self.find_seeded_peaks(wavenumber, intensity_smoothed, seed_positions)
            print(f"Found {len(peaks)} initial peaks")
            
            # Merge duplicate peaks
            peaks = self.merge_duplicate_peaks(peaks)
            print(f"After merging: {len(peaks)} peaks")
            
            # Classify peaks
            peaks = self.classify_peaks(peaks)
            
            # Print graphene ranges for reference
            print("Looking for graphene peaks in ranges:")
            for band, (min_pos, max_pos) in self.graphene_ranges.items():
                print(f"  {band}: {min_pos}-{max_pos} cm⁻¹")
            print(f"Save all peaks: {save_all_peaks}")
            
            # Print peak information
            graphene_peaks = 0
            for i, peak in enumerate(peaks, 1):
                classification = peak['classification']
                if 'graphene' in classification:
                    graphene_peaks += 1
                
                merged_info = f" (merged from {peak.get('merged_from', 1)} peaks)" if 'merged_from' in peak else ""
                print(f"Peak {i}: {peak['center_cm']:.1f} cm⁻¹, FWHM={peak['fwhm_cm']:.1f}, "
                      f"classified as: {classification}{merged_info}")
            
            # Create results dictionary
            results = {
                'source_file': filename,
                'num_total_peaks': len(peaks),
                'num_graphene_peaks': graphene_peaks,
                'all_peaks': peaks
            }
            
            # Add individual peak data for compatibility
            for i, peak in enumerate(peaks, 1):
                results[f'peak_{i}_center'] = peak['center_cm']
                results[f'peak_{i}_fwhm'] = peak['fwhm_cm']
                results[f'peak_{i}_amplitude'] = peak['amplitude']
            
            # Save results
            base_name = os.path.splitext(os.path.basename(filename))[0]
            features_file = f"{output_dir}/features/{base_name}_features.json"
            
            with open(features_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Saved features → {features_file}")
            
            # Create and save plot
            plot_file = f"{output_dir}/images/{base_name}_analysis.png"
            self.plot_spectrum(wavenumber, intensity_raw, intensity_baseline_removed, 
                              intensity_smoothed, peaks, plot_file)
            
            # Summary
            print("=== Summary ===")
            print(f"Total peaks found: {len(peaks)}")
            print(f"Graphene peaks: {graphene_peaks}")
            print(f"Features saved: {len(results)}")
            
            if graphene_peaks == 0:
                print("⚠️  No graphene peaks detected in expected ranges!")
                print("Your data may be from a different material, or you may need to:")
                print("- Check if you have the right data file")
                print("- Adjust the peak ranges")
                print(f"- Set SAVE_ALL_PEAKS = True to analyze whatever material this is")
                
                # Suggest possible materials
                self.suggest_material(peaks)
            
            return results
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            
            # Provide more detailed debugging information
            if os.path.exists(filename):
                try:
                    file_size = os.path.getsize(filename)
                    print(f"File exists, size: {file_size} bytes")
                    
                    # Try to read first few lines as text
                    with open(filename, 'rb') as f:
                        first_lines = f.read(200)
                    print(f"First 200 bytes: {first_lines}")
                    
                except Exception as debug_error:
                    print(f"Could not read file for debugging: {debug_error}")
            else:
                print(f"File does not exist: {filename}")
                print(f"Current working directory: {os.getcwd()}")
                print(f"Files in current directory: {os.listdir('.')}")
            
            raise
    
    def suggest_material(self, peaks: List[Dict]):
        """Suggest possible materials based on peak positions"""
        peak_positions = [p['center_cm'] for p in peaks]
        
        print("\n🔍 Material identification suggestions:")
        for material, ref_positions in self.material_database.items():
            matches = 0
            for ref_pos in ref_positions:
                for peak_pos in peak_positions:
                    if abs(peak_pos - ref_pos) <= 10:  # 10 cm⁻¹ tolerance
                        matches += 1
                        break
            
            if matches > 0:
                confidence = matches / len(ref_positions)
                print(f"  {material.replace('_', ' ').title()}: {matches}/{len(ref_positions)} peaks match "
                      f"(confidence: {confidence:.1%})")
    
    def plot_spectrum(self, wavenumber: np.ndarray, intensity_raw: np.ndarray, 
                     intensity_baseline: np.ndarray, intensity_smoothed: np.ndarray, 
                     peaks: List[Dict], filename: str):
        """Create comprehensive spectrum plot with peak annotations"""
        
        plt.figure(figsize=(12, 8))
        
        # Plot spectra
        plt.plot(wavenumber, intensity_raw, 'lightblue', alpha=0.7, label='raw')
        plt.plot(wavenumber, intensity_baseline, 'orange', alpha=0.8, label='baseline-removed')
        plt.plot(wavenumber, intensity_smoothed, 'green', linewidth=2, label='smoothed')
        
        # Annotate peaks
        for peak in peaks:
            center = peak['center_cm']
            # Find closest point in spectrum for annotation height
            idx = np.argmin(np.abs(wavenumber - center))
            height = intensity_smoothed[idx]
            
            # Plot peak marker
            plt.plot(center, height, 'x', markersize=8, color='red', markeredgewidth=2)
            
            # Annotate with classification
            classification = peak['classification']
            if classification != 'other':
                label = classification.replace('_', '\n')
            else:
                label = f"{center:.0f}"
            
            plt.annotate(label, xy=(center, height), xytext=(center, height * 1.1),
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.xlabel('Raman shift (cm⁻¹)')
        plt.ylabel('Intensity (a.u.)')
        plt.title('Raman Spectrum Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(wavenumber.max(), wavenumber.min())  # Reverse x-axis
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot     → {filename}")

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = RamanAnalyzer()
    
    # Define seeded peak positions (from your output)
    seeded_peaks = [190.6, 383.6, 403.2, 452.1, 519.0, 568.7, 755.6]
    
    # Analyze spectrum
    try:
        results = analyzer.analyze_spectrum(
            filename="Gr_YC_2.txt",  # Replace with your file
            seed_positions=seeded_peaks,
            output_dir="output",
            save_all_peaks=True
        )
        
        print(f"\n✅ Analysis complete! Results saved.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Please check your data file path and format.")