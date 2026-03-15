import os
import glob
import numpy as np
import csv
from typing import Dict

class RFMeasurementGenerator:
    """Generate synthetic RF two-port S-parameter measurements with topology support."""

    def __init__(self, z0: float = 50.0):
        self.z0 = z0

    @staticmethod
    def _freq_vector(f_start: float, f_stop: float, n_points: int):
        """Create logarithmic frequency vector."""
        freqs = np.logspace(np.log10(f_start), np.log10(f_stop), n_points)
        return freqs, 2 * np.pi * freqs

    @staticmethod
    def _abcd_series_rlc_vec(omega: np.ndarray, r: float, l: float, c: float) -> np.ndarray:
        """ABCD matrix for a series-connected RLC branch (Band-pass behavior)."""
        z = r + 1j * omega * l + 1 / (1j * omega * c) 
        N = omega.size
        abcd = np.zeros((N, 2, 2), dtype=complex)
        abcd[:, 0, 0] = 1
        abcd[:, 0, 1] = z
        abcd[:, 1, 0] = 0
        abcd[:, 1, 1] = 1
        return abcd

    @staticmethod
    def _abcd_shunt_rlc_vec(omega: np.ndarray, r: float, l: float, c: float) -> np.ndarray:
        """ABCD matrix for a shunt-connected RLC branch to ground (Notch/Band-stop behavior)."""
        z = r + 1j * omega * l + 1 / (1j * omega * c)
        y = 1.0 / z  # Admittance
        N = omega.size
        abcd = np.zeros((N, 2, 2), dtype=complex)
        abcd[:, 0, 0] = 1
        abcd[:, 0, 1] = 0
        abcd[:, 1, 0] = y
        abcd[:, 1, 1] = 1
        return abcd

    @staticmethod
    def _abcd_shunt_rc_vec(omega: np.ndarray, r: float, c: float) -> np.ndarray:
        """ABCD matrix for fixture pads - shunt RC element."""
        y = 1.0 / r + 1j * omega * c
        N = omega.size
        abcd = np.zeros((N, 2, 2), dtype=complex)
        abcd[:, 0, 0] = 1
        abcd[:, 0, 1] = 0
        abcd[:, 1, 0] = y
        abcd[:, 1, 1] = 1
        return abcd

    @staticmethod
    def _abcd_to_s_vec(abcd: np.ndarray, z0: float) -> np.ndarray:
        """Convert ABCD matrices to S-parameter matrices."""
        a, b, c, d = abcd[:, 0, 0], abcd[:, 0, 1], abcd[:, 1, 0], abcd[:, 1, 1]
        delta = a + b / z0 + c * z0 + d
        s11 = (a + b / z0 - c * z0 - d) / delta
        s12 = 2 * (a * d - b * c) / delta
        s21 = 2 / delta
        s22 = (-a + b / z0 - c * z0 + d) / delta
        
        s = np.zeros((abcd.shape[0], 2, 2), dtype=complex)
        s[:, 0, 0], s[:, 0, 1] = s11, s12
        s[:, 1, 0], s[:, 1, 1] = s21, s22
        return s

    def _add_complex_noise(self, sparams: np.ndarray, noise_level: float, rng: np.random.Generator) -> np.ndarray:
        """Add additive complex Gaussian noise to S-parameters."""
        if noise_level == 0.0: return sparams
        noise = rng.normal(scale=noise_level, size=sparams.shape) + 1j * rng.normal(scale=noise_level, size=sparams.shape)
        return sparams + noise

    def _save_sparams_csv(self, filename: str, freqs: np.ndarray, sparams: np.ndarray):
        """Save S-parameters to a CSV file in Real/Imag format."""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frequency_Hz', 'S11_real', 'S11_imag', 'S12_real', 'S12_imag', 
                             'S21_real', 'S21_imag', 'S22_real', 'S22_imag'])
            for i in range(len(freqs)):
                writer.writerow([
                    freqs[i],
                    sparams[i, 0, 0].real, sparams[i, 0, 0].imag,
                    sparams[i, 0, 1].real, sparams[i, 0, 1].imag,
                    sparams[i, 1, 0].real, sparams[i, 1, 0].imag,
                    sparams[i, 1, 1].real, sparams[i, 1, 1].imag
                ])

    def generate_coherent_datasets(self, f_start: float, f_stop: float, n_points: int,
                                   dut_params: Dict[str, float], pad_params: Dict[str, float],
                                   noise_level: float, seed: int, output_dir: str,
                                   topology: str = 'series'):
        """Generate fixture, raw, and golden datasets based on selected topology."""
        freqs, omega = self._freq_vector(f_start, f_stop, n_points)
        
        # Define Fixture (PAD)
        pad_abcd = self._abcd_shunt_rc_vec(omega, pad_params['r'], pad_params['c'])
        
        # Define DUT based on requested topology
        if topology.lower() == 'shunt':
            dut_abcd = self._abcd_shunt_rlc_vec(omega, dut_params['r'], dut_params['l'], dut_params['c'])
        else:
            dut_abcd = self._abcd_series_rlc_vec(omega, dut_params['r'], dut_params['l'], dut_params['c'])
        
        # Composite system: PAD -> DUT -> PAD
        total_abcd = pad_abcd @ dut_abcd @ pad_abcd
        
        # Convert to S-parameters
        s_fixture = self._abcd_to_s_vec(pad_abcd, self.z0)
        s_total = self._abcd_to_s_vec(total_abcd, self.z0)
        s_golden = self._abcd_to_s_vec(dut_abcd, self.z0)

        # Add Noise
        rng = np.random.default_rng(seed)
        s_fixture_noisy = self._add_complex_noise(s_fixture, noise_level, rng)
        s_total_noisy = self._add_complex_noise(s_total, noise_level, rng)

        # Save files
        os.makedirs(output_dir, exist_ok=True)
        self._save_sparams_csv(os.path.join(output_dir, 'fixture.csv'), freqs, s_fixture_noisy)
        self._save_sparams_csv(os.path.join(output_dir, 'dut_plus_fixture.csv'), freqs, s_total_noisy)
        self._save_sparams_csv(os.path.join(output_dir, 'golden_dut.csv'), freqs, s_golden)

if __name__ == "__main__":
    output_dir = "../measurement_data"

    # Cleanup old data
    for file_path in glob.glob(os.path.join(output_dir, "*")):
        if os.path.isfile(file_path):
            os.remove(file_path)

    generator = RFMeasurementGenerator(z0=50.0)
    
    # Example: Generating a Notch Filter (Shunt Topology)
    generator.generate_coherent_datasets(
        f_start=1e6,
        f_stop=10e9,
        n_points=1001,
        dut_params={'r': 1.0, 'l': 1e-9, 'c': 1e-12},
        pad_params={'r': 1000.0, 'c': 1e-12},
        noise_level=0.0035,
        seed=42,
        output_dir=output_dir,
        topology='shunt' # Switch to 'series' for Band-pass
    )
    print(f"Data successfully generated in {output_dir}")