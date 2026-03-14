import os
import numpy as np
import csv
from typing import Dict

"""

"""
class RFMeasurementGenerator:
    """Generate synthetic RF two-port S-parameter measurements.

    This class focuses on generation only (no de-embedding or analysis).
    """

    def __init__(self, z0: float = 50.0):
        self.z0 = z0

    # --- Model computations (vectorized over frequency) ---
    @staticmethod
    def _freq_vector(f_start: float, f_stop: float, n_points: int) -> (np.ndarray, np.ndarray):
        freqs = np.logspace(np.log10(f_start), np.log10(f_stop), n_points)
        return freqs, 2 * np.pi * freqs

    @staticmethod
    def _abcd_series_rlc_vec(omega: np.ndarray, r: float, l: float, c: float) -> np.ndarray:
        """Vectorized ABCD for series RLC: returns array (N,2,2)."""
        z = r + 1j * omega * l + 1 / (1j * omega * c) 
        N = omega.size
        abcd = np.zeros((N, 2, 2), dtype=complex)
        abcd[:, 0, 0] = 1
        abcd[:, 0, 1] = z
        abcd[:, 1, 0] = 0
        abcd[:, 1, 1] = 1
        return abcd

    @staticmethod
    def _abcd_shunt_rc_vec(omega: np.ndarray, r: float, c: float) -> np.ndarray:
        """Vectorized ABCD for shunt RC: returns array (N,2,2)."""
        y = 1.0 / r + 1j * omega * c
        N = omega.size
        abcd = np.zeros((N, 2, 2), dtype=complex)
        abcd[:, 0, 0] = 1
        abcd[:, 0, 1] = 0
        abcd[:, 1, 0] = y
        abcd[:, 1, 1] = 1
        return abcd

    # --- ABCD -> S conversion (vectorized) ---
    @staticmethod
    def _abcd_to_s_vec(abcd: np.ndarray, z0: float) -> np.ndarray:
        """Convert array of ABCD matrices (N,2,2) to S-parameters (N,2,2)."""
        a = abcd[:, 0, 0]
        b = abcd[:, 0, 1]
        c = abcd[:, 1, 0]
        d = abcd[:, 1, 1]

        delta = a + b / z0 + c * z0 + d

        s11 = (a + b / z0 - c * z0 - d) / delta
        s12 = 2 * (a * d - b * c) / delta
        s21 = 2 / delta
        s22 = (-a + b / z0 + c * z0 - d) / delta

        N = abcd.shape[0]
        s = np.zeros((N, 2, 2), dtype=complex)
        s[:, 0, 0] = s11
        s[:, 0, 1] = s12
        s[:, 1, 0] = s21
        s[:, 1, 1] = s22
        return s

    # --- Noise addition ---
    @staticmethod
    def _add_complex_noise(sparams: np.ndarray, noise_level: float, rng: np.random.Generator) -> np.ndarray:
        """Add independent complex Gaussian noise to sparams (std = noise_level)."""
        N = sparams.shape[0]
        noise = rng.normal(scale=noise_level, size=(N, 2, 2)) + 1j * rng.normal(scale=noise_level, size=(N, 2, 2))
        return sparams + noise

    # --- CSV export ---
    @staticmethod
    def _save_sparams_csv(filepath: str, frequencies: np.ndarray, sparams: np.ndarray) -> None:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frequency_Hz',
                'S11_real', 'S11_imag',
                'S21_real', 'S21_imag',
                'S12_real', 'S12_imag',
                'S22_real', 'S22_imag'
            ])
            for freq, s in zip(frequencies, sparams):
                writer.writerow([
                    freq,
                    s[0, 0].real, s[0, 0].imag,
                    s[1, 0].real, s[1, 0].imag,
                    s[0, 1].real, s[0, 1].imag,
                    s[1, 1].real, s[1, 1].imag,
                ])

    # --- Public coherent dataset generator ---
    def generate_coherent_datasets(self,
                                   f_start: float,
                                   f_stop: float,
                                   n_points: int,
                                   dut_params: Dict[str, float],
                                   pad_params: Dict[str, float],
                                   noise_level: float,
                                   seed: int,
                                   output_dir: str) -> None:
        """Generate three coherent datasets and save them as CSV files.

        Files produced in output_dir:
            fixture.csv           -> PAD @ PAD (with noise)
            dut_plus_fixture.csv  -> PAD @ DUT @ PAD (with noise)
            golden_dut.csv        -> DUT only (no noise)

        dut_params must contain: 'r', 'l', 'c'
        pad_params must contain: 'r', 'c'
        """
        freqs, omega = self._freq_vector(f_start, f_stop, n_points)

        # Compute models once (vectorized over frequency)
        dut_abcd = self._abcd_series_rlc_vec(omega, dut_params['r'], dut_params['l'], dut_params['c'])
        pad_abcd = self._abcd_shunt_rc_vec(omega, pad_params['r'], pad_params['c'])

        # Build requested cascades per-frequency
        fixture_abcd = np.matmul(pad_abcd, pad_abcd)
        total_abcd = np.matmul(np.matmul(pad_abcd, dut_abcd), pad_abcd)
        golden_abcd = dut_abcd

        # Convert to S-parameters
        s_fixture = self._abcd_to_s_vec(fixture_abcd, self.z0)
        s_total = self._abcd_to_s_vec(total_abcd, self.z0)
        s_golden = self._abcd_to_s_vec(golden_abcd, self.z0)

        # Add noise only to fixture and total, reproducibly
        rng = np.random.default_rng(seed)
        s_fixture_noisy = self._add_complex_noise(s_fixture, noise_level, rng)
        s_total_noisy = self._add_complex_noise(s_total, noise_level, rng)

        # Ensure output directory exists and save files
        os.makedirs(output_dir, exist_ok=True)
        self._save_sparams_csv(os.path.join(output_dir, 'fixture.csv'), freqs, s_fixture_noisy)
        self._save_sparams_csv(os.path.join(output_dir, 'dut_plus_fixture.csv'), freqs, s_total_noisy)
        self._save_sparams_csv(os.path.join(output_dir, 'golden_dut.csv'), freqs, s_golden)



if __name__ == "__main__":
    # Example usage for coherent dataset generation
    generator = RFMeasurementGenerator(z0=50.0)

    generator.generate_coherent_datasets(
        f_start=1e6,
        f_stop=10e9,
        n_points=1001,
        dut_params={'r': 1.0, 'l': 1e-9, 'c': 1e-12},
        pad_params={'r': 10.0, 'c': 1e-12},
        noise_level=0.01,
        seed=42,
        output_dir="./measurement_data"
    )

    print("Wrote fixture.csv, dut_plus_fixture.csv, golden_dut.csv to ./measurement_data")
