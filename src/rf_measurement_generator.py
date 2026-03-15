import os
import glob
import numpy as np
import csv
from typing import Dict

class RFMeasurementGenerator:
    """Generate synthetic RF two-port S-parameter measurements."""

    def __init__(self, z0: float = 50.0):
        self.z0 = z0

    @staticmethod
    def _freq_vector(f_start: float, f_stop: float, n_points: int):
        freqs = np.logspace(np.log10(f_start), np.log10(f_stop), n_points)
        return freqs, 2 * np.pi * freqs

    @staticmethod
    def _abcd_series_rlc_vec(omega: np.ndarray, r: float, l: float, c: float) -> np.ndarray:
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
        a = abcd[:, 0, 0]
        b = abcd[:, 0, 1]
        c = abcd[:, 1, 0]
        d = abcd[:, 1, 1]

        delta = a + b / z0 + c * z0 + d

        s11 = (a + b / z0 - c * z0 - d) / delta
        s12 = 2 * (a * d - b * c) / delta
        s21 = 2 / delta
        s22 = (-a + b / z0 - c * z0 + d) / delta

        N = abcd.shape[0]
        s = np.zeros((N, 2, 2), dtype=complex)
        s[:, 0, 0] = s11
        s[:, 0, 1] = s12
        s[:, 1, 0] = s21
        s[:, 1, 1] = s22
        return s

    @staticmethod
    def _add_complex_noise(sparams: np.ndarray, noise_level: float, rng: np.random.Generator) -> np.ndarray:
        if noise_level == 0.0:
            return sparams
        N = sparams.shape[0]
        noise = rng.normal(scale=noise_level, size=(N, 2, 2)) + 1j * rng.normal(scale=noise_level, size=(N, 2, 2))
        return sparams + noise

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

    def generate_coherent_datasets(self, f_start: float, f_stop: float, n_points: int,
                                   dut_params: Dict[str, float], pad_params: Dict[str, float],
                                   noise_level: float, seed: int, output_dir: str) -> None:
        
        freqs, omega = self._freq_vector(f_start, f_stop, n_points)

        dut_abcd = self._abcd_series_rlc_vec(omega, dut_params['r'], dut_params['l'], dut_params['c'])
        pad_abcd = self._abcd_shunt_rc_vec(omega, pad_params['r'], pad_params['c'])

        # ФИКС: Фикстурата е само един PAD
        fixture_abcd = pad_abcd 
        total_abcd = np.matmul(np.matmul(pad_abcd, dut_abcd), pad_abcd)
        golden_abcd = dut_abcd

        s_fixture = self._abcd_to_s_vec(fixture_abcd, self.z0)
        s_total = self._abcd_to_s_vec(total_abcd, self.z0)
        s_golden = self._abcd_to_s_vec(golden_abcd, self.z0)

        rng = np.random.default_rng(seed)
        s_fixture_noisy = self._add_complex_noise(s_fixture, noise_level, rng)
        s_total_noisy = self._add_complex_noise(s_total, noise_level, rng)

        os.makedirs(output_dir, exist_ok=True)
        self._save_sparams_csv(os.path.join(output_dir, 'fixture.csv'), freqs, s_fixture_noisy)
        self._save_sparams_csv(os.path.join(output_dir, 'dut_plus_fixture.csv'), freqs, s_total_noisy)
        self._save_sparams_csv(os.path.join(output_dir, 'golden_dut.csv'), freqs, s_golden)


if __name__ == "__main__":

    output_dir = "../measurement_data"

    # Delete all files in the directory
    for file_path in glob.glob(os.path.join(output_dir, "*")):
        if os.path.isfile(file_path):
            print(f"Deleting file: {file_path}")
            os.remove(file_path)


    generator = RFMeasurementGenerator(z0=50.0)
    generator.generate_coherent_datasets(
        f_start=1e6,
        f_stop=10e9,
        n_points=1001,
        dut_params={'r': 1.0, 'l': 1e-9, 'c': 3e-12},
        pad_params={'r': 1000.0, 'c': 1e-12},
        noise_level=0.005,  
        seed=42,
        output_dir="../measurement_data"
    )
    print("Files successfully generated in ../measurement_data")