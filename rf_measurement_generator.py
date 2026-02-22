import numpy as np
import csv
from typing import Tuple


class RFMeasurementGenerator:
    """Generate synthetic RF two-port S-parameter measurements."""
    
    def __init__(self, 
                 f_start: float = 1e6, 
                 f_stop: float = 10e9, 
                 num_points: int = 101,
                 z0: float = 50.0):
        """
        Initialize RF measurement generator.
        
        Args:
            f_start: Start frequency in Hz
            f_stop: Stop frequency in Hz
            num_points: Number of frequency points
            z0: Characteristic impedance in Ohms
        """
        self.f_start = f_start
        self.f_stop = f_stop
        self.num_points = num_points
        self.z0 = z0
        
    def generate_measurement(self,
                            output_file: str = "rf_measurements.csv",
                            dut_r: float = 1.0,
                            dut_l: float = 1e-9,
                            dut_c: float = 1e-12,
                            pad_r: float = 10.0,
                            pad_c: float = 1e-12,
                            noise_level: float = 0.01) -> None:
        """
        Generate synthetic RF measurement data and save to CSV.
        
        Args:
            output_file: Output CSV filename
            dut_r: DUT series resistance in Ohms
            dut_l: DUT series inductance in Henries
            dut_c: DUT series capacitance in Farads
            pad_r: Pad shunt resistance in Ohms
            pad_c: Pad shunt capacitance in Farads
            noise_level: Noise standard deviation (complex)
        """
        # Create logarithmic frequency sweep
        frequencies = np.logspace(np.log10(self.f_start), 
                                   np.log10(self.f_stop), 
                                   self.num_points)
        omega = 2 * np.pi * frequencies
        
        # Storage for S-parameters
        s11_data = np.zeros(self.num_points, dtype=complex)
        s21_data = np.zeros(self.num_points, dtype=complex)
        s12_data = np.zeros(self.num_points, dtype=complex)
        s22_data = np.zeros(self.num_points, dtype=complex)
        
        # Process each frequency
        for idx, (f, w) in enumerate(zip(frequencies, omega)):
            # Build ABCD matrices
            abcd_dut = self._abcd_series_rlc(w, dut_r, dut_l, dut_c)
            abcd_pad1 = self._abcd_shunt_rc(w, pad_r, pad_c)
            abcd_pad2 = self._abcd_shunt_rc(w, pad_r, pad_c)
            
            # Cascade: PAD1 + DUT + PAD2
            abcd_total = abcd_pad1 @ abcd_dut @ abcd_pad2
            
            # Convert to S-parameters
            s_params = self._abcd_to_s(abcd_total, self.z0)
            
            s11_data[idx] = s_params[0, 0]
            s21_data[idx] = s_params[1, 0]
            s12_data[idx] = s_params[0, 1]
            s22_data[idx] = s_params[1, 1]
        
        # Add small complex Gaussian noise
        s11_data += noise_level * (np.random.randn(self.num_points) + 
                                    1j * np.random.randn(self.num_points))
        s21_data += noise_level * (np.random.randn(self.num_points) + 
                                    1j * np.random.randn(self.num_points))
        s12_data += noise_level * (np.random.randn(self.num_points) + 
                                    1j * np.random.randn(self.num_points))
        s22_data += noise_level * (np.random.randn(self.num_points) + 
                                    1j * np.random.randn(self.num_points))
        
        # Save to CSV
        self._save_to_csv(output_file, frequencies, s11_data, s21_data, s12_data, s22_data)
    
    @staticmethod
    def _abcd_series_rlc(omega: float, r: float, l: float, c: float) -> np.ndarray:
        """
        Create ABCD matrix for series RLC element.
        
        For series RLC: Z = R + j*omega*L + 1/(j*omega*C)
        ABCD = [[1, Z], [0, 1]]
        """
        z = r + 1j * omega * l + 1 / (1j * omega * c)
        return np.array([[1, z], [0, 1]], dtype=complex)
    
    @staticmethod
    def _abcd_shunt_rc(omega: float, r: float, c: float) -> np.ndarray:
        """
        Create ABCD matrix for shunt RC element.
        
        For shunt RC: Y = 1/R + j*omega*C
        ABCD = [[1, 0], [Y, 1]]
        """
        y = 1/r + 1j * omega * c
        return np.array([[1, 0], [y, 1]], dtype=complex)
    
    @staticmethod
    def _abcd_to_s(abcd: np.ndarray, z0: float) -> np.ndarray:
        """
        Convert ABCD matrix to S-parameters (2x2).
        
        Assuming both ports have same characteristic impedance Z0.
        """
        a, b, c, d = abcd[0, 0], abcd[0, 1], abcd[1, 0], abcd[1, 1]
        
        delta = a + b/z0 + c*z0 + d
        
        s11 = (a + b/z0 - c*z0 - d) / delta
        s12 = 2 * (a*d - b*c) / delta
        s21 = 2 / delta
        s22 = (-a + b/z0 + c*z0 - d) / delta
        
        return np.array([[s11, s12], [s21, s22]], dtype=complex)
    
    @staticmethod
    def _save_to_csv(filename: str, frequencies: np.ndarray, 
                     s11: np.ndarray, s21: np.ndarray, 
                     s12: np.ndarray, s22: np.ndarray) -> None:
        """Save S-parameters to CSV file."""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frequency_Hz', 
                            'S11_real', 'S11_imag',
                            'S21_real', 'S21_imag',
                            'S12_real', 'S12_imag',
                            'S22_real', 'S22_imag'])
            
            for freq, s11_val, s21_val, s12_val, s22_val in zip(
                frequencies, s11, s21, s12, s22):
                writer.writerow([
                    freq,
                    s11_val.real, s11_val.imag,
                    s21_val.real, s21_val.imag,
                    s12_val.real, s12_val.imag,
                    s22_val.real, s22_val.imag
                ])


if __name__ == "__main__":
    # Example usage
    generator = RFMeasurementGenerator(
        f_start=1e6,      # 1 MHz
        f_stop=10e9,      # 10 GHz
        num_points=101,   # 101 frequency points
        z0=50.0           # 50 Ohm
    )
    
    generator.generate_measurement(
        output_file="./measurement_data/rf_measurements.csv",
        dut_r=1.0,        # 1 Ohm
        dut_l=1e-9,       # 1 nH
        dut_c=1e-12,      # 1 pF
        pad_r=10.0,       # 10 Ohm
        pad_c=1e-12,      # 1 pF
        noise_level=0.01  # 1% noise
    )
    
    print("RF measurements saved to ./measurement_data/rf_measurements.csv")
