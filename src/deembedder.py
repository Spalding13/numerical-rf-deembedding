import numpy as np
import pandas as pd

def s_to_abcd(s_params, z0=50):
    """
    Converts S-parameters to ABCD (cascade) matrices.
    ABCD matrices are highly useful because when RF components are cascaded,
    their matrices can simply be multiplied together.
    
    Parameters:
    - s_params: A tuple of 4 numpy arrays (s11, s12, s21, s22) representing S-params for each frequency.
    - z0: System characteristic impedance (default is 50 Ohms).
    """
    s11, s12, s21, s22 = s_params
    
    # Common denominator for the ABCD conversion formulas
    denom = 2 * s21
    
    # Standard RF engineering formulas to convert S to ABCD parameters
    a = ((1 + s11) * (1 - s22) + s12 * s21) / denom
    b = z0 * ((1 + s11) * (1 + s22) - s12 * s21) / denom
    c = (1 / z0) * ((1 - s11) * (1 - s22) - s12 * s21) / denom
    d = ((1 - s11) * (1 + s22) + s12 * s21) / denom

    # Stack the calculated arrays into a 3D matrix of shape (N, 2, 2),
    # where N is the number of frequency points.
    return np.stack([np.stack([a, b], axis=-1), 
                     np.stack([c, d], axis=-1)], axis=-2)


def abcd_to_s(abcd, z0=50):
    """
    Converts ABCD matrices back to S-parameters.
    This is used after performing mathematical operations (like de-embedding)
    in the ABCD domain, to return the data to a format suitable for plotting (S-parameters).
    """
    # Extract individual elements from the 3D array for all frequencies simultaneously
    a, b = abcd[:, 0, 0], abcd[:, 0, 1]
    c, d = abcd[:, 1, 0], abcd[:, 1, 1]
    
    # Calculate the 'delta' denominator for the conversion formulas
    delta = a + b / z0 + c * z0 + d
    
    # Conversion formulas from ABCD to S-parameters
    s11 = (a + b / z0 - c * z0 - d) / delta
    s12 = 2 * (a * d - b * c) / delta
    s21 = 2 / delta
    
    # FIXED SIGN FOR S22: Ensures physically accurate reflection coefficient at port 2
    s22 = (-a + b / z0 - c * z0 + d) / delta
    
    # Reconstruct the 3D matrix of shape (N, 2, 2)
    return np.stack([np.stack([s11, s12], axis=-1), 
                     np.stack([s21, s22], axis=-1)], axis=-2)


def deembed_dut(fixture_csv, total_csv, z0=50):
    """
    Performs the core de-embedding operation.
    Loads the fixture and total measurement data, converts them to the ABCD domain,
    mathematically removes the fixture from both sides, and returns the intrinsic DUT.
    """
    # 1. Load the raw data from the CSV files
    df_fix = pd.read_csv(fixture_csv)
    df_tot = pd.read_csv(total_csv)
    
    # Helper function to reconstruct complex numbers from separated real and imaginary columns
    def get_complex_s(df):
        s11 = df['S11_real'] + 1j * df['S11_imag']
        s12 = df['S12_real'] + 1j * df['S12_imag']
        s21 = df['S21_real'] + 1j * df['S21_imag']
        s22 = df['S22_real'] + 1j * df['S22_imag']
        return (s11.values, s12.values, s21.values, s22.values)

    # 2. Convert the complex S-parameters into ABCD matrices
    abcd_fix = s_to_abcd(get_complex_s(df_fix), z0)
    abcd_tot = s_to_abcd(get_complex_s(df_tot), z0)
    
    # 3. De-embedding math
    # Since our PAD is a parallel RC element, its matrix is perfectly symmetric.
    # Therefore, we can safely use the exact same inverted matrix for both the left and right sides.
    abcd_fix_inv = np.linalg.inv(abcd_fix)
    
    # Intrinsic DUT = Inv(Fixture) * Total * Inv(Fixture)
    # The '@' operator performs batch matrix multiplication for all frequency points
    abcd_intrinsic = abcd_fix_inv @ abcd_tot @ abcd_fix_inv
    
    return abcd_intrinsic