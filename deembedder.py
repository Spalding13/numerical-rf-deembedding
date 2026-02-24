import numpy as np
import pandas as pd

"""Convert S-parameters to ABCD matrices (vectorized)."""
def s_to_abcd(s_params, z0=50):
    s11, s12, s21, s22 = s_params
    denom = 2 * s21
    
    a = ((1 + s11) * (1 - s22) + s12 * s21) / denom
    
    b = z0 * ((1 + s11) * (1 + s22) - s12 * s21) / denom
    
    c = (1 / z0) * ((1 - s11) * (1 - s22) - s12 * s21) / denom
    
    d = ((1 - s11) * (1 + s22) + s12 * s21) / denom
    


    # Result is (N, 2, 2) matrix
    return np.stack([np.stack([a, b], axis=-1), 
                     np.stack([c, d], axis=-1)], axis=-2)

"""De-embedding function: takes fixture and total S-parameter CSVs, returns intrinsic ABCD matrices."""
def deembed_dut(fixture_csv, total_csv, z0=50):
    # Load data
    df_fix = pd.read_csv(fixture_csv)
    df_tot = pd.read_csv(total_csv)
    
    # 2. Реконструкция на комплексни S-параметри
    def get_complex_s(df):
        s11 = df['S11_real'] + 1j*df['S11_imag']
        s12 = df['S12_real'] + 1j*df['S12_imag']
        s21 = df['S21_real'] + 1j*df['S21_imag']
        s22 = df['S22_real'] + 1j*df['S22_imag']
        return (s11.values, s12.values, s21.values, s22.values)

    # 3. Transform and operfomr the demmbedding part

    abcd_fix = s_to_abcd(get_complex_s(df_fix), z0)
    abcd_tot = s_to_abcd(get_complex_s(df_tot), z0)
    
    # Invert the fixture ABCD matrix
    abcd_fix_inv = np.linalg.inv(abcd_fix)
    
    # Intrinsic = Inv(Fix) * Total * Inv(Fix)
    abcd_intrinsic = abcd_fix_inv @ abcd_tot @ abcd_fix_inv
    
    return abcd_intrinsic 


