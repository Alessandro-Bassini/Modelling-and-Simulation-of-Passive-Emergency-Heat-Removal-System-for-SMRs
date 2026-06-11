import config
from scipy.optimize import least_squares

from components import (
    model_inlet_header,
    model_orifice,
    model_heated_section,
    model_unheated_section,
    model_generic_elbow,
    model_horizontal_riser,
    model_vertical_riser,
    model_siphon,
    model_condenser_detailed,
    model_vertical_downcomer_a,
    model_horizontal_downcomer,
    model_vertical_downcomer_b
)

# TARGET SETUP
TARGET_VOLUME = 0.025 # m3
FILLING_RATIO = 0.49
RHO_REF = 1000.0 # kg/m3 (Cold water reference)
TARGET_MASS = TARGET_VOLUME * FILLING_RATIO * RHO_REF

def solve_full_loop(x):
    """
    x = [m_dot, P_start, h_start]
    Returns residuals [delta_P, delta_h, delta_mass]
    """
    m_dot_guess = x[0]
    P_start_guess = x[1]
    h_start_guess = x[2]

    # Constraints (soft)
    if m_dot_guess <= 0.0001: m_dot_guess = 0.0001
    if P_start_guess < 1e5: P_start_guess = 1e5
    if h_start_guess < 100000: h_start_guess = 100000

    # 1. Inlet Header
    out_1 = model_inlet_header(m_dot_guess, P_start_guess, h_start_guess)

    # 2. Orifice
    out_2 = model_orifice(out_1)

    # 3. Heated Section (34 kW)
    out_3 = model_heated_section(out_2, Q_total_kW=33.6)

    # 4. Unheated Section
    out_4 = model_unheated_section(out_3)

    # 5. Elbow 1
    out_5 = model_generic_elbow(out_4, L=0.6, Angle=90.0, name="Elbow 1"  )
    
    # 6. Horizontal Riser
    out_6 = model_horizontal_riser(out_5, 9.45)
    
    # 7. Elbow 2
    out_7 = model_generic_elbow(out_6, L=0.2, Angle=90.0, name="Elbow 2" )
    
    # 8. Vertical Riser
    out_8 = model_vertical_riser(out_7, 10.7, 87)
    
    # 9. Siphon
    out_9 = model_siphon(out_8)

    # 10. Condenser
    res_cond = model_condenser_detailed(out_9, T_pool_C=100.0)

    # 11. DOWNCOMER A
    res_dwn_a = model_vertical_downcomer_a(res_cond)

    # 12. Elbow 3
    out_11 = model_generic_elbow(res_dwn_a, L=0.2, Angle=90.0, name="Elbow 3" )
    
    # 13. HORIZONTAL DOWNCOMER
    res_dwn_h = model_horizontal_downcomer(out_11)

    # 14. Elbow 4
    out_12 = model_generic_elbow(res_dwn_h, L=0.2, Angle=90.0, name="Elbow 4" )
    
    # 15. DOWNCOMER B
    res_dwn_b = model_vertical_downcomer_b(out_12)

    # --- CLOSURE ---
    # Final state should match initial state for steady loop
    P_final = res_dwn_b['P']
    h_final = res_dwn_b['h']
    
    # --- TOTAL MASS ---
    total_mass = (
        out_1['mass'] + out_2['mass'] + out_3['mass'] + out_4['mass'] +
        out_5['mass'] + out_6['mass'] + out_7['mass'] + out_8['mass'] +
        out_9['mass'] + res_cond['mass'] + res_dwn_a['mass'] + out_11['mass'] +
        res_dwn_h['mass'] + out_12['mass'] + res_dwn_b['mass']
    )
    
    # Residuals
    # Scaling factors are important for least_squares
    res_P = (P_final - P_start_guess) / 1000.0 # kPa error
    res_h = (h_final - h_start_guess) / 1000.0 # kJ/kg error
    res_M = (total_mass - TARGET_MASS) * 10    # Amplify mass error slightly
    
    return [res_P, res_h, res_M]

if __name__ == "__main__":
    print(f"Target Mass: {TARGET_MASS:.4f} kg (Vol={TARGET_VOLUME}, FR={FILLING_RATIO}, RhoRef={RHO_REF})")
    
    # Initial Guess (from valid point)
    x0 = [0.04406, 1628.69 * 1000, 692 * 1000]
    
    print("Starting optimization...")
    # Run optimization with PRINT_REPORT = False (default is True in config, so let's temporarily set it to False)
    config.PRINT_REPORT = False
    
    res = least_squares(solve_full_loop, x0, bounds=([0.001, 1e5, 1e5], [1.0, 50e5, 3000e3]), verbose=2, ftol=1e-4)
    
    print("\nOptimization Finished.")
    print("Success:", res.success)
    print("Message:", res.message)
    print(f"Solution: m_dot={res.x[0]:.5f} kg/s, P_start={res.x[1]/1000:.2f} kPa, h_start={res.x[2]/1000:.2f} kJ/kg")
    
    # Run Final Report with printing enabled
    config.PRINT_REPORT = True
    print("\n" + "="*80)
    print("FINAL CONVERGED STATE REPORT")
    print("="*80)
    residuals = solve_full_loop(res.x)
    
    print("\nFINAL CHECK:")
    print(f"Residual P: {residuals[0]:.8f} kPa")
    print(f"Residual h: {residuals[1]:.8f} kJ/kg")
    print(f"Residual M: {residuals[2]/10:.8f} kg")
