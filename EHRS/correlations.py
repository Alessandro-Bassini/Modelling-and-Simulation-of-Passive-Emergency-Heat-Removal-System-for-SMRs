def get_two_phase_friction_multiplier(x, rho_l, rho_v, mu_l, mu_v, G, D, sigma=0.03):
    """Correlazione di Friedel (1979) per perdite di carico bifase."""
    if x <= 0.001: return 1.0
    if x >= 0.999: 
        # Rapporto teorico dP_go / dP_lo per continuità
        return (rho_l / rho_v) * (mu_v / mu_l)**0.25 

    Re_lo = (G * D) / mu_l
    Re_go = (G * D) / mu_v
    
    f_lo = 0.079 * Re_lo**(-0.25) if Re_lo > 0 else 0.01
    f_go = 0.079 * Re_go**(-0.25) if Re_go > 0 else 0.01

    rho_r = rho_l / rho_v
    
    E = (1 - x)**2 + x**2 * (rho_r * (f_go / f_lo))
    F = x**0.78 * (1 - x)**0.224
    
    rho_h = 1.0 / (x/rho_v + (1-x)/rho_l)
    We = (G**2 * D) / (sigma * rho_h)
    Fr = G**2 / (9.81 * D * rho_h**2)
    
    # Protezione divisione zero
    denom = (Fr**0.045 * We**0.035)
    if denom < 1e-6: denom = 1e-6
    
    H = (rho_r)**0.91 * (mu_v / mu_l)**0.19 * (1 - mu_v / mu_l)**0.7
    phi_sq = E + 3.24 * F * H / denom
    
    return max(1.0, phi_sq)


def get_variable_U_loss(T_fluid_K):
    """
    Calcola U_loss variabile in base alla temperatura locale.
    Basato sull'osservazione sperimentale:
    - T ~ 155°C (Run 34kW) -> U = 9.0
    - T ~ 175°C (Run 44kW) -> U = 11.0
    """
    T_C = T_fluid_K - 273.15
    
    # Interpolazione Lineare
    # Slope = (11 - 9) / (175 - 155) = 2 / 20 = 0.1
    slope = 0.05
    
    # Punto base: 155°C -> 9.0
    U_calc = 9.0 + slope * (T_C - 155.0)
    
    # Mettiamo dei limiti di sicurezza (Clamping) per evitare valori assurdi
    # Minimo 8.0, Massimo 13.0
    return max(8.0, min(13.0, U_calc))
