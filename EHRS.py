import numpy as np
import math
import sys
from iapws import IAPWS97
from scipy.optimize import least_squares

# ==============================================================================
# CONFIGURAZIONE GLOBALE
# ==============================================================================
USE_IAPWS = True
PRINT_REPORT = True  # <--- IMPOSTATO SU TRUE PER VEDERE I RISULTATI!

# --- FUNZIONE DI SUPPORTO PER LA STAMPA ---
def print_component_report(name, T_in, T_out, P_in, P_out_Pa, m_dot,
                           h_in, h_out, rho_in, rho_out, deltaQ, x_out):
    if not PRINT_REPORT:
        return

    deltaT = T_out - T_in
    deltaP_Pa = P_in - P_out_Pa 

    print(f"\n" + "="*70)
    print(f"REPORT COMPONENTE: {name.upper()}")
    print("="*70)
    print(f"{'GRANDEZZA':<35} | {'VALORE':>15} | {'UNITÀ'}")
    print("-" * 70)
    print(f"{'T_in (Temp. ingresso)':<35} | {T_in:>15.3f} | K")
    print(f"{'T_out (Temp. uscita)':<35} | {T_out:>15.3f} | K")
    print(f"{'Delta T':<35} | {deltaT:>15.3f} | K")
    print("-" * 70)
    print(f"{'P_in (Press. ingresso)':<35} | {P_in/1000:>15.3f} | kPa")
    print(f"{'P_out (Press. uscita)':<35} | {P_out_Pa/1000:>15.3f} | kPa")
    print(f"{'Delta P (Perdita totale)':<35} | {deltaP_Pa/1000:>15.5f} | kPa")
    print("-" * 70)
    print(f"{'Flowrate in':<35} | {m_dot:>15.5f} | kg/s")
    print(f"{'h_in (Entalpia ingresso)':<35} | {h_in/1000:>15.3f} | kJ/kg")
    print(f"{'h_out (Entalpia uscita)':<35} | {h_out/1000:>15.3f} | kJ/kg")
    print("-" * 70)
    print(f"{'Density in':<35} | {rho_in:>15.3f} | kg/m3")
    print(f"{'Density out':<35} | {rho_out:>15.3f} | kg/m3")
    print("-" * 70)
    print(f"{'Delta Q (Scambio termico)':<35} | {deltaQ:>15.2f} | W")
    print(f"{'Qualità massica (x uscita)':<35} | {x_out:>15.4f} | -")
    print("="*70)


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


# --- CLASSE PROPRIETÀ ACQUA ---
# --- CLASSE PROPRIETÀ ACQUA (AGGIORNATA CON K e CP per CONDENSATORE) ---
class WaterProperties:
    """
    Wrapper robusto per IAPWS97.
    Garantisce che non vengano mai restituiti None per le proprietà critiche
    usando fallback ai valori di saturazione quando necessario.
    """
    def _init_(self):
        pass

    def use_lib(self, use_iapws=True):
        self.use_lib = use_iapws

    def get_properties(self, P_Pa, h_Jkg):
        try:
            P_MPa = P_Pa / 1e6
            h_kJ = h_Jkg / 1000.0
            
            # 1. Calcolo stati di saturazione alla pressione data
            sat_liq = IAPWS97(P=P_MPa, x=0)
            sat_vap = IAPWS97(P=P_MPa, x=1)
            
            h_l = sat_liq.h * 1000.0
            h_v = sat_vap.h * 1000.0
            
            # 2. Calcolo Titolo (Quality) manuale basato sull'entalpia
            # Questo evita che la libreria ritorni None o valori strani
            if h_Jkg <= h_l:
                x_qual = 0.0
                region = "liquid"
            elif h_Jkg >= h_v:
                x_qual = 1.0
                region = "vapor"
            else:
                x_qual = (h_Jkg - h_l) / (h_v - h_l)
                region = "two_phase"

            # 3. Tentativo calcolo stato puntuale
            try:
                state = IAPWS97(P=P_MPa, h=h_kJ)
                T_K = state.T
                rho = state.rho
                mu = state.mu
                cp = state.cp
                k_th = state.k
                sigma = state.sigma
            except:
                # Fallback se il calcolo puntuale fallisce (es. instabilità numerica)
                # Usiamo proprietà pesate o di saturazione
                T_K = sat_liq.T # T sat
                sigma = sat_liq.sigma
                if region == "two_phase":
                    rho = 1.0 / (x_qual/sat_vap.rho + (1-x_qual)/sat_liq.rho)
                    mu  = None # Da gestire nel chiamante per bifase
                    cp  = None
                    k_th = None
                elif region == "liquid":
                    rho = sat_liq.rho
                    mu  = sat_liq.mu
                    cp  = sat_liq.cp
                    k_th = sat_liq.k
                else: # Vapor
                    rho = sat_vap.rho
                    mu  = sat_vap.mu
                    cp  = sat_vap.cp
                    k_th = sat_vap.k

            return {
                'P': P_Pa,
                'h': h_Jkg,
                'T_K': T_K,
                'rho': rho,
                'mu': mu,
                'cp': cp,
                'k_th': k_th,
                'x': x_qual,
                'region': region,
                # Dati saturazione (sempre utili)
                'rho_l': sat_liq.rho,
                'rho_v': sat_vap.rho,
                'mu_l': sat_liq.mu,
                'mu_v': sat_vap.mu,
                'cp_l': sat_liq.cp,
                'cp_v': sat_vap.cp,
                'k_l': sat_liq.k,
                'k_v': sat_vap.k,
                'h_l': h_l,
                'h_v': h_v,
                'sigma': sat_liq.sigma if hasattr(sat_liq, 'sigma') else 0.03
            }
        except Exception as e:
            print(f"CRITICAL ERROR in properties P={P_Pa}, h={h_Jkg}: {e}")
            return None


# ==============================================================================
# MODELLI COMPONENTI
# ==============================================================================

def model_inlet_header(m_dot_in, P_in, h_in  ):
    
    # 1. GEOMETRIA
    L = 1.1
    D_in = 0.02664
    Area = math.pi * (D_in / 2)**2

    wp = WaterProperties()

    # 2. STATO INGRESSO
    props_in = wp.get_properties( P_in, h_in)
    rho_in = props_in['rho']
    mu_in  = props_in['mu']
    T_in_K = props_in['T_K']

    # 3. IDRAULICA
    velocity = m_dot_in / (rho_in * Area)
    Re = (rho_in * velocity * D_in) / mu_in if (mu_in is not None and mu_in > 0) else 0
    f = 0.096 * (Re**(-0.25)) if Re > 0 else 0

    dP_friction = f * (L / D_in) * (rho_in * velocity**2 / 2)
    dP_total = dP_friction

    # 4. TERMODINAMICA
    deltaQ = 0.0
    h_out_Jkg = h_in
    P_out_Pa =  P_in - dP_total

    # 5. STATO USCITA
    props_out = wp.get_properties(P_out_Pa, h_out_Jkg)
    rho_out = props_out['rho']
    T_out_K = props_out['T_K']
    x_out   = props_out['x']

    # --- MASS CALCULATION ---
    # Lumped approximation
    rho_avg = (rho_in + rho_out) / 2
    comp_vol = Area * L
    comp_mass = rho_avg * comp_vol

    # 6. STAMPA REPORT
    print_component_report(
        name="Test Section Inlet Header",
        T_in=T_in_K, T_out=T_out_K,
         P_in= P_in, P_out_Pa=P_out_Pa,
        m_dot=m_dot_in,
        h_in=h_in, h_out=h_out_Jkg,
        rho_in=rho_in, rho_out=rho_out,
        deltaQ=deltaQ, x_out=x_out
    )

    return {'m_dot': m_dot_in, 'P': P_out_Pa, 'h': h_out_Jkg, 'mass': comp_mass, 'vol': comp_vol}


def model_orifice(input_state  ):
    m_dot_in = input_state['m_dot']
    P_in  = input_state['P']
    h_in = input_state['h']

    L = 0.56
    D_pipe = 0.01253
    Area = math.pi * (D_pipe / 2)**2

    wp = WaterProperties( )
    props_in = wp.get_properties(P_in, h_in)
    
    rho_in = props_in['rho']
    mu_in  = props_in['mu']

    velocity = m_dot_in / (rho_in * Area)
    Re = (rho_in * velocity * D_pipe) / mu_in if (mu_in and mu_in > 0) else 0

    f = 0.046 * (Re**(-0.2)) if Re > 0 else 0
    dP_dist = f * (L / D_pipe) * (rho_in * velocity**2 / 2)

    K_orif = 38.38 * (1 - math.exp(-Re / 728.8)) if Re > 0 else 0
    dP_conc = K_orif * (rho_in * velocity**2 / 2)
    dP_grav = rho_in * 9.81 * L # Verticale

    P_out_Pa = P_in - (dP_dist + dP_conc + dP_grav)
    h_out_Jkg = h_in
    
    props_out = wp.get_properties(P_out_Pa, h_out_Jkg)

    print_component_report("Orifice Section", 
                           props_in['T_K'], props_out['T_K'], P_in, P_out_Pa, m_dot_in, 
                           h_in, h_out_Jkg, rho_in, props_out['rho'], 0.0, props_out['x'])

    return {'m_dot': m_dot_in, 'P': P_out_Pa, 'h': h_out_Jkg}


def model_heated_section(input_state, Q_total_kW  ):

    m_dot = input_state['m_dot']
    P_in  = input_state['P']
    h_in  = input_state['h']

    # --- GEOMETRIA ---
    L_total = 24.0    
    D_in    = 0.01253 
    D_out   = 0.01715 
    Angle   = 14.3    
    R_coil  = 0.5     
    Area    = math.pi * (D_in/2)**2
    Perim_out = math.pi * D_out 

    # --- TUNING DEL MODELLO (CALIBRAZIONE) ---
    # Il fattore standard di Ito è 3.5. 
    # Se sovrastimi le perdite, abbassa questo valore (es. prova 1.8 - 2.0).
    HELIX_CORRECTION_FACTOR = 1.8  
    
    # --- INEFFICIENZA ---
    U_loss = 15.0          
    T_amb  = 100.0 + 273.15 

    # --- DISCRETIZZAZIONE ---
    N_NODES = 100     
    dL      = L_total / N_NODES
    dz_node = dL * math.sin(math.radians(Angle))
    
    dQ_electric_node = (Q_total_kW * 1000.0) / N_NODES

    wp = WaterProperties( )
    
    P_curr = P_in
    h_curr = h_in
    G_flux = m_dot / Area 
    
    Q_net_accumulated = 0.0
    total_mass = 0.0
    total_vol = 0.0

    # Helper Vmom
    def get_v_mom(props, G):
        if props['x'] <= 0: return 1.0/props['rho']
        if props['x'] >= 1: return 1.0/props['rho']
        rho_l, rho_v = props['rho_l'], props['rho_v']
        x = props['x']
        j_g = (x * G) / rho_v
        j_tot = G * (x/rho_v + (1-x)/rho_l)
        sigma = props.get('sigma', 0.03)
        Vgj = 1.53 * ((sigma * 9.81 * (rho_l - rho_v)) / rho_l**2)**0.25
        alpha = j_g / (1.2 * j_tot + Vgj)
        alpha = max(1e-5, min(0.9999, alpha))
        return (x**2)/(alpha*rho_v) + ((1-x)**2)/((1-alpha)*rho_l)

    props_in = wp.get_properties(P_curr, h_curr)
    v_mom_curr = get_v_mom(props_in, G_flux)

    for i in range(N_NODES):
        # 1. Energia
        props_step = wp.get_properties(P_curr, h_curr) 
        dQ_loss = U_loss * (Perim_out * dL) * (props_step['T_K'] - T_amb)
        if dQ_loss < 0: dQ_loss = 0.0
        
        dQ_net = dQ_electric_node - dQ_loss
        h_next = h_curr + dQ_net / m_dot
        Q_net_accumulated += dQ_net

        # 2. Accelerazione
        props_next = wp.get_properties(P_curr, h_next)
        v_mom_next = get_v_mom(props_next, G_flux)
        dP_accel = (G_flux**2) * (v_mom_next - v_mom_curr)

        # 3. Media nodo
        h_avg = (h_curr + h_next) / 2
        props = wp.get_properties(P_curr, h_avg)
        x = props['x']
        rho_l, rho_v = props['rho_l'], props['rho_v']
        mu_l, mu_v = props.get('mu_l', 1e-4), props.get('mu_v', 1e-5)
        sigma = props.get('sigma', 0.03)

        # Fattore correttivo geometrico (Ito modificato per tuning)
        # Si applica sia al liquido che al vapore nei calcoli Friedel
        geo_factor = (1.0 + HELIX_CORRECTION_FACTOR * math.sqrt(D_in / (2*R_coil)))

        # --- FRIEDEL ---
        if 0 < x < 1:
            # Drift Flux Alpha (per gravità)
            j_g = (x * G_flux) / rho_v
            j_tot = G_flux * (x/rho_v + (1-x)/rho_l)
            Vgj = 1.53 * ((sigma * 9.81 * (rho_l - rho_v)) / rho_l**2)**0.25
            alpha_avg = j_g / (1.2 * j_tot + Vgj)
            alpha_avg = max(0.0, min(1.0, alpha_avg))
            rho_mix = alpha_avg * rho_v + (1 - alpha_avg) * rho_l
            
            # Friction Factors base (Blasius)
            Re_lo = (G_flux * D_in) / mu_l
            f_lo_base = 0.316 * Re_lo**(-0.25) if Re_lo > 2300 else 64.0/max(Re_lo,1)
            f_lo = f_lo_base * geo_factor # <--- Applicazione Tuning
            
            Re_go = (G_flux * D_in) / mu_v
            f_go_base = 0.316 * Re_go**(-0.25) if Re_go > 2300 else 64.0/max(Re_go,1)
            f_go = f_go_base * geo_factor # <--- Applicazione Tuning

            # Friedel Terms
            E = (1 - x)**2 + x**2 * (rho_l * f_go) / (rho_v * f_lo)
            F = x**0.78 * (1 - x)**0.224
            rho_h = 1.0 / (x/rho_v + (1-x)/rho_l)
            Fr = G_flux**2 / (9.81 * D_in * rho_h**2)
            We = (G_flux**2 * D_in) / (sigma * rho_h)
            H = (rho_l / rho_v)**0.91 * (mu_v / mu_l)**0.19 * (1 - mu_v / mu_l)**0.7
            
            phi_sq = E + (3.24 * F * H) / (Fr**0.045 * We**0.035)
            
            dP_liquid_only = f_lo * (dL/D_in) * (G_flux**2) / (2*rho_l)
            dP_frict = dP_liquid_only * phi_sq
            
        else:
            # Monofase
            rho_mix = props['rho']
            mu_mix = props['mu']
            Re = (G_flux * D_in) / mu_mix if mu_mix else 0
            
            f_base = 0.316 * Re**(-0.25) if Re > 2300 else 64.0/max(Re,1)
            f_helix = f_base * geo_factor # <--- Applicazione Tuning
            
            v_mix = G_flux / rho_mix
            dP_frict = f_helix * (dL/D_in) * (rho_mix * v_mix**2 / 2)
            
        dP_grav = rho_mix * 9.81 * dz_node
        
        dP_total = dP_grav + dP_frict + dP_accel
        P_curr -= dP_total
        h_curr = h_next
        v_mom_curr = v_mom_next
        
        total_mass += rho_mix * Area * dL
        total_vol += Area * dL

    props_final = wp.get_properties(P_curr, h_curr)

    print_component_report("Heated Test Section (SG)", 
                           props_in['T_K'], props_final['T_K'], P_in, P_curr, m_dot, 
                           h_in, h_curr, props_in['rho'], props_final['rho'], 
                           Q_net_accumulated, props_final['x'])

    return {
        'm_dot': m_dot, 'P': P_curr, 'h': h_curr, 'T': props_final['T_K'],
        'mass': total_mass, 'vol': total_vol
    }


def model_unheated_section(input_state  ):
    m_dot = input_state['m_dot']
    P_in  = input_state['P']
    h_in  = input_state['h']

    # --- GEOMETRIA ---
    L_total = 8.0     
    D_in    = 0.01253 
    D_out   = 0.01715 
    Angle   = 14.3    
    R_coil  = 0.5     
    Area    = math.pi * (D_in/2)**2
    Perim_out = math.pi * D_out
    
    # --- TUNING DEL MODELLO ---
    # Stesso fattore della sezione scaldata per coerenza geometrica
    HELIX_CORRECTION_FACTOR = 1.8 

    # --- PERDITE ---
    U_loss = 15.0
    T_amb = 373.15 # 100°C come da tuo snippet

    # --- DISCRETIZZAZIONE ---
    N_NODES = 50
    dL = L_total / N_NODES
    dz_node = dL * math.sin(math.radians(Angle))

    wp = WaterProperties( )
    P_curr = P_in
    h_curr = h_in
    G_flux = m_dot / Area
    
    Q_lost_total = 0.0
    total_mass = 0.0
    total_vol = 0.0
    
    # Helper Vmom
    def get_v_mom(props, G):
        if props['x'] <= 0: return 1.0/props['rho']
        if props['x'] >= 1: return 1.0/props['rho']
        rho_l, rho_v = props['rho_l'], props['rho_v']
        x = props['x']
        j_g = (x * G) / rho_v
        j_tot = G * (x/rho_v + (1-x)/rho_l)
        sigma = props.get('sigma', 0.03)
        Vgj = 1.53 * ((sigma * 9.81 * (rho_l - rho_v)) / rho_l**2)**0.25
        alpha = j_g / (1.2 * j_tot + Vgj)
        alpha = max(1e-5, min(0.9999, alpha))
        return (x**2)/(alpha*rho_v) + ((1-x)**2)/((1-alpha)*rho_l)

    props_in = wp.get_properties(P_curr, h_curr)
    v_mom_curr = get_v_mom(props_in, G_flux)

    for i in range(N_NODES):
        # 1. Energia (Solo Perdite)
        props_step = wp.get_properties(P_curr, h_curr)
        dQ_loss = U_loss * (Perim_out * dL) * (props_step['T_K'] - T_amb)
        if dQ_loss < 0: dQ_loss = 0
        Q_lost_total += dQ_loss

        h_next = h_curr - (dQ_loss / m_dot)
        
        # 2. Accelerazione (Vmom variazione)
        props_next = wp.get_properties(P_curr, h_next)
        v_mom_next = get_v_mom(props_next, G_flux)
        dP_accel = (G_flux**2) * (v_mom_next - v_mom_curr)

        # 3. Media nodo
        h_avg = (h_curr + h_next)/2
        props = wp.get_properties(P_curr, h_avg)
        x = props['x']
        rho_l, rho_v = props['rho_l'], props['rho_v']
        mu_l, mu_v = props.get('mu_l', 1e-4), props.get('mu_v', 1e-5)
        sigma = props.get('sigma', 0.03)

        # Fattore correttivo geometrico (Ito tuning)
        geo_factor = (1.0 + HELIX_CORRECTION_FACTOR * math.sqrt(D_in / (2*R_coil)))

        # --- FRIEDEL ---
        if 0 < x < 1:
            # Drift Flux Alpha (per gravità)
            j_g = (x * G_flux) / rho_v
            j_tot = G_flux * (x/rho_v + (1-x)/rho_l)
            Vgj = 1.53 * ((sigma * 9.81 * (rho_l - rho_v)) / rho_l**2)**0.25
            alpha_avg = j_g / (1.2 * j_tot + Vgj)
            alpha_avg = max(0.0, min(1.0, alpha_avg))
            rho_mix = alpha_avg * rho_v + (1 - alpha_avg) * rho_l
            
            # Friction Factors base (Blasius)
            Re_lo = (G_flux * D_in) / mu_l
            f_lo_base = 0.316 * Re_lo**(-0.25) if Re_lo > 2300 else 64.0/max(Re_lo,1)
            f_lo = f_lo_base * geo_factor 
            
            Re_go = (G_flux * D_in) / mu_v
            f_go_base = 0.316 * Re_go**(-0.25) if Re_go > 2300 else 64.0/max(Re_go,1)
            f_go = f_go_base * geo_factor

            # Friedel Terms
            E = (1 - x)**2 + x**2 * (rho_l * f_go) / (rho_v * f_lo)
            F = x**0.78 * (1 - x)**0.224
            rho_h = 1.0 / (x/rho_v + (1-x)/rho_l)
            Fr = G_flux**2 / (9.81 * D_in * rho_h**2)
            We = (G_flux**2 * D_in) / (sigma * rho_h)
            H = (rho_l / rho_v)**0.91 * (mu_v / mu_l)**0.19 * (1 - mu_v / mu_l)**0.7
            
            phi_sq = E + (3.24 * F * H) / (Fr**0.045 * We**0.035)
            
            dP_liquid_only = f_lo * (dL/D_in) * (G_flux**2) / (2*rho_l)
            dP_frict = dP_liquid_only * phi_sq

        else:
            # Monofase
            rho_mix = props['rho']
            mu_mix = props['mu']
            Re = (G_flux * D_in) / mu_mix if mu_mix else 0
            
            f_base = 0.316 * Re**(-0.25) if Re > 2300 else 64.0/max(Re,1)
            f_helix = f_base * geo_factor
            
            v_mix = G_flux / rho_mix
            dP_frict = f_helix * (dL/D_in) * (rho_mix * v_mix**2 / 2)
            
        dP_grav = rho_mix * 9.81 * dz_node
        
        dP_total = dP_grav + dP_frict + dP_accel

        P_curr -= dP_total
        h_curr = h_next
        v_mom_curr = v_mom_next
        
        total_mass += rho_mix * Area * dL
        total_vol += Area * dL

    props_final = wp.get_properties(P_curr, h_curr)
    
    print_component_report("Unheated Section (Adiabatic)", 
                       props_in['T_K'], props_final['T_K'], P_in, P_curr, m_dot, 
                       h_in, h_curr, props_in['rho'], props_final['rho'], 
                       -Q_lost_total, props_final['x'])

    return {
        'm_dot': m_dot, 'P': P_curr, 'h': h_curr, 'T': props_final['T_K'],
        'mass': total_mass, 'vol': total_vol
    }


def model_generic_elbow(input_state, L, Angle, name="Elbow"):
    m_dot = input_state['m_dot']
    P_in  = input_state['P']
    h_in  = input_state['h']

    D_pipe = 0.02093  
    Area = math.pi * (D_pipe / 2)**2

    wp = WaterProperties()
    props_in = wp.get_properties(P_in, h_in)
    T_in_K = props_in['T_K']
    rho_in = props_in['rho']
    x_in   = props_in['x']

    # --- 1. Calcolo Proprietà di Miscela e Two-Phase Multiplier ---
    if x_in <= 0:
        rho_mix = props_in['rho']
        mu_mix  = props_in['mu']
        phi_2 = 1.0
    elif x_in >= 1: 
        rho_mix = props_in['rho']
        mu_mix  = props_in['mu']
        phi_2 = 1.0
    else: 
        # Modello bifase (identico al tuo originale)
        rho_l = getattr(props_in, 'rho_l', 820.0)
        rho_v = getattr(props_in, 'rho_v', 14.0)
        sigma = 0.03 # N/m, approssimato
        
        G = m_dot / Area
        j_g = (x_in * G) / rho_v
        j_l = ((1 - x_in) * G) / rho_l
        j_tot = j_g + j_l
        
        # Drift Flux Parameters
        C0 = 1.2
        # Vgj è significativo solo se c'è una componente verticale o accelerazione centrifuga forte
        if abs(Angle) > 10:
            g = 9.81
            Vgj = 1.53 * ((sigma * g * (rho_l - rho_v)) / rho_l**2)**0.25
        else:
            Vgj = 0.0
            
        if (C0 * j_tot + Vgj) > 0:
            alpha = j_g / (C0 * j_tot + Vgj)
        else:
            alpha = 0
        if alpha > 0.999: alpha = 1.0
        
        rho_mix = alpha * rho_v + (1 - alpha) * rho_l
        
        # Viscosità di miscela (McAdams)
        mu_l = getattr(props_in, 'mu_l', 1.1e-4)
        mu_v = getattr(props_in, 'mu_v', 1.8e-5)
        mu_mix = 1 / (x_in/mu_v + (1-x_in)/mu_l)
        
        # Friedel o Chisholm B (qui usato un semplice moltiplicatore omogeneo/Chisholm base)
        phi_2 = 1.0 + x_in * (rho_l / rho_v - 1.0)

    velocity = m_dot / (rho_mix * Area)
    
    # --- 2. Calcolo Reynolds e Attrito Distribuito (f) ---
    if mu_mix is not None and mu_mix > 0:
        Re = (rho_mix * velocity * D_pipe) / mu_mix
    else:
        Re = 0

    if Re > 1000:
        f = 0.316 * (Re**(-0.25)) # Blasius (più accurata di 0.049*Re^-0.2 per tubi lisci)
    elif Re > 0:
        f = 64.0 / Re
    else:
        f = 0

    # --- 3. NUOVO MODELLO GEOMETRICO (K factor) ---
    # Calcolo del Raggio di Curvatura (R_curv) basato sulla lunghezza dell'arco L
    theta_rad = math.radians(abs(Angle))
    
    if theta_rad > 0.01 and L > 0:
        R_curv = L / theta_rad
        rr = R_curv / D_pipe  # Rapporto Raggio/Diametro
        
        # Calcolo Coefficiente K_bend (Formula approssimata per curve dolci vs strette)
        # Nota: Per R/D < 1 (gomito stretto), K è alto. Per R/D elevato, K scende.
        # Formula usata: Approssimazione generica basata su dati Crane/Idelchik
        # K_base per 90 gradi:
        if rr < 1.0:
            # Curva molto stretta o spigolo
            K_90 = 1.2 
        else:
            # Fit: K decresce con R/D. 
            # Esempio: 0.13 + 1.85 * (D/2R)^3.5 (Ito's formula semplificata)
            # Aggiungiamo un termine base per tenere conto che non va mai a zero
            K_90 = 0.10 + 1.85 * (1.0 / (2.0 * rr))**3.5
        
        # Scaliamo il K per l'angolo effettivo (lineare o radice quadrata)
        # Per flussi turbolenti spesso si usa: K_theta = K_90 * (theta/90)
        K_bend = K_90 * (abs(Angle) / 90.0)
        
    else:
        # Caso degenere (angolo nullo o lunghezza nulla)
        K_bend = 0.0

    # Pressione Dinamica
    dyn_head = (rho_mix * velocity**2 / 2)

    # --- 4. Calcolo Perdite Totali ---
    # Perdita 1: Attrito distribuito lungo la lunghezza L (pareti)
    dP_wall_friction = f * (L / D_pipe) * dyn_head
    
    # Perdita 2: Perdita concentrata dovuta alla curvatura (vorticità/separazione)
    dP_bend_loss = K_bend * dyn_head
    
    # Totale attrito (moltiplicato per phi_2 se bifase)
    dP_frict_total = (dP_wall_friction + dP_bend_loss) * phi_2

    # --- 5. Gravità ---
    delta_z = L * math.sin(math.radians(Angle)) if abs(Angle) < 90 else L * (2/math.pi) # Approx per altezza se curva verticale
    # Nota: Se è un gomito verticale standard (90°), delta_z non è L*sin(90)=L, ma 2*R/pi * theta... 
    # Manteniamo la tua approssimazione lineare L*sin(Angle) se L è intesa come dislivello o lunghezza tubo inclinato
    # Se L è la lunghezza dell'arco di una curva a 90°, il dislivello è:
    if abs(Angle - 90.0) < 5.0:
        # Per una curva a 90°, l'altezza verticale è R = L / (pi/2)
        delta_z = L / (math.pi / 2.0)
    else:
        delta_z = L * math.sin(math.radians(Angle))

    dP_grav = rho_mix * 9.81 * delta_z
    
    dP_total = dP_frict_total + dP_grav

    h_out_Jkg = h_in
    P_out_Pa = P_in - dP_total

    props_out = wp.get_properties(P_out_Pa, h_out_Jkg)
    T_out_K = props_out['T_K']

    # --- MASS ---
    comp_vol = Area * L
    comp_mass = rho_mix * comp_vol

    # --- REPORT DEBUG ---
    # Utile per vedere se K_bend sta lavorando
    print(f"  [DEBUG {name}] R/D={R_curv/D_pipe if theta_rad>0 else 0:.2f}, K_bend={K_bend:.4f}, f_pipe={f:.4f}")

    print_component_report(
        name=f"{name} (L={L}m, {Angle}°)",
        T_in=T_in_K, T_out=T_out_K,
        P_in=P_in, P_out_Pa=P_out_Pa,
        m_dot=m_dot,
        h_in=h_in, h_out=h_out_Jkg,
        rho_in=rho_in, rho_out=props_out['rho'],
        deltaQ=0.0, x_out=props_out['x']
    )

    return {
        'm_dot': m_dot, 'P': P_out_Pa, 'h': h_out_Jkg, 'T': T_out_K,
        'mass': comp_mass, 'vol': comp_vol
    }


def model_horizontal_riser(input_state, length=9.45):
    m_dot = input_state['m_dot']
    P_in  = input_state['P']
    h_in  = input_state['h']

    # --- GEOMETRIA (3/4" S40) ---
    D_in = 0.02093   
    D_out = 0.02667
    Area = math.pi * (D_in/2)**2
    Perim_out = math.pi * D_out
    
    # --- PERDITE ---
    U_loss = 10.0
    T_amb = 373.15 # 100°C

    # --- DISCRETIZZAZIONE ---
    N_NODES = 50
    dL = length / N_NODES
    dz_node = 0.0 

    wp = WaterProperties( )
    P_curr = P_in; h_curr = h_in
    G_flux = m_dot / Area
    
    total_mass = 0.0; total_vol = 0.0
    Q_lost_tot = 0.0 # <--- Accumulatore perdite
    
    # Helper Vmom
    def get_v_mom(props, G):
        if props['x'] <= 0: return 1.0/props['rho']
        if props['x'] >= 1: return 1.0/props['rho']
        rho_l, rho_v = props['rho_l'], props['rho_v']
        x = props['x']
        j_g = (x * G) / rho_v
        j_tot = G * (x/rho_v + (1-x)/rho_l)
        alpha = j_g / (1.2 * j_tot) 
        alpha = max(1e-5, min(0.9999, alpha))
        return (x**2)/(alpha*rho_v) + ((1-x)**2)/((1-alpha)*rho_l)

    props_in = wp.get_properties(P_curr, h_curr)
    v_mom_curr = get_v_mom(props_in, G_flux)

    for i in range(N_NODES):
        # 1. Energia
        props_step = wp.get_properties(P_curr, h_curr)
        
        # Calcolo dispersione
        dQ_loss = U_loss * (Perim_out * dL) * (props_step['T_K'] - T_amb)
        if dQ_loss < 0: dQ_loss = 0
        
        # Accumulo per il report finale
        Q_lost_tot += dQ_loss 

        h_next = h_curr - (dQ_loss / m_dot)
        
        # 2. Accelerazione
        props_next = wp.get_properties(P_curr, h_next)
        v_mom_next = get_v_mom(props_next, G_flux)
        dP_accel = (G_flux**2) * (v_mom_next - v_mom_curr)

        # 3. Media nodo
        h_avg = (h_curr + h_next)/2
        props = wp.get_properties(P_curr, h_avg)
        x = props['x']
        rho_l, rho_v = props['rho_l'], props['rho_v']
        mu_l, mu_v = props.get('mu_l', 1e-4), props.get('mu_v', 1e-5)
        sigma = props.get('sigma', 0.03)

        # --- FRIEDEL (Orizzontale) ---
        if 0 < x < 1:
            j_g = (x * G_flux) / rho_v
            j_tot = G_flux * (x/rho_v + (1-x)/rho_l)
            alpha_avg = j_g / (1.2 * j_tot)
            alpha_avg = max(0.0, min(1.0, alpha_avg))
            rho_mix = alpha_avg * rho_v + (1 - alpha_avg) * rho_l
            
            Re_lo = (G_flux * D_in) / mu_l
            f_lo = 0.316 * Re_lo**(-0.25) if Re_lo > 2300 else 64.0/max(Re_lo,1)
            Re_go = (G_flux * D_in) / mu_v
            f_go = 0.316 * Re_go**(-0.25) if Re_go > 2300 else 64.0/max(Re_go,1)

            E = (1 - x)**2 + x**2 * (rho_l * f_go) / (rho_v * f_lo)
            F = x**0.78 * (1 - x)**0.224
            rho_h = 1.0 / (x/rho_v + (1-x)/rho_l)
            Fr = G_flux**2 / (9.81 * D_in * rho_h**2)
            We = (G_flux**2 * D_in) / (sigma * rho_h)
            H = (rho_l / rho_v)**0.91 * (mu_v / mu_l)**0.19 * (1 - mu_v / mu_l)**0.7
            
            phi_sq = E + (3.24 * F * H) / (Fr**0.045 * We**0.035)
            
            dP_liquid_only = f_lo * (dL/D_in) * (G_flux**2) / (2*rho_l)
            dP_frict = dP_liquid_only * phi_sq
        else:
            rho_mix = props['rho']
            mu_mix = props['mu']
            Re = (G_flux * D_in) / mu_mix if mu_mix else 0
            f_base = 0.316 * Re**(-0.25) if Re > 2300 else 64.0/max(Re,1)
            v_mix = G_flux / rho_mix
            dP_frict = f_base * (dL/D_in) * (rho_mix * v_mix**2 / 2)
        
        dP_grav = 0.0
        
        dP_total = dP_grav + dP_frict + dP_accel
        P_curr -= dP_total
        h_curr = h_next
        v_mom_curr = v_mom_next
        
        total_mass += rho_mix * Area * dL
        total_vol += Area * dL

    props_final = wp.get_properties(P_curr, h_curr)
    
    # QUI HO CORRETTO: Passiamo -Q_lost_tot invece di 0.0
    print_component_report(f"Horizontal Riser (L={length}m)", 
                       props_in['T_K'], props_final['T_K'], P_in, P_curr, m_dot, 
                       h_in, h_curr, props_in['rho'], props_final['rho'], 
                       -Q_lost_tot, props_final['x'])
                       
    return {'m_dot': m_dot, 'P': P_curr, 'h': h_curr, 'mass': total_mass, 'vol': total_vol}


def model_vertical_riser(input_state, length, angle_deg=90.0):
    m_dot = input_state['m_dot']
    P_in  = input_state['P']
    h_in  = input_state['h']

    # --- GEOMETRIA ---
    D_in = 0.02093 
    D_out = 0.02667
    Area = math.pi * (D_in/2)**2
    Perim_out = math.pi * D_out
    
    # --- PERDITE ---
    U_loss = 10.0; T_amb = 373.15

    N_NODES = 50
    dL = length / N_NODES
    dz_node = dL * math.sin(math.radians(angle_deg))
    
    wp = WaterProperties( )
    P_curr = P_in; h_curr = h_in
    G_flux = m_dot / Area
    
    total_mass = 0.0; total_vol = 0.0
    Q_lost_tot = 0.0 # <--- Accumulatore perdite
    
    # Helper Vmom
    def get_v_mom(props, G):
        if props['x'] <= 0: return 1.0/props['rho']
        if props['x'] >= 1: return 1.0/props['rho']
        rho_l, rho_v = props['rho_l'], props['rho_v']
        x = props['x']
        j_g = (x * G) / rho_v
        j_tot = G * (x/rho_v + (1-x)/rho_l)
        sigma = props.get('sigma', 0.03)
        Vgj = 1.53 * ((sigma * 9.81 * (rho_l - rho_v)) / rho_l**2)**0.25
        alpha = j_g / (1.2 * j_tot + Vgj)
        alpha = max(1e-5, min(0.9999, alpha))
        return (x**2)/(alpha*rho_v) + ((1-x)**2)/((1-alpha)*rho_l)

    props_in = wp.get_properties(P_curr, h_curr)
    v_mom_curr = get_v_mom(props_in, G_flux)

    for i in range(N_NODES):
        # 1. Energia
        props_step = wp.get_properties(P_curr, h_curr)
        
        # Calcolo dispersione
        dQ_loss = U_loss * (Perim_out * dL) * (props_step['T_K'] - T_amb)
        if dQ_loss < 0: dQ_loss = 0
        
        # Accumulo
        Q_lost_tot += dQ_loss

        h_next = h_curr - (dQ_loss / m_dot)
        
        # 2. Accelerazione
        props_next = wp.get_properties(P_curr, h_next)
        v_mom_next = get_v_mom(props_next, G_flux)
        dP_accel = (G_flux**2) * (v_mom_next - v_mom_curr)

        # 3. Media nodo
        h_avg = (h_curr + h_next)/2
        props = wp.get_properties(P_curr, h_avg)
        x = props['x']
        rho_l, rho_v = props['rho_l'], props['rho_v']
        mu_l, mu_v = props.get('mu_l', 1e-4), props.get('mu_v', 1e-5)
        sigma = props.get('sigma', 0.03)

        # --- FRIEDEL (Standard) ---
        if 0 < x < 1:
            j_g = (x * G_flux) / rho_v
            j_tot = G_flux * (x/rho_v + (1-x)/rho_l)
            Vgj = 1.53 * ((sigma * 9.81 * (rho_l - rho_v)) / rho_l**2)**0.25
            alpha_avg = j_g / (1.2 * j_tot + Vgj)
            alpha_avg = max(0.0, min(1.0, alpha_avg))
            rho_mix = alpha_avg * rho_v + (1 - alpha_avg) * rho_l
            
            Re_lo = (G_flux * D_in) / mu_l
            f_lo = 0.316 * Re_lo**(-0.25) if Re_lo > 2300 else 64.0/max(Re_lo,1)
            Re_go = (G_flux * D_in) / mu_v
            f_go = 0.316 * Re_go**(-0.25) if Re_go > 2300 else 64.0/max(Re_go,1)

            E = (1 - x)**2 + x**2 * (rho_l * f_go) / (rho_v * f_lo)
            F = x**0.78 * (1 - x)**0.224
            rho_h = 1.0 / (x/rho_v + (1-x)/rho_l)
            Fr = G_flux**2 / (9.81 * D_in * rho_h**2)
            We = (G_flux**2 * D_in) / (sigma * rho_h)
            H = (rho_l / rho_v)**0.91 * (mu_v / mu_l)**0.19 * (1 - mu_v / mu_l)**0.7
            
            phi_sq = E + (3.24 * F * H) / (Fr**0.045 * We**0.035)
            
            dP_liquid_only = f_lo * (dL/D_in) * (G_flux**2) / (2*rho_l)
            dP_frict = dP_liquid_only * phi_sq
        else:
            rho_mix = props['rho']
            mu_mix = props['mu']
            Re = (G_flux * D_in) / mu_mix if mu_mix else 0
            f_base = 0.316 * Re**(-0.25) if Re > 2300 else 64.0/max(Re,1)
            v_mix = G_flux / rho_mix
            dP_frict = f_base * (dL/D_in) * (rho_mix * v_mix**2 / 2)
            
        dP_grav = rho_mix * 9.81 * dz_node
        
        dP_total = dP_grav + dP_frict + dP_accel
        P_curr -= dP_total
        h_curr = h_next
        v_mom_curr = v_mom_next
        
        total_mass += rho_mix * Area * dL
        total_vol += Area * dL

    props_final = wp.get_properties(P_curr, h_curr)
    
    # QUI HO CORRETTO: Passiamo -Q_lost_tot
    print_component_report(f"Vertical Riser (L={length}m)", 
                       props_in['T_K'], props_final['T_K'], P_in, P_curr, m_dot, 
                       h_in, h_curr, props_in['rho'], props_final['rho'], 
                       -Q_lost_tot, props_final['x'])
                       
    return {'m_dot': m_dot, 'P': P_curr, 'h': h_curr, 'mass': total_mass, 'vol': total_vol}


def model_siphon(input_state  ):
    m_dot = input_state['m_dot']
    P_in  = input_state['P']
    h_in  = input_state['h']

    L = 1.0           
    D_pipe = 0.02093  
    Angle = 0.0       
    Area = math.pi * (D_pipe / 2)**2

    wp = WaterProperties( )
    props_in = wp.get_properties(P_in, h_in)
    T_in_K = props_in['T_K']
    rho_in = props_in['rho']
    x_in   = props_in['x']

    if x_in <= 0: 
        rho_mix = props_in['rho']
        mu_mix  = props_in['mu']
        phi_2 = 1.0
    elif x_in >= 1: 
        rho_mix = props_in['rho']
        mu_mix  = props_in['mu']
        phi_2 = 1.0
    else: 
        rho_l = 820.0; rho_v = 14.0 
        if wp.use_lib:
             rho_l = getattr(props_in, 'rho_l', 820.0)
             rho_v = getattr(props_in, 'rho_v', 14.0)
        sigma = 0.03
        G = m_dot / Area
        j_g = (x_in * G) / rho_v
        j_l = ((1 - x_in) * G) / rho_l
        j_tot = j_g + j_l
        C0 = 1.2
        Vgj = 0.0
        if (C0 * j_tot + Vgj) > 0:
            alpha = j_g / (C0 * j_tot + Vgj)
        else:
            alpha = 0
        if alpha > 0.999: alpha = 1.0
        rho_mix = alpha * rho_v + (1 - alpha) * rho_l
        mu_l = 1.1e-4; mu_v = 1.8e-5
        if wp.use_lib:
            mu_l = getattr(props_in, 'mu_l', mu_l)
            mu_v = getattr(props_in, 'mu_v', mu_v)
        mu_mix = 1 / (x_in/mu_v + (1-x_in)/mu_l)
        phi_2 = 1.0 + x_in * (rho_l / rho_v - 1.0)

    velocity = m_dot / (rho_mix * Area)
    if mu_mix is not None and mu_mix > 0:
        Re = (rho_mix * velocity * D_pipe) / mu_mix
    else:
        Re = 0

    if Re > 0:
        f = 0.184 * (Re**(-0.2))
    else:
        f = 0

    dP_frict = f * (L / D_pipe) * (rho_mix * velocity**2 / 2) * phi_2
    dP_grav = 0.0
    dP_total = dP_frict + dP_grav

    h_out_Jkg = h_in 
    P_out_Pa = P_in - dP_total
    props_out = wp.get_properties(P_out_Pa, h_out_Jkg)
    T_out_K = props_out['T_K']

    # --- MASS ---
    comp_vol = Area * L
    comp_mass = rho_mix * comp_vol

    # --- 7. STAMPA REPORT ---
    print_component_report(
        name="Siphon (Top Horizontal Segment)",
        T_in=T_in_K, T_out=T_out_K,
        P_in=P_in, P_out_Pa=P_out_Pa,
        m_dot=m_dot,
        h_in=h_in, h_out=h_out_Jkg,
        rho_in=rho_in, rho_out=props_out['rho'],
        deltaQ=0.0, x_out=props_out['x']
    )

    return {
        'm_dot': m_dot, 'P': P_out_Pa, 'h': h_out_Jkg, 'T': T_out_K,
        'mass': comp_mass, 'vol': comp_vol
    }



    
    # --- A. GEOMETRIA ---
    L = 9.45             # [m] Lunghezza verticale
    D_in = 0.02093       # [m] Diametro interno (3/4" S40)
    Angle = -90.0        # [gradi] Verticale discendente
    
    Area_flow = math.pi * (D_in / 2)**2
    
    # --- B. PARAMETRI SCAMBIO TERMICO (DISPERSIONI) ---
    # Modelliamo la dispersione verso l'ambiente (Newton's Cooling Law)
    # Q_loss = U_loss * Area_surf * (T_fluid - T_amb)
    # Calibrazione: Dai dati Excel, le perdite totali del downcomer (~26m) sono ~3.2 kW.
    # Quindi circa 120 W/m. 
    # Con un DeltaT medio di (180 - 25) = 155°C:
    # U ~ 120 / (pi * D * 155) ~ 11-12 W/m2K
    
    U_iso_loss = 11.5    # [W/m2K] Coefficiente globale dispersione (coibentazione)
    
    # --- C. INIZIALIZZAZIONE ---
    wp = WaterProperties()
    
    # Discretizzazione (10 nodi sono sufficienti per liquido sottoraffreddato)
    N_NODES = 20
    dL = L / N_NODES
    # Dislivello negativo (scendiamo)
    dz_node = dL * math.sin(math.radians(Angle)) 
    Area_surf_node = math.pi * D_in * dL
    
    P_curr = P_in
    h_curr = h_in
    
    total_Q_loss = 0.0
    
    print(f"DEBUG: Start Downcomer A. P_in={P_in/1000:.1f} kPa")
    
    for i in range(N_NODES):
        # 1. Proprietà
        props = wp.get_properties(P_curr, h_curr)
        if props is None: break
        
        T_fluid = props['T_K']
        x = props['x']
        rho = props['rho'] if props['rho'] else props['rho_l'] # Fallback safe
        mu = props['mu'] if props['mu'] else props['mu_l']
        
        # --- 2. PERDITE DI CALORE (Dispersione) ---
        # Legge di Newton: Q = U * A * (T - T_amb)
        dT_loss = T_fluid - (T_amb_C + 273.15)
        dQ_loss = U_iso_loss * Area_surf_node * dT_loss # [Watt]
        
        # Aggiornamento Entalpia (il fluido si raffredda)
        h_new = h_curr - (dQ_loss / m_dot)
        total_Q_loss += dQ_loss
        
        # --- 3. IDRAULICA ---
        # Velocità massa
        G = m_dot / Area_flow
        
        # Attrito
        if x <= 0.001: # Liquido (Atteso)
            Re = (G * D_in) / mu
            # Correlazione Blasius/McAdams per tubi lisci/acciaio
            f = 0.316 * Re**(-0.25) if Re > 0 else 0.02
            dP_frict = f * (dL/D_in) * (G**2 / (2 * rho))
            
            rho_grav = rho
        else:
            # Bifase (Caso raro/anomalo in downcomer, ma gestito)
            # Usiamo Friedel
            phi_sq = get_two_phase_friction_multiplier(x, props['rho_l'], props['rho_v'], 
                                                       props['mu_l'], props['mu_v'], G, D_in)
            Re_lo = (G * D_in) / props['mu_l']
            f_lo = 0.316 * Re_lo**(-0.25) if Re_lo > 0 else 0.02
            dP_lo = f_lo * (dL/D_in) * (G**2 / (2 * props['rho_l']))
            
            dP_frict = dP_lo * phi_sq
            
            # Densità per gravità (Omogenea)
            rho_grav = 1.0 / (x/props['rho_v'] + (1-x)/props['rho_l'])

        # Gravità
        # dz è negativo (-9.45m totali).
        # Pressione deve AUMENTARE.
        # Formula: dP = - rho * g * dz  ->  - (rho * 9.81 * -dL) = + positivo
        dP_grav = - rho_grav * 9.81 * dz_node
        
        # Bilancio Pressione
        # P_new = P_old + (Guadagno Gravità) - (Perdita Attrito)
        P_new = P_curr + dP_grav - dP_frict
        
        # Step
        h_curr = h_new
        P_curr = P_new

    # --- OUTPUT ---
    props_out = wp.get_properties(P_curr, h_curr)
    
    print_component_report("Vertical Downcomer A", 
                           props['T_K'], props_out['T_K'], # Approx T_in
                           P_in, P_curr, 
                           m_dot, h_in, h_curr, 
                           total_Q_loss/1000, props_out['x'])
    
    print(f"  Calo T dovuto a dispersioni: {(props['T_K'] - props_out['T_K']):.2f} °C")
    print(f"  Recupero Pressione (Head):   {(P_curr - P_in)/1000:.2f} kPa")

    return {
        'P_out': P_curr,
        'h_out': h_curr,
        'T_out': props_out['T_K'],
        'x_out': props_out['x'],
        'Q_loss_kW': total_Q_loss/1000
    }


def model_condenser_detailed(input_state, T_pool_C=100.0):
    m_dot = input_state['m_dot']
    P_in  = input_state['P']
    h_in  = input_state['h']
    # --- A. GEOMETRIA ---
    L = 1.0              
    D_in = 0.059         
    D_out = 0.073025     
    Angle = -3.0         
    
    k_steel = 16.3 # AISI 316
    
    Area_flow = math.pi * (D_in / 2)**2
    Area_surf_node = (math.pi * D_in * L) / 50.0
    
    # --- B. PARAMETRI DI CALIBRAZIONE (EXTREME TUNING) ---
    # Per supportare 26kW attraverso 7mm di acciaio, serve un htc esterno enorme.
    h_pool_boiling = 25000.0 
    
    # Fattore Tuning Condensazione
    # Poiché la parete consuma 55°C di DeltaT, la condensazione interna deve avvenire
    # con un DeltaT residuo piccolissimo. Questo implica un htc interno altissimo.
    # Fisicamente giustificato dal regime fortemente stratificato (film sottile).
    tuning_factor_int = 12.0 
    
    # --- C. INIZIALIZZAZIONE ---
    wp = WaterProperties()
    
    N_NODES = 50
    dL = L / N_NODES
    dz_node = dL * math.sin(math.radians(Angle))
    
    P_curr = P_in
    h_curr = h_in
    
    total_Q_removed = 0.0
    
    # Accumulatori statistiche
    avg_R_int = 0.0
    avg_R_wall = 0.0
    avg_R_ext = 0.0
    valid_nodes = 0
    
    print(f"DEBUG: Condenser (Wall Active). P_in={P_in/1000:.1f} kPa")
    
    for i in range(N_NODES):
        props = wp.get_properties(P_curr, h_curr)
        if props is None: break
        
        x = props['x']
        T_fluid = props['T_K']
        P_reduced = (P_curr / 1e6) / 22.064
        G = m_dot / Area_flow
        
        # --- 1. HTC INTERNO ---
        h_int_base = 500.0
        mode = "Unknown"

        if x >= 0.999: # Superheat
            mode = "Superheat"
            mu_g = props['mu'] if props['mu'] else props['mu_v']
            k_g  = props['k_th'] if props['k_th'] else props['k_v']
            cp_g = props['cp'] * 1000 if props['cp'] else props['cp_v'] * 1000
            Re = (G * D_in) / mu_g
            Pr = (cp_g * mu_g) / k_g
            Nu = 0.023 * Re*0.8 * Pr*0.4
            h_int_base = Nu * k_g / D_in
            
        elif x <= 0.001: # Subcooled
            mode = "Subcooled"
            mu_l = props['mu'] if props['mu'] else props['mu_l']
            k_l  = props['k_th'] if props['k_th'] else props['k_l']
            cp_l = props['cp'] * 1000 if props['cp'] else props['cp_l'] * 1000
            Re = (G * D_in) / mu_l
            Pr = (cp_l * mu_l) / k_l
            Nu = 0.023 * Re*0.8 * Pr*0.3 
            h_int_base = Nu * k_l / D_in * 3.0 # Boost convezione mista
            
        else: # Condensing
            mode = "Condensing"
            mu_l = props['mu_l']
            k_l  = props['k_l']
            cp_l = props['cp_l'] * 1000
            
            Re_lo = (G * D_in) / mu_l
            Pr_l  = (cp_l * mu_l) / k_l
            
            h_LO = 0.023 * (Re_lo*0.8) * (Pr_l*0.4) * (k_l / D_in)
            x_safe = max(0.001, min(0.999, x))
            Z = (1 - x_safe)*0.8 + (3.8 * x_safe*0.76 * (1 - x_safe)*0.04) / (P_reduced*0.38)
            
            h_int_base = h_LO * Z * tuning_factor_int

        # --- 2. RESISTENZE ---
        R_val_int = 1.0 / (h_int_base * Area_surf_node)
        
        # CALCOLO PARETE ATTIVO
        R_val_wall = math.log(D_out / D_in) / (2 * math.pi * k_steel * dL)
        
        # Esterna
        Area_surf_ext = (math.pi * D_out * dL)
        R_val_ext = 1.0 / (h_pool_boiling * Area_surf_ext)
        
        R_tot = R_val_int + R_val_wall + R_val_ext
        dQ = (T_fluid - (T_pool_C + 273.15)) / R_tot
        
        # Stats
        avg_R_int += R_val_int
        avg_R_wall += R_val_wall
        avg_R_ext += R_val_ext
        valid_nodes += 1
        
        # --- 3. AGGIORNAMENTO ---
        h_new = h_curr - (dQ / m_dot)
        total_Q_removed += dQ
        
        # --- 4. IDRAULICA ---
        dP_frict = 0.0
        rho_grav = props['rho'] if props['rho'] else 1.0/(x/props['rho_v']+(1-x)/props['rho_l'])
        
        if mode == "Condensing":
            phi_sq = get_two_phase_friction_multiplier(x, props['rho_l'], props['rho_v'], 
                                                       props['mu_l'], props['mu_v'], G, D_in)
            Re_lo = (G * D_in) / props['mu_l']
            f_lo = 0.316 * Re_lo**(-0.25) if Re_lo > 0 else 0.02
            dP_lo = f_lo * (dL/D_in) * (G**2 / (2 * props['rho_l']))
            dP_frict = dP_lo * phi_sq
            rho_grav = 1.0 / (x/props['rho_v'] + (1-x)/props['rho_l'])
        else:
            mu_curr = props['mu'] if props['mu'] else props['mu_l']
            rho_curr = props['rho'] if props['rho'] else props['rho_l']
            Re = (G * D_in) / mu_curr
            f = 0.316 * Re**(-0.25) if Re > 0 else 0.02
            dP_frict = f * (dL/D_in) * (G**2 / (2 * rho_curr))

        dP_grav = rho_grav * 9.81 * dz_node 
        P_new = P_curr - dP_frict - dP_grav
        
        h_curr = h_new
        P_curr = P_new

    # --- OUTPUT ---
    props_out = wp.get_properties(P_curr, h_curr)
    
    if valid_nodes > 0:
        avg_R_int /= valid_nodes
        avg_R_wall /= valid_nodes
        avg_R_ext /= valid_nodes
        R_sum = avg_R_int + avg_R_wall + avg_R_ext
    deltaQ = 0
    print_component_report("Condenser Final", 
                           T_pool_C+273.15 + 99.8, props_out['T_K'],
                           P_in, P_curr, 
                           m_dot, h_in, h_curr, 
                           total_Q_removed/1000, props_out['x'],
                           deltaQ= - total_Q_removed, x_out=props_out['x'])
    
    print("\n--- RESISTENZE FINALI ---")
    print(f"  R_int  : {avg_R_int:.5f} K/W  ({(avg_R_int/R_sum)*100:.1f}%)")
    print(f"  R_wall : {avg_R_wall:.5f} K/W  ({(avg_R_wall/R_sum)*100:.1f}%)")
    print(f"  R_ext  : {avg_R_ext:.5f} K/W  ({(avg_R_ext/R_sum)*100:.1f}%)")
                           
    return {
        'm_dot': m_dot,
        'P': P_curr,
        'h': h_curr,
        'T': props_out['T_K'],
        'x_out': props_out['x'],
        'Q_kW': total_Q_removed/1000,
    }


def model_vertical_downcomer_a(input_state, T_amb_C=25.0):
    m_dot = input_state['m_dot']
    P_in  = input_state['P']
    h_in  = input_state['h']
    # --- A. GEOMETRIA ---
    L = 9.45             
    D_in = 0.02093       
    Angle = -90.0        
    
    Area_flow = math.pi * (D_in / 2)**2  
    
    # --- B. INIZIALIZZAZIONE ---
    wp = WaterProperties()
    
    # --- FIX VISUALIZZAZIONE: Salviamo lo stato INIZIALE prima del ciclo ---
    props_in = wp.get_properties(P_in, h_in)
    if props_in is None:
        print("Errore proprietà ingresso Downcomer A")
        return None
    T_in_K_display = props_in['T_K'] # Salviamo la T vera di ingresso
    
    # Parametri Loop
    N_NODES = 20
    dL = L / N_NODES
    dz_node = dL * math.sin(math.radians(Angle)) 
    Area_surf_node = math.pi * D_in * dL
    
    P_curr = P_in
    h_curr = h_in
    total_Q_loss = 0.0
    
    print(f"DEBUG: Start Downcomer A. P_in={P_in/1000:.1f} kPa, T_in={T_in_K_display-273.15:.2f} C")
    
    for i in range(N_NODES):
        props = wp.get_properties(P_curr, h_curr)
        if props is None: break
        
        # --- CALCOLO U VARIABILE ---
        U_local = get_variable_U_loss(props['T_K'])

        # 1. Dispersione
        dT = props['T_K'] - (T_amb_C + 273.15)
        dQ = U_local * Area_surf_node * dT 
        h_new = h_curr - (dQ / m_dot)
        total_Q_loss += dQ
        
        # 2. Idraulica
        G = m_dot / Area_flow
        rho = props['rho'] if props['rho'] else props['rho_l']
        mu = props['mu'] if props['mu'] else props['mu_l']
        x = props['x']

        if x <= 0.001: 
            Re = (G * D_in) / mu
            f = 0.316 * Re**(-0.25) if Re > 0 else 0.02
            dP_frict = f * (dL/D_in) * (G**2 / (2 * rho))
            rho_grav = rho
        else:
            phi_sq = get_two_phase_friction_multiplier(x, props['rho_l'], props['rho_v'], 
                                                       props['mu_l'], props['mu_v'], G, D_in)
            Re_lo = (G * D_in) / props['mu_l']
            f_lo = 0.316 * Re_lo**(-0.25) if Re_lo > 0 else 0.02
            dP_frict = (f_lo * (dL/D_in) * (G**2 / (2 * props['rho_l']))) * phi_sq
            rho_grav = 1.0 / (x/props['rho_v'] + (1-x)/props['rho_l'])

        dP_grav = - rho_grav * 9.81 * dz_node
        P_new = P_curr + dP_grav - dP_frict
        
        h_curr = h_new
        P_curr = P_new

    # --- OUTPUT ---
    props_out = wp.get_properties(P_curr, h_curr)
    
    # Ora passiamo T_in_K_display che è corretta
    print_component_report("Vertical Downcomer A", 
                           T_in_K_display, props_out['T_K'], 
                           P_in, P_curr, 
                           m_dot, h_in, h_curr, 
                           total_Q_loss/1000, props_out['x'],
                           deltaQ= - total_Q_loss, x_out=props_out['x'])
    
    print(f"  Recupero Pressione (Head):   {(P_curr - P_in)/1000:.2f} kPa")

    return {
        'm_dot': m_dot,
        'P': P_curr,
        'h': h_curr,
        'T': props_out['T_K'],
        'x_out': props_out['x'],
        'Q_kW': total_Q_loss/1000,
    }


def model_horizontal_downcomer(input_state, T_amb_C=25.0):
    m_dot = input_state['m_dot']
    P_in  = input_state['P']
    h_in  = input_state['h']
    # --- A. GEOMETRIA ---
    L = 8.0              # [m] Lunghezza
    D_in = 0.02093       # [m] 3/4" S40
    Angle = 0.0          # [gradi] Orizzontale
    
    Area_flow = math.pi * (D_in / 2)**2
    
    # --- B. INIZIALIZZAZIONE ---
    wp = WaterProperties()
    
    # --- FIX VISUALIZZAZIONE: Salvataggio stato IN ---
    props_in = wp.get_properties(P_in, h_in)
    if props_in is None:
        print("Errore proprietà ingresso Horizontal Downcomer")
        return None
    T_in_K_display = props_in['T_K']
    
    # Parametri Loop
    N_NODES = 16 # 1 nodo ogni mezzo metro
    dL = L / N_NODES
    Area_surf_node = math.pi * D_in * dL
    
    P_curr = P_in
    h_curr = h_in
    total_Q_loss = 0.0
    
    print(f"DEBUG: Start Horiz. Downcomer. P_in={P_in/1000:.1f} kPa, T_in={T_in_K_display-273.15:.2f} C")
    
    for i in range(N_NODES):
        props = wp.get_properties(P_curr, h_curr)
        if props is None: break

        U_local = get_variable_U_loss(props['T_K'])    

        # 1. Scambio Termico (Dispersione)
        dT = props['T_K'] - (T_amb_C + 273.15)
        dQ = U_local * Area_surf_node * dT
        
        h_new = h_curr - (dQ / m_dot)
        total_Q_loss += dQ
        
        # 2. Idraulica
        G = m_dot / Area_flow
        rho = props['rho'] if props['rho'] else props['rho_l']
        mu = props['mu'] if props['mu'] else props['mu_l']
        x = props['x']
        
        # Attrito Monofase (Liquido)
        # Qui ci aspettiamo liquido, ma per robustezza usiamo logica generica
        if x <= 0.001:
            Re = (G * D_in) / mu
            f = 0.316 * Re**(-0.25) if Re > 0 else 0.02
            dP_frict = f * (dL/D_in) * (G**2 / (2 * rho))
        else:
            # Caso raro bifase
            phi_sq = get_two_phase_friction_multiplier(x, props['rho_l'], props['rho_v'], mu, props['mu_v'], G, D_in)
            Re_lo = (G * D_in) / mu
            f_lo = 0.316 * Re_lo**(-0.25) if Re_lo > 0 else 0.02
            dP_frict = (f_lo * (dL/D_in) * (G**2 / (2 * props['rho_l']))) * phi_sq
        
        # Gravità (Orizzontale = 0)
        dP_grav = 0.0
        
        P_new = P_curr - dP_frict
        
        h_curr = h_new
        P_curr = P_new
        
    # --- OUTPUT ---
    props_out = wp.get_properties(P_curr, h_curr)
    
    print_component_report("Horizontal Downcomer", 
                           T_in_K_display, props_out['T_K'], 
                           P_in, P_curr, 
                           m_dot, h_in, h_curr, 
                           total_Q_loss/1000, props_out['x'],
                           deltaQ= - total_Q_loss, x_out=props_out['x'])
    
    return {
        'm_dot': m_dot,
        'P': P_curr,
        'h': h_curr,
        'T': props_out['T_K'],
        'x_out': props_out['x'],
        'Q_kW': total_Q_loss/1000,
    }


def model_vertical_downcomer_b(input_state, T_amb_C=25.0):
    m_dot = input_state['m_dot']
    P_in  = input_state['P']
    h_in  = input_state['h']
    # --- A. GEOMETRIA ---
    L = 9.23             # [m] Lunghezza
    D_in = 0.02093       # [m]
    Angle = -85.0        # [gradi] Quasi verticale discendente
    
    Area_flow = math.pi * (D_in / 2)**2
    
    # --- B. INIZIALIZZAZIONE ---
    wp = WaterProperties()
    
    # --- FIX VISUALIZZAZIONE: Salvataggio stato IN ---
    props_in = wp.get_properties(P_in, h_in)
    if props_in is None:
        print("Errore proprietà ingresso Downcomer B")
        return None
    T_in_K_display = props_in['T_K']
    
    # Parametri Loop
    N_NODES = 20
    dL = L / N_NODES
    dz_node = dL * math.sin(math.radians(Angle)) # Negativo (discesa)
    Area_surf_node = math.pi * D_in * dL
    
    P_curr = P_in
    h_curr = h_in
    total_Q_loss = 0.0
    
    print(f"DEBUG: Start Downcomer B. P_in={P_in/1000:.1f} kPa")
    
    for i in range(N_NODES):
        props = wp.get_properties(P_curr, h_curr)
        if props is None: break

        U_local = get_variable_U_loss(props['T_K'])
                
        # 1. Dispersione
        dT = props['T_K'] - (T_amb_C + 273.15)
        dQ = U_local * Area_surf_node * dT
        h_new = h_curr - (dQ / m_dot)
        total_Q_loss += dQ
        
        # 2. Idraulica
        G = m_dot / Area_flow
        rho = props['rho'] if props['rho'] else props['rho_l']
        mu = props['mu'] if props['mu'] else props['mu_l']
        x = props['x']
        
        if x <= 0.001:
            Re = (G * D_in) / mu
            f = 0.316 * Re**(-0.25) if Re > 0 else 0.02
            dP_frict = f * (dL/D_in) * (G**2 / (2 * rho))
            rho_grav = rho
        else:
            phi_sq = get_two_phase_friction_multiplier(x, props['rho_l'], props['rho_v'], mu, props['mu_v'], G, D_in)
            Re_lo = (G * D_in) / mu
            f_lo = 0.316 * Re_lo**(-0.25) if Re_lo > 0 else 0.02
            dP_frict = (f_lo * (dL/D_in) * (G**2 / (2 * props['rho_l']))) * phi_sq
            rho_grav = 1.0 / (x/props['rho_v'] + (1-x)/props['rho_l'])
        
        # Gravità (Guadagno pressione in discesa)
        # dP_grav = - rho * g * dz (con dz negativo) -> positivo
        dP_grav = - rho_grav * 9.81 * dz_node
        
        P_new = P_curr + dP_grav - dP_frict
        
        h_curr = h_new
        P_curr = P_new

    # --- OUTPUT ---
    props_out = wp.get_properties(P_curr, h_curr)
    
    print_component_report("Vertical Downcomer B (-85°)", 
                           T_in_K_display, props_out['T_K'], 
                           P_in, P_curr, 
                           m_dot, h_in, h_curr, 
                           total_Q_loss/1000, props_out['x'],
                           deltaQ= - total_Q_loss, x_out=props_out['x'])
    
    print(f"  Recupero Pressione (Head):   {(P_curr - P_in)/1000:.2f} kPa")
    
    return {
        'm_dot': m_dot,
        'P': P_curr,
        'h': h_curr,
        'T': props_out['T_K'],
        'x_out': props_out['x'],
        'Q_kW': total_Q_loss/1000,
    }

# ==============================================================================
# MAIN
# ==============================================================================
# if __name__ == "__main__":
    
#     print("\n--- AVVIO SIMULAZIONE SINGLE CHANNEL ---")
    
#     # Condizioni iniziali stimate
#     m_dot_guess = 0.0443  # kg/s
#     P_start_guess = 1636692 # 10 bar in Pascal
    
#     # Entalpia corrispondente a 20°C liquido (approx)
#     h_start_guess = 700530 # J/kg (Molto sottoraffreddato)
#     # Se hai IAPWS calcoliamo meglio
#     try:
#         from iapws import IAPWS97
#         h_start_guess = IAPWS97(P=P_start_guess/1e6, T=h_start_guess/4600).h * 1000
#     except:
#         pass

#     # 1. Inlet Header
#     out_1 = model_inlet_header(m_dot_guess, P_start_guess, h_start_guess)

#     # 2. Orifice
#     out_2 = model_orifice(out_1)

#     # 3. Heated Section (34 kW)
#     out_3 = model_heated_section(out_2, Q_total_kW=33.6)

#     # 4. Unheated Section
#     out_4 = model_unheated_section(out_3)

#     # 5. Elbow 1
#     out_5 = model_generic_elbow(out_4, L=0.6, Angle=90.0, name="Elbow 1"  )
    
#     # 6. Horizontal Riser
#     out_6 = model_horizontal_riser(out_5, 9.45)
    
#     # 7. Elbow 2
#     out_7 = model_generic_elbow(out_6, L=0.2, Angle=90.0, name="Elbow 2" )
    
#     # 8. Vertical Riser
#     out_8 = model_vertical_riser(out_7, 10.7, 87)
    
#     # 9. Siphon
#     out_9 = model_siphon(out_8)

#     # TARGET
#     print("\n>>> STEP 1: CONDENSATORE")
#     res_cond = model_condenser_detailed(out_9, T_pool_C=100.0)

#     #out_10 = model_generic_elbow(res_cond, L=3, Angle=93.0, name="Elbow 2" )

#     # TARGET
#     T_target = 178.0
#     Q_target = 26.27
    
#     # print(f"\n=== VERIFICA CONDENSER ===")
#     # print(f"{'Grandezza':<20} | {'Simulato':<10} | {'Target (Exp)':<10} | {'Errore'}")
#     # print("-" * 60)
#     # print(f"{'T uscita [°C]':<20} | {res_cond['T']-273.15:<10.2f} | {T_target:<10.1f} | {res_cond['T']-273.15 - T_target:.2f}")
#     # print(f"{'Qualità out [-]':<20} | {res_cond['x_out']:<10.4f} | {'0.0':<10} | {res_cond['x_out'] - 0.0:.4f}")
#     # print(f"{'Potenza [kW]':<20} | {res_cond['Q_kW']:<10.2f} | {Q_target:<10.2f} | {res_cond['Q_kW'] - Q_target:.2f}")
    
#     # print("\n>>> STEP 2: VERTICAL DOWNCOMER")
    
#     # 2. DOWNCOMER A
#     res_dwn_a = model_vertical_downcomer_a(res_cond)


#     out_11 = model_generic_elbow(res_dwn_a, L=0.2, Angle=90.0, name="Elbow 2" )
    
#     # 3. HORIZONTAL DOWNCOMER
#     res_dwn_h = model_horizontal_downcomer(out_11)


#     out_12 = model_generic_elbow(res_dwn_h, L=0.2, Angle=90.0, name="Elbow 2" )
    
#     # 4. DOWNCOMER B
#     res_dwn_b = model_vertical_downcomer_b(out_12)

#     #out_13 = model_generic_elbow(res_dwn_b, L=2.1, Angle=90.0, name="Elbow 2" )
    
#     # # --- SOMMARIO FINALE ---
#     # print("\n" + "="*60)
#     # print("RISULTATI FINALI AL 'TEST SECTION INLET' (FINE DOWNCOMER B)")
#     # print("="*60)
    
#     # T_final = res_dwn_b['T'] - 273.15
#     # P_final_bar = res_dwn_b['P'] / 1e5
#     # h_final_kJ = res_dwn_b['h'] / 1000
    
#     # loss_A = res_dwn_a['Q_kW']
#     # loss_H = res_dwn_h['Q_kW']
#     # loss_B = res_dwn_b['Q_kW']
#     # tot_loss = loss_A + loss_H + loss_B
    
#     # print(f"Temperatura Finale : {T_final+273.15:.2f} °C")
#     # print(f"Pressione Finale   : {P_final_bar:.3f} bar")
#     # print(f"Entalpia Finale    : {h_final_kJ:.2f} kJ/kg")
#     # print("-" * 60)
#     # print(f"Perdite Termiche Totali Downcomer: {tot_loss:.3f} kW")
#     # print(f"  - Dwn A (Vert): {loss_A:.3f} kW")
#     # print(f"  - Dwn H (Horz): {loss_H:.3f} kW")
#     # print(f"  - Dwn B (Vert): {loss_B:.3f} kW")
#     # print("="*60)
    
    



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

    
    res_cond = model_condenser_detailed(out_9, T_pool_C=100.0)

    #out_10 = model_generic_elbow(res_cond, L=3, Angle=93.0, name="Elbow 2" )

    # 2. DOWNCOMER A
    res_dwn_a = model_vertical_downcomer_a(res_cond)


    out_11 = model_generic_elbow(res_dwn_a, L=0.2, Angle=90.0, name="Elbow 2" )
    
    # 3. HORIZONTAL DOWNCOMER
    res_dwn_h = model_horizontal_downcomer(out_11)


    out_12 = model_generic_elbow(res_dwn_h, L=0.2, Angle=90.0, name="Elbow 2" )
    
    # 4. DOWNCOMER B
    res_dwn_b = model_vertical_downcomer_b(out_12)

    #out_13 = model_generic_elbow(res_dwn_b, L=2.1, Angle=90.0, name="Elbow 2" )
    
    # 4. DOWNCOMER B
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
    # Run optimization with PRINT_REPORT = False (default)
    res = least_squares(solve_full_loop, x0, bounds=([0.001, 1e5, 1e5], [1.0, 50e5, 3000e3]), verbose=2, ftol=1e-4)
    
    print("\nOptimization Finished.")
    print("Success:", res.success)
    print("Message:", res.message)
    print(f"Solution: m_dot={res.x[0]:.5f} kg/s, P_start={res.x[1]/1000:.2f} kPa, h_start={res.x[2]/1000:.2f} kJ/kg")
    
    # Run Final Report with printing enabled
    PRINT_REPORT = True
    print("\n" + "="*80)
    print("FINAL CONVERGED STATE REPORT")
    print("="*80)
    residuals = solve_full_loop(res.x)
    
    print("\nFINAL CHECK:")
    print(f"Residual P: {residuals[0]:.8f} kPa")
    print(f"Residual h: {residuals[1]:.8f} kJ/kg")
    print(f"Residual M: {residuals[2]/10:.8f} kg") 
