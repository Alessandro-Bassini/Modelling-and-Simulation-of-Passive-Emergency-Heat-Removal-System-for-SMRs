import math
from water_properties import WaterProperties
from utils import print_component_report
from correlations import get_two_phase_friction_multiplier, get_variable_U_loss

# ==============================================================================
# MODELLI COMPONENTI
# ==============================================================================

def model_inlet_header(m_dot_in, P_in, h_in):
    # 1. GEOMETRIA
    L = 1.1
    D_in = 0.02664
    Area = math.pi * (D_in / 2)**2

    wp = WaterProperties()

    # 2. STATO INGRESSO
    props_in = wp.get_properties(P_in, h_in)
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


def model_orifice(input_state):
    m_dot_in = input_state['m_dot']
    P_in  = input_state['P']
    h_in = input_state['h']

    L = 0.56
    D_pipe = 0.01253
    Area = math.pi * (D_pipe / 2)**2

    wp = WaterProperties()
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
    
    comp_vol = Area * L
    comp_mass = rho_in * comp_vol

    print_component_report("Orifice Section", 
                           props_in['T_K'], props_out['T_K'], P_in, P_out_Pa, m_dot_in, 
                           h_in, h_out_Jkg, rho_in, props_out['rho'], 0.0, props_out['x'])

    return {'m_dot': m_dot_in, 'P': P_out_Pa, 'h': h_out_Jkg, 'mass': comp_mass, 'vol': comp_vol}


def model_heated_section(input_state, Q_total_kW):
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
    HELIX_CORRECTION_FACTOR = 1.8  
    
    # --- INEFFICIENZA ---
    U_loss = 15.0          
    T_amb  = 100.0 + 273.15 

    # --- DISCRETIZZAZIONE ---
    N_NODES = 100     
    dL      = L_total / N_NODES
    dz_node = dL * math.sin(math.radians(Angle))
    
    dQ_electric_node = (Q_total_kW * 1000.0) / N_NODES

    wp = WaterProperties()
    
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
        geo_factor = (1.0 + HELIX_CORRECTION_FACTOR * math.sqrt(D_in / (2*R_coil)))

        # --- FRIEDEL ---
        if 0 < x < 1:
            j_g = (x * G_flux) / rho_v
            j_tot = G_flux * (x/rho_v + (1-x)/rho_l)
            Vgj = 1.53 * ((sigma * 9.81 * (rho_l - rho_v)) / rho_l**2)**0.25
            alpha_avg = j_g / (1.2 * j_tot + Vgj)
            alpha_avg = max(0.0, min(1.0, alpha_avg))
            rho_mix = alpha_avg * rho_v + (1 - alpha_avg) * rho_l
            
            Re_lo = (G_flux * D_in) / mu_l
            f_lo_base = 0.316 * Re_lo**(-0.25) if Re_lo > 2300 else 64.0/max(Re_lo,1)
            f_lo = f_lo_base * geo_factor 
            
            Re_go = (G_flux * D_in) / mu_v
            f_go_base = 0.316 * Re_go**(-0.25) if Re_go > 2300 else 64.0/max(Re_go,1)
            f_go = f_go_base * geo_factor 

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

    print_component_report("Heated Test Section (SG)", 
                           props_in['T_K'], props_final['T_K'], P_in, P_curr, m_dot, 
                           h_in, h_curr, props_in['rho'], props_final['rho'], 
                           Q_net_accumulated, props_final['x'])

    return {
        'm_dot': m_dot, 'P': P_curr, 'h': h_curr, 'T': props_final['T_K'],
        'mass': total_mass, 'vol': total_vol
    }


def model_unheated_section(input_state):
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
    HELIX_CORRECTION_FACTOR = 1.8 

    # --- PERDITE ---
    U_loss = 15.0
    T_amb = 373.15 # 100°C come da tuo snippet

    # --- DISCRETIZZAZIONE ---
    N_NODES = 50
    dL = L_total / N_NODES
    dz_node = dL * math.sin(math.radians(Angle))

    wp = WaterProperties()
    P_curr = P_in
    h_curr = h_in
    G_flux = m_dot / Area
    
    Q_lost_total = 0.0
    total_mass = 0.0
    total_vol = 0.0
    
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
        props_step = wp.get_properties(P_curr, h_curr)
        dQ_loss = U_loss * (Perim_out * dL) * (props_step['T_K'] - T_amb)
        if dQ_loss < 0: dQ_loss = 0
        Q_lost_total += dQ_loss

        h_next = h_curr - (dQ_loss / m_dot)
        
        props_next = wp.get_properties(P_curr, h_next)
        v_mom_next = get_v_mom(props_next, G_flux)
        dP_accel = (G_flux**2) * (v_mom_next - v_mom_curr)

        h_avg = (h_curr + h_next)/2
        props = wp.get_properties(P_curr, h_avg)
        x = props['x']
        rho_l, rho_v = props['rho_l'], props['rho_v']
        mu_l, mu_v = props.get('mu_l', 1e-4), props.get('mu_v', 1e-5)
        sigma = props.get('sigma', 0.03)

        geo_factor = (1.0 + HELIX_CORRECTION_FACTOR * math.sqrt(D_in / (2*R_coil)))

        if 0 < x < 1:
            j_g = (x * G_flux) / rho_v
            j_tot = G_flux * (x/rho_v + (1-x)/rho_l)
            Vgj = 1.53 * ((sigma * 9.81 * (rho_l - rho_v)) / rho_l**2)**0.25
            alpha_avg = j_g / (1.2 * j_tot + Vgj)
            alpha_avg = max(0.0, min(1.0, alpha_avg))
            rho_mix = alpha_avg * rho_v + (1 - alpha_avg) * rho_l
            
            Re_lo = (G_flux * D_in) / mu_l
            f_lo_base = 0.316 * Re_lo**(-0.25) if Re_lo > 2300 else 64.0/max(Re_lo,1)
            f_lo = f_lo_base * geo_factor 
            
            Re_go = (G_flux * D_in) / mu_v
            f_go_base = 0.316 * Re_go**(-0.25) if Re_go > 2300 else 64.0/max(Re_go,1)
            f_go = f_go_base * geo_factor

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

    if x_in <= 0:
        rho_mix = props_in['rho']
        mu_mix  = props_in['mu']
        phi_2 = 1.0
    elif x_in >= 1: 
        rho_mix = props_in['rho']
        mu_mix  = props_in['mu']
        phi_2 = 1.0
    else: 
        rho_l = getattr(props_in, 'rho_l', 820.0)
        rho_v = getattr(props_in, 'rho_v', 14.0)
        sigma = 0.03
        
        G = m_dot / Area
        j_g = (x_in * G) / rho_v
        j_l = ((1 - x_in) * G) / rho_l
        j_tot = j_g + j_l
        
        C0 = 1.2
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
        
        mu_l = getattr(props_in, 'mu_l', 1.1e-4)
        mu_v = getattr(props_in, 'mu_v', 1.8e-5)
        mu_mix = 1 / (x_in/mu_v + (1-x_in)/mu_l)
        
        phi_2 = 1.0 + x_in * (rho_l / rho_v - 1.0)

    velocity = m_dot / (rho_mix * Area)
    
    if mu_mix is not None and mu_mix > 0:
        Re = (rho_mix * velocity * D_pipe) / mu_mix
    else:
        Re = 0

    if Re > 1000:
        f = 0.316 * (Re**(-0.25))
    elif Re > 0:
        f = 64.0 / Re
    else:
        f = 0

    theta_rad = math.radians(abs(Angle))
    
    if theta_rad > 0.01 and L > 0:
        R_curv = L / theta_rad
        rr = R_curv / D_pipe 
        
        if rr < 1.0:
            K_90 = 1.2 
        else:
            K_90 = 0.10 + 1.85 * (1.0 / (2.0 * rr))**3.5
        
        K_bend = K_90 * (abs(Angle) / 90.0)
    else:
        K_bend = 0.0

    dyn_head = (rho_mix * velocity**2 / 2)
    dP_wall_friction = f * (L / D_pipe) * dyn_head
    dP_bend_loss = K_bend * dyn_head
    dP_frict_total = (dP_wall_friction + dP_bend_loss) * phi_2

    if abs(Angle - 90.0) < 5.0:
        delta_z = L / (math.pi / 2.0)
    else:
        delta_z = L * math.sin(math.radians(Angle))

    dP_grav = rho_mix * 9.81 * delta_z
    dP_total = dP_frict_total + dP_grav

    h_out_Jkg = h_in
    P_out_Pa = P_in - dP_total

    props_out = wp.get_properties(P_out_Pa, h_out_Jkg)
    T_out_K = props_out['T_K']

    comp_vol = Area * L
    comp_mass = rho_mix * comp_vol

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

    D_in = 0.02093   
    D_out = 0.02667
    Area = math.pi * (D_in/2)**2
    Perim_out = math.pi * D_out
    
    U_loss = 10.0
    T_amb = 373.15 

    N_NODES = 50
    dL = length / N_NODES
    dz_node = 0.0 

    wp = WaterProperties()
    P_curr = P_in; h_curr = h_in
    G_flux = m_dot / Area
    
    total_mass = 0.0; total_vol = 0.0
    Q_lost_tot = 0.0 
    
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
        props_step = wp.get_properties(P_curr, h_curr)
        
        dQ_loss = U_loss * (Perim_out * dL) * (props_step['T_K'] - T_amb)
        if dQ_loss < 0: dQ_loss = 0
        
        Q_lost_tot += dQ_loss 

        h_next = h_curr - (dQ_loss / m_dot)
        
        props_next = wp.get_properties(P_curr, h_next)
        v_mom_next = get_v_mom(props_next, G_flux)
        dP_accel = (G_flux**2) * (v_mom_next - v_mom_curr)

        h_avg = (h_curr + h_next)/2
        props = wp.get_properties(P_curr, h_avg)
        x = props['x']
        rho_l, rho_v = props['rho_l'], props['rho_v']
        mu_l, mu_v = props.get('mu_l', 1e-4), props.get('mu_v', 1e-5)
        sigma = props.get('sigma', 0.03)

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
    
    print_component_report(f"Horizontal Riser (L={length}m)", 
                       props_in['T_K'], props_final['T_K'], P_in, P_curr, m_dot, 
                       h_in, h_curr, props_in['rho'], props_final['rho'], 
                       -Q_lost_tot, props_final['x'])
                       
    return {'m_dot': m_dot, 'P': P_curr, 'h': h_curr, 'mass': total_mass, 'vol': total_vol}


def model_vertical_riser(input_state, length, angle_deg=90.0):
    m_dot = input_state['m_dot']
    P_in  = input_state['P']
    h_in  = input_state['h']

    D_in = 0.02093 
    D_out = 0.02667
    Area = math.pi * (D_in/2)**2
    Perim_out = math.pi * D_out
    
    U_loss = 10.0; T_amb = 373.15

    N_NODES = 50
    dL = length / N_NODES
    dz_node = dL * math.sin(math.radians(angle_deg))
    
    wp = WaterProperties()
    P_curr = P_in; h_curr = h_in
    G_flux = m_dot / Area
    
    total_mass = 0.0; total_vol = 0.0
    Q_lost_tot = 0.0 
    
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
        props_step = wp.get_properties(P_curr, h_curr)
        
        dQ_loss = U_loss * (Perim_out * dL) * (props_step['T_K'] - T_amb)
        if dQ_loss < 0: dQ_loss = 0
        
        Q_lost_tot += dQ_loss

        h_next = h_curr - (dQ_loss / m_dot)
        
        props_next = wp.get_properties(P_curr, h_next)
        v_mom_next = get_v_mom(props_next, G_flux)
        dP_accel = (G_flux**2) * (v_mom_next - v_mom_curr)

        h_avg = (h_curr + h_next)/2
        props = wp.get_properties(P_curr, h_avg)
        x = props['x']
        rho_l, rho_v = props['rho_l'], props['rho_v']
        mu_l, mu_v = props.get('mu_l', 1e-4), props.get('mu_v', 1e-5)
        sigma = props.get('sigma', 0.03)

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
    
    print_component_report(f"Vertical Riser (L={length}m)", 
                       props_in['T_K'], props_final['T_K'], P_in, P_curr, m_dot, 
                       h_in, h_curr, props_in['rho'], props_final['rho'], 
                       -Q_lost_tot, props_final['x'])
                       
    return {'m_dot': m_dot, 'P': P_curr, 'h': h_curr, 'mass': total_mass, 'vol': total_vol}


def model_siphon(input_state):
    m_dot = input_state['m_dot']
    P_in  = input_state['P']
    h_in  = input_state['h']

    L = 1.0           
    D_pipe = 0.02093  
    Angle = 0.0       
    Area = math.pi * (D_pipe / 2)**2

    wp = WaterProperties()
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
        if wp._use_lib:
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
        if wp._use_lib:
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

    comp_vol = Area * L
    comp_mass = rho_mix * comp_vol

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


def model_condenser_detailed(input_state, T_pool_C=100.0):
    m_dot = input_state['m_dot']
    P_in  = input_state['P']
    h_in  = input_state['h']
    
    L = 1.0              
    D_in = 0.059         
    D_out = 0.073025     
    Angle = -3.0         
    
    k_steel = 16.3 
    
    Area_flow = math.pi * (D_in / 2)**2
    Area_surf_node = (math.pi * D_in * L) / 50.0
    
    h_pool_boiling = 25000.0 
    tuning_factor_int = 12.0 
    
    wp = WaterProperties()
    
    N_NODES = 50
    dL = L / N_NODES
    dz_node = dL * math.sin(math.radians(Angle))
    
    P_curr = P_in
    h_curr = h_in
    
    total_Q_removed = 0.0
    
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
        
        h_int_base = 500.0
        mode = "Unknown"

        if x >= 0.999: 
            mode = "Superheat"
            mu_g = props['mu'] if props['mu'] else props['mu_v']
            k_g  = props['k_th'] if props['k_th'] else props['k_v']
            cp_g = props['cp'] * 1000 if props['cp'] else props['cp_v'] * 1000
            Re = (G * D_in) / mu_g
            Pr = (cp_g * mu_g) / k_g
            Nu = 0.023 * Re*0.8 * Pr*0.4
            h_int_base = Nu * k_g / D_in
            
        elif x <= 0.001: 
            mode = "Subcooled"
            mu_l = props['mu'] if props['mu'] else props['mu_l']
            k_l  = props['k_th'] if props['k_th'] else props['k_l']
            cp_l = props['cp'] * 1000 if props['cp'] else props['cp_l'] * 1000
            Re = (G * D_in) / mu_l
            Pr = (cp_l * mu_l) / k_l
            Nu = 0.023 * Re*0.8 * Pr*0.3 
            h_int_base = Nu * k_l / D_in * 3.0 
            
        else: 
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

        R_val_int = 1.0 / (h_int_base * Area_surf_node)
        R_val_wall = math.log(D_out / D_in) / (2 * math.pi * k_steel * dL)
        Area_surf_ext = (math.pi * D_out * dL)
        R_val_ext = 1.0 / (h_pool_boiling * Area_surf_ext)
        
        R_tot = R_val_int + R_val_wall + R_val_ext
        dQ = (T_fluid - (T_pool_C + 273.15)) / R_tot
        
        avg_R_int += R_val_int
        avg_R_wall += R_val_wall
        avg_R_ext += R_val_ext
        valid_nodes += 1
        
        h_new = h_curr - (dQ / m_dot)
        total_Q_removed += dQ
        
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

    props_out = wp.get_properties(P_curr, h_curr)
    
    if valid_nodes > 0:
        avg_R_int /= valid_nodes
        avg_R_wall /= valid_nodes
        avg_R_ext /= valid_nodes
        R_sum = avg_R_int + avg_R_wall + avg_R_ext

    comp_vol = Area_flow * L
    comp_mass = (wp.get_properties(P_in, h_in)['rho'] + props_out['rho'])/2 * comp_vol

    print_component_report("Condenser Final", 
                           T_pool_C+273.15 + 99.8, props_out['T_K'],
                           P_in, P_curr, 
                           m_dot, h_in, h_curr, 
                           wp.get_properties(P_in, h_in)['rho'], props_out['rho'],
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
        'mass': comp_mass,
        'vol': comp_vol
    }


def model_vertical_downcomer_a(input_state, T_amb_C=25.0):
    m_dot = input_state['m_dot']
    P_in  = input_state['P']
    h_in  = input_state['h']
    
    L = 9.45             
    D_in = 0.02093       
    Angle = -90.0        
    
    Area_flow = math.pi * (D_in / 2)**2  
    
    wp = WaterProperties()
    
    props_in = wp.get_properties(P_in, h_in)
    if props_in is None:
        print("Errore proprietà ingresso Downcomer A")
        return None
    T_in_K_display = props_in['T_K'] 
    
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
        
        U_local = get_variable_U_loss(props['T_K'])

        dT = props['T_K'] - (T_amb_C + 273.15)
        dQ = U_local * Area_surf_node * dT 
        h_new = h_curr - (dQ / m_dot)
        total_Q_loss += dQ
        
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

    props_out = wp.get_properties(P_curr, h_curr)

    comp_vol = Area_flow * L
    comp_mass = (props_in['rho'] + props_out['rho'])/2 * comp_vol
    
    print_component_report("Vertical Downcomer A", 
                           T_in_K_display, props_out['T_K'], 
                           P_in, P_curr, 
                           m_dot, h_in, h_curr, 
                           props_in['rho'], props_out['rho'],
                           deltaQ= - total_Q_loss, x_out=props_out['x'])
    
    print(f"  Recupero Pressione (Head):   {(P_curr - P_in)/1000:.2f} kPa")

    return {
        'm_dot': m_dot,
        'P': P_curr,
        'h': h_curr,
        'T': props_out['T_K'],
        'x_out': props_out['x'],
        'Q_kW': total_Q_loss/1000,
        'mass': comp_mass,
        'vol': comp_vol
    }


def model_horizontal_downcomer(input_state, T_amb_C=25.0):
    m_dot = input_state['m_dot']
    P_in  = input_state['P']
    h_in  = input_state['h']
    
    L = 8.0              
    D_in = 0.02093       
    Angle = 0.0          
    
    Area_flow = math.pi * (D_in / 2)**2
    
    wp = WaterProperties()
    
    props_in = wp.get_properties(P_in, h_in)
    if props_in is None:
        print("Errore proprietà ingresso Horizontal Downcomer")
        return None
    T_in_K_display = props_in['T_K']
    
    N_NODES = 16 
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

        dT = props['T_K'] - (T_amb_C + 273.15)
        dQ = U_local * Area_surf_node * dT
        
        h_new = h_curr - (dQ / m_dot)
        total_Q_loss += dQ
        
        G = m_dot / Area_flow
        rho = props['rho'] if props['rho'] else props['rho_l']
        mu = props['mu'] if props['mu'] else props['mu_l']
        x = props['x']
        
        if x <= 0.001:
            Re = (G * D_in) / mu
            f = 0.316 * Re**(-0.25) if Re > 0 else 0.02
            dP_frict = f * (dL/D_in) * (G**2 / (2 * rho))
        else:
            phi_sq = get_two_phase_friction_multiplier(x, props['rho_l'], props['rho_v'], mu, props['mu_v'], G, D_in)
            Re_lo = (G * D_in) / mu
            f_lo = 0.316 * Re_lo**(-0.25) if Re_lo > 0 else 0.02
            dP_frict = (f_lo * (dL/D_in) * (G**2 / (2 * props['rho_l']))) * phi_sq
        
        dP_grav = 0.0
        
        P_new = P_curr - dP_frict
        
        h_curr = h_new
        P_curr = P_new
        
    props_out = wp.get_properties(P_curr, h_curr)

    comp_vol = Area_flow * L
    comp_mass = (props_in['rho'] + props_out['rho'])/2 * comp_vol
    
    print_component_report("Horizontal Downcomer", 
                           T_in_K_display, props_out['T_K'], 
                           P_in, P_curr, 
                           m_dot, h_in, h_curr, 
                           props_in['rho'], props_out['rho'],
                           deltaQ= - total_Q_loss, x_out=props_out['x'])
    
    return {
        'm_dot': m_dot,
        'P': P_curr,
        'h': h_curr,
        'T': props_out['T_K'],
        'x_out': props_out['x'],
        'Q_kW': total_Q_loss/1000,
        'mass': comp_mass,
        'vol': comp_vol
    }


def model_vertical_downcomer_b(input_state, T_amb_C=25.0):
    m_dot = input_state['m_dot']
    P_in  = input_state['P']
    h_in  = input_state['h']
    
    L = 9.23             
    D_in = 0.02093       
    Angle = -85.0        
    
    Area_flow = math.pi * (D_in / 2)**2
    
    wp = WaterProperties()
    
    props_in = wp.get_properties(P_in, h_in)
    if props_in is None:
        print("Errore proprietà ingresso Downcomer B")
        return None
    T_in_K_display = props_in['T_K']
    
    N_NODES = 20
    dL = L / N_NODES
    dz_node = dL * math.sin(math.radians(Angle)) 
    Area_surf_node = math.pi * D_in * dL
    
    P_curr = P_in
    h_curr = h_in
    total_Q_loss = 0.0
    
    print(f"DEBUG: Start Downcomer B. P_in={P_in/1000:.1f} kPa")
    
    for i in range(N_NODES):
        props = wp.get_properties(P_curr, h_curr)
        if props is None: break

        U_local = get_variable_U_loss(props['T_K'])
                
        dT = props['T_K'] - (T_amb_C + 273.15)
        dQ = U_local * Area_surf_node * dT
        h_new = h_curr - (dQ / m_dot)
        total_Q_loss += dQ
        
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
        
        dP_grav = - rho_grav * 9.81 * dz_node
        
        P_new = P_curr + dP_grav - dP_frict
        
        h_curr = h_new
        P_curr = P_new

    props_out = wp.get_properties(P_curr, h_curr)
    
    comp_vol = Area_flow * L
    comp_mass = (props_in['rho'] + props_out['rho'])/2 * comp_vol

    print_component_report("Vertical Downcomer B (-85°)", 
                           T_in_K_display, props_out['T_K'], 
                           P_in, P_curr, 
                           m_dot, h_in, h_curr, 
                           props_in['rho'], props_out['rho'],
                           deltaQ= - total_Q_loss, x_out=props_out['x'])
    
    print(f"  Recupero Pressione (Head):   {(P_curr - P_in)/1000:.2f} kPa")
    
    return {
        'm_dot': m_dot,
        'P': P_curr,
        'h': h_curr,
        'T': props_out['T_K'],
        'x_out': props_out['x'],
        'Q_kW': total_Q_loss/1000,
        'mass': comp_mass,
        'vol': comp_vol
    }
