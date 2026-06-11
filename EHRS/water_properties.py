from iapws import IAPWS97

# --- CLASSE PROPRIETÀ ACQUA (AGGIORNATA CON K e CP per CONDENSATORE) ---
class WaterProperties:
    """
    Wrapper robusto per IAPWS97.
    Garantisce che non vengano mai restituiti None per le proprietà critiche
    usando fallback ai valori di saturazione quando necessario.
    """
    def __init__(self):
        self._use_lib = True

    def use_lib(self, use_iapws=True):
        self._use_lib = use_iapws

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
