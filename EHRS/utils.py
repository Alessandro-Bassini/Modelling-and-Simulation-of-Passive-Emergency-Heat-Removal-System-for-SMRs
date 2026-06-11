import config

# --- FUNZIONE DI SUPPORTO PER LA STAMPA ---
def print_component_report(name, T_in, T_out, P_in, P_out_Pa, m_dot,
                           h_in, h_out, rho_in, rho_out, deltaQ, x_out):
    if not config.PRINT_REPORT:
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
