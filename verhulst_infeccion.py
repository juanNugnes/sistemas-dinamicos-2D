#!/usr/bin/env python3
"""Panel interactivo para ensayar un modelo Verhulst de infecciones.
Permite ajustar tasa de contagio, capacidad de carga, recuperacion e
intervenciones como confinamientos o vacunacion constante.
"""
import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

FIRMA = "Autores: Nugnes, Sturba, Quatraro"


def _entry(parent, label, value, width=8):
    fila = tk.Frame(parent)
    fila.pack(anchor='w', pady=1)
    tk.Label(fila, text=label).pack(side=tk.LEFT)
    entry = tk.Entry(fila, width=width)
    entry.pack(side=tk.LEFT)
    entry.insert(0, str(value))
    return entry


def lanzar_app():
    root = tk.Tk()
    root.title('Modelo Verhulst de infeccion')
    root.geometry('1320x820')
    root.minsize(1150, 680)

    contenedor = tk.Frame(root)
    contenedor.pack(fill=tk.BOTH, expand=True)

    panel_izq = tk.Frame(contenedor, padx=10, pady=10)
    panel_izq.pack(side=tk.LEFT, fill=tk.Y)
    panel_der = tk.Frame(contenedor)
    panel_der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    tk.Label(panel_izq, text='Condiciones iniciales', font=('Consolas', 11, 'bold')).pack(anchor='w')
    entry_I0 = _entry(panel_izq, 'Infectados iniciales:', 50)
    entry_cap = _entry(panel_izq, 'Capacidad (K):', 1000)
    entry_pob = _entry(panel_izq, 'Poblacion total:', 5000)

    tk.Label(panel_izq, text='Parámetros dinamicos', font=('Consolas', 11, 'bold')).pack(anchor='w', pady=(6,0))
    entry_beta = _entry(panel_izq, 'Contagio base (r):', 0.6)
    entry_gamma = _entry(panel_izq, 'Recuperacion (g):', 0.2)
    entry_shock = _entry(panel_izq, 'Impulso (casos/dia):', 5)

    tk.Label(panel_izq, text='Intervenciones', font=('Consolas', 11, 'bold')).pack(anchor='w', pady=(6,0))
    conf_var = tk.BooleanVar(value=True)
    tk.Checkbutton(panel_izq, text='Aplicar confinamiento', variable=conf_var).pack(anchor='w')
    entry_conf_day = _entry(panel_izq, 'Dia de inicio:', 15, width=6)
    entry_conf_drop = _entry(panel_izq, 'Reduccion (%):', 40, width=6)

    vac_var = tk.BooleanVar(value=False)
    tk.Checkbutton(panel_izq, text='Vacunacion constante', variable=vac_var).pack(anchor='w', pady=(4,0))
    entry_vac = _entry(panel_izq, 'Vacunas por dia:', 2, width=6)

    entry_tmax = _entry(panel_izq, 'Dias a simular:', 120, width=6)

    figura, (ax_inf, ax_metrics) = plt.subplots(2, 1, figsize=(7,8))
    ax_reff = ax_metrics.twinx()
    figura.tight_layout(pad=3.0)
    canvas = FigureCanvasTkAgg(figura, master=panel_der)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    resumen = tk.Text(panel_izq, width=44, height=20, font=('Consolas', 10), bg='#f3f3f3')
    resumen.pack(fill=tk.X, pady=(8,0))

    def simular():
        try:
            I0 = max(float(entry_I0.get()), 0.0)
            K = max(float(entry_cap.get()), 1.0)
            poblacion = max(float(entry_pob.get()), K)
            beta = max(float(entry_beta.get()), 0.0)
            gamma = max(float(entry_gamma.get()), 0.0)
            shock = float(entry_shock.get())
            dias = max(float(entry_tmax.get()), 1.0)
            conf_day = max(float(entry_conf_day.get()), 0.0)
            conf_drop = np.clip(float(entry_conf_drop.get())/100.0, 0.0, 0.95)
            vac_rate = max(float(entry_vac.get()), 0.0) if vac_var.get() else 0.0

            def beta_eff(t):
                factor = beta
                if conf_var.get() and t >= conf_day:
                    factor *= (1.0 - conf_drop)
                return factor

            def rhs(t, z):
                I = max(z[0], 0.0)
                contagio = beta_eff(t) * I * (1.0 - I/max(K, 1e-6))
                recupera = gamma * I
                vacunacion = vac_rate * I / max(K, 1.0)
                impulso = shock * np.exp(-0.15 * t)
                dI = contagio - recupera - vacunacion + impulso
                return [dI]

            sol = solve_ivp(rhs, [0, dias], [I0], t_eval=np.linspace(0, dias, 400))
            if not sol.success:
                raise RuntimeError('Integrador no convergio')

            I = np.clip(sol.y[0], 0, poblacion)
            nuevos = np.gradient(I, sol.t, edge_order=2)
            nuevos = np.maximum(nuevos, 0)
            S = np.maximum(K - I, 0)
            reff = []
            for t_val, I_val in zip(sol.t, I):
                denom = gamma + vac_rate
                if denom == 0:
                    reff.append(0.0)
                else:
                    reff.append(beta_eff(t_val) * (1 - I_val/max(K,1.0)) / denom)
            reff = np.array(reff)

            ax_inf.clear(); ax_metrics.clear(); ax_reff.clear()
            ax_inf.plot(sol.t, I, label='Infectados', color='firebrick')
            ax_inf.plot(sol.t, S, label='Reservorio disponible', color='steelblue')
            ax_inf.set_xlabel('Tiempo (dias)')
            ax_inf.set_ylabel('Personas')
            ax_inf.set_title('Dinamica Verhulst')
            ax_inf.legend(); ax_inf.grid(True, alpha=0.3)

            ax_metrics.bar(sol.t, nuevos, width=dias/400, color='darkorange', alpha=0.7, label='Casos nuevos')
            ax_metrics.set_xlabel('Tiempo (dias)')
            ax_metrics.set_ylabel('Casos/dia')
            ax_metrics.set_title('Casos diarios y R efectivo')
            ax_metrics.grid(True, alpha=0.3)

            ax_reff.plot(sol.t, reff, color='darkslateblue', label='R efectivo')
            ax_reff.axhline(1.0, color='gray', linestyle='--', linewidth=1)
            ax_reff.set_ylabel('R efectivo')

            ax_metrics.legend([ax_metrics.containers[0], ax_reff.get_lines()[0]], ['Casos nuevos', 'R efectivo'], loc='upper right')
            canvas.draw()

            pico = np.max(I)
            dia_pico = float(sol.t[np.argmax(I)])
            resumen.delete('1.0', tk.END)
            resumen.insert(tk.END, f'Capacidad K = {K}\n')
            resumen.insert(tk.END, f'Pico infectados: {pico:.0f} (dia {dia_pico:.1f})\n')
            resumen.insert(tk.END, f'Casos finales: {I[-1]:.0f}\n')
            resumen.insert(tk.END, f'Reservorio final: {S[-1]:.0f}\n')
            resumen.insert(tk.END, f'R efectivo minimo: {np.min(reff):.2f}\n')
            resumen.insert(tk.END, f'R efectivo maximo: {np.max(reff):.2f}\n')
            resumen.insert(tk.END, f'Casos diarios max: {np.max(nuevos):.1f}\n')
            if conf_var.get():
                resumen.insert(tk.END, f'Confinamiento desde dia {conf_day} (-{conf_drop*100:.0f}% r)\n')
            if vac_var.get():
                resumen.insert(tk.END, f'Vacunacion constante: {vac_rate} por dia\n')
            if shock != 0:
                resumen.insert(tk.END, f'Impulso inicial: {shock} casos/dia\n')
        except Exception as exc:
            messagebox.showerror('Error', f'Simulacion: {exc}')

    tk.Button(panel_izq, text='Simular brote', bg='#00695C', fg='white', command=simular).pack(anchor='w', pady=10)
    tk.Label(root, text=FIRMA, font=('Consolas', 9, 'italic')).pack(side=tk.BOTTOM, pady=6)

    root.mainloop()


if __name__ == '__main__':
    lanzar_app()
