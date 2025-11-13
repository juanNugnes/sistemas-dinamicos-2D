#!/usr/bin/env python3
"""Laboratorio simple para explorar dinamicas tipo Romeo y Julieta.
Permite ajustar sensibilidades propias y cruzadas, escoger modos narrativos
y visualizar la evolucion temporal y en plano de fases.
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


def _labeled_entry(parent, texto, valor, ancho=8):
    fila = tk.Frame(parent)
    fila.pack(anchor='w', pady=1)
    tk.Label(fila, text=texto).pack(side=tk.LEFT)
    entry = tk.Entry(fila, width=ancho)
    entry.pack(side=tk.LEFT)
    entry.insert(0, str(valor))
    return entry


def lanzar_app():
    root = tk.Tk()
    root.title('Laboratorio Romeo & Julieta')
    root.geometry('1280x780')
    root.minsize(1100, 650)

    contenedor = tk.Frame(root)
    contenedor.pack(fill=tk.BOTH, expand=True)

    panel_izq = tk.Frame(contenedor, padx=10, pady=10)
    panel_izq.pack(side=tk.LEFT, fill=tk.Y)
    panel_der = tk.Frame(contenedor)
    panel_der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    tk.Label(panel_izq, text='Estado inicial', font=('Consolas', 11, 'bold')).pack(anchor='w')
    entry_R0 = _labeled_entry(panel_izq, 'Romeo (R0):', 1.0)
    entry_J0 = _labeled_entry(panel_izq, 'Julieta (J0):', -0.5)

    tk.Label(panel_izq, text='Sensibilidad propia', font=('Consolas', 11, 'bold')).pack(anchor='w', pady=(6,0))
    entry_alpha = _labeled_entry(panel_izq, 'a (Romeo):', 0.1)
    entry_delta = _labeled_entry(panel_izq, 'd (Julieta):', -0.1)

    tk.Label(panel_izq, text='Influencia cruzada', font=('Consolas', 11, 'bold')).pack(anchor='w', pady=(6,0))
    entry_beta = _labeled_entry(panel_izq, 'b (Julieta sobre R):', 1.0)
    entry_gamma = _labeled_entry(panel_izq, 'c (Romeo sobre J):', -1.0)

    tk.Label(panel_izq, text='Modo narrativo', font=('Consolas', 11, 'bold')).pack(anchor='w', pady=(6,0))
    modo_var = tk.StringVar(value='clasico')
    modos = [
        ('Clasico', 'clasico'),
        ('Impulsivo', 'impulsivo'),
        ('Dramatico', 'dramatico')
    ]
    for texto, val in modos:
        tk.Radiobutton(panel_izq, text=texto, variable=modo_var, value=val).pack(anchor='w')

    sat_var = tk.BooleanVar(value=False)
    tk.Checkbutton(panel_izq, text='Limitar emociones (tanh)', variable=sat_var).pack(anchor='w', pady=(4,0))
    entry_sat = _labeled_entry(panel_izq, 'Escala saturacion:', 3.0, ancho=6)

    entry_boost = _labeled_entry(panel_izq, 'Impulso externo:', 0.0, ancho=6)
    entry_tmax = _labeled_entry(panel_izq, 'Duracion (dias):', 50, ancho=6)

    figura, (ax_t, ax_phase) = plt.subplots(2, 1, figsize=(7,8))
    figura.tight_layout(pad=3.0)
    canvas = FigureCanvasTkAgg(figura, master=panel_der)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    resumen = tk.Text(panel_izq, width=42, height=18, font=('Consolas', 10), bg='#f3f3f3')
    resumen.pack(fill=tk.X, pady=(8,0))

    def simular():
        try:
            R0 = float(entry_R0.get())
            J0 = float(entry_J0.get())
            a = float(entry_alpha.get())
            d = float(entry_delta.get())
            b = float(entry_beta.get())
            c = float(entry_gamma.get())
            impulso = float(entry_boost.get())
            t_final = max(float(entry_tmax.get()), 1.0)
            sat = max(float(entry_sat.get()), 0.1)
            modo = modo_var.get()

            def rhs(t, z):
                R, J = z
                a_eff, d_eff = a, d
                b_eff, c_eff = b, c
                forc = impulso
                if modo == 'impulsivo':
                    b_eff *= 1.4
                    c_eff *= 1.4
                    forc += 0.2*np.sin(0.4*t)
                elif modo == 'dramatico':
                    a_eff -= 0.15
                    d_eff -= 0.15
                    forc += 0.1*np.cos(0.25*t)
                if sat_var.get():
                    R_lin = sat * np.tanh(R/sat)
                    J_lin = sat * np.tanh(J/sat)
                else:
                    R_lin = R
                    J_lin = J
                dR = a_eff * R_lin + b_eff * J_lin + forc
                dJ = c_eff * R_lin + d_eff * J_lin - forc
                return [dR, dJ]

            sol = solve_ivp(rhs, [0, t_final], [R0, J0], t_eval=np.linspace(0, t_final, 400))
            if not sol.success:
                raise RuntimeError('Integrador no convergio')

            R = sol.y[0]
            J = sol.y[1]
            energia = 0.5*(R**2 + J**2)
            balance = R - J

            ax_t.clear(); ax_phase.clear()
            ax_t.plot(sol.t, R, label='Romeo', color='royalblue')
            ax_t.plot(sol.t, J, label='Julieta', color='crimson')
            ax_t.set_xlabel('Tiempo'); ax_t.set_ylabel('Intensidad emocional')
            ax_t.set_title('Evolucion de sentimientos')
            ax_t.legend(); ax_t.grid(True, alpha=0.3)

            ax_phase.plot(R, J, color='purple', lw=2)
            ax_phase.scatter([R0], [J0], color='black', marker='o', label='Inicio')
            ax_phase.scatter([R[-1]], [J[-1]], color='black', marker='x', label='Final')
            ax_phase.set_xlabel('Romeo'); ax_phase.set_ylabel('Julieta')
            ax_phase.set_title('Plano de fases')
            ax_phase.legend(); ax_phase.grid(True, alpha=0.3)
            canvas.draw()

            resumen.delete('1.0', tk.END)
            resumen.insert(tk.END, 'Modo {}\n'.format(modo))
            resumen.insert(tk.END, 'Parametros a={}, b={}, c={}, d={}\n'.format(a, b, c, d))
            resumen.insert(tk.END, 'Inicio R={:.2f} J={:.2f}\n'.format(R0, J0))
            resumen.insert(tk.END, 'Final  R={:.2f} J={:.2f}\n'.format(R[-1], J[-1]))
            resumen.insert(tk.END, 'Balance final R-J = {:.2f}\n'.format(balance[-1]))
            resumen.insert(tk.END, 'Energia max {:.2f}\n'.format(np.max(energia)))
            if sat_var.get():
                resumen.insert(tk.END, 'Saturacion activada (escala {})\n'.format(sat))
            if impulso != 0.0:
                resumen.insert(tk.END, 'Impulso externo = {}\n'.format(impulso))
        except Exception as exc:
            messagebox.showerror('Error', f'Simulacion: {exc}')

    tk.Button(panel_izq, text='Simular historia', bg='#9C27B0', fg='white', command=simular).pack(anchor='w', pady=8)
    tk.Label(root, text=FIRMA, font=('Consolas', 9, 'italic')).pack(side=tk.BOTTOM, pady=6)

    root.mainloop()


if __name__ == '__main__':
    lanzar_app()
