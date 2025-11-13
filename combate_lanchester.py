#!/usr/bin/env python3
"""Panel interactivo para ensayar escenarios tipo Lanchester extendidos.
Incluye variaciones lineales, cuadraticas y mixtas con fatiga, refuerzos
constantes, y moduladores estrategicos para cada bando.
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


def _labeled_entry(parent, texto, valor, ancho=10):
    fila = tk.Frame(parent)
    fila.pack(anchor='w', pady=1)
    tk.Label(fila, text=texto).pack(side=tk.LEFT)
    entry = tk.Entry(fila, width=ancho)
    entry.pack(side=tk.LEFT)
    entry.insert(0, str(valor))
    return entry


def lanzar_app():
    root = tk.Tk()
    root.title('Laboratorio de enfrentamientos')
    root.geometry('1350x820')
    root.minsize(1200, 700)

    marco = tk.Frame(root)
    marco.pack(fill=tk.BOTH, expand=True)

    panel_izq = tk.Frame(marco, padx=10, pady=10)
    panel_izq.pack(side=tk.LEFT, fill=tk.Y)
    panel_der = tk.Frame(marco)
    panel_der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    tk.Label(panel_izq, text='Escenario inicial', font=('Consolas', 11, 'bold')).pack(anchor='w')
    entry_B0 = _labeled_entry(panel_izq, 'Ejercito Celeste :', 500)
    entry_R0 = _labeled_entry(panel_izq, 'Ejercito Verde :', 500)

    tk.Label(panel_izq, text='Coeficientes de enfrentamiento', font=('Consolas', 11, 'bold')).pack(anchor='w', pady=(6,0))
    entry_alpha = _labeled_entry(panel_izq, 'σ (Celeste sobre Verde) :', 0.001)
    entry_beta  = _labeled_entry(panel_izq, 'τ (Verde sobre Celeste) :', 0.001)

    tk.Label(panel_izq, text='Moduladores estrategicos', font=('Consolas', 11, 'bold')).pack(anchor='w', pady=(6,0))
    entry_sup_blue = _labeled_entry(panel_izq, 'Sinergia Celeste :', 4.0, ancho=6)
    entry_sup_red  = _labeled_entry(panel_izq, 'Adaptacion Verde :', 1.0, ancho=6)

    tk.Label(panel_izq, text='Tipo de modelo', font=('Consolas', 11, 'bold')).pack(anchor='w', pady=(6,0))
    modelo_var = tk.StringVar(value='linear')
    opciones = [
        ('Disperso (lineal)', 'linear'),
        ('Potenciado (cuadratico)', 'quadratic'),
        ('Asimetrico (Celeste cuadratico)', 'mixed')
    ]
    for texto, val in opciones:
        tk.Radiobutton(panel_izq, text=texto, variable=modelo_var, value=val).pack(anchor='w')

    tk.Label(panel_izq, text='Resistencia operativa', font=('Consolas', 11, 'bold')).pack(anchor='w', pady=(6,0))
    fatiga_var = tk.BooleanVar(value=False)
    tk.Checkbutton(panel_izq, text='Incluir fatiga', variable=fatiga_var).pack(anchor='w')
    entry_fatiga_blue = _labeled_entry(panel_izq, 'tasa Celeste :', 0, ancho=6)
    entry_fatiga_red  = _labeled_entry(panel_izq, 'tasa Verde :', 0, ancho=6)

    tk.Label(panel_izq, text='Apoyo logistico', font=('Consolas', 11, 'bold')).pack(anchor='w', pady=(6,0))
    ref_var = tk.BooleanVar(value=False)
    tk.Checkbutton(panel_izq, text='Agregar refuerzos constantes', variable=ref_var).pack(anchor='w')
    entry_ref_blue = _labeled_entry(panel_izq, 'Celeste (+/tiempo):', 0, ancho=6)
    entry_ref_red  = _labeled_entry(panel_izq, 'Verde (+/tiempo):', 0, ancho=6)

    entry_tmax = _labeled_entry(panel_izq, 'Duracion max (dias):', 200, ancho=8)

    figura, (ax_t, ax_losses) = plt.subplots(2, 1, figsize=(7,8))
    figura.tight_layout(pad=3.0)
    canvas = FigureCanvasTkAgg(figura, master=panel_der)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    resumen_txt = tk.Text(panel_izq, width=45, height=20, font=('Consolas', 10), bg='#f3f3f3')
    resumen_txt.pack(fill=tk.X, pady=(8,0))

    def simular():
        try:
            B0 = max(float(entry_B0.get()), 0.0)
            R0 = max(float(entry_R0.get()), 0.0)
            alpha = float(entry_alpha.get())
            beta = float(entry_beta.get())
            t_final = max(float(entry_tmax.get()), 1.0)
            sinergia = max(float(entry_sup_blue.get()), 0.0)
            adaptacion = max(float(entry_sup_red.get()), 0.0)
            ref_b = float(entry_ref_blue.get()) if ref_var.get() else 0.0
            ref_r = float(entry_ref_red.get()) if ref_var.get() else 0.0
            fat_b = float(entry_fatiga_blue.get()) if fatiga_var.get() else 0.0
            fat_r = float(entry_fatiga_red.get()) if fatiga_var.get() else 0.0
            modelo = modelo_var.get()

            def rhs(t, z):
                B, R = max(z[0], 0.0), max(z[1], 0.0)
                eff_alpha = alpha * sinergia
                eff_beta = beta * adaptacion
                if modelo == 'linear':
                    dB = -eff_beta * R - fat_b * B + ref_b
                    dR = -eff_alpha * B - fat_r * R + ref_r
                elif modelo == 'quadratic':
                    dB = -eff_beta * R * B - fat_b * B + ref_b
                    dR = -eff_alpha * B * R - fat_r * R + ref_r
                else:  # mixed
                    dB = -eff_beta * R * B - fat_b * B + ref_b
                    dR = -eff_alpha * B - fat_r * R + ref_r
                return [dB, dR]

            sol = solve_ivp(rhs, [0, t_final], [B0, R0], t_eval=np.linspace(0, t_final, 400))
            if not sol.success:
                raise RuntimeError('Integrador no convergio')
            B = np.clip(sol.y[0], 0, None)
            R = np.clip(sol.y[1], 0, None)
            Bf, Rf = float(B[-1]), float(R[-1])
            perdidas_b = B0 - Bf
            perdidas_r = R0 - Rf
            ganador = 'Celeste' if Bf > Rf else ('Verde' if Rf > Bf else 'Empate')

            ax_t.clear(); ax_losses.clear()
            ax_t.plot(sol.t, B, label='Ejercito Celeste', color='deepskyblue')
            ax_t.plot(sol.t, R, label='Ejercito Verde', color='seagreen')
            ax_t.set_xlabel('Tiempo'); ax_t.set_ylabel('Efectivos'); ax_t.set_title('Trayectoria de ejercitos')
            ax_t.legend(); ax_t.grid(True, alpha=0.3)

            relacion = np.divide(B, np.maximum(R, 1e-6))
            perdidas_c = np.maximum(B0 - B, 0)
            perdidas_v = np.maximum(R0 - R, 0)
            ax_losses.plot(sol.t, perdidas_c, label='Bajas Celeste', color='dodgerblue')
            ax_losses.plot(sol.t, perdidas_v, label='Bajas Verde', color='darkgreen')
            ax_losses.fill_between(sol.t, perdidas_c, perdidas_v, where=perdidas_c>=perdidas_v,
                                   color='dodgerblue', alpha=0.08)
            ax_losses.fill_between(sol.t, perdidas_c, perdidas_v, where=perdidas_v>perdidas_c,
                                   color='darkgreen', alpha=0.08)
            ax_losses.set_xlabel('Tiempo'); ax_losses.set_ylabel('Efectivos perdidos')
            ax_losses.set_title('Perdidas acumuladas')
            ax_losses.legend(loc='best')
            ax_losses.grid(True, alpha=0.3)
            canvas.draw()

            resumen = []
            resumen.append('Modelo {} | σ={} τ={}'.format(modelo, alpha, beta))
            resumen.append('Moduladores -> Celeste {} | Verde {}'.format(sinergia, adaptacion))
            resumen.append('Inicio: Ejercito Celeste {:.0f}  Ejercito Verde {:.0f}'.format(B0, R0))
            resumen.append('Final : Ejercito Celeste {:.2f}  Ejercito Verde {:.2f}'.format(Bf, Rf))
            resumen.append('Perdidas -> Celeste {:.2f}  Verde {:.2f}'.format(perdidas_b, perdidas_r))
            resumen.append('Resultado: {}'.format(ganador))
            if fatiga_var.get():
                resumen.append('Fatiga aplicada Celeste={} Verde={}'.format(fat_b, fat_r))
            if ref_var.get():
                resumen.append('Refuerzos por tiempo Celeste={} Verde={}'.format(ref_b, ref_r))
            max_rel = float(np.max(relacion))
            resumen.append('Relacion max Celeste/Verde: {:.2f}'.format(max_rel))
            resumen_txt.delete('1.0', tk.END)
            resumen_txt.insert(tk.END, '\n'.join(resumen))
        except Exception as exc:
            messagebox.showerror('Error', f'Simulacion: {exc}')

    tk.Button(panel_izq, text='Simular combate', bg='#4CAF50', fg='white', command=simular).pack(anchor='w', pady=10)
    tk.Label(root, text=FIRMA, font=('Consolas', 9, 'italic')).pack(side=tk.BOTTOM, pady=6)

    root.mainloop()


if __name__ == '__main__':
    lanzar_app()
