import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox
from scipy.integrate import solve_ivp

# ============================================================
# 🔹 Ventana MODO NUMÉRICO CLÁSICO
# ============================================================
def abrir_modo_numerico():
    wn = tk.Toplevel()
    wn.title("Sistema dinámico 2D - Modo Numérico clásico")
    wn.geometry("1200x900")

    # --- Dividir ventana: gráfico a la izquierda / análisis a la derecha
    frame_main = tk.Frame(wn)
    frame_main.pack(fill=tk.BOTH, expand=True)

    frame_left = tk.Frame(frame_main)
    frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    frame_right = tk.Frame(frame_main, padx=10, pady=10, bg="#f4f4f4")
    frame_right.pack(side=tk.RIGHT, fill=tk.Y)

    # === Entradas ===
    frame_inputs = tk.Frame(frame_left)
    frame_inputs.pack(pady=5)

    tk.Label(frame_inputs, text="dx =", font=("Consolas", 12)).grid(row=0, column=0)
    entry_fx = tk.Entry(frame_inputs, width=25, font=("Consolas", 12))
    entry_fx.grid(row=0, column=1)
    entry_fx.insert(0, "2*x - 5*y")

    tk.Label(frame_inputs, text="dy =", font=("Consolas", 12)).grid(row=1, column=0)
    entry_fy = tk.Entry(frame_inputs, width=25, font=("Consolas", 12))
    entry_fy.grid(row=1, column=1)
    entry_fy.insert(0, "4*x - 2*y")

    tk.Label(frame_inputs, text="x₀:", font=("Consolas", 12)).grid(row=0, column=2)
    entry_x0 = tk.Entry(frame_inputs, width=8, font=("Consolas", 12))
    entry_x0.grid(row=0, column=3)
    entry_x0.insert(0, "1")

    tk.Label(frame_inputs, text="y₀:", font=("Consolas", 12)).grid(row=1, column=2)
    entry_y0 = tk.Entry(frame_inputs, width=8, font=("Consolas", 12))
    entry_y0.grid(row=1, column=3)
    entry_y0.insert(0, "1")

    # === Canvas ===
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    canvas = FigureCanvasTkAgg(fig, master=frame_left)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)

    # === Panel derecho para mostrar resultados ===
    tk.Label(frame_right, text="📊 Análisis local del sistema:", bg="#f4f4f4",
             font=("Consolas", 12, "bold")).pack(anchor="w", pady=(0, 5))
    text_info = tk.Text(frame_right, width=45, height=35, font=("Consolas", 11), bg="#f9f9f9")
    text_info.pack(fill=tk.BOTH, expand=True)

    # === Función principal ===
    def graficar_numerico():
        fx_text = entry_fx.get().strip()
        fy_text = entry_fy.get().strip()
        text_info.delete("1.0", tk.END)

        try:
            x0 = float(entry_x0.get())
            y0 = float(entry_y0.get())
        except ValueError:
            messagebox.showerror("Error", "x₀ e y₀ deben ser números.")
            return

        def system(t, XY):
            x, y = XY
            dx = eval(fx_text, {"x": x, "y": y, "np": np})
            dy = eval(fy_text, {"x": x, "y": y, "np": np})
            return [dx, dy]

        # === Jacobiano numérico en (0,0) ===
        try:
            h = 1e-5
            dfdx = (eval(fx_text, {"x": h, "y": 0}) - eval(fx_text, {"x": -h, "y": 0})) / (2*h)
            dfdy = (eval(fx_text, {"x": 0, "y": h}) - eval(fx_text, {"x": 0, "y": -h})) / (2*h)
            dgdx = (eval(fy_text, {"x": h, "y": 0}) - eval(fy_text, {"x": -h, "y": 0})) / (2*h)
            dgdy = (eval(fy_text, {"x": 0, "y": h}) - eval(fy_text, {"x": 0, "y": -h})) / (2*h)
            A = np.array([[dfdx, dfdy], [dgdx, dgdy]])
            valores, vectores = np.linalg.eig(A)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron calcular autovalores:\n{e}")
            return

        alpha = np.real(valores[0])
        beta = np.imag(valores[0])
        gamma = np.sqrt(alpha**2 + beta**2)
        tau = np.trace(A)
        Delta = np.linalg.det(A)
        D = tau**2 - 4*Delta

        # Mostrar info numérica
        text_info.insert(tk.END, f"Matriz Jacobiana (en (0,0)):\n{A}\n\n")
        text_info.insert(tk.END, f"Autovalores:\n λ₁ = {valores[0]:.4f}\n λ₂ = {valores[1]:.4f}\n\n")
        text_info.insert(tk.END, f"Partes reales e imaginarias:\n α = {alpha:.4f}\n β = {beta:.4f}\n\n")
        text_info.insert(tk.END, f"Autovectores:\n{vectores}\n\n")
        text_info.insert(tk.END, f"Invariantes:\n τ = {tau:.4f}\n Δ = {Delta:.4f}\n D = {D:.4f}\n\n")

        # Clasificación
        if Delta < 0:
            tipo = "Silla"
        elif D < 0:
            tipo = "Foco" if tau != 0 else "Centro"
        else:
            tipo = "Nodo"
        text_info.insert(tk.END, f"Tipo de equilibrio: {tipo}\n")

        # === Campo vectorial ===
        ax.clear()
        x_min, x_max = -10, 10
        y_min, y_max = -10, 10
        n_points = 60
        X, Y = np.meshgrid(np.linspace(x_min, x_max, n_points),
                           np.linspace(y_min, y_max, n_points))
        U, V = np.zeros_like(X), np.zeros_like(Y)
        for i in range(n_points):
            for j in range(n_points):
                dx, dy = system(0, [X[i, j], Y[i, j]])
                norm = np.sqrt(dx**2 + dy**2)
                if norm != 0:
                    U[i, j], V[i, j] = dx/norm, dy/norm

        ax.quiver(X, Y, U, V, color='gray', alpha=0.4, scale=25, scale_units='xy')

        # === Nulclinas ===
        try:
            cont1 = ax.contour(X, Y, U, levels=[0], colors='blue', linewidths=2, linestyles='--')
            cont2 = ax.contour(X, Y, V, levels=[0], colors='red', linewidths=2, linestyles='--')
            ax.clabel(cont1, fmt='dx/dt=0', colors='blue', fontsize=9)
            ax.clabel(cont2, fmt='dy/dt=0', colors='red', fontsize=9)
        except Exception as e:
            print("Error calculando nulclinas:", e)

        # === Trayectoria temporal ===
        sol = solve_ivp(system, [0, 20], [x0, y0], t_eval=np.linspace(0, 20, 1000))
        if sol.success and sol.y.size > 0:
            x_traj, y_traj = sol.y
            ax.plot(x_traj, y_traj, color='green', lw=2, label="Trayectoria")
            ax.scatter(x_traj[0], y_traj[0], color='black', s=50, label="Inicial")

        # Configuración final
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Campo de direcciones, nulclinas y trayectoria")
        ax.grid(True)
        ax.legend()
        canvas.draw()

    # === Botón ===
    tk.Button(frame_left, text="GRAFICAR", bg="#4CAF50", fg="white",
              font=("Consolas", 14, "bold"), command=graficar_numerico).pack(pady=10)


# ============================================================
# 🔹 Ventana MODO SIMBÓLICO / NO LINEAL
# ============================================================
def abrir_modo_simbolico():
    ws = tk.Toplevel()
    ws.title("Sistema dinámico 2D - Modo Simbólico / No lineal")
    ws.geometry("1200x900")

    # --- Dividimos en dos secciones: izquierda (gráfico) / derecha (info)
    frame_main = tk.Frame(ws)
    frame_main.pack(fill=tk.BOTH, expand=True)

    frame_left = tk.Frame(frame_main)
    frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    frame_right = tk.Frame(frame_main, padx=10, pady=10, bg="#f4f4f4")
    frame_right.pack(side=tk.RIGHT, fill=tk.Y)

    # Entradas arriba del gráfico
    frame_inputs = tk.Frame(frame_left)
    frame_inputs.pack(pady=5)

    tk.Label(frame_inputs, text="dx =", font=("Consolas", 12)).grid(row=0, column=0)
    entry_fx = tk.Entry(frame_inputs, width=25, font=("Consolas", 12))
    entry_fx.grid(row=0, column=1)
    entry_fx.insert(0, "y")

    tk.Label(frame_inputs, text="dy =", font=("Consolas", 12)).grid(row=1, column=0)
    entry_fy = tk.Entry(frame_inputs, width=25, font=("Consolas", 12))
    entry_fy.grid(row=1, column=1)
    entry_fy.insert(0, "-x")

    # Figura principal
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    canvas = FigureCanvasTkAgg(fig, master=frame_left)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)

    # Panel derecho: resultados textuales
    tk.Label(frame_right, text="📊 Puntos de equilibrio y análisis:", bg="#f4f4f4",
             font=("Consolas", 12, "bold")).pack(anchor="w", pady=(0, 5))
    text_result = tk.Text(frame_right, width=45, height=35, font=("Consolas", 11), bg="#f9f9f9")
    text_result.pack(fill=tk.BOTH, expand=True)

    def graficar_simbolico():
        fx_text = entry_fx.get().strip()
        fy_text = entry_fy.get().strip()
        text_result.delete("1.0", tk.END)  # limpiar resultados previos

        x, y = sp.symbols("x y")
        try:
            f_expr = sp.sympify(fx_text)
            g_expr = sp.sympify(fy_text)
        except Exception as e:
            messagebox.showerror("Error", f"Expresiones inválidas: {e}")
            return

        # --- Cálculos simbólicos ---
        nullcline_x = sp.solve(sp.Eq(f_expr, 0), y)
        nullcline_y = sp.solve(sp.Eq(g_expr, 0), x)
        equilibria = sp.solve([sp.Eq(f_expr, 0), sp.Eq(g_expr, 0)], (x, y), dict=True)
        J = sp.Matrix([[sp.diff(f_expr, x), sp.diff(f_expr, y)],
                       [sp.diff(g_expr, x), sp.diff(g_expr, y)]])

        ax.clear()
        f_np = sp.lambdify((x, y), f_expr, modules='numpy')
        g_np = sp.lambdify((x, y), g_expr, modules='numpy')

        x_vals = np.linspace(-5, 5, 200)
        y_vals = np.linspace(-5, 5, 200)
        X, Y = np.meshgrid(x_vals, y_vals)
        U = f_np(X, Y)
        V = g_np(X, Y)

        ax.streamplot(X, Y, U, V, color='gray', density=1.5, linewidth=1)

        # Nullclinas
        for sol in nullcline_x:
            try:
                y_fun = sp.lambdify(x, sol, modules='numpy')
                ax.plot(x_vals, y_fun(x_vals), 'b--', label='dx/dt = 0')
            except Exception:
                pass
        for sol in nullcline_y:
            try:
                x_fun = sp.lambdify(y, sol, modules='numpy')
                ax.plot(x_fun(y_vals), y_vals, 'r--', label='dy/dt = 0')
            except Exception:
                pass

        # Equilibrios + texto lateral
        if equilibria:
            for i, eq in enumerate(equilibria, start=1):
                xe = float(eq[x])
                ye = float(eq[y])
                J_eval = J.subs(eq)
                eigvals = list(J_eval.eigenvals().keys())

                tau = np.real(sum([complex(ev.evalf()) for ev in eigvals]))
                Delta = np.real(np.prod([complex(ev.evalf()) for ev in eigvals]))
                D = tau**2 - 4 * Delta
                if Delta < 0:
                    tipo = "Silla"
                elif D < 0:
                    tipo = "Foco" if tau != 0 else "Centro"
                else:
                    tipo = "Nodo"

                ax.scatter(xe, ye, color='black', s=60)
                ax.text(xe + 0.2, ye + 0.2, f"{tipo}", fontsize=9)

                text_result.insert(tk.END,
                    f"🔹 Punto de equilibrio {i}:\n"
                    f"   (x, y) = ({xe:.3f}, {ye:.3f})\n"
                    f"   τ = {tau:.4f}\n"
                    f"   Δ = {Delta:.4f}\n"
                    f"   D = {D:.4f}\n"
                    f"   Tipo: {tipo}\n"
                    f"----------------------------------------\n"
                )
        else:
            text_result.insert(tk.END, "⚠️ No se encontraron puntos de equilibrio.\n")

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.grid(True)
        ax.set_title("Modo Simbólico / No lineal")
        ax.legend(fontsize=8)
        canvas.draw()

    tk.Button(frame_left, text="GRAFICAR", bg="#4CAF50", fg="white",
              font=("Consolas", 14, "bold"), command=graficar_simbolico).pack(pady=10)





# ============================================================
# 🧭 Ventana modo no homogeneo
# ============================================================

def abrir_modo_no_homogeneo():
    wh = tk.Toplevel()
    wh.title("Sistema dinámico 2D - Modo No homogéneo")
    wh.geometry("1200x900")

    # --- Dividir ventana
    frame_main = tk.Frame(wh)
    frame_main.pack(fill=tk.BOTH, expand=True)

    frame_left = tk.Frame(frame_main)
    frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    frame_right = tk.Frame(frame_main, padx=10, pady=10, bg="#f4f4f4")
    frame_right.pack(side=tk.RIGHT, fill=tk.Y)

    # === Entradas ===
    frame_inputs = tk.Frame(frame_left)
    frame_inputs.pack(pady=5)

    tk.Label(frame_inputs, text="dx =", font=("Consolas", 12)).grid(row=0, column=0)
    entry_fx = tk.Entry(frame_inputs, width=25, font=("Consolas", 12))
    entry_fx.grid(row=0, column=1)
    entry_fx.insert(0, "2*x - 5*y")

    tk.Label(frame_inputs, text="dy =", font=("Consolas", 12)).grid(row=1, column=0)
    entry_fy = tk.Entry(frame_inputs, width=25, font=("Consolas", 12))
    entry_fy.grid(row=1, column=1)
    entry_fy.insert(0, "4*x - 2*y")

    tk.Label(frame_inputs, text="f(t) =", font=("Consolas", 12)).grid(row=2, column=0)
    entry_ft = tk.Entry(frame_inputs, width=25, font=("Consolas", 12))
    entry_ft.grid(row=2, column=1)
    entry_ft.insert(0, "np.sin(t)")

    tk.Label(frame_inputs, text="g(t) =", font=("Consolas", 12)).grid(row=3, column=0)
    entry_gt = tk.Entry(frame_inputs, width=25, font=("Consolas", 12))
    entry_gt.grid(row=3, column=1)
    entry_gt.insert(0, "np.cos(t)")

    tk.Label(frame_inputs, text="x₀:", font=("Consolas", 12)).grid(row=0, column=2)
    entry_x0 = tk.Entry(frame_inputs, width=8, font=("Consolas", 12))
    entry_x0.grid(row=0, column=3)
    entry_x0.insert(0, "1")

    tk.Label(frame_inputs, text="y₀:", font=("Consolas", 12)).grid(row=1, column=2)
    entry_y0 = tk.Entry(frame_inputs, width=8, font=("Consolas", 12))
    entry_y0.grid(row=1, column=3)
    entry_y0.insert(0, "1")

    tk.Label(frame_inputs, text="Periodo T:", font=("Consolas", 12)).grid(row=2, column=2)
    entry_T = tk.Entry(frame_inputs, width=8, font=("Consolas", 12))
    entry_T.grid(row=2, column=3)
    entry_T.insert(0, "6.283")

    show_poincare = tk.BooleanVar()
    tk.Checkbutton(frame_inputs, text="Mostrar sección de Poincaré",
                   variable=show_poincare, font=("Consolas", 12)).grid(row=3, column=2, columnspan=2)

    # === Canvas ===
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    canvas = FigureCanvasTkAgg(fig, master=frame_left)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)

    # === Panel derecho ===
    tk.Label(frame_right, text="📊 Análisis del sistema no homogéneo:",
             bg="#f4f4f4", font=("Consolas", 12, "bold")).pack(anchor="w", pady=(0, 5))
    text_info = tk.Text(frame_right, width=45, height=35, font=("Consolas", 11), bg="#f9f9f9")
    text_info.pack(fill=tk.BOTH, expand=True)

    # === Función principal ===
    def graficar_no_homogeneo():
        fx_text = entry_fx.get().strip()
        fy_text = entry_fy.get().strip()
        ft_text = entry_ft.get().strip()
        gt_text = entry_gt.get().strip()
        text_info.delete("1.0", tk.END)

        try:
            x0 = float(entry_x0.get())
            y0 = float(entry_y0.get())
        except ValueError:
            messagebox.showerror("Error", "x₀ e y₀ deben ser números.")
            return

        def system(t, XY):
            x, y = XY
            dx = eval(fx_text, {"x": x, "y": y, "t": t, "np": np})
            dy = eval(fy_text, {"x": x, "y": y, "t": t, "np": np})
            dx += eval(ft_text, {"t": t, "np": np})
            dy += eval(gt_text, {"t": t, "np": np})
            return [dx, dy]

        # --- Jacobiano homogéneo (sin f(t), g(t))
        h = 1e-5
        dfdx = (eval(fx_text, {"x": h, "y": 0}) - eval(fx_text, {"x": -h, "y": 0})) / (2*h)
        dfdy = (eval(fx_text, {"x": 0, "y": h}) - eval(fx_text, {"x": 0, "y": -h})) / (2*h)
        dgdx = (eval(fy_text, {"x": h, "y": 0}) - eval(fy_text, {"x": -h, "y": 0})) / (2*h)
        dgdy = (eval(fy_text, {"x": 0, "y": h}) - eval(fy_text, {"x": 0, "y": -h})) / (2*h)
        A = np.array([[dfdx, dfdy], [dgdx, dgdy]])
        valores, vectores = np.linalg.eig(A)
        tau = np.trace(A)
        Delta = np.linalg.det(A)
        D = tau**2 - 4*Delta

        text_info.insert(tk.END, f"Matriz Jacobiana (parte homogénea):\n{A}\n\n")
        text_info.insert(tk.END, f"Autovalores:\n λ₁={valores[0]:.4f}, λ₂={valores[1]:.4f}\n")
        text_info.insert(tk.END, f"Invariantes: τ={tau:.4f}, Δ={Delta:.4f}, D={D:.4f}\n\n")

        # === Campo vectorial ===
        ax.clear()
        x_min, x_max = -10, 10
        y_min, y_max = -10, 10
        n_points = 60
        X, Y = np.meshgrid(np.linspace(x_min, x_max, n_points),
                           np.linspace(y_min, y_max, n_points))
        U, V = np.zeros_like(X), np.zeros_like(Y)
        for i in range(n_points):
            for j in range(n_points):
                dx, dy = system(0, [X[i, j], Y[i, j]])
                norm = np.sqrt(dx**2 + dy**2)
                if norm != 0:
                    U[i, j], V[i, j] = dx/norm, dy/norm

        ax.quiver(X, Y, U, V, color='gray', alpha=0.4, scale=25, scale_units='xy')
        ax.contour(X, Y, U, levels=[0], colors='blue', linestyles='--')
        ax.contour(X, Y, V, levels=[0], colors='red', linestyles='--')

        # === Trayectoria temporal ===
        T = float(entry_T.get())
        t_final = 10*T if show_poincare.get() else 20
        sol = solve_ivp(system, [0, t_final], [x0, y0], t_eval=np.linspace(0, t_final, 1500))

        if sol.success and sol.y.size > 0:
            x_traj, y_traj = sol.y
            ax.plot(x_traj, y_traj, color='green', lw=2, label="Trayectoria")
            ax.scatter(x_traj[0], y_traj[0], color='black', s=50, label="Inicial")

        # === Poincaré ===
        if show_poincare.get():
            x_p, y_p = [], []
            for n in range(1, 11):
                tn = n*T
                xn = np.interp(tn, sol.t, sol.y[0])
                yn = np.interp(tn, sol.t, sol.y[1])
                x_p.append(xn)
                y_p.append(yn)
            ax.scatter(x_p, y_p, color='magenta', s=50, zorder=4, label="Poincaré")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True)
        ax.set_title("Campo y trayectoria (No homogéneo)")
        ax.legend()
        canvas.draw()

    # === FRAME BOTÓN (separado para que no se tape con el gráfico) ===
    frame_button = tk.Frame(frame_left)
    frame_button.pack(pady=10)

    btn_graficar = tk.Button(frame_button, text="GRAFICAR", bg="#4CAF50", fg="white",
                             font=("Consolas", 14, "bold"), width=20, height=1,
                             command=graficar_no_homogeneo)
    btn_graficar.pack()




# ============================================================
# 🧭 Ventana principal (selector de modo)
# ============================================================
root = tk.Tk()
root.title("Selector de modo")
root.geometry("400x250")

tk.Label(root, text="Elegí el tipo de sistema dinámico", font=("Consolas", 13, "bold")).pack(pady=20)

tk.Button(root, text="🔹 Modo Numérico clásico", font=("Consolas", 12), width=30,
          command=abrir_modo_numerico).pack(pady=10)

tk.Button(root, text="🔸 Modo Simbólico / No lineal", font=("Consolas", 12), width=30,
          command=abrir_modo_simbolico).pack(pady=10)

tk.Button(root, text="🔸 Modo No homogeneo", font=("Consolas", 12), width=30,
          command=abrir_modo_no_homogeneo).pack(pady=10)

tk.Label(root, text="(Cada modo se abre en una ventana separada)", font=("Consolas", 10, "italic")).pack(pady=15)

root.mainloop()
