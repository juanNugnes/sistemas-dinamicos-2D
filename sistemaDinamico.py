import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox
from scipy.integrate import solve_ivp

def _best_integer_scale(real_vec: np.ndarray, imag_vec: np.ndarray, max_scale: int = 20):
    """Busca un factor entero pequeño que deje real_vec e imag_vec cercanos a enteros.
    Devuelve (scale, real_int_vec, imag_int_vec)."""
    best_scale = 1
    best_score = float('inf')
    best_real = np.round(real_vec).astype(int)
    best_imag = np.round(imag_vec).astype(int)

    for k in range(1, max_scale + 1):
        r = real_vec * k
        im = imag_vec * k
        r_int = np.round(r)
        im_int = np.round(im)
        score = np.sum(np.abs(r - r_int)) + np.sum(np.abs(im - im_int))
        if score < best_score:
            best_score = score
            best_scale = k
            best_real = r_int.astype(int)
            best_imag = im_int.astype(int)
            if score < 1e-9:
                break
    # Reducir por mcd común si aplica
    try:
        from math import gcd
        def _gcd_vec(v):
            vals = [abs(int(x)) for x in v if abs(int(x)) > 0]
            if not vals:
                return 1
            g = vals[0]
            for t in vals[1:]:
                g = gcd(g, t)
            return g if g > 0 else 1
        g_common = _gcd_vec(best_real) if np.any(best_real) else 1
        g_common = gcd(g_common, _gcd_vec(best_imag)) if np.any(best_imag) else g_common
        if g_common > 1:
            best_real = (best_real // g_common).astype(int)
            best_imag = (best_imag // g_common).astype(int)
    except Exception:
        pass
    return int(best_scale), best_real, best_imag


def calcular_autovectores(A, valores):
    """
    Calcula los autovectores resolviendo (A - λI)v = 0 para cada autovalor.
    Retorna autovectores complejos cuando hay autovalores complejos.
    
    Para una matriz 2x2 [[a, b], [c, d]] y autovalor λ, resolvemos:
    (a-λ)v₁ + b*v₂ = 0
    c*v₁ + (d-λ)*v₂ = 0
    """
    vectores = []
    vectores_complejos = []
    # Caso especial: eigenvalor doble y A ≈ λI → infinitos autovectores (base canónica)
    try:
        if len(valores) >= 2 and np.allclose(valores[0], valores[1]) and \
           np.allclose(A - float(valores[0]) * np.eye(2), 0, atol=1e-10):
            return np.array([[1, 0], [0, 1]]).T, [np.array([1+0j, 0+0j]), np.array([0+0j, 1+0j])]
    except Exception:
        pass
    
    for lambda_val in valores:
        # Matriz (A - λI)
        A_minus_lambda = A - lambda_val * np.eye(2, dtype=complex)
        
        a_minus_lambda = A_minus_lambda[0, 0]
        b = A_minus_lambda[0, 1]
        c = A_minus_lambda[1, 0]
        d_minus_lambda = A_minus_lambda[1, 1]
        
        # Resolvemos el sistema homogéneo (A - λI)v = 0
        v1, v2 = 0.0+0j, 0.0+0j
        
        if abs(b) > 1e-10:  # b != 0, usar primera fila: (a-λ)v₁ + b*v₂ = 0
            v1 = 1.0+0j
            v2 = -a_minus_lambda / b
        elif abs(a_minus_lambda) > 1e-10:  # a-λ != 0
            v2 = 1.0+0j
            v1 = -b / a_minus_lambda
        elif abs(c) > 1e-10:  # usar segunda fila: c*v₁ + (d-λ)*v₂ = 0
            v1 = 1.0+0j
            v2 = -c / d_minus_lambda if abs(d_minus_lambda) > 1e-10 else 0.0+0j
        elif abs(d_minus_lambda) > 1e-10:  # d-λ != 0
            v2 = 1.0+0j
            v1 = -d_minus_lambda / c if abs(c) > 1e-10 else 0.0+0j
        else:
            # Caso degenerado: ambas filas son cero, cualquier vector es autovector
            v1 = 1.0+0j
            v2 = 0.0+0j
        
        vec = np.array([v1, v2], dtype=complex)
        
        # Guardar el vector complejo
        vectores_complejos.append(vec)
        
        # Si el autovalor es complejo, guardar parte real e imaginaria por separado
        # Si es real, simplificar a enteros cuando sea posible
        if np.iscomplexobj(lambda_val) or abs(lambda_val.imag) > 1e-10:
            # Autovalor complejo: guardar como está
            vectores.append(vec)
        else:
            # Autovalor real: simplificar
            vec_real = np.real(vec)
            v1_real, v2_real = float(vec_real[0]), float(vec_real[1])
            
            if abs(v1_real) > 1e-10 and abs(v2_real) > 1e-10:
                # Verificar si son enteros
                if np.allclose(vec_real, np.round(vec_real), atol=1e-6):
                    vec_simplified = np.round(vec_real).astype(int)
                else:
                    # Buscar factor común
                    for scale in [1, 2, 3, 4, 5, 10]:
                        vec_scaled = vec_real * scale
                        if np.allclose(vec_scaled, np.round(vec_scaled), atol=1e-6):
                            vec_simplified = np.round(vec_scaled).astype(int)
                            break
                    else:
                        vec_simplified = np.round(vec_real * 10) / 10
                vectores.append(vec_simplified)
            elif abs(v1_real) < 1e-10:
                vectores.append(np.array([0, 1], dtype=int))
            elif abs(v2_real) < 1e-10:
                vectores.append(np.array([1, 0], dtype=int))
            else:
                vectores.append(vec_real)
    
    return np.array(vectores).T, vectores_complejos


def formatear_autovector_complejo(vec_complex: np.ndarray) -> str:
    """Devuelve string del autovector complejo con enteros bonitos: (a,b) + (c,d)i.
    Preferencia: a1==0 y b2==0 si es posible (forma (0,a2) + (b1,0)i)."""
    # Probar rotaciones por {1, i, -1, -i} y elegir la mejor según prioridad
    best = None
    for c in [1+0j, 1j, -1+0j, -1j]:
        v = c * vec_complex
        real_vec = np.real(v)
        imag_vec = np.imag(v)
        _, r_int, i_int = _best_integer_scale(real_vec, imag_vec, max_scale=50)
        score = int(np.sum(np.abs(r_int)) + np.sum(np.abs(i_int)))
        zeros = int(np.sum(r_int == 0) + np.sum(i_int == 0))
        prefer_shape = 0 if (int(r_int[0]) == 0 and int(i_int[1]) == 0) else 1
        key = (prefer_shape, score, -zeros)
        if best is None or key < best[0]:
            best = (key, r_int, i_int)
    _, r_int, i_int = best
    # Si no hay parte imaginaria, mostrar solo real
    if np.all(i_int == 0):
        return f"({int(r_int[0])}, {int(r_int[1])})"
    return f"({int(r_int[0])}, {int(r_int[1])}) + ({int(i_int[0])}, {int(i_int[1])})i"

def canonicalizar_a_b(vec_complex: np.ndarray):
    """Devuelve (a1,a2,b1,b2) enteros "bonitos" tras rotar por {1,i,-1,-i}.
    Preferencia: a1==0 y b2==0 si es posible."""
    best = None
    for c in [1+0j, 1j, -1+0j, -1j]:
        v = c * vec_complex
        real_vec = np.real(v)
        imag_vec = np.imag(v)
        _, r_int, i_int = _best_integer_scale(real_vec, imag_vec, max_scale=50)
        score = int(np.sum(np.abs(r_int)) + np.sum(np.abs(i_int)))
        zeros = int(np.sum(r_int == 0) + np.sum(i_int == 0))
        prefer_shape = 0 if (int(r_int[0]) == 0 and int(i_int[1]) == 0) else 1
        key = (prefer_shape, score, -zeros)
        if best is None or key < best[0]:
            best = (key, r_int, i_int)
    r_int = best[1]; i_int = best[2]
    return int(r_int[0]), int(r_int[1]), int(i_int[0]), int(i_int[1])

def best_integer_scale_real(real_vec: np.ndarray, max_scale: int = 50):
    """Escala un vector real a enteros "bonitos" con un pequeño factor entero."""
    best = None
    for k in range(1, max_scale + 1):
        r = real_vec * k
        r_int = np.round(r)
        score = float(np.sum(np.abs(r - r_int)))
        if best is None or score < best[0]:
            best = (score, r_int.astype(int), k)
            if score < 1e-9:
                break
    # Reducir por mcd
    vec = best[1]
    try:
        from math import gcd
        vals = [abs(int(x)) for x in vec if abs(int(x)) > 0]
        if vals:
            g = vals[0]
            for t in vals[1:]:
                g = gcd(g, t)
            if g > 1:
                vec = (vec // g).astype(int)
    except Exception:
        pass
    return vec

def canonicalize_real_eigenvector(v_real: np.ndarray) -> np.ndarray:
    vec = best_integer_scale_real(v_real)
    # Regla de signo: preferir segunda componente >= 0
    if int(vec[1]) < 0:
        vec = -vec
    # Si ambos 0 (no debería), devolver (1,0)
    if vec[0] == 0 and vec[1] == 0:
        vec = np.array([1, 0])
    return vec.astype(int)

def compute_generalized_w(B: np.ndarray, v_int: np.ndarray) -> np.ndarray:
    """Encuentra w entero pequeño tal que B w = v_int (rank 1)."""
    from fractions import Fraction
    # Buscar una fila no nula
    for r in range(B.shape[0]):
        r1, r2 = B[r, 0], B[r, 1]
        if abs(r1) > 1e-12 or abs(r2) > 1e-12:
            vr = float(v_int[0] if r == 0 else v_int[1])
            if abs(r1) > 1e-12:
                # Fijar w2 = 1
                w2 = 1.0
                w1 = (vr - r2 * w2) / r1
            else:
                # r2 != 0, fijar w1 = 1
                w1 = 1.0
                w2 = (vr - r1 * w1) / r2
            # Convertir a fracciones simples y escalar a enteros
            f1 = Fraction(w1).limit_denominator(10)
            f2 = Fraction(w2).limit_denominator(10)
            num1, den1 = f1.numerator, f1.denominator
            num2, den2 = f2.numerator, f2.denominator
            # mcm de denominadores
            def lcm(a, b):
                from math import gcd
                return abs(a*b)//gcd(a, b)
            L = lcm(den1, den2)
            w_int = np.array([num1 * (L // den1), num2 * (L // den2)], dtype=int)
            # Verificación rápida; si falla, usar lstsq y redondear
            if np.allclose(B @ w_int, v_int, atol=1e-8):
                return w_int
            # Fallback
            try:
                w_ls = np.linalg.lstsq(B, v_int.astype(float), rcond=None)[0]
                w_int = best_integer_scale_real(w_ls)
            except Exception:
                w_int = np.array([0, 0])
            return w_int
    return np.array([0, 0])

# ============================================================
# 🔹 Ventana MODO NUMÉRICO CLÁSICO
# ============================================================
def abrir_modo_numerico():
    wn = tk.Toplevel()
    wn.title("Sistema dinámico 2D - Modo Numérico clásico")
    wn.geometry("1200x900")
    wn.minsize(800, 600)  # Ensure minimum window size

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

    # === Botones ===
    btns = tk.Frame(frame_left, relief=tk.RAISED, bd=2, bg="#e0e0e0")
    btns.pack(pady=15, fill=tk.X)
    tk.Button(btns, text="GRAFICAR", bg="#4CAF50", fg="white",
              font=("Consolas", 14, "bold"), command=lambda: graficar_numerico(), 
              width=15, height=2).pack(side=tk.LEFT, padx=10, expand=True)
    tk.Button(btns, text="VERIFICAR", bg="#2196F3", fg="white",
              font=("Consolas", 14, "bold"), command=lambda: verificar_solucion(),
              width=15, height=2).pack(side=tk.LEFT, padx=10, expand=True)

    # === Opciones de analisis ===
    opts_frame = tk.Frame(frame_left)
    opts_frame.pack(pady=(10, 0), anchor="w")
    var_mostrar_niveles_H = tk.BooleanVar(value=True)
    tk.Checkbutton(
        opts_frame,
        text="Mostrar niveles de H si es Hamiltoniano",
        variable=var_mostrar_niveles_H,
        font=("Consolas", 10)
    ).pack(side=tk.LEFT)

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
            valores = np.linalg.eigvals(A)
            vectores, vectores_complejos = calcular_autovectores(A, valores)
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
        
        # Mostrar autovectores según si son complejos o reales
        text_info.insert(tk.END, "Autovectores:\n")
        if np.iscomplexobj(valores[0]) or abs(valores[0].imag) > 1e-10:
            # Para par complejo, mostrar solo uno en forma (a,b) + (c,d)i (tomamos λ con Im>0)
            # Seleccionamos el vector correspondiente al autovalor de parte imaginaria positiva
            idx = 0
            if np.imag(valores[0]) < 0 and len(vectores_complejos) > 1:
                idx = 1
            text_info.insert(tk.END, f"v = {formatear_autovector_complejo(vectores_complejos[idx])}\n")
        else:
            # Caso real: mostrar vectores simplificados
            text_info.insert(tk.END, f"{vectores}\n")
        text_info.insert(tk.END, "\n")
        text_info.insert(tk.END, f"Invariantes:\n τ = {tau:.4f}\n Δ = {Delta:.4f}\n D = {D:.4f}\n\n")

        # Clasificación
        if Delta < 0:
            tipo = "Silla"
        elif D < 0:
            tipo = "Foco" if tau != 0 else "Centro"
        else:
            tipo = "Nodo"
        text_info.insert(tk.END, f"Tipo de equilibrio: {tipo}\n\n")
        
        # === Solución analítica ===
        try:
            # Verificar si el sistema es lineal (puede calcularse analíticamente)
            # Solo funciona para sistemas lineales homogéneos
            lambda1, lambda2 = valores[0], valores[1]
            is_complex = np.iscomplexobj(lambda1) or abs(lambda1.imag) > 1e-10
            
            # Formatear la solución analítica
            text_info.insert(tk.END, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            text_info.insert(tk.END, "📐 SOLUCIÓN ANALÍTICA:\n\n")
            
            if is_complex:
                # Autovalores complejos conjugados: λ = α ± βi
                # Solución: x(t) = e^(αt)[A₁cos(βt) + A₂sin(βt)]
                #           y(t) = e^(αt)[B₁cos(βt) + B₂sin(βt)]
                alpha = lambda1.real
                beta = abs(lambda1.imag)  # Tomamos el valor absoluto
                
                # usar el autovector asociado al autovalor con Im>0
                v_complex = vectores_complejos[0 if np.imag(valores[0])>0 else 1]
                v_real = np.real(v_complex)
                v_imag = np.imag(v_complex)
                
                v1_real_x, v1_real_y = float(v_real[0]), float(v_real[1])
                v1_imag_x, v1_imag_y = float(v_imag[0]), float(v_imag[1])
                
                # Para autovalores complejos conjugados, la solución es:
                # x(t) = e^(αt)[v_real_x * cos(βt) - v_imag_x * sin(βt)] * C1 + 
                #        e^(αt)[v_real_x * sin(βt) + v_imag_x * cos(βt)] * C2
                # Similar para y(t)
                
                # Resolver condiciones iniciales:
                # En t=0: x(0) = v_real_x * C1 + v_imag_x * C2 = x0
                #         y(0) = v_real_y * C1 + v_imag_y * C2 = y0
                
                det = v1_real_x * v1_imag_y - v1_imag_x * v1_real_y
                if abs(det) > 1e-10:
                    C1 = (x0 * v1_imag_y - y0 * v1_imag_x) / det
                    C2 = (v1_real_x * y0 - v1_real_y * x0) / det
                    
                    # Mostrar solución simbólica con C1, C2, y vectores enteros a⃗, b⃗
                    vec_arrow = "\u20D7"  # flecha sobre la letra
                    # Canonicalizar a⃗, b⃗ con rotación {1,i,-1,-i}
                    a1, a2, b1, b2 = canonicalizar_a_b(v_complex)
                    
                    text_info.insert(tk.END, f"Vectores base (enteros): v = a{vec_arrow} + b{vec_arrow}·i, con a{vec_arrow},b{vec_arrow}∈R²\n")
                    text_info.insert(tk.END, f"a{vec_arrow} = ({a1}, {a2}),  b{vec_arrow} = ({b1}, {b2})\n")
                    text_info.insert(tk.END, f"α = {alpha:.4f}, β = {beta:.4f}\n\n")
                    text_info.insert(tk.END,
                        "X(t) = e^{αt}[ C₁ (a⃗·cos(βt) − b⃗·sin(βt)) + C₂ (a⃗·sin(βt) + b⃗·cos(βt)) ]\n"
                        .replace('α', f"{alpha:.4f}").replace('β', f"{beta:.4f}"))
                    text_info.insert(tk.END, "Componentes:\n")
                    # Forma simplificada agrupando cos y sen
                    cos_t = f"cos({beta:.4f}t)"
                    sen_t = f"sen({beta:.4f}t)"
                    # Coeficientes agrupados
                    cx = f"({a1}·C₁ + {b1}·C₂)"
                    sx = f"({a1}·C₂ − {b1}·C₁)"
                    cy = f"({a2}·C₁ + {b2}·C₂)"
                    sy = f"({a2}·C₂ − {b2}·C₁)"
                    if abs(alpha) < 1e-10:
                        text_info.insert(tk.END, f"x(t) = {cx}·{cos_t} + {sx}·{sen_t}\n")
                        text_info.insert(tk.END, f"y(t) = {cy}·{cos_t} + {sy}·{sen_t}\n\n")
                    else:
                        text_info.insert(tk.END, f"x(t) = e^({alpha:.4f}t)[ {cx}·{cos_t} + {sx}·{sen_t} ]\n")
                        text_info.insert(tk.END, f"y(t) = e^({alpha:.4f}t)[ {cy}·{cos_t} + {sy}·{sen_t} ]\n\n")
            else:
                # Autovalores reales
                v1, v2 = vectores[:, 0], vectores[:, 1]
                v1_x, v1_y = float(v1[0]), float(v1[1])
                v2_x, v2_y = float(v2[0]), float(v2[1])

                if abs(lambda1 - lambda2) < 1e-9:
                    # λ repetido. Revisar si hay sólo 1 autovector (defectivo)
                    if abs(v1_x * v2_y - v2_x * v1_y) < 1e-9:
                        # Sistema no diagonalizable - obtener vector generalizado w: (A-λI) w = v
                        B = A - float(lambda1) * np.eye(2)
                        v_int = canonicalize_real_eigenvector(np.array([v1_x, v1_y]))
                        w_int = compute_generalized_w(B, v_int)
                        vx_i, vy_i = int(v_int[0]), int(v_int[1])
                        wx_i, wy_i = int(w_int[0]), int(w_int[1])

                        text_info.insert(tk.END, f"Caso no diagonalizable (Jordan): λ = {lambda1:.4f}\n")
                        text_info.insert(tk.END, f"v = ({vx_i}, {vy_i}),  w = ({wx_i}, {wy_i})  ( (A-λI)w = v )\n")
                        text_info.insert(tk.END, "X(t) = e^{λt}[ C₁ v + C₂ (t v + w) ]\n".replace('λ', f"{lambda1:.4f}"))
                        text_info.insert(tk.END, "Componentes:\n")
                        text_info.insert(tk.END, f"x(t) = e^({lambda1:.4f}·t)[ C₁·{vx_i} + C₂·({vx_i}·t + {wx_i}) ]\n")
                        text_info.insert(tk.END, f"y(t) = e^({lambda1:.4f}·t)[ C₁·{vy_i} + C₂·({vy_i}·t + {wy_i}) ]\n")
                    else:
                        # Diagonalizable aunque repetido
                        text_info.insert(tk.END, f"X(t) = C₁ e^({lambda1:.4f}·t) v₁ + C₂ e^({lambda2:.4f}·t) v₂\n")
                        text_info.insert(tk.END, f"v₁ = ({v1_x:.4f}, {v1_y:.4f}), v₂ = ({v2_x:.4f}, {v2_y:.4f})\n")
                        text_info.insert(tk.END, "Componentes:\n")
                        text_info.insert(tk.END, f"x(t) = C₁·{v1_x:.4f}·e^({lambda1:.4f}·t) + C₂·{v2_x:.4f}·e^({lambda2:.4f}·t)\n")
                        text_info.insert(tk.END, f"y(t) = C₁·{v1_y:.4f}·e^({lambda1:.4f}·t) + C₂·{v2_y:.4f}·e^({lambda2:.4f}·t)\n")
                else:
                    # Reales distintos (estándar)
                    text_info.insert(tk.END, f"X(t) = C₁ e^({lambda1:.4f}·t) v₁ + C₂ e^({lambda2:.4f}·t) v₂\n")
                    text_info.insert(tk.END, f"v₁ = ({v1_x:.4f}, {v1_y:.4f}), v₂ = ({v2_x:.4f}, {v2_y:.4f})\n")
                    text_info.insert(tk.END, "Componentes:\n")
                    text_info.insert(tk.END, f"x(t) = C₁·{v1_x:.4f}·e^({lambda1:.4f}·t) + C₂·{v2_x:.4f}·e^({lambda2:.4f}·t)\n")
                    text_info.insert(tk.END, f"y(t) = C₁·{v1_y:.4f}·e^({lambda1:.4f}·t) + C₂·{v2_y:.4f}·e^({lambda2:.4f}·t)\n")
        except Exception as e:
            text_info.insert(tk.END, f"\n⚠️ No se pudo calcular solución analítica: {e}\n")

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

        # === Diagrama de fase (múltiples trayectorias) ===
        # Dibujar múltiples trayectorias para visualizar el comportamiento completo
        t_span = 20
        t_eval = np.linspace(0, t_span, 1000)
        
        # Trayectoria principal (condiciones iniciales del usuario)
        sol = solve_ivp(system, [0, t_span], [x0, y0], t_eval=t_eval)
        if sol.success and sol.y.size > 0:
            x_traj, y_traj = sol.y
            ax.plot(x_traj, y_traj, color='green', lw=2.5, label="Trayectoria", zorder=5)
            ax.scatter(x_traj[0], y_traj[0], color='black', s=80, label="Inicial", zorder=6)
            
            # Agregar flechas a la trayectoria principal
            # Seleccionar puntos equidistantes a lo largo de la trayectoria
            num_arrows = min(15, len(x_traj) // 10)
            arrow_indices = np.linspace(0, len(x_traj)-2, num_arrows, dtype=int)
            for idx in arrow_indices:
                if idx < len(x_traj) - 1:
                    dx_arrow = x_traj[idx+1] - x_traj[idx]
                    dy_arrow = y_traj[idx+1] - y_traj[idx]
                    # Normalizar para que las flechas tengan tamaño consistente
                    norm_arrow = np.sqrt(dx_arrow**2 + dy_arrow**2)
                    if norm_arrow > 1e-10:
                        dx_arrow_norm = dx_arrow / norm_arrow * 0.3
                        dy_arrow_norm = dy_arrow / norm_arrow * 0.3
                        ax.arrow(x_traj[idx], y_traj[idx], dx_arrow_norm, dy_arrow_norm,
                                head_width=0.15, head_length=0.1, fc='green', ec='green', 
                                alpha=0.8, zorder=6)
        
        # Trayectorias adicionales para el diagrama de fase
        # Generar puntos iniciales distribuidos alrededor del origen
        phase_points = []
        
        # Puntos en círculos concéntricos alrededor del origen (más densidad)
        for r in [0.5, 1.0, 1.5, 2.5]:
            for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                x_init = r * np.cos(angle)
                y_init = r * np.sin(angle)
                phase_points.append((x_init, y_init))
        
        # Puntos adicionales cerca de los ejes
        for coord in [-3, -2, -1, 1, 2, 3]:
            phase_points.append((coord, 0))
            phase_points.append((0, coord))
        
        # Agregar puntos cerca de los autovectores (usar parte real para visualización)
        if len(vectores) >= 2:
            v1, v2 = vectores[:, 0], vectores[:, 1]
            # Tomar parte real para visualización
            v1_x = float(np.real(v1[0]))
            v1_y = float(np.real(v1[1]))
            v2_x = float(np.real(v2[0]))
            v2_y = float(np.real(v2[1]))
            
            # Normalizar autovectores para visualización
            norm1 = np.sqrt(v1_x**2 + v1_y**2)
            norm2 = np.sqrt(v2_x**2 + v2_y**2)
            if norm1 > 1e-10:
                v1_x_norm, v1_y_norm = v1_x/norm1, v1_y/norm1
                for scale in [-2, -1, 1, 2]:
                    phase_points.append((scale * v1_x_norm, scale * v1_y_norm))
            if norm2 > 1e-10:
                v2_x_norm, v2_y_norm = v2_x/norm2, v2_y/norm2
                for scale in [-2, -1, 1, 2]:
                    phase_points.append((scale * v2_x_norm, scale * v2_y_norm))
        
        # Dibujar trayectorias del diagrama de fase
        colors_phase = plt.cm.viridis(np.linspace(0.3, 0.8, len(phase_points)))
        for i, (x_init, y_init) in enumerate(phase_points):
            # Saltar si está muy cerca de la trayectoria principal
            if np.sqrt((x_init - x0)**2 + (y_init - y0)**2) < 0.1:
                continue
            
            try:
                sol_phase = solve_ivp(system, [0, t_span], [x_init, y_init], 
                                     t_eval=t_eval, dense_output=True)
                if sol_phase.success and sol_phase.y.size > 0:
                    x_ph, y_ph = sol_phase.y
                    # Mantener el tramo dentro de límites
                    mask = (np.abs(x_ph) < 15) & (np.abs(y_ph) < 15)
                    if np.any(mask):
                        end_idx = np.argmax(~mask)
                        if end_idx == 0:
                            end_idx = len(x_ph)
                        ax.plot(x_ph[:end_idx], y_ph[:end_idx], color=colors_phase[i], lw=1, alpha=0.6, zorder=1)

                        # Agregar flechas sobre ese tramo
                        num_arrows_ph = min(8, max(1, end_idx // 15))
                        arrow_indices_ph = np.linspace(0, max(0, end_idx-2), num_arrows_ph, dtype=int)
                        for idx_ph in arrow_indices_ph:
                            if idx_ph < end_idx - 1:
                                dx_arrow_ph = x_ph[idx_ph+1] - x_ph[idx_ph]
                                dy_arrow_ph = y_ph[idx_ph+1] - y_ph[idx_ph]
                                norm_arrow_ph = np.sqrt(dx_arrow_ph**2 + dy_arrow_ph**2)
                                if norm_arrow_ph > 1e-10:
                                    dx_arrow_norm_ph = dx_arrow_ph / norm_arrow_ph * 0.2
                                    dy_arrow_norm_ph = dy_arrow_ph / norm_arrow_ph * 0.2
                                    ax.arrow(x_ph[idx_ph], y_ph[idx_ph], dx_arrow_norm_ph, dy_arrow_norm_ph,
                                            head_width=0.1, head_length=0.08, fc=colors_phase[i], 
                                            ec=colors_phase[i], alpha=0.5, zorder=2)

                # Integración hacia atrás en el tiempo para completar espirales
                sol_back = solve_ivp(system, [0, -t_span], [x_init, y_init],
                                     t_eval=np.linspace(0, -t_span, 1000))
                if sol_back.success and sol_back.y.size > 0:
                    xb, yb = sol_back.y
                    maskb = (np.abs(xb) < 15) & (np.abs(yb) < 15)
                    if np.any(maskb):
                        endb = np.argmax(~maskb)
                        if endb == 0:
                            endb = len(xb)
                        ax.plot(xb[:endb], yb[:endb], color=colors_phase[i], lw=1, alpha=0.6, zorder=1)
            except:
                pass
        
        # Dibujar dirección de los autovectores (líneas que muestran las direcciones principales)
        if len(vectores) >= 2:
            v1, v2 = vectores[:, 0], vectores[:, 1]
            # Tomar parte real para visualización
            v1_x = float(np.real(v1[0]))
            v1_y = float(np.real(v1[1]))
            v2_x = float(np.real(v2[0]))
            v2_y = float(np.real(v2[1]))
            
            # Normalizar para visualización
            scale_vec = 5.0
            if np.sqrt(v1_x**2 + v1_y**2) > 1e-10:
                ax.plot([0, scale_vec*v1_x], [0, scale_vec*v1_y], 
                       'orange', linewidth=2, linestyle=':', alpha=0.7, 
                       label='Autovector 1', zorder=3)
            if np.sqrt(v2_x**2 + v2_y**2) > 1e-10:
                ax.plot([0, scale_vec*v2_x], [0, scale_vec*v2_y], 
                       'purple', linewidth=2, linestyle=':', alpha=0.7, 
                       label='Autovector 2', zorder=3)

        # Configuración final
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Diagrama de fase: Campo de direcciones, nulclinas y trayectorias")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        canvas.draw()

    # === Verificación simbólica ===
    def verificar_solucion():
        text_info.insert(tk.END, "\n🧪 Derivadas simbólicas de x(t) y y(t):\n")
        try:
            fx_text = entry_fx.get().strip()
            fy_text = entry_fy.get().strip()
            # Jacobiano en (0,0)
            h = 1e-5
            dfdx = (eval(fx_text, {"x": h, "y": 0, "np": np}) - eval(fx_text, {"x": -h, "y": 0, "np": np})) / (2*h)
            dfdy = (eval(fx_text, {"x": 0, "y": h, "np": np}) - eval(fx_text, {"x": 0, "y": -h, "np": np})) / (2*h)
            dgdx = (eval(fy_text, {"x": h, "y": 0, "np": np}) - eval(fy_text, {"x": -h, "y": 0, "np": np})) / (2*h)
            dgdy = (eval(fy_text, {"x": 0, "y": h, "np": np}) - eval(fy_text, {"x": 0, "y": -h, "np": np})) / (2*h)
            A = np.array([[dfdx, dfdy], [dgdx, dgdy]], dtype=float)
            valores = np.linalg.eigvals(A)
            vect_pres, vect_comp = calcular_autovectores(A, valores)

            t = sp.symbols('t')
            C1, C2 = sp.symbols('C1 C2')

            if np.iscomplexobj(valores[0]) or abs(np.imag(valores[0])) > 1e-10:
                alpha = float(np.real(valores[0]))
                beta = abs(float(np.imag(valores[0])))
                v_complex = vect_comp[0]
                v_real = np.real(v_complex)
                v_imag = np.imag(v_complex)
                _, r_int, i_int = _best_integer_scale(v_real, v_imag, max_scale=50)
                a1, a2 = int(r_int[0]), int(r_int[1])
                b1, b2 = int(i_int[0]), int(i_int[1])
                cos = sp.cos(beta*t)
                sin = sp.sin(beta*t)
                # Si alpha es prácticamente 0, no incluimos el factor exponencial
                if abs(alpha) < 1e-12:
                    x_expr = (a1*C1 + b1*C2)*cos + (a1*C2 - b1*C1)*sin
                    y_expr = (a2*C1 + b2*C2)*cos + (a2*C2 - b2*C1)*sin
                else:
                    x_expr = sp.exp(alpha*t) * ( (a1*C1 + b1*C2)*cos + (a1*C2 - b1*C1)*sin )
                    y_expr = sp.exp(alpha*t) * ( (a2*C1 + b2*C2)*cos + (a2*C2 - b2*C1)*sin )
            else:
                v1 = vect_pres[:, 0]
                v2 = vect_pres[:, 1]
                lam1 = float(valores[0]); lam2 = float(valores[1])
                if abs(lam1 - lam2) < 1e-9 and abs(float(v1[0])*float(v2[1]) - float(v2[0])*float(v1[1])) < 1e-9:
                    # Generalizado: (A-λI) w = v1
                    B = A - lam1*np.eye(2)
                    v_int = canonicalize_real_eigenvector(np.array([float(v1[0]), float(v1[1])]))
                    w_int = compute_generalized_w(B, v_int)
                    x_expr = sp.exp(lam1*t) * ( C1*int(v_int[0]) + C2*(int(v_int[0])*t + int(w_int[0])) )
                    y_expr = sp.exp(lam1*t) * ( C1*int(v_int[1]) + C2*(int(v_int[1])*t + int(w_int[1])) )
                else:
                    x_expr = C1*float(v1[0])*sp.exp(lam1*t) + C2*float(v2[0])*sp.exp(lam2*t)
                    y_expr = C1*float(v1[1])*sp.exp(lam1*t) + C2*float(v2[1])*sp.exp(lam2*t)

            dxdt = sp.simplify(sp.diff(x_expr, t))
            dydt = sp.simplify(sp.diff(y_expr, t))

            def _fmt(expr: sp.Expr) -> str:
                s = sp.sstr(expr)
                # sstr usa sin/cos, convertimos a "sen" para consistencia visual
                s = s.replace('sin', 'sen')
                return s

            text_info.insert(tk.END, f"dx/dt = {_fmt(dxdt)}\n")
            text_info.insert(tk.END, f"dy/dt = {_fmt(dydt)}\n")
        except Exception as e:
            text_info.insert(tk.END, f"⚠️ Error en verificación: {e}\n")



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
# 🔸 Ventana MODO SISTEMAS NO LINEALES
# ============================================================

def abrir_modo_no_lineal(p_dx=None, p_dy=None, p_x0=None, p_y0=None):
    wnl = tk.Toplevel()
    wnl.title("Análisis de Sistemas Dinámicos No Lineales")
    wnl.geometry("1500x1000")
    wnl.minsize(1200, 800)

    # --- Dividir ventana
    frame_main = tk.Frame(wnl)
    frame_main.pack(fill=tk.BOTH, expand=True)

    frame_left = tk.Frame(frame_main)
    frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    frame_right = tk.Frame(frame_main, padx=10, pady=10, bg="#f4f4f4")
    frame_right.pack(side=tk.RIGHT, fill=tk.Y)

    # === Entradas ===
    frame_inputs = tk.Frame(frame_left)
    frame_inputs.pack(pady=10)

    # Sistema no lineal
    tk.Label(frame_inputs, text="Sistema No Lineal:", font=("Consolas", 12, "bold")).grid(row=0, column=0, columnspan=4, sticky="w")
    
    tk.Label(frame_inputs, text="dx/dt =", font=("Consolas", 10)).grid(row=1, column=0)
    entry_dx = tk.Entry(frame_inputs, width=30, font=("Consolas", 10))
    entry_dx.grid(row=1, column=1, columnspan=2)
    entry_dx.insert(0, "y")
    
    tk.Label(frame_inputs, text="dy/dt =", font=("Consolas", 10)).grid(row=2, column=0)
    entry_dy = tk.Entry(frame_inputs, width=30, font=("Consolas", 10))
    entry_dy.grid(row=2, column=1, columnspan=2)
    entry_dy.insert(0, "x*(3-x)")

    # Condiciones iniciales para simulación
    tk.Label(frame_inputs, text="Condiciones iniciales:", font=("Consolas", 12, "bold")).grid(row=3, column=0, columnspan=4, sticky="w", pady=(10,0))
    
    tk.Label(frame_inputs, text="x₀:", font=("Consolas", 10)).grid(row=4, column=0)
    entry_x0 = tk.Entry(frame_inputs, width=8, font=("Consolas", 10))
    entry_x0.grid(row=4, column=1)
    entry_x0.insert(0, "0.5")

    tk.Label(frame_inputs, text="y₀:", font=("Consolas", 10)).grid(row=4, column=2)
    entry_y0 = tk.Entry(frame_inputs, width=8, font=("Consolas", 10))
    entry_y0.grid(row=4, column=3)
    entry_y0.insert(0, "0.5")

    # Sobrescribir con parámetros si fueron provistos
    try:
        if p_dx is not None:
            entry_dx.delete(0, tk.END); entry_dx.insert(0, str(p_dx))
        if p_dy is not None:
            entry_dy.delete(0, tk.END); entry_dy.insert(0, str(p_dy))
        if p_x0 is not None:
            entry_x0.delete(0, tk.END); entry_x0.insert(0, str(p_x0))
        if p_y0 is not None:
            entry_y0.delete(0, tk.END); entry_y0.insert(0, str(p_y0))
    except Exception:
        pass

    # Ejemplos predefinidos de la imagen
    frame_ejemplos = tk.Frame(frame_inputs)
    frame_ejemplos.grid(row=5, column=0, columnspan=4, pady=10, sticky="ew")
    
    tk.Label(frame_ejemplos, text="Ejemplos de la imagen:", font=("Consolas", 11, "bold")).pack(anchor="w")
    
    def cargar_ejemplo_nl(num):
        ejemplos = {
            1: {"dx": "y", "dy": "x*(3-x)", "x0": "0.5", "y0": "0.5"},  # Ejemplo 1
            2: {"dx": "y", "dy": "x**3 - x", "x0": "0.1", "y0": "0.1"},  # Ejemplo 2
            3: {"dx": "y - x", "dy": "x**2 - 1", "x0": "0", "y0": "0"},  # Ejemplo 3
            4: {"dx": "x*y", "dy": "x**2 + y**2 - 1", "x0": "0.5", "y0": "0.5"},  # Ejemplo 4
            11: {"dx": "-x + x*y", "dy": "-2*x + x*y", "x0": "1", "y0": "1"},  # Ejemplo 11
            12: {"dx": "x**2 + y**2 - 2", "dy": "x**2 - y**2", "x0": "1", "y0": "1"},  # Ejemplo 12
            13: {"dx": "14*x - 0.5*x**2 - x*y", "dy": "16*y - 0.5*y**2 - x*y", "x0": "5", "y0": "5"},  # Ejemplo 13
            14: {"dx": "x*(3-x) - 2*x*y", "dy": "y*(2-y) - x*y", "x0": "1", "y0": "1"}  # Ejemplo 14
        }
        if num in ejemplos:
            ej = ejemplos[num]
            entry_dx.delete(0, tk.END); entry_dx.insert(0, ej["dx"])
            entry_dy.delete(0, tk.END); entry_dy.insert(0, ej["dy"])
            entry_x0.delete(0, tk.END); entry_x0.insert(0, ej["x0"])
            entry_y0.delete(0, tk.END); entry_y0.insert(0, ej["y0"])

    btn_frame = tk.Frame(frame_ejemplos)
    btn_frame.pack(fill="x")
    for i in [1, 2, 3, 4, 11, 12, 13, 14]:
        tk.Button(btn_frame, text=f"Ej{i}", font=("Consolas", 8), width=4,
                 command=lambda x=i: cargar_ejemplo_nl(x)).pack(side=tk.LEFT, padx=1)

    # Botón con nombre para el caso Romeo–Julieta
    tk.Button(
        btn_frame,
        text="Romeo y Julieta",
        font=("Consolas", 8),
        command=lambda: (
            entry_dx.delete(0, tk.END), entry_dx.insert(0, "x*(1 - x**2 - y**2)"),
            entry_dy.delete(0, tk.END), entry_dy.insert(0, "-y*(1 - x**2 - y**2)"),
            entry_x0.delete(0, tk.END), entry_x0.insert(0, "0.5"),
            entry_y0.delete(0, tk.END), entry_y0.insert(0, "0.5")
        )
    ).pack(side=tk.LEFT, padx=6)

    # === Botón único ===
    btn_frame = tk.Frame(frame_left)
    btn_frame.pack(pady=20)
    
    tk.Button(btn_frame, text="🚀 ANALIZAR SISTEMA COMPLETO", bg="#4CAF50", fg="white",
              font=("Consolas", 14, "bold"), command=lambda: analizar_sistema_completo(),
              width=35, height=3, relief=tk.RAISED, bd=3).pack()

    # === Canvas ===
    fig, ax = plt.subplots(figsize=(8, 8))
    canvas = FigureCanvasTkAgg(fig, master=frame_left)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)

    # === Panel derecho ===
    tk.Label(frame_right, text="📊 Análisis de Sistema No Lineal:",
             bg="#f4f4f4", font=("Consolas", 12, "bold")).pack(anchor="w", pady=(0, 5))
    text_info = tk.Text(frame_right, width=55, height=45, font=("Consolas", 9), bg="#f9f9f9")
    text_info.pack(fill=tk.BOTH, expand=True)

    # Variables globales para almacenar resultados
    equilibrios = []
    jacobianos = []
    autovalores_puntos = []

    # === Función única: Análisis completo integrado ===
    def analizar_sistema_completo():
        global equilibrios, jacobianos, autovalores_puntos
        equilibrios = []
        jacobianos = []
        autovalores_puntos = []
        
        try:
            text_info.delete("1.0", tk.END)
            dx_expr = entry_dx.get().strip()
            dy_expr = entry_dy.get().strip()
            x0 = float(entry_x0.get())
            y0 = float(entry_y0.get())
            
            # ═══════════════════════════════════════════════
            # 1️⃣ PASO 1: CÁLCULO DE PUNTOS DE EQUILIBRIO
            # ═══════════════════════════════════════════════
            text_info.insert(tk.END, "═══════════════════════════════════════════════\n")
            text_info.insert(tk.END, "1️⃣ PASO 1: CÁLCULO DE PUNTOS DE EQUILIBRIO\n")
            text_info.insert(tk.END, "═══════════════════════════════════════════════\n\n")
            
            text_info.insert(tk.END, f"Sistema:\n")
            text_info.insert(tk.END, f"dx/dt = {dx_expr}\n")
            text_info.insert(tk.END, f"dy/dt = {dy_expr}\n\n")
            
            text_info.insert(tk.END, "Condición de equilibrio: dx/dt = 0 y dy/dt = 0\n\n")
            
            # Resolver usando sympy
            x, y = sp.symbols('x y')
            try:
                dx_sympy = sp.sympify(dx_expr.replace('**', '^').replace('^', '**'))
                dy_sympy = sp.sympify(dy_expr.replace('**', '^').replace('^', '**'))
                
                text_info.insert(tk.END, "Resolviendo:\n")
                text_info.insert(tk.END, f"{dx_sympy} = 0\n")
                text_info.insert(tk.END, f"{dy_sympy} = 0\n\n")
                
                soluciones = sp.solve([dx_sympy, dy_sympy], [x, y])

                # --- Analisis Hamiltoniano / Conservatividad ---
                text_info.insert(tk.END, "\n-- PASO: Analisis Hamiltoniano / Conservatividad\n")
                try:
                    div_expr = sp.simplify(sp.diff(dx_sympy, x) + sp.diff(dy_sympy, y))
                except Exception:
                    div_expr = None
                H_expr = None
                es_hamiltoniano = False
                if div_expr is not None and sp.simplify(div_expr) == 0:
                    es_hamiltoniano = True
                    text_info.insert(tk.END, "   Condicion f_x + g_y = 0 (divergencia nula).\n")
                    # Intentar construir H: H_y = f = dx_sympy y H_x = -g = -dy_sympy
                    try:
                        # Ruta A: integrar f respecto a y
                        H_cand = sp.integrate(dx_sympy, (y))
                        Cprime = sp.simplify(-dy_sympy - sp.diff(H_cand, x))
                        if not (Cprime.has(y)):
                            Cx = sp.integrate(Cprime, (x))
                            H_expr = sp.simplify(H_cand + Cx)
                        else:
                            # Ruta B: integrar -g respecto a x
                            H_cand2 = -sp.integrate(dy_sympy, (x))
                            Cprime_y = sp.simplify(dx_sympy - sp.diff(H_cand2, y))
                            if not (Cprime_y.has(x)):
                                Cy = sp.integrate(Cprime_y, (y))
                                H_expr = sp.simplify(H_cand2 + Cy)
                    except Exception:
                        H_expr = None
                    if H_expr is not None:
                        text_info.insert(tk.END, f"\n   Sistema Hamiltoniano. Integral primera H(x,y) (hasta cte):\n   H(x,y) = {sp.simplify(H_expr)}\n")
                    else:
                        text_info.insert(tk.END, "\n   Sistema Hamiltoniano (divergencia nula), pero no se encontro H cerrada simbolica.\n")
                else:
                    if div_expr is None:
                        text_info.insert(tk.END, "   No fue posible calcular la divergencia.\n")
                    else:
                        text_info.insert(tk.END, f"\n   No Hamiltoniano (divergencia = {sp.simplify(div_expr)} != 0).\n")
                
                if isinstance(soluciones, list):
                    for i, sol in enumerate(soluciones):
                        if isinstance(sol, tuple) and len(sol) == 2:
                            try:
                                x_val = float(sol[0]) if sol[0].is_real else complex(sol[0])
                                y_val = float(sol[1]) if sol[1].is_real else complex(sol[1])
                                if isinstance(x_val, (int, float)) and isinstance(y_val, (int, float)):
                                    equilibrios.append((x_val, y_val))
                                    text_info.insert(tk.END, f"P{i+1} = ({x_val:.4f}, {y_val:.4f})\n")
                            except:
                                pass
                elif isinstance(soluciones, dict):
                    try:
                        x_val = float(soluciones[x]) if soluciones[x].is_real else None
                        y_val = float(soluciones[y]) if soluciones[y].is_real else None
                        if x_val is not None and y_val is not None:
                            equilibrios.append((x_val, y_val))
                            text_info.insert(tk.END, f"P1 = ({x_val:.4f}, {y_val:.4f})\n")
                    except:
                        pass
            except:
                pass
            
            # Método numérico como respaldo
            if not equilibrios:
                text_info.insert(tk.END, "\n⚠️ Usando método numérico...\n")
                from scipy.optimize import fsolve
                
                def sistema_eq(vars):
                    x_val, y_val = vars
                    dx_val = eval(dx_expr, {"x": x_val, "y": y_val, "np": np})
                    dy_val = eval(dy_expr, {"x": x_val, "y": y_val, "np": np})
                    return [dx_val, dy_val]
                
                puntos_iniciales = [(0, 0), (1, 1), (-1, -1), (2, 2), (-2, -2), (0, 1), (1, 0)]
                
                for punto in puntos_iniciales:
                    try:
                        sol = fsolve(sistema_eq, punto, xtol=1e-10)
                        residuo = sistema_eq(sol)
                        if abs(residuo[0]) < 1e-8 and abs(residuo[1]) < 1e-8:
                            nuevo = True
                            for eq_existente in equilibrios:
                                if abs(sol[0] - eq_existente[0]) < 1e-6 and abs(sol[1] - eq_existente[1]) < 1e-6:
                                    nuevo = False
                                    break
                            if nuevo:
                                equilibrios.append((sol[0], sol[1]))
                    except:
                        pass
                
                for i, eq in enumerate(equilibrios):
                    text_info.insert(tk.END, f"P{i+1} = ({eq[0]:.4f}, {eq[1]:.4f})\n")
            
            text_info.insert(tk.END, f"\n✅ Se encontraron {len(equilibrios)} punto(s) de equilibrio.\n\n")
            text_info.update()
            
            # ═══════════════════════════════════════════════
            # 2️⃣ PASO 2: LINEARIZACIÓN EN PUNTOS CRÍTICOS
            # ═══════════════════════════════════════════════
            if equilibrios:
                text_info.insert(tk.END, "═══════════════════════════════════════════════\n")
                text_info.insert(tk.END, "2️⃣ PASO 2: LINEARIZACIÓN EN PUNTOS CRÍTICOS\n")
                text_info.insert(tk.END, "═══════════════════════════════════════════════\n\n")
                
                # Calcular Jacobiano simbólico
                x, y = sp.symbols('x y')
                dx_sympy = sp.sympify(dx_expr.replace('**', '^').replace('^', '**'))
                dy_sympy = sp.sympify(dy_expr.replace('**', '^').replace('^', '**'))
                
                # Derivadas parciales
                dfdx = sp.diff(dx_sympy, x)
                dfdy = sp.diff(dx_sympy, y)
                dgdx = sp.diff(dy_sympy, x)
                dgdy = sp.diff(dy_sympy, y)
                
                text_info.insert(tk.END, "Matriz Jacobiana J(x,y):\n")
                text_info.insert(tk.END, f"J = [∂f/∂x  ∂f/∂y] = [{dfdx}  {dfdy}]\n")
                text_info.insert(tk.END, f"    [∂g/∂x  ∂g/∂y]   [{dgdx}  {dgdy}]\n\n")
                
                # Evaluar en cada punto de equilibrio
                for i, (x_eq, y_eq) in enumerate(equilibrios):
                    text_info.insert(tk.END, f"En P{i+1} = ({x_eq:.4f}, {y_eq:.4f}):\n")
                    
                    # Evaluar derivadas en el punto
                    j11 = float(dfdx.subs([(x, x_eq), (y, y_eq)]))
                    j12 = float(dfdy.subs([(x, x_eq), (y, y_eq)]))
                    j21 = float(dgdx.subs([(x, x_eq), (y, y_eq)]))
                    j22 = float(dgdy.subs([(x, x_eq), (y, y_eq)]))
                    
                    J = np.array([[j11, j12], [j21, j22]])
                    jacobianos.append(J)
                    
                    text_info.insert(tk.END, f"J({x_eq:.4f}, {y_eq:.4f}) = [{j11:8.4f}  {j12:8.4f}]\n")
                    text_info.insert(tk.END, f"                      [{j21:8.4f}  {j22:8.4f}]\n\n")
                
                text_info.update()
            
            # ═══════════════════════════════════════════════
            # 3️⃣ PASO 3: AUTOVALORES Y AUTOVECTORES
            # ═══════════════════════════════════════════════
            if jacobianos:
                text_info.insert(tk.END, "═══════════════════════════════════════════════\n")
                text_info.insert(tk.END, "3️⃣ PASO 3: AUTOVALORES Y AUTOVECTORES\n")
                text_info.insert(tk.END, "═══════════════════════════════════════════════\n\n")
                
                for i, (J, eq) in enumerate(zip(jacobianos, equilibrios)):
                    text_info.insert(tk.END, f"Punto P{i+1} = ({eq[0]:.4f}, {eq[1]:.4f}):\n")
                    
                    # Calcular autovalores y autovectores
                    valores, vectores = np.linalg.eig(J)
                    autovalores_puntos.append((valores, vectores))
                    
                    text_info.insert(tk.END, f"Autovalores:\n")
                    text_info.insert(tk.END, f"λ₁ = {valores[0]:.4f}\n")
                    text_info.insert(tk.END, f"λ₂ = {valores[1]:.4f}\n\n")
                    
                    # Clasificación del punto
                    lambda1, lambda2 = valores[0], valores[1]
                    det_J = np.linalg.det(J)
                    tr_J = np.trace(J)
                    
                    text_info.insert(tk.END, f"Determinante: det(J) = {det_J:.4f}\n")
                    text_info.insert(tk.END, f"Traza: tr(J) = {tr_J:.4f}\n")
                    
                    # Clasificación
                    if abs(np.imag(lambda1)) > 1e-10:  # Autovalores complejos
                        alpha = np.real(lambda1)
                        if alpha < -1e-10:
                            tipo = "FOCO ESTABLE (espiral convergente)"
                        elif alpha > 1e-10:
                            tipo = "FOCO INESTABLE (espiral divergente)"
                        else:
                            tipo = "CENTRO (órbitas cerradas)"
                    else:  # Autovalores reales
                        if det_J < 0:
                            tipo = "SILLA (punto de silla)"
                        elif det_J > 0:
                            if tr_J < 0:
                                tipo = "NODO ESTABLE"
                            elif tr_J > 0:
                                tipo = "NODO INESTABLE"
                            else:
                                tipo = "CASO MARGINAL"
                        else:
                            tipo = "CASO DEGENERADO"
                    
                    # Determinar si es hiperbólico
                    es_hiperbolico = all(abs(np.real(val)) > 1e-10 for val in valores)
                    
                    text_info.insert(tk.END, f"Tipo: {tipo}\n")
                    text_info.insert(tk.END, f"Hiperbólico: {'SÍ' if es_hiperbolico else 'NO'}\n")
                    
                    if not es_hiperbolico:
                        text_info.insert(tk.END, "⚠️ PUNTO NO HIPERBÓLICO - Se requiere análisis adicional\n")
                    
                    text_info.insert(tk.END, "\n" + "─"*50 + "\n")
                
                text_info.update()
            
            # ═══════════════════════════════════════════════
            # 4️⃣ PASO 4: ANÁLISIS GLOBAL Y RESUMEN
            # ═══════════════════════════════════════════════
            if autovalores_puntos:
                text_info.insert(tk.END, "═══════════════════════════════════════════════\n")
                text_info.insert(tk.END, "4️⃣ PASO 4: ANÁLISIS GLOBAL Y RESUMEN\n")
                text_info.insert(tk.END, "═══════════════════════════════════════════════\n\n")
                
                # Resumen de estabilidad
                text_info.insert(tk.END, "RESUMEN DE ESTABILIDAD:\n")
                for i, (eq, (valores, _)) in enumerate(zip(equilibrios, autovalores_puntos)):
                    max_real = max(np.real(valores))
                    if max_real < -1e-10:
                        estabilidad = "ESTABLE"
                        emoji = "✅"
                    elif max_real > 1e-10:
                        estabilidad = "INESTABLE"
                        emoji = "❌"
                    else:
                        estabilidad = "MARGINALMENTE ESTABLE"
                        emoji = "⚠️"
                    
                    text_info.insert(tk.END, f"{emoji} P{i+1} = ({eq[0]:.4f}, {eq[1]:.4f}) → {estabilidad}\n")
                
                text_info.insert(tk.END, "\n")
                text_info.update()
            
            # ═══════════════════════════════════════════════
            # 5️⃣ PASO 5: GENERACIÓN DEL DIAGRAMA DE FASE
            # ═══════════════════════════════════════════════
            text_info.insert(tk.END, "═══════════════════════════════════════════════\n")
            text_info.insert(tk.END, "5️⃣ PASO 5: GENERACIÓN DEL DIAGRAMA DE FASE\n")
            text_info.insert(tk.END, "═══════════════════════════════════════════════\n\n")
            
            def sistema_nl(t, XY):
                x, y = XY
                dx = eval(dx_expr, {"x": x, "y": y, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                dy = eval(dy_expr, {"x": x, "y": y, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                return [dx, dy]
            
            # Limpiar gráfico
            ax.clear()
            
            # Determinar rango apropiado basado en puntos de equilibrio
            if equilibrios:
                x_coords = [eq[0] for eq in equilibrios]
                y_coords = [eq[1] for eq in equilibrios]
                x_center = np.mean(x_coords)
                y_center = np.mean(y_coords)
                x_range_eq = max(abs(max(x_coords) - x_center), abs(min(x_coords) - x_center), 2)
                y_range_eq = max(abs(max(y_coords) - y_center), abs(min(y_coords) - y_center), 2)
                x_lim = max(x_range_eq * 2, 3)
                y_lim = max(y_range_eq * 2, 3)
            else:
                x_center, y_center = 0, 0
                x_lim, y_lim = 5, 5
            
            # Campo vectorial
            x_range = np.linspace(x_center - x_lim, x_center + x_lim, 20)
            y_range = np.linspace(y_center - y_lim, y_center + y_lim, 20)
            X, Y = np.meshgrid(x_range, y_range)
            U, V = np.zeros_like(X), np.zeros_like(Y)
            
            for i in range(len(x_range)):
                for j in range(len(y_range)):
                    try:
                        dXY = sistema_nl(0, [X[j,i], Y[j,i]])
                        U[j,i], V[j,i] = dXY[0], dXY[1]
                    except:
                        U[j,i], V[j,i] = 0, 0
            
            # Normalizar para mejor visualización
            M = np.sqrt(U**2 + V**2)
            M[M == 0] = 1
            U_norm, V_norm = U/M, V/M
            
            ax.quiver(X, Y, U_norm, V_norm, alpha=0.5, color='gray', scale=25, width=0.003)

            # Superponer curvas de nivel de H si corresponde
            try:
                if 'H_expr' in locals() and H_expr is not None and var_mostrar_niveles_H.get():
                    H_np = sp.lambdify((x, y), H_expr, modules=['numpy'])
                    HH = H_np(X, Y)
                    if np.iscomplexobj(HH):
                        HH = np.real(HH)
                    finite_vals = HH[np.isfinite(HH)]
                    if finite_vals.size > 0:
                        vmin, vmax = np.percentile(finite_vals, [10, 90])
                        if vmin != vmax:
                            levels = np.linspace(vmin, vmax, 8)
                            cs = ax.contour(X, Y, HH, levels=levels, cmap='coolwarm', alpha=0.6)
                            ax.clabel(cs, inline=1, fontsize=8)
            except Exception:
                pass
            
            # Puntos de equilibrio con colores según estabilidad
            if equilibrios and autovalores_puntos:
                for i, (eq, (valores, _)) in enumerate(zip(equilibrios, autovalores_puntos)):
                    max_real = max(np.real(valores))
                    if max_real < -1e-10:
                        color = 'green'
                        marker = 'o'
                    elif max_real > 1e-10:
                        color = 'red' 
                        marker = 's'
                    else:
                        color = 'orange'
                        marker = '^'
                    
                    ax.scatter(eq[0], eq[1], color=color, s=150, zorder=5, 
                              marker=marker, edgecolors='black', linewidth=2,
                              label=f'P{i+1} ({["Estable", "Inestable", "Marginal"][0 if max_real < -1e-10 else 1 if max_real > 1e-10 else 2]})' if i < 3 else "")
            
            # Trayectorias desde múltiples condiciones iniciales
            t_span = [0, 15]
            t_eval = np.linspace(0, 15, 1000)
            
            # Trayectoria principal (condición inicial del usuario)
            try:
                sol = solve_ivp(sistema_nl, t_span, [x0, y0], t_eval=t_eval, dense_output=True)
                if sol.success:
                    ax.plot(sol.y[0], sol.y[1], 'b-', linewidth=3, label='Trayectoria principal', alpha=0.8)
                    ax.scatter(x0, y0, color='blue', s=120, zorder=6, marker='s', 
                              edgecolors='white', linewidth=2, label='Condición inicial')
            except:
                pass
            
            # Trayectorias adicionales desde puntos estratégicos
            condiciones_extra = []
            
            # Agregar puntos cerca de equilibrios
            for eq in equilibrios[:4]:  # Máximo 4 equilibrios para no saturar
                for delta in [0.1, -0.1]:
                    condiciones_extra.extend([
                        (eq[0] + delta, eq[1]), 
                        (eq[0], eq[1] + delta),
                        (eq[0] + delta, eq[1] + delta)
                    ])
            
            # Agregar algunos puntos adicionales
            condiciones_extra.extend([
                (x_center + x_lim*0.3, y_center + y_lim*0.3),
                (x_center - x_lim*0.3, y_center - y_lim*0.3),
                (x_center + x_lim*0.5, y_center),
                (x_center, y_center + y_lim*0.5),
                (x_center - x_lim*0.5, y_center),
                (x_center, y_center - y_lim*0.5)
            ])
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(condiciones_extra)))
            
            for (xi, yi), color in zip(condiciones_extra[:12], colors[:12]):  # Limitar a 12 trayectorias
                try:
                    sol_extra = solve_ivp(sistema_nl, t_span, [xi, yi], t_eval=t_eval)
                    if sol_extra.success and len(sol_extra.y[0]) > 10:
                        # Filtrar puntos que se van al infinito
                        mask = (np.abs(sol_extra.y[0]) < x_center + 2*x_lim) & (np.abs(sol_extra.y[1]) < y_center + 2*y_lim)
                        if np.any(mask):
                            end_idx = np.argmax(~mask) if not np.all(mask) else len(sol_extra.y[0])
                            if end_idx == 0:
                                end_idx = len(sol_extra.y[0])
                            ax.plot(sol_extra.y[0][:end_idx], sol_extra.y[1][:end_idx], 
                                   color=color, alpha=0.6, linewidth=1.5)
                except:
                    pass
            
            # Configurar gráfico
            ax.set_xlim(x_center - x_lim, x_center + x_lim)
            ax.set_ylim(y_center - y_lim, y_center + y_lim)
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.set_title(f'Diagrama de Fase: dx/dt = {dx_expr}, dy/dt = {dy_expr}', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
            
            # Validacion visual extra para no homogeneas
            try:
                es_constante = True
                f1_0 = eval(f1_text, {"t": 0, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                f1_1 = eval(f1_text, {"t": 1, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                f2_0 = eval(f2_text, {"t": 0, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                f2_1 = eval(f2_text, {"t": 1, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                if abs(f1_0 - f1_1) > 1e-10 or abs(f2_0 - f2_1) > 1e-10:
                    es_constante = False
                if es_constante:
                    f_const = np.array([float(f1_0), float(f2_0)])
                    if abs(np.linalg.det(A)) > 1e-12:
                        Xp = np.linalg.solve(-A, f_const)
                        ax.scatter(Xp[0], Xp[1], s=120, c='k', marker='X', label='Equilibrio Xp', zorder=6)
                    # Nulclinas afines en la ventana actual
                    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
                    X2, Y2 = np.meshgrid(np.linspace(xmin, xmax, 30), np.linspace(ymin, ymax, 30))
                    U2 = A[0,0]*X2 + A[0,1]*Y2 + f_const[0]
                    V2 = A[1,0]*X2 + A[1,1]*Y2 + f_const[1]
                    try:
                        c1 = ax.contour(X2, Y2, U2, levels=[0], colors='blue', linestyles='--', linewidths=1.2)
                        c2 = ax.contour(X2, Y2, V2, levels=[0], colors='red', linestyles='--', linewidths=1.2)
                        ax.clabel(c1, fmt={'0': 'dx/dt=0'}, fontsize=8)
                        ax.clabel(c2, fmt={'0': 'dy/dt=0'}, fontsize=8)
                    except Exception:
                        pass
                else:
                    ax.text(0.02, 0.98, 'Campo mostrado en t=0 (no autonomo)', transform=ax.transAxes,
                            va='top', ha='left', fontsize=8, color='gray')
            except Exception:
                pass

            canvas.draw()
            
            # Mensaje final
            text_info.insert(tk.END, "✅ ANÁLISIS COMPLETO FINALIZADO\n")
            text_info.insert(tk.END, f"📊 Encontrados {len(equilibrios)} punto(s) de equilibrio\n")
            text_info.insert(tk.END, f"🎨 Diagrama de fase generado con trayectorias múltiples\n\n")
            
        except Exception as e:
            text_info.insert(tk.END, f"❌ Error en análisis completo: {e}\n")
            messagebox.showerror("Error", f"Error en análisis: {e}")


# ============================================================
# 🧭 Ventana modo no homogeneo
# ============================================================

def abrir_modo_no_homogeneo(p_A=None, p_f=None, p_x0=None):
    wh = tk.Toplevel()
    wh.title("Sistema dinámico 2D - Modo No homogéneo X' = AX + f(t)")
    wh.geometry("1400x950")
    wh.minsize(1200, 800)

    # --- Dividir ventana
    frame_main = tk.Frame(wh)
    frame_main.pack(fill=tk.BOTH, expand=True)

    frame_left = tk.Frame(frame_main)
    frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    frame_right = tk.Frame(frame_main, padx=10, pady=10, bg="#f4f4f4")
    frame_right.pack(side=tk.RIGHT, fill=tk.Y)

    # === Entradas ===
    frame_inputs = tk.Frame(frame_left)
    frame_inputs.pack(pady=10)

    # Matriz A (parte homogénea)
    tk.Label(frame_inputs, text="Matriz A:", font=("Consolas", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
    
    tk.Label(frame_inputs, text="a₁₁:", font=("Consolas", 10)).grid(row=1, column=0)
    entry_a11 = tk.Entry(frame_inputs, width=8, font=("Consolas", 10))
    entry_a11.grid(row=1, column=1)
    entry_a11.insert(0, "0")
    
    tk.Label(frame_inputs, text="a₁₂:", font=("Consolas", 10)).grid(row=1, column=2)
    entry_a12 = tk.Entry(frame_inputs, width=8, font=("Consolas", 10))
    entry_a12.grid(row=1, column=3)
    entry_a12.insert(0, "-1")
    
    tk.Label(frame_inputs, text="a₂₁:", font=("Consolas", 10)).grid(row=2, column=0)
    entry_a21 = tk.Entry(frame_inputs, width=8, font=("Consolas", 10))
    entry_a21.grid(row=2, column=1)
    entry_a21.insert(0, "1")
    
    tk.Label(frame_inputs, text="a₂₂:", font=("Consolas", 10)).grid(row=2, column=2)
    entry_a22 = tk.Entry(frame_inputs, width=8, font=("Consolas", 10))
    entry_a22.grid(row=2, column=3)
    entry_a22.insert(0, "0")

    # Función forzante f(t)
    tk.Label(frame_inputs, text="Función forzante f(t):", font=("Consolas", 12, "bold")).grid(row=3, column=0, columnspan=4, sticky="w", pady=(10,0))
    
    tk.Label(frame_inputs, text="f₁(t):", font=("Consolas", 10)).grid(row=4, column=0)
    entry_f1 = tk.Entry(frame_inputs, width=20, font=("Consolas", 10))
    entry_f1.grid(row=4, column=1, columnspan=2)
    entry_f1.insert(0, "1")
    
    tk.Label(frame_inputs, text="f₂(t):", font=("Consolas", 10)).grid(row=5, column=0)
    entry_f2 = tk.Entry(frame_inputs, width=20, font=("Consolas", 10))
    entry_f2.grid(row=5, column=1, columnspan=2)
    entry_f2.insert(0, "0")

    # Condiciones iniciales
    tk.Label(frame_inputs, text="Condiciones iniciales:", font=("Consolas", 12, "bold")).grid(row=6, column=0, columnspan=4, sticky="w", pady=(10,0))
    
    tk.Label(frame_inputs, text="x₀:", font=("Consolas", 10)).grid(row=7, column=0)
    entry_x0 = tk.Entry(frame_inputs, width=8, font=("Consolas", 10))
    entry_x0.grid(row=7, column=1)
    entry_x0.insert(0, "1")

    tk.Label(frame_inputs, text="y₀:", font=("Consolas", 10)).grid(row=7, column=2)
    entry_y0 = tk.Entry(frame_inputs, width=8, font=("Consolas", 10))
    entry_y0.grid(row=7, column=3)
    entry_y0.insert(0, "0")

    # Sobrescribir con parámetros iniciales si se proveen
    try:
        if isinstance(p_A, (list, tuple)) and len(p_A) == 2 and len(p_A[0]) == 2 and len(p_A[1]) == 2:
            a11, a12 = p_A[0][0], p_A[0][1]
            a21, a22 = p_A[1][0], p_A[1][1]
            entry_a11.delete(0, tk.END); entry_a11.insert(0, str(a11))
            entry_a12.delete(0, tk.END); entry_a12.insert(0, str(a12))
            entry_a21.delete(0, tk.END); entry_a21.insert(0, str(a21))
            entry_a22.delete(0, tk.END); entry_a22.insert(0, str(a22))
        if isinstance(p_f, (list, tuple)) and len(p_f) == 2:
            entry_f1.delete(0, tk.END); entry_f1.insert(0, str(p_f[0]))
            entry_f2.delete(0, tk.END); entry_f2.insert(0, str(p_f[1]))
        if isinstance(p_x0, (list, tuple)) and len(p_x0) == 2:
            entry_x0.delete(0, tk.END); entry_x0.insert(0, str(p_x0[0]))
            entry_y0.delete(0, tk.END); entry_y0.insert(0, str(p_x0[1]))
    except Exception:
        pass

    # Ejemplos predefinidos
    frame_ejemplos = tk.Frame(frame_inputs)
    frame_ejemplos.grid(row=8, column=0, columnspan=4, pady=10, sticky="ew")
    
    tk.Label(frame_ejemplos, text="Ejemplos:", font=("Consolas", 11, "bold")).pack(anchor="w")
    
    def cargar_ejemplo(num):
        ejemplos = {
            1: {"A": [[0, -1], [1, 0]], "f": ["1", "0"], "x0": [1, 0]},  # Ejemplo 1 de la imagen
            2: {"A": [[1, 2], [2, 1]], "f": ["1", "-3"], "x0": [0, 0]},   # Ejemplo 2
            3: {"A": [[4, -1], [2, 1]], "f": ["0", "1"], "x0": [0, 0]}, # Ejemplo 3
            4: {"A": [[-2, 1], [1, -2]], "f": ["1", "0"], "x0": [0, 0]}, # Ejemplo 4
            5: {"A": [[-1, 2], [0, -1]], "f": ["-8", "0"], "x0": [0, 0]} # Ejemplo 5
        }
        ej = ejemplos[num]
        entry_a11.delete(0, tk.END); entry_a11.insert(0, str(ej["A"][0][0]))
        entry_a12.delete(0, tk.END); entry_a12.insert(0, str(ej["A"][0][1]))
        entry_a21.delete(0, tk.END); entry_a21.insert(0, str(ej["A"][1][0]))
        entry_a22.delete(0, tk.END); entry_a22.insert(0, str(ej["A"][1][1]))
        entry_f1.delete(0, tk.END); entry_f1.insert(0, ej["f"][0])
        entry_f2.delete(0, tk.END); entry_f2.insert(0, ej["f"][1])
        entry_x0.delete(0, tk.END); entry_x0.insert(0, str(ej["x0"][0]))
        entry_y0.delete(0, tk.END); entry_y0.insert(0, str(ej["x0"][1]))

    btn_frame = tk.Frame(frame_ejemplos)
    btn_frame.pack(fill="x")
    for i in range(1, 6):
        tk.Button(btn_frame, text=f"Ej{i}", font=("Consolas", 9), width=4,
                 command=lambda x=i: cargar_ejemplo(x)).pack(side=tk.LEFT, padx=2)

    # Botón con nombre para el caso de batalla (Lanchester)
    def _set_lanchester():
        entry_a11.delete(0, tk.END); entry_a11.insert(0, "0")
        entry_a12.delete(0, tk.END); entry_a12.insert(0, "-0.1")
        entry_a21.delete(0, tk.END); entry_a21.insert(0, "-0.2")
        entry_a22.delete(0, tk.END); entry_a22.insert(0, "0")
        entry_f1.delete(0, tk.END); entry_f1.insert(0, "0")
        entry_f2.delete(0, tk.END); entry_f2.insert(0, "0")
        entry_x0.delete(0, tk.END); entry_x0.insert(0, "500")
        entry_y0.delete(0, tk.END); entry_y0.insert(0, "400")
    tk.Button(btn_frame, text="Batalla (Lanchester)", font=("Consolas", 9),
              command=_set_lanchester).pack(side=tk.LEFT, padx=6)

    # === Botones ===
    btns = tk.Frame(frame_left, relief=tk.RAISED, bd=2, bg="#e0e0e0")
    btns.pack(pady=15, fill=tk.X)
    tk.Button(btns, text="RESOLVER ANALÍTICAMENTE", bg="#4CAF50", fg="white",
              font=("Consolas", 12, "bold"), command=lambda: resolver_no_homogeneo(), 
              width=25, height=2).pack(side=tk.LEFT, padx=10, expand=True)
    tk.Button(btns, text="GRAFICAR COMPARACIÓN", bg="#2196F3", fg="white",
              font=("Consolas", 12, "bold"), command=lambda: graficar_comparacion(entry_a11, entry_a12, entry_a21, entry_a22, entry_f1, entry_f2, entry_x0, entry_y0, ax, canvas, entry_T_poincare, entry_t0_poincare, entry_N_poincare, var_mostrar_poincare),
              width=25, height=2).pack(side=tk.LEFT, padx=10, expand=True)

    # === Sección de Poincaré (si f(t) es periódica) ===
    frame_poin = tk.Frame(frame_left)
    frame_poin.pack(pady=(0, 8), fill=tk.X)
    tk.Label(frame_poin, text="Poincaré (f(t) periódica):", font=("Consolas", 10, "bold")).grid(row=0, column=0, sticky='w', padx=4)
    tk.Label(frame_poin, text="T:").grid(row=1, column=0, sticky='e')
    entry_T_poincare = tk.Entry(frame_poin, width=8)
    entry_T_poincare.grid(row=1, column=1)
    entry_T_poincare.insert(0, "6.28318")
    tk.Label(frame_poin, text="t0:").grid(row=1, column=2, sticky='e')
    entry_t0_poincare = tk.Entry(frame_poin, width=8)
    entry_t0_poincare.grid(row=1, column=3)
    entry_t0_poincare.insert(0, "0")
    tk.Label(frame_poin, text="N:").grid(row=1, column=4, sticky='e')
    entry_N_poincare = tk.Entry(frame_poin, width=8)
    entry_N_poincare.grid(row=1, column=5)
    entry_N_poincare.insert(0, "10")
    var_mostrar_poincare = tk.BooleanVar(value=False)
    tk.Checkbutton(frame_poin, text="Mostrar Poincaré", variable=var_mostrar_poincare).grid(row=1, column=6, padx=8)

    # === Canvas ===
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    canvas = FigureCanvasTkAgg(fig, master=frame_left)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)

    # === Panel derecho ===
    tk.Label(frame_right, text="📊 Análisis del sistema no homogéneo:",
             bg="#f4f4f4", font=("Consolas", 12, "bold")).pack(anchor="w", pady=(0, 5))
    text_info = tk.Text(frame_right, width=50, height=40, font=("Consolas", 10), bg="#f9f9f9")
    text_info.pack(fill=tk.BOTH, expand=True)

    # === Función de resolución analítica ===
    def resolver_no_homogeneo():
        text_info.delete("1.0", tk.END)
        try:
            # Obtener matriz A
            a11 = float(entry_a11.get())
            a12 = float(entry_a12.get())
            a21 = float(entry_a21.get())
            a22 = float(entry_a22.get())
            A = np.array([[a11, a12], [a21, a22]])
            
            # Obtener función forzante f(t)
            f1_text = entry_f1.get().strip().replace(',', '.')
            f2_text = entry_f2.get().strip().replace(',', '.')
            
            # Condiciones iniciales
            x0 = float(entry_x0.get())
            y0 = float(entry_y0.get())
            
            text_info.insert(tk.END, "═══════════════════════════════════════════════\n")
            text_info.insert(tk.END, "🔧 RESOLUCIÓN ANALÍTICA DE SISTEMA NO HOMOGÉNEO\n")
            text_info.insert(tk.END, "═══════════════════════════════════════════════\n\n")
            
            # Mostrar el sistema
            text_info.insert(tk.END, f"Sistema: X' = AX + f(t)\n\n")
            text_info.insert(tk.END, f"Matriz A = [{a11:g}  {a12:g}]\n")
            text_info.insert(tk.END, f"           [{a21:g}  {a22:g}]\n\n")
            text_info.insert(tk.END, f"f(t) = [{f1_text}]\n")
            text_info.insert(tk.END, f"       [{f2_text}]\n\n")
            text_info.insert(tk.END, f"Condiciones iniciales: X(0) = [{x0:g}, {y0:g}]ᵀ\n\n")
            
            # 1. Resolver sistema homogéneo X' = AX
            text_info.insert(tk.END, "━━━ PASO 1: Sistema homogéneo X' = AX ━━━\n\n")
            
            valores = np.linalg.eigvals(A)
            vectores, vectores_complejos = calcular_autovectores(A, valores)
            
            text_info.insert(tk.END, f"Autovalores: λ₁ = {valores[0]:.4f}, λ₂ = {valores[1]:.4f}\n")
            
            # Mostrar autovectores
            if np.iscomplexobj(valores[0]) or abs(valores[0].imag) > 1e-10:
                idx = 0 if np.imag(valores[0]) > 0 else 1
                text_info.insert(tk.END, f"Autovector: v = {formatear_autovector_complejo(vectores_complejos[idx])}\n\n")
            else:
                text_info.insert(tk.END, f"Autovectores: v₁ = [{vectores[0,0]:.4f}, {vectores[1,0]:.4f}]ᵀ\n")
                text_info.insert(tk.END, f"              v₂ = [{vectores[0,1]:.4f}, {vectores[1,1]:.4f}]ᵀ\n\n")
            
            # Solución homogénea
            text_info.insert(tk.END, "Solución homogénea Xₕ(t):\n")
            
            if np.iscomplexobj(valores[0]) or abs(valores[0].imag) > 1e-10:
                # Caso complejo
                alpha = valores[0].real
                beta = abs(valores[0].imag)
                v_complex = vectores_complejos[0 if np.imag(valores[0]) > 0 else 1]
                a1, a2, b1, b2 = canonicalizar_a_b(v_complex)
                
                text_info.insert(tk.END, f"α = {alpha:.4f}, β = {beta:.4f}\n")
                text_info.insert(tk.END, f"a⃗ = ({a1}, {a2}), b⃗ = ({b1}, {b2})\n")
                if abs(alpha) < 1e-12:
                    text_info.insert(tk.END, f"Xₕ(t) = C₁[a⃗cos({beta:.4f}t) - b⃗sen({beta:.4f}t)] + C₂[a⃗sen({beta:.4f}t) + b⃗cos({beta:.4f}t)]\n\n")
                else:
                    text_info.insert(tk.END, f"Xₕ(t) = e^({alpha:.4f}t)[C₁[a⃗cos({beta:.4f}t) - b⃗sen({beta:.4f}t)] + C₂[a⃗sen({beta:.4f}t) + b⃗cos({beta:.4f}t)]]\n\n")
            else:
                # Caso real
                lambda1, lambda2 = valores[0], valores[1]
                v1, v2 = vectores[:, 0], vectores[:, 1]
                text_info.insert(tk.END, f"Xₕ(t) = C₁e^({lambda1:.4f}t)[{v1[0]:.4f}, {v1[1]:.4f}]ᵀ + C₂e^({lambda2:.4f}t)[{v2[0]:.4f}, {v2[1]:.4f}]ᵀ\n\n")
            
            # 2. Encontrar solución particular
            text_info.insert(tk.END, "━━━ PASO 2: Solución particular Xₚ(t) ━━━\n\n")
            
            # Análisis de la función forzante
            text_info.insert(tk.END, "Análisis de f(t):\n")
            
            # Determinar tipo de función forzante
            es_constante = True
            es_polinomial = False
            es_exponencial = False
            es_trigonometrica = False
            
            try:
                # Evaluar en varios puntos para detectar el tipo
                f1_val_0 = eval(f1_text.replace('t', '0'), {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                f1_val_1 = eval(f1_text.replace('t', '1'), {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                f2_val_0 = eval(f2_text.replace('t', '0'), {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                f2_val_1 = eval(f2_text.replace('t', '1'), {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                
                if abs(f1_val_0 - f1_val_1) > 1e-10 or abs(f2_val_0 - f2_val_1) > 1e-10:
                    es_constante = False
                
                # Detectar tipos específicos
                if 'sin' in f1_text or 'cos' in f1_text or 'sin' in f2_text or 'cos' in f2_text:
                    es_trigonometrica = True
                elif 'exp' in f1_text or 'exp' in f2_text:
                    es_exponencial = True
                elif 't' in f1_text or 't' in f2_text:
                    es_polinomial = True
                    
            except:
                pass
            
            if es_constante:
                text_info.insert(tk.END, f"f(t) es constante: f(t) = [{f1_val_0:.4f}, {f2_val_0:.4f}]ᵀ\n")
                text_info.insert(tk.END, "Método: Xₚ = constante\n")
                text_info.insert(tk.END, "Sustituyendo 0 = AXₚ + f(t):\n")
                
                # Resolver -AXₚ = f
                try:
                    f_const = np.array([f1_val_0, f2_val_0])
                    if abs(np.linalg.det(A)) > 1e-10:
                        Xp = np.linalg.solve(-A, f_const)
                        text_info.insert(tk.END, f"Xₚ = [{Xp[0]:.4f}, {Xp[1]:.4f}]ᵀ\n\n")
                    else:
                        text_info.insert(tk.END, "Sistema singular - solución particular no única\n\n")
                except:
                    text_info.insert(tk.END, "Error al calcular solución particular\n\n")
                    
            elif es_trigonometrica:
                text_info.insert(tk.END, "f(t) contiene funciones trigonométricas\n")
                text_info.insert(tk.END, "Método: Xₚ = a⃗cos(ωt) + b⃗sen(ωt)\n\n")
            elif es_exponencial:
                text_info.insert(tk.END, "f(t) contiene funciones exponenciales\n")
                text_info.insert(tk.END, "Método: Xₚ = a⃗e^(rt)\n\n")
            elif es_polinomial:
                text_info.insert(tk.END, "f(t) es polinomial en t\n")
                text_info.insert(tk.END, "Método: Xₚ = a⃗t^n + b⃗t^(n-1) + ...\n\n")
            
            # 3. Solución general
            text_info.insert(tk.END, "━━━ PASO 3: Solución general ━━━\n\n")
            text_info.insert(tk.END, "X(t) = Xₕ(t) + Xₚ(t)\n\n")
            
            # 4. Aplicar condiciones iniciales
            text_info.insert(tk.END, "━━━ PASO 4: Condiciones iniciales ━━━\n\n")
            text_info.insert(tk.END, f"X(0) = [{x0:g}, {y0:g}]ᵀ\n")
            text_info.insert(tk.END, "Resolver para C₁ y C₂...\n\n")
            
            # 5. Análisis de comportamiento
            text_info.insert(tk.END, "━━━ PASO 5: Análisis de comportamiento ━━━\n\n")
            
            # Determinar estabilidad del sistema homogéneo
            max_real_part = max(np.real(valores))
            if max_real_part < -1e-10:
                comportamiento = "ESTABLE - La perturbación f(t) no cambia la estabilidad asintótica"
            elif max_real_part > 1e-10:
                comportamiento = "INESTABLE - La perturbación f(t) puede ser dominada por el crecimiento exponencial"
            else:
                comportamiento = "MARGINALMENTE ESTABLE - La perturbación f(t) puede cambiar significativamente el comportamiento"
            
            text_info.insert(tk.END, f"Comportamiento: {comportamiento}\n\n")
            
            # Análisis de preservación/ruptura
            text_info.insert(tk.END, "🔍 PRESERVACIÓN vs RUPTURA DE COMPORTAMIENTOS:\n\n")
            
            if es_constante:
                text_info.insert(tk.END, "• Perturbación CONSTANTE:\n")
                text_info.insert(tk.END, "  - Desplaza el punto de equilibrio\n")
                text_info.insert(tk.END, "  - PRESERVA el tipo de estabilidad\n")
                text_info.insert(tk.END, "  - Las trayectorias mantienen su forma cualitativa\n\n")
            elif es_trigonometrica:
                text_info.insert(tk.END, "• Perturbación PERIÓDICA:\n")
                text_info.insert(tk.END, "  - Introduce oscilaciones forzadas\n")
                if max_real_part < 0:
                    text_info.insert(tk.END, "  - PRESERVA estabilidad (oscilaciones amortiguadas)\n")
                else:
                    text_info.insert(tk.END, "  - Puede ROMPER estabilidad (resonancia)\n")
                text_info.insert(tk.END, "  - Posible aparición de ciclos límite\n\n")
            elif es_exponencial:
                text_info.insert(tk.END, "• Perturbación EXPONENCIAL:\n")
                text_info.insert(tk.END, "  - Puede ROMPER completamente el comportamiento\n")
                text_info.insert(tk.END, "  - Competencia entre crecimiento homogéneo y forzante\n\n")
            
        except Exception as e:
            text_info.insert(tk.END, f"❌ Error en el análisis: {e}\n")

    # === Función de graficación comparativa ===
def graficar_comparacion(entry_a11, entry_a12, entry_a21, entry_a22, entry_f1, entry_f2, entry_x0, entry_y0, ax, canvas, entry_T_poincare, entry_t0_poincare, entry_N_poincare, var_mostrar_poincare):
        try:
            # Obtener parámetros
            a11 = float(entry_a11.get())
            a12 = float(entry_a12.get())
            a21 = float(entry_a21.get())
            a22 = float(entry_a22.get())
            A = np.array([[a11, a12], [a21, a22]])
            
            f1_text = entry_f1.get().strip().replace(',', '.')
            f2_text = entry_f2.get().strip().replace(',', '.')
            x0 = float(entry_x0.get())
            y0 = float(entry_y0.get())
            
            # Sistema homogéneo
            def sistema_homogeneo(t, XY):
                XY = np.asarray(XY, dtype=float).reshape(2,)
                return A @ XY
            
            # Sistema no homogéneo
            def sistema_no_homogeneo(t, XY):
                f1_val = eval(f1_text, {"t": t, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                f2_val = eval(f2_text, {"t": t, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                try:
                    f_vec = np.array([float(f1_val), float(f2_val)], dtype=float)
                except Exception:
                    raise ValueError("Expresión de f(t) inválida. Usa punto decimal (ej. 1.9) y expresiones escalares para cada componente.")
                XY = np.asarray(XY, dtype=float).reshape(2,)
                out = A @ XY + f_vec
                # Chequeo de finitud para evitar warnings silenciosos
                if not np.all(np.isfinite(out)):
                    raise ValueError("El campo produjo valores no finitos (revise f(t) y condiciones)")
                return out
            
            # Resolver ambos sistemas
            # Preparar horizonte según Poincaré
            try:
                T_p = float(entry_T_poincare.get())
                t0_p = float(entry_t0_poincare.get())
                N_p = int(entry_N_poincare.get())
                mostrar_p = bool(var_mostrar_poincare.get())
            except Exception:
                T_p, t0_p, N_p, mostrar_p = 0.0, 0.0, 0, False
            t_end = 10.0
            if mostrar_p and (T_p is not None) and T_p > 0 and (N_p is not None) and N_p > 0:
                t_end = max(10.0, t0_p + N_p * T_p)
            t_span = [0, t_end]
            # Mantener densidad razonable para evitar bloqueos con T grande o N alto
            n_eval = max(1000, int(200 * t_end))
            n_eval = min(n_eval, 5000)
            t_eval = np.linspace(0, t_end, n_eval)
            
            sol_hom = solve_ivp(sistema_homogeneo, t_span, [x0, y0], t_eval=t_eval, dense_output=True)
            sol_no_hom = solve_ivp(sistema_no_homogeneo, t_span, [x0, y0], t_eval=t_eval, dense_output=True)
            
            # Graficar
            ax.clear()
            
            if sol_hom.success and sol_no_hom.success:
                # Trayectorias en el plano de fase
                ax.plot(sol_hom.y[0], sol_hom.y[1], 'b--', linewidth=2, label='Sistema homogéneo', alpha=0.7)
                ax.plot(sol_no_hom.y[0], sol_no_hom.y[1], 'r-', linewidth=2, label='Sistema no homogéneo')
                
                # Puntos iniciales
                ax.scatter(x0, y0, color='black', s=100, zorder=5, label='Condición inicial')
                
                # Campo vectorial del sistema no homogéneo (en t=0)
                x_range = np.linspace(-5, 5, 15)
                y_range = np.linspace(-5, 5, 15)
                X, Y = np.meshgrid(x_range, y_range)
                U, V = np.zeros_like(X), np.zeros_like(Y)
                
                for i in range(len(x_range)):
                    for j in range(len(y_range)):
                        dXY = sistema_no_homogeneo(0, [X[j,i], Y[j,i]])
                        U[j,i], V[j,i] = dXY[0], dXY[1]
                
                # Normalizar para mejor visualización
                M = np.sqrt(U**2 + V**2)
                M[M == 0] = 1
                U_norm, V_norm = U/M, V/M
                
                ax.quiver(X, Y, U_norm, V_norm, alpha=0.3, color='gray', scale=20)
                
                ax.set_xlabel('x(t)')
                ax.set_ylabel('y(t)')
                ax.set_title('Comparación: Sistema homogéneo vs No homogéneo')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Ajustar límites
                all_x = np.concatenate([sol_hom.y[0], sol_no_hom.y[0]])
                all_y = np.concatenate([sol_hom.y[1], sol_no_hom.y[1]])
                margin = 0.1
                x_range = np.ptp(all_x)
                y_range = np.ptp(all_y)
                ax.set_xlim(np.min(all_x) - margin*x_range, np.max(all_x) + margin*x_range)
                ax.set_ylim(np.min(all_y) - margin*y_range, np.max(all_y) + margin*y_range)

                # Sección de Poincaré (muestra puntos estroboscópicos de no homogéneo)
                try:
                    if mostrar_p and T_p > 0 and N_p > 0 and hasattr(sol_no_hom, 'sol') and (sol_no_hom.sol is not None):
                        t_points = [t0_p + k * T_p for k in range(N_p + 1)]
                        t_min, t_max = float(sol_no_hom.t[0]), float(sol_no_hom.t[-1])
                        t_points = [tp for tp in t_points if tp >= t_min and tp <= t_max]
                        if len(t_points) > 0:
                            XYp = np.array([sol_no_hom.sol(tp) for tp in t_points])
                            xp, yp = XYp[:,0], XYp[:,1]
                            ax.scatter(xp, yp, s=28, c='magenta', edgecolors='white', linewidths=0.6,
                                       label='Poincaré (no homog.)', zorder=7)
                except Exception:
                    pass
                
            canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en la graficación: {e}")




# ============================================================
# 🧭 Ventana principal (selector de modo)
# ============================================================
root = tk.Tk()
root.title("Selector de modo")
root.geometry("420x340")
try:
    root.minsize(420, 340)
except Exception:
    pass

tk.Label(root, text="Elegí el tipo de sistema dinámico", font=("Consolas", 13, "bold")).pack(pady=20)

tk.Button(root, text="🔹 Modo Numérico clásico", font=("Consolas", 12), width=30,
          command=abrir_modo_numerico).pack(pady=10)

tk.Button(root, text="🔸 Modo Sistemas No Lineales", font=("Consolas", 12), width=30,
          command=abrir_modo_no_lineal).pack(pady=10)

tk.Button(root, text="🔸 Modo No homogeneo", font=("Consolas", 12), width=30,
          command=abrir_modo_no_homogeneo).pack(pady=10)

tk.Label(root, text="", font=("Consolas", 10, "italic")).pack(pady=15)

root.mainloop()
