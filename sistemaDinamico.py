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

    # === Botones ===
    btns = tk.Frame(frame_left)
    btns.pack(pady=10)
    tk.Button(btns, text="GRAFICAR", bg="#4CAF50", fg="white",
              font=("Consolas", 14, "bold"), command=graficar_numerico).pack(side=tk.LEFT, padx=6)
    tk.Button(btns, text="VERIFICAR", bg="#2196F3", fg="white",
              font=("Consolas", 14, "bold"), command=verificar_solucion).pack(side=tk.LEFT, padx=6)


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
        valores = np.linalg.eigvals(A)
        vectores, vectores_complejos = calcular_autovectores(A, valores)
        tau = np.trace(A)
        Delta = np.linalg.det(A)
        D = tau**2 - 4*Delta

        text_info.insert(tk.END, f"Matriz Jacobiana (parte homogénea):\n{A}\n\n")
        text_info.insert(tk.END, f"Autovalores:\n λ₁={valores[0]:.4f}, λ₂={valores[1]:.4f}\n")
        # Mostrar autovectores en el mismo formato
        if np.iscomplexobj(valores[0]) or abs(valores[0].imag) > 1e-10:
            text_info.insert(tk.END, f"Autovector:\n{formatear_autovector_complejo(vectores_complejos[0])}\n\n")
        else:
            text_info.insert(tk.END, f"Autovectores:\n{vectores}\n\n")
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
