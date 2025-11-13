# Sistemas Dinámicos – Guía rápida

Este repositorio contiene un conjunto de herramientas interactivas para analizar sistemas dinámicos lineales, no lineales, no homogéneos y modelos especiales (bifurcaciones 1D, combate, romance e infecciones). Todos los módulos están escritos en Python/Tkinter y comparten dependencias científicas comunes.

## Requisitos

- Python 3.9+ (se probó con 3.10)
- Paquetes: `numpy`, `sympy`, `matplotlib`, `scipy`

Instala dependencias con:

```bash
pip install numpy sympy matplotlib scipy
```

## Inicio rápido

1. Abre una terminal en la carpeta `c:\sistemas-dinamicos`.
2. Ejecuta `python sistemaDinamico.py`.
3. Se abrirá el selector principal; cada botón lanza una ventana distinta:
   - `Modo numerico clasico`: análisis lineal/homogéneo.
   - `Modo Sistemas No Lineales`: análisis completo con equilibrio, linealización, campo de fases y herramientas de Poincaré.
   - `Modo No homogeneo`: comparación lineal vs no homogéneo con excitaciones.
   - `Bifurcaciones 1D`: (ventana separada) analiza ecuaciones autónomas 1D y construye diagramas de bifurcación.
   - `Escenario Combate (Lanchester)`: abre `combate_lanchester.py`.
   - `Romeo & Julieta`: abre `romeo_julieta.py`.
   - `Modelo Verhulst infeccion`: abre `verhulst_infeccion.py`.

Cada ventana funciona de forma independiente; puedes abrir varias en paralelo desde el menú.

## Módulos incluidos

| Archivo | Descripción |
| ------- | ----------- |
| `sistemaDinamico.py` | Aplicación principal con menú y modos clásicos. |
| `combate_lanchester.py` | Laboratorio de enfrentamientos (modelos Lanchester extendidos). |
| `romeo_julieta.py` | Simulador de dinámicas afectivas tipo Romeo-Julieta. |
| `verhulst_infeccion.py` | Modelo logístico de infecciones con intervenciones. |

### Ejecutar módulos de forma directa

Todos los scripts pueden abrirse sin pasar por el menú:

```bash
python combate_lanchester.py
python romeo_julieta.py
python verhulst_infeccion.py
```

## Consejos de uso

- Guarda los parámetros antes de cerrar cada ventana si quieres repetir un escenario.
- Algunas funciones (por ejemplo, integraciones largas o diagramas de bifurcación densos) pueden tardar; espera a que los gráficos se actualicen antes de lanzar otra simulación.
- Si un modo no abre por falta de paquetes, revisa las dependencias listadas arriba.

¡Listo! Con esto deberías poder explorar cada modo y extender el código según tus necesidades.***
