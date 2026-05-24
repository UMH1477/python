# ============================================================
# FUNCIONES COMUNES REUTILIZABLES PARA MONTE CARLO Y TABLAS
# ============================================================

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import t



# ============================================================
# FUNCIONES TEÓRICAS COMUNES PARA REDES ABIERTAS M/M/s
# ============================================================

def resolver_ecuaciones_trafico(gamma_vec, P):
    """
    Resuelve las ecuaciones de tráfico de una red abierta:

        lambda = gamma + lambda P

    En forma matricial:

        lambda = gamma (I - P)^(-1)
    """

    gamma_vec = np.asarray(gamma_vec, dtype=float)
    P = np.asarray(P, dtype=float)

    I = np.eye(len(gamma_vec))

    return gamma_vec @ np.linalg.inv(I - P)


def metricas_mm_s(lambd, mu, s):
    """
    Calcula las métricas teóricas de una cola M/M/s.
    """

    rho = lambd / (s * mu)

    if rho >= 1:
        return {
            "lambda_efectiva": lambd,
            "tasa_salida": np.nan,
            "rho": rho,
            "L": np.inf,
            "Lq": np.inf,
            "W": np.inf,
            "Wq": np.inf
        }

    a = lambd / mu

    suma = sum(
        a**n / math.factorial(n)
        for n in range(s)
    )

    ultimo = a**s / (
        math.factorial(s) * (1 - rho)
    )

    p0 = 1 / (suma + ultimo)

    Lq = (
        p0 * a**s * rho
        / (math.factorial(s) * (1 - rho)**2)
    )

    L = Lq + a
    Wq = Lq / lambd
    W = Wq + 1 / mu

    return {
        "lambda_efectiva": lambd,
        "tasa_salida": lambd,
        "rho": rho,
        "L": L,
        "Lq": Lq,
        "W": W,
        "Wq": Wq
    }


# ------------------------------------------------------------
# 1. Nodos del parque
# ------------------------------------------------------------

NOMBRES_NODOS = [
    "Montaña Rusa",
    "Tiovivo",
    "Flotador Acuático",
    "Fantasía",
    "Montaña Fantasma",
    "Viaje a la Luna"
]

N_NODOS = len(NOMBRES_NODOS)


# ------------------------------------------------------------
# 2. Duraciones base de las atracciones
# ------------------------------------------------------------

# Duración fija de disfrute de la atracción, en minutos.
# Corrección importante:
# El Flotador Acuático dura 1.5 minutos.
DURACION_ATRACCION = np.array([
    2.0,          # Montaña Rusa
    3.0,          # Tiovivo
    1.5,          # Flotador Acuático
    5.0,          # Fantasía
    1.5,          # Montaña Fantasma
    100 / 60      # Viaje a la Luna
], dtype=float)


# ------------------------------------------------------------
# 1. Resumen Monte Carlo de una métrica
# ------------------------------------------------------------

def resumen_monte_carlo(valores, confianza=0.95):
    """
    Calcula la estimación Monte Carlo, el error Monte Carlo
    y el intervalo de confianza al nivel indicado.

    Parámetros
    ----------
    valores : list, array
        Valores de una misma métrica en las diferentes réplicas.
    confianza : float
        Nivel de confianza. Por defecto, 0.95.

    Devuelve
    --------
    dict
        media, error_mc, ic_inf, ic_sup, n.
    """

    valores = np.asarray(valores, dtype=float)
    valores = valores[~np.isnan(valores)]

    n = len(valores)

    if n == 0:
        return {
            "media": np.nan,
            "error_mc": np.nan,
            "ic_inf": np.nan,
            "ic_sup": np.nan,
            "n": 0
        }

    media = np.mean(valores)

    if n == 1:
        return {
            "media": media,
            "error_mc": np.nan,
            "ic_inf": np.nan,
            "ic_sup": np.nan,
            "n": n
        }

    error_mc = np.std(valores, ddof=1) / np.sqrt(n)
    alpha = 1 - confianza
    critico = t.ppf(1 - alpha / 2, df=n - 1)

    return {
        "media": media,
        "error_mc": error_mc,
        "ic_inf": media - critico * error_mc,
        "ic_sup": media + critico * error_mc,
        "n": n
    }


# ------------------------------------------------------------
# 2. Agregación de réplicas Monte Carlo
# ------------------------------------------------------------

def agregar_replicas(resultados_replicas, confianza=0.95):
    """
    Agrega las métricas locales y globales obtenidas en varias
    réplicas independientes.

    Cada réplica debe tener la estructura:

        resultado["locales"]:
            DataFrame con columnas ['metrica', 'nodo', 'valor']

        resultado["globales"]:
            DataFrame con columnas ['metrica', 'nodo', 'valor']

    Devuelve
    --------
    dict con:
        - locales: resumen MC por métrica y nodo
        - globales: resumen MC por métrica global
        - replicas: resultados originales
    """

    if len(resultados_replicas) == 0:
        raise ValueError("La lista resultados_replicas está vacía.")

    # --------------------------------------------------------
    # Métricas locales
    # --------------------------------------------------------

    filas_locales = []

    claves_locales = (
        resultados_replicas[0]["locales"]
        [["metrica", "nodo"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    for _, clave in claves_locales.iterrows():

        metrica = clave["metrica"]
        nodo = clave["nodo"]

        valores = []

        for resultado in resultados_replicas:

            df = resultado["locales"]

            seleccion = df.loc[
                (df["metrica"] == metrica) &
                (df["nodo"] == nodo),
                "valor"
            ]

            if len(seleccion) == 0:
                valores.append(np.nan)
            else:
                valores.append(seleccion.iloc[0])

        resumen = resumen_monte_carlo(
            valores=valores,
            confianza=confianza
        )

        filas_locales.append({
            "metrica": metrica,
            "nodo": nodo,
            **resumen
        })

    resumen_locales = pd.DataFrame(filas_locales)

    # --------------------------------------------------------
    # Métricas globales
    # --------------------------------------------------------

    filas_globales = []

    claves_globales = (
        resultados_replicas[0]["globales"]
        [["metrica", "nodo"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    for _, clave in claves_globales.iterrows():

        metrica = clave["metrica"]
        nodo = clave["nodo"]

        valores = []

        for resultado in resultados_replicas:

            df = resultado["globales"]

            seleccion = df.loc[
                (df["metrica"] == metrica) &
                (df["nodo"] == nodo),
                "valor"
            ]

            if len(seleccion) == 0:
                valores.append(np.nan)
            else:
                valores.append(seleccion.iloc[0])

        resumen = resumen_monte_carlo(
            valores=valores,
            confianza=confianza
        )

        filas_globales.append({
            "metrica": metrica,
            "nodo": nodo,
            **resumen
        })

    resumen_globales = pd.DataFrame(filas_globales)

    return {
        "locales": resumen_locales,
        "globales": resumen_globales,
        "replicas": resultados_replicas
    }


# ------------------------------------------------------------
# 3. Formateo de estimaciones Monte Carlo
# ------------------------------------------------------------

def formatear_estimacion_mc(fila, decimales=4, mostrar="ic"):
    """
    Formatea una estimación Monte Carlo.

    mostrar='ic':
        media [ic_inf - ic_sup]

    mostrar='error':
        media ± error_mc
    """

    media = fila["media"]
    error_mc = fila["error_mc"]
    ic_inf = fila["ic_inf"]
    ic_sup = fila["ic_sup"]

    if pd.isna(media):
        return "—"

    if mostrar == "error":
        if pd.isna(error_mc):
            return f"{media:.{decimales}f} ± —"
        return f"{media:.{decimales}f} ± {error_mc:.{decimales}f}"

    if pd.isna(ic_inf) or pd.isna(ic_sup):
        return f"{media:.{decimales}f} [— - —]"

    return (
        f"{media:.{decimales}f} "
        f"[{ic_inf:.{decimales}f} - {ic_sup:.{decimales}f}]"
    )


# ------------------------------------------------------------
# 4. Tabla de métricas locales
# ------------------------------------------------------------

def tabla_metricas_locales_mc(
    resultado_mc,
    nombres_nodos=None,
    decimales=4,
    mostrar="ic"
):
    """
    Genera la tabla de métricas locales.

    Filas:
        métricas locales.

    Columnas:
        nodos / atracciones.

    Celdas:
        estimación Monte Carlo con IC 95% o error Monte Carlo.
    """

    df = resultado_mc["locales"].copy()

    df["estimacion"] = df.apply(
        lambda fila: formatear_estimacion_mc(
            fila,
            decimales=decimales,
            mostrar=mostrar
        ),
        axis=1
    )

    tabla = df.pivot(
        index="metrica",
        columns="nodo",
        values="estimacion"
    )

    if nombres_nodos is not None:
        columnas = [nodo for nodo in nombres_nodos if nodo in tabla.columns]
        tabla = tabla[columnas]

    return tabla


# ------------------------------------------------------------
# 5. Tabla de métricas globales
# ------------------------------------------------------------

def tabla_metricas_globales_mc(
    resultado_mc,
    nombre_modelo="Modelo",
    decimales=4,
    mostrar="ic"
):
    """
    Genera la tabla de métricas globales.

    Filas:
        métricas globales.

    Columnas:
        modelo.

    Celdas:
        estimación Monte Carlo con IC 95% o error Monte Carlo.
    """

    df = resultado_mc["globales"].copy()

    df["estimacion"] = df.apply(
        lambda fila: formatear_estimacion_mc(
            fila,
            decimales=decimales,
            mostrar=mostrar
        ),
        axis=1
    )

    tabla = df.pivot(
        index="metrica",
        columns="nodo",
        values="estimacion"
    )

    # Normalmente la columna se llama "Global".
    # La renombramos con el nombre del modelo para que la tabla sea clara.
    if "Global" in tabla.columns:
        tabla = tabla.rename(columns={"Global": nombre_modelo})

    return tabla


# ------------------------------------------------------------
# 6. Estilo visual común para tablas
# ------------------------------------------------------------

def estilizar_tabla_mc(tabla, titulo="Estimaciones Monte Carlo"):
    """
    Aplica un estilo visual homogéneo a las tablas Monte Carlo.
    """

    return (
        tabla.style
        .set_caption(titulo)
        .set_properties(**{
            "text-align": "center",
            "white-space": "pre-wrap",
            "border": "1px solid #D6DBDF",
            "padding": "6px"
        })
        .set_table_styles([
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("font-weight", "bold"),
                    ("font-size", "15px"),
                    ("padding", "8px")
                ]
            },
            {
                "selector": "th",
                "props": [
                    ("background-color", "#DDEBF7"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("border", "1px solid #D6DBDF"),
                    ("padding", "6px")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid #D6DBDF"),
                    ("padding", "6px")
                ]
            }
        ])
        .apply(
            lambda fila: [
                "background-color: #F8FBFD"
                if i % 2 == 0
                else "background-color: white"
                for i in range(len(fila))
            ],
            axis=1
        )
    )


# ------------------------------------------------------------
# 7. Mostrar tablas locales y globales de un modelo
# ------------------------------------------------------------

def mostrar_tablas_modelo_mc(
    resultado_mc,
    nombres_nodos,
    nombre_modelo="Modelo",
    decimales=4,
    mostrar="ic"
):
    """
    Muestra, separadamente, la tabla de métricas locales
    y la tabla de métricas globales.

    Esta será la función estándar de visualización para todos
    los modelos.
    """

    tabla_local = tabla_metricas_locales_mc(
        resultado_mc=resultado_mc,
        nombres_nodos=nombres_nodos,
        decimales=decimales,
        mostrar=mostrar
    )

    tabla_global = tabla_metricas_globales_mc(
        resultado_mc=resultado_mc,
        nombre_modelo=nombre_modelo,
        decimales=decimales,
        mostrar=mostrar
    )

    display(
        estilizar_tabla_mc(
            tabla_local,
            titulo=f"{nombre_modelo}. Métricas locales por nodo"
        )
    )

    display(
        estilizar_tabla_mc(
            tabla_global,
            titulo=f"{nombre_modelo}. Métricas globales del parque"
        )
    )

    return tabla_local, tabla_global

# ------------------------------------------------------------
# 8. Comparación gráfica entre dos modelos
# ------------------------------------------------------------

def graficar_comparacion_local(
    resultado_a,
    nombre_a,
    resultado_b,
    nombre_b,
    nombres_nodos,
    metricas=None,
    max_columnas=4
):
    """
    Compara dos modelos a nivel local.

    Para cada métrica común se representa el comportamiento por nodo.
    Se usan solo las estimaciones Monte Carlo, no los errores ni los IC.
    """

    locales_a = resultado_a["locales"]
    locales_b = resultado_b["locales"]

    metricas_comunes = sorted(
        set(locales_a["metrica"]).intersection(set(locales_b["metrica"]))
    )

    if metricas is None:
        metricas = metricas_comunes
    else:
        metricas = [m for m in metricas if m in metricas_comunes]

    if len(metricas) == 0:
        print("No hay métricas locales comunes para comparar.")
        return

    n_metricas = len(metricas)
    n_cols = min(max_columnas, n_metricas)
    n_rows = math.ceil(n_metricas / n_cols)

    colores = ["#8DB6CD", "#F4A7A1"]

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.2 * n_rows),
        squeeze=False
    )

    axes = axes.flatten()

    x = np.arange(len(nombres_nodos))
    etiquetas_x = [n.replace(" ", "\n") for n in nombres_nodos]

    for k, metrica in enumerate(metricas):

        ax = axes[k]

        valores_a = (
            locales_a[locales_a["metrica"] == metrica]
            .set_index("nodo")
            .reindex(nombres_nodos)["media"]
        )

        valores_b = (
            locales_b[locales_b["metrica"] == metrica]
            .set_index("nodo")
            .reindex(nombres_nodos)["media"]
        )

        ax.plot(
            x,
            valores_a,
            marker="o",
            linewidth=2,
            color=colores[0],
            label=nombre_a
        )

        ax.plot(
            x,
            valores_b,
            marker="o",
            linewidth=2,
            color=colores[1],
            label=nombre_b
        )

        ax.set_title(metrica, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(etiquetas_x, fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    for j in range(n_metricas, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Comparación local: {nombre_a} vs {nombre_b}",
        fontsize=13,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()


def graficar_comparacion_global(resultado_a, nombre_a, resultado_b, nombre_b):
    """
    Compara dos modelos a nivel global en una sola fila con dos columnas.
    """
    globales_a = resultado_a["globales"]
    globales_b = resultado_b["globales"]

    metricas_comunes = sorted(
        set(globales_a["metrica"]).intersection(set(globales_b["metrica"]))
    )

    metricas_tamano = [
        m for m in metricas_comunes
        if m.startswith("L") or "visitantes" in m.lower() or "cola" in m.lower()
    ]

    metricas_tiempo = [
        m for m in metricas_comunes
        if m.startswith("W") or "tiempo" in m.lower()
        or "espera" in m.lower() or "permanencia" in m.lower()
    ]

    bloques = [
        ("Tamaños", metricas_tamano),
        ("Tiempos", metricas_tiempo)
    ]

    colores = ["#8DB6CD", "#F4A7A1"]

    # Creamos la figura con 1 fila y 2 columnas
    # Ajustamos el ancho (figsize) para que quepan ambos
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    axes = axes.flatten()

    for i, (titulo, metricas) in enumerate(bloques):
        ax = axes[i]

        if len(metricas) == 0:
            ax.set_title(f"Sin datos para {titulo}")
            ax.axis('off') # Oculta el eje si no hay datos
            continue

        valores_a = (
            globales_a
            .set_index("metrica")
            .reindex(metricas)["media"]
        )

        valores_b = (
            globales_b
            .set_index("metrica")
            .reindex(metricas)["media"]
        )

        x = np.arange(len(metricas))

        ax.bar(
            x - 0.18,
            valores_a,
            width=0.36,
            color=colores[0],
            label=nombre_a
        )

        ax.bar(
            x + 0.18,
            valores_b,
            width=0.36,
            color=colores[1],
            label=nombre_b
        )

        ax.set_title(
            f"Comparación global - {titulo}",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(metricas, fontsize=9, rotation=45 if len(metricas) > 4 else 0)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()

    # Ajuste final de espacio entre los dos subplots
    plt.tight_layout()
    plt.show()


def comparar_modelos(
    resultado_a,
    nombre_a,
    resultado_b,
    nombre_b,
    nombres_nodos,
    metricas_locales=None
):
    """
    Función envoltorio para comparar dos modelos.
    """

    graficar_comparacion_local(
        resultado_a=resultado_a,
        nombre_a=nombre_a,
        resultado_b=resultado_b,
        nombre_b=nombre_b,
        nombres_nodos=nombres_nodos,
        metricas=metricas_locales
    )

    graficar_comparacion_global(
        resultado_a=resultado_a,
        nombre_a=nombre_a,
        resultado_b=resultado_b,
        nombre_b=nombre_b
    )

# ============================================================
# BLOQUE COMÚN AÑADIDO DESDE EL MODELO 2
# Servicio por lotes
# ============================================================


# ------------------------------------------------------------
# 3. Servicio por lotes
# ------------------------------------------------------------

# Capacidad de cada lote individual.
# En el Flotador Acuático cada flotador tiene capacidad 2.
CAPACIDAD_LOTE = np.array([
    24,   # Montaña Rusa: dos vagones de 12
    35,   # Tiovivo
    2,    # Flotador Acuático: cada flotador lleva 2
    60,   # Fantasía
    16,   # Montaña Fantasma
    20    # Viaje a la Luna
], dtype=int)

# Número de lotes/ciclos paralelos.
# El Flotador Acuático tiene 10 flotadores en paralelo.
NUM_LOTES_PARALELOS = np.array([
    1,
    1,
    10,
    1,
    1,
    1
], dtype=int)

# Capacidad total simultánea de cada nodo.
CAPACIDAD_TOTAL = CAPACIDAD_LOTE * NUM_LOTES_PARALELOS


# ------------------------------------------------------------
# 4. Tiempo máximo de espera para arrancar lote
# ------------------------------------------------------------

# Tiempo máximo que espera el primer visitante del lote antes de arrancar.
# Si la cola alcanza la capacidad del lote antes, el lote arranca inmediatamente.
TIEMPO_MAX_ESPERA_LOTE = np.array([
    1.0,   # Montaña Rusa
    2.0,   # Tiovivo
    3.0,   # Flotador Acuático
    3.0,   # Fantasía
    1.0,   # Montaña Fantasma
    1.0    # Viaje a la Luna
], dtype=float)

# ============================================================
# BLOQUE COMÚN AÑADIDO DESDE EL MODELO 3
# Embarque y desembarque
# ============================================================

# ------------------------------------------------------------
# 5. Parámetros de embarque
# ------------------------------------------------------------

# B_i(n) = b0_i + b1_i * n
EMBARQUE_B0 = np.array([
    0.40,  # Montaña Rusa
    0.50,  # Tiovivo
    0.25,  # Flotador Acuático
    0.70,  # Fantasía
    0.35,  # Montaña Fantasma
    0.45   # Viaje a la Luna
], dtype=float)

EMBARQUE_B1 = np.array([
    0.045,  # Montaña Rusa
    0.030,  # Tiovivo
    0.080,  # Flotador Acuático
    0.025,  # Fantasía
    0.035,  # Montaña Fantasma
    0.040   # Viaje a la Luna
], dtype=float)


# ------------------------------------------------------------
# 6. Parámetros de desembarque
# ------------------------------------------------------------

# D_i(n) = d0_i + d1_i * n
DESEMBARQUE_D0 = np.array([
    0.25,  # Montaña Rusa
    0.30,  # Tiovivo
    0.20,  # Flotador Acuático
    0.40,  # Fantasía
    0.25,  # Montaña Fantasma
    0.30   # Viaje a la Luna
], dtype=float)

DESEMBARQUE_D1 = np.array([
    0.025,  # Montaña Rusa
    0.015,  # Tiovivo
    0.050,  # Flotador Acuático
    0.015,  # Fantasía
    0.020,  # Montaña Fantasma
    0.020   # Viaje a la Luna
], dtype=float)


# ------------------------------------------------------------
# 7. Funciones comunes de embarque/desembarque
# ------------------------------------------------------------

def tiempo_embarque(i, n_visitantes):
    """
    Tiempo de embarque del lote en el nodo i.

    Se calcula como:

        B_i(n) = b0_i + b1_i * n

    donde n es el número real de visitantes que entran en el lote.
    """

    return EMBARQUE_B0[i] + EMBARQUE_B1[i] * n_visitantes


def tiempo_desembarque(i, n_visitantes):
    """
    Tiempo de desembarque del lote en el nodo i.

    Se calcula como:

        D_i(n) = d0_i + d1_i * n

    donde n es el número real de visitantes que salen del lote.
    """

    return DESEMBARQUE_D0[i] + DESEMBARQUE_D1[i] * n_visitantes


def tiempo_ciclo_atraccion(i, n_visitantes):
    """
    Tiempo total de ciclo de un lote en el nodo i.

    Incluye:
        - embarque,
        - disfrute de la atracción,
        - desembarque.
    """

    return (
        tiempo_embarque(i, n_visitantes)
        + DURACION_ATRACCION[i]
        + tiempo_desembarque(i, n_visitantes)
    )


# ------------------------------------------------------------
# 8. Encaminamiento común
# ------------------------------------------------------------

def parametros_encaminamiento_uniforme(gamma_hora=600):
    """
    Define el encaminamiento uniforme del parque.

    Llegadas externas:
        - gamma_hora visitantes/hora en total;
        - reparto uniforme entre los nodos.

    Tras cada atracción:
        - probabilidad 1/6 de salir;
        - probabilidad 1/6 de ir a cada una de las otras 5 atracciones.
    """

    gamma_total = gamma_hora / 60
    gamma_vec = np.repeat(gamma_total / N_NODOS, N_NODOS)

    P = np.zeros((N_NODOS, N_NODOS))

    for i in range(N_NODOS):
        for j in range(N_NODOS):
            if i != j:
                P[i, j] = 1 / N_NODOS

    return gamma_total, gamma_vec, P


def elegir_siguiente_nodo_uniforme(i_actual, rng):
    """
    Elige el siguiente destino después de completar una atracción.

    Opciones:
        - las otras 5 atracciones;
        - salida del parque.

    Devuelve:
        -1 si el visitante sale del parque;
         j si el visitante va al nodo j.
    """

    opciones = [j for j in range(N_NODOS) if j != i_actual] + [-1]

    return int(rng.choice(opciones))

# ============================================================
# BLOQUE COMÚN AÑADIDO DESDE EL MODELO 4
# Desplazamientos aleatorios entre atracciones
# ============================================================

# Parámetros de la distribución triangular de desplazamiento, en minutos.
# T_mov ~ Triangular(minimo, moda, maximo)
PARAMETROS_DESPLAZAMIENTO_TRIANGULAR = {
    "left": 1.0,
    "mode": 2.0,
    "right": 4.0
}


def tiempo_desplazamiento_triangular(rng, parametros=None):
    """
    Genera un tiempo de desplazamiento entre atracciones.

    Por defecto:
        T_mov ~ Triangular(1, 2, 4), en minutos.

    Parámetros
    ----------
    rng : np.random.Generator
        Generador aleatorio de NumPy.
    parametros : dict, opcional
        Diccionario con claves left, mode y right.

    Devuelve
    --------
    float
        Tiempo de desplazamiento generado.
    """

    if parametros is None:
        parametros = PARAMETROS_DESPLAZAMIENTO_TRIANGULAR

    return float(
        rng.triangular(
            left=parametros["left"],
            mode=parametros["mode"],
            right=parametros["right"]
        )
    )

# ============================================================
# BLOQUE COMÚN AÑADIDO DESDE EL MODELO 5
# Llegadas no estacionarias y elección adaptativa de destino
# ============================================================

import numpy as np


# ------------------------------------------------------------
# 1. Llegadas externas por franjas horarias
# ------------------------------------------------------------

# Cada tupla contiene:
#     inicio_franja, fin_franja, tasa_total_visitantes_hora
FRANJAS_LLEGADAS_MODELO5 = [
    (0, 60, 300),
    (60, 180, 650),
    (180, 300, 450),
    (300, 480, 700),
    (480, 600, 250)
]


def gamma_hora_t(t, franjas=FRANJAS_LLEGADAS_MODELO5):
    """
    Devuelve la tasa total externa de llegada al parque en visitantes/hora
    en el instante t.

    Si t queda fuera de las franjas definidas, devuelve 0.
    """

    for inicio, fin, gamma_hora in franjas:
        if inicio <= t < fin:
            return gamma_hora

    return 0.0


def gamma_minuto_t(t, franjas=FRANJAS_LLEGADAS_MODELO5):
    """
    Devuelve la tasa total externa de llegada al parque en visitantes/minuto
    en el instante t.
    """

    return gamma_hora_t(t, franjas=franjas) / 60.0


def gamma_nodo_minuto_t(t, i, franjas=FRANJAS_LLEGADAS_MODELO5):
    """
    Devuelve la tasa externa de llegada al nodo i en visitantes/minuto.

    El reparto inicial entre atracciones se mantiene equitativo.
    """

    return gamma_minuto_t(t, franjas=franjas) / N_NODOS


def fin_franja_actual(t, franjas=FRANJAS_LLEGADAS_MODELO5):
    """
    Devuelve el instante final de la franja horaria en la que se encuentra t.

    Si t no pertenece a ninguna franja, devuelve infinito.
    """

    for inicio, fin, gamma_hora in franjas:
        if inicio <= t < fin:
            return fin

    return np.inf


# ------------------------------------------------------------
# 2. Popularidad y coordenadas espaciales
# ------------------------------------------------------------

POPULARIDAD_MODELO5 = np.array([
    1.40,   # Montaña Rusa
    0.75,   # Tiovivo
    1.10,   # Flotador Acuático
    0.90,   # Fantasía
    1.20,   # Montaña Fantasma
    1.00    # Viaje a la Luna
], dtype=float)


COORDENADAS_MODELO5 = np.array([
    [0, 0],   # Montaña Rusa
    [2, 1],   # Tiovivo
    [5, 0],   # Flotador Acuático
    [1, 4],   # Fantasía
    [4, 4],   # Montaña Fantasma
    [6, 3]    # Viaje a la Luna
], dtype=float)


def calcular_matriz_distancias_corregidas(coordenadas):
    """
    Calcula la matriz d*_{ij} = 1 + d_{ij}, donde d_{ij}
    es la distancia euclídea entre las atracciones i y j.
    """

    coordenadas = np.asarray(coordenadas, dtype=float)

    n = coordenadas.shape[0]
    distancias = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            distancia_euclidea = np.sqrt(
                (coordenadas[i, 0] - coordenadas[j, 0])**2
                + (coordenadas[i, 1] - coordenadas[j, 1])**2
            )
            distancias[i, j] = 1.0 + distancia_euclidea

    return distancias


DISTANCIAS_CORREGIDAS_MODELO5 = calcular_matriz_distancias_corregidas(
    COORDENADAS_MODELO5
)


# ------------------------------------------------------------
# 3. Sensibilidad de la elección adaptativa
# ------------------------------------------------------------

ALPHA_POPULARIDAD_MODELO5 = 1.2
BETA_DISTANCIA_MODELO5 = 0.8
THETA_ESPERA_MODELO5 = 0.06


# ------------------------------------------------------------
# 4. Probabilidad de salida dependiente de visitas completadas
# ------------------------------------------------------------

P_SALIDA_BASE_MODELO5 = 0.05
P_SALIDA_INCREMENTO_MODELO5 = 0.04
P_SALIDA_MAX_MODELO5 = 0.35


def probabilidad_salida_por_visitas(
    visitas_completadas,
    p_base=P_SALIDA_BASE_MODELO5,
    incremento=P_SALIDA_INCREMENTO_MODELO5,
    p_max=P_SALIDA_MAX_MODELO5
):
    """
    Calcula la probabilidad de abandonar el parque después de haber
    completado v atracciones.

        p_salida(v) = min(0.05 + 0.04 v, 0.35)
    """

    return min(
        p_base + incremento * visitas_completadas,
        p_max
    )


# ------------------------------------------------------------
# 5. Capacidad media aproximada y espera estimada
# ------------------------------------------------------------

def capacidad_media_aproximada_por_nodo():
    """
    Calcula la capacidad media aproximada de cada nodo en visitantes/minuto.

    Se utiliza:

        capacidad_media_j =
            capacidad simultánea total_j / ciclo_lleno_j

    donde ciclo_lleno_j incluye:
        - embarque con lote lleno;
        - duración fija de la atracción;
        - desembarque con lote lleno.

    En el Flotador Acuático, la capacidad simultánea total ya recoge
    los 10 flotadores paralelos de capacidad 2.
    """

    capacidad_media = np.zeros(N_NODOS)

    for j in range(N_NODOS):

        ciclo_lleno_j = tiempo_ciclo_atraccion(
            i=j,
            n_visitantes=CAPACIDAD_LOTE[j]
        )

        capacidad_media[j] = CAPACIDAD_TOTAL[j] / ciclo_lleno_j

    return capacidad_media


CAPACIDAD_MEDIA_APROXIMADA = capacidad_media_aproximada_por_nodo()


def espera_estimada_por_cola(en_cola, capacidad_media=CAPACIDAD_MEDIA_APROXIMADA):
    """
    Estima la espera en cola de cada nodo a partir del número de visitantes
    esperando físicamente en cola.

        Wq_estimada_j(t) = cola_j(t) / capacidad_media_j
    """

    en_cola = np.asarray(en_cola, dtype=float)
    capacidad_media = np.asarray(capacidad_media, dtype=float)

    return np.divide(
        en_cola,
        capacidad_media,
        out=np.zeros_like(en_cola, dtype=float),
        where=capacidad_media > 0
    )


# ------------------------------------------------------------
# 6. Elección adaptativa de destino
# ------------------------------------------------------------

def elegir_destino_adaptativo(
    i_actual,
    en_cola,
    rng,
    popularidad=POPULARIDAD_MODELO5,
    distancias=DISTANCIAS_CORREGIDAS_MODELO5,
    alpha=ALPHA_POPULARIDAD_MODELO5,
    beta=BETA_DISTANCIA_MODELO5,
    theta=THETA_ESPERA_MODELO5,
    capacidad_media=CAPACIDAD_MEDIA_APROXIMADA
):
    """
    Elige la siguiente atracción entre todas las distintas de la actual.

    La probabilidad de elegir j desde i es proporcional a:

        A_j^alpha · d_ij^(-beta) · exp(-theta · Wq_estimada_j)

    Devuelve
    --------
    destino : int
        Índice del nodo elegido.
    espera_estimada_destino : float
        Espera estimada del destino elegido en el instante de decisión.
    probabilidades : np.array
        Vector de probabilidades sobre todos los nodos. En el nodo actual
        queda probabilidad 0.
    """

    espera_estimada = espera_estimada_por_cola(
        en_cola=en_cola,
        capacidad_media=capacidad_media
    )

    candidatos = np.array(
        [j for j in range(N_NODOS) if j != i_actual],
        dtype=int
    )

    pesos = (
        popularidad[candidatos] ** alpha
        * distancias[i_actual, candidatos] ** (-beta)
        * np.exp(-theta * espera_estimada[candidatos])
    )

    if np.sum(pesos) <= 0 or np.any(~np.isfinite(pesos)):
        pesos = np.ones(len(candidatos), dtype=float)

    probabilidades_candidatos = pesos / np.sum(pesos)

    destino = int(
        rng.choice(
            candidatos,
            p=probabilidades_candidatos
        )
    )

    probabilidades = np.zeros(N_NODOS, dtype=float)
    probabilidades[candidatos] = probabilidades_candidatos

    return destino, espera_estimada[destino], probabilidades



# ============================================================
# AMPLIACIÓN DEL BLOQUE COMÚN REUTILIZABLE para el MODELO 6
# Perfiles, franjas, decisiones por utilidad y visualización MC
# ============================================================

import math
import numpy as np
import pandas as pd

# ============================================================
# MODELO 6. MODELO 5 + PERFILES + TOLERANCIA A LA ESPERA
#           + MÉTRICAS POR FRANJA, PERFIL Y PERFIL/FRANJA
# ============================================================

PERFILES_MODELO6 = [
    "Familias",
    "Jóvenes",
    "Intensivos",
    "Relajados",
]

PROB_PERFILES_MODELO6 = np.array([0.35, 0.30, 0.20, 0.15], dtype=float)
PROB_PERFILES_MODELO6 = PROB_PERFILES_MODELO6 / PROB_PERFILES_MODELO6.sum()

# theta, beta, alpha y objetivo de atracciones completadas.
THETA_PERFIL_MODELO6 = np.array([0.08, 0.04, 0.10, 0.06], dtype=float)
BETA_PERFIL_MODELO6 = np.array([1.00, 0.60, 0.40, 1.20], dtype=float)
ALPHA_PERFIL_MODELO6 = np.array([1.10, 1.30, 1.40, 1.00], dtype=float)
OBJETIVO_PERFIL_MODELO6 = np.array([4, 6, 7, 3], dtype=int)

# Tolerancia individual a la espera: Triangular(min, moda, max), en minutos.
TOLERANCIA_TRIANGULAR_MODELO6 = np.array([
    [6, 12, 20],
    [10, 20, 35],
    [8, 15, 25],
    [5, 10, 18],
], dtype=float)

# Popularidad A_{k,j}: filas = perfiles, columnas = atracciones.
POPULARIDAD_PERFIL_MODELO6 = np.array([
    [0.80, 1.40, 1.10, 1.50, 0.70, 0.90],
    [1.60, 0.40, 1.20, 0.70, 1.50, 1.30],
    [1.50, 0.50, 1.00, 0.80, 1.30, 1.20],
    [0.90, 1.00, 1.10, 1.30, 0.80, 0.90],
], dtype=float)

# Probabilidad de salida p_k(v,w) = min(p0_k + a_k*v + b_k*w, pmax_k)
# w se mide en minutos. No se divide entre 60.
P0_SALIDA_PERFIL_MODELO6 = np.array([0.06, 0.03, 0.02, 0.08], dtype=float)
A_SALIDA_PERFIL_MODELO6 = np.array([0.05, 0.03, 0.02, 0.06], dtype=float)
B_SALIDA_PERFIL_MODELO6 = np.array([0.20, 0.08, 0.12, 0.12], dtype=float)
PMAX_SALIDA_PERFIL_MODELO6 = np.array([0.45, 0.30, 0.25, 0.50], dtype=float)



# ------------------------------------------------------------
# 1. Utilidades generales para nombres, perfiles y franjas
# ------------------------------------------------------------

def id_texto(x):
    """Convierte una etiqueta en identificador simple para nombres de métricas."""
    return (
        str(x).lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def etiquetas_franjas(franjas):
    """Devuelve etiquetas tipo '0-120' a partir de una lista de franjas (inicio, fin, tasa)."""
    return [f"{int(a)}-{int(b)}" for a, b, *_ in franjas]


def indice_franja(t, franjas):
    """Devuelve el índice de la franja que contiene el instante t."""
    for r, franja in enumerate(franjas):
        inicio, fin = franja[0], franja[1]
        if inicio <= t < fin:
            return r
    return None


def duracion_franja_en_horizonte(r, tiempo_limite, franjas):
    """Duración efectiva de una franja dentro del horizonte simulado."""
    inicio, fin = franjas[r][0], franjas[r][1]
    return max(0.0, min(tiempo_limite, fin) - min(tiempo_limite, inicio))


def recorrer_intervalos_franja(t0, t1, franjas):
    """Genera pares (indice_franja, duracion) al repartir [t0, t1] entre franjas."""
    if t1 <= t0:
        return

    for r, franja in enumerate(franjas):
        inicio, fin = franja[0], franja[1]
        a = max(t0, inicio)
        b = min(t1, fin)
        if b > a:
            yield r, b - a


# ------------------------------------------------------------
# 2. Utilidades generales de muestreo y decisión
# ------------------------------------------------------------

def normalizar_probabilidades(pesos):
    """Normaliza pesos no negativos para obtener probabilidades."""
    pesos = np.asarray(pesos, dtype=float)
    total = pesos.sum()
    if total <= 0:
        raise ValueError("Los pesos deben sumar una cantidad positiva.")
    return pesos / total


def elegir_indice_ponderado(rng, pesos):
    """Elige un índice usando pesos no necesariamente normalizados."""
    prob = normalizar_probabilidades(pesos)
    return int(rng.choice(len(prob), p=prob))


def generar_triangular(rng, parametros):
    """Genera una observación triangular a partir de (mínimo, moda, máximo)."""
    a, m, b = parametros
    return float(rng.triangular(left=a, mode=m, right=b))


def probabilidad_salida_lineal_capada(p0, a, b, pmax, visitas_completadas, espera_acumulada):
    """
    Probabilidad de salida lineal y capada.

    Usa la forma:
        min(p0 + a * visitas_completadas + b * espera_acumulada, pmax)

    La escala de espera_acumulada debe ser coherente con la definición de b.
    """
    p = p0 + a * visitas_completadas + b * espera_acumulada
    return float(min(p, pmax))


def elegir_destino_por_utilidad_tolerancia(
    i_actual,
    en_cola,
    perfil,
    tolerancia,
    popularidad_perfil,
    alpha_perfil,
    beta_perfil,
    theta_perfil,
    distancias,
    capacidad_media,
    espera_estimada_func,
):
    """
    Regla genérica de elección:
      1. calcula utilidad para cada destino distinto del actual;
      2. ordena destinos por utilidad decreciente;
      3. acepta el primer destino con espera estimada <= tolerancia;
      4. si ninguno es tolerable, devuelve como reserva el de menor espera.

    Devuelve:
        destino, espera_estimada_destino, sin_destino_tolerable, vector_utilidad
    """
    n = len(en_cola)
    espera_estimada = espera_estimada_func(en_cola, capacidad_media)
    candidatos = [j for j in range(n) if j != i_actual]

    alpha = alpha_perfil[perfil]
    beta = beta_perfil[perfil]
    theta = theta_perfil[perfil]
    popularidad = popularidad_perfil[perfil]

    utilidad = np.full(n, -np.inf, dtype=float)

    for j in candidatos:
        utilidad[j] = (
            alpha * np.log(popularidad[j])
            - beta * np.log(distancias[i_actual, j])
            - theta * espera_estimada[j]
        )

    candidatos_ordenados = sorted(candidatos, key=lambda j: utilidad[j], reverse=True)

    for j in candidatos_ordenados:
        if espera_estimada[j] <= tolerancia:
            return int(j), float(espera_estimada[j]), False, utilidad

    destino_menor_espera = int(min(candidatos, key=lambda j: espera_estimada[j]))
    return destino_menor_espera, float(espera_estimada[destino_menor_espera]), True, utilidad


# ------------------------------------------------------------
# 3. Tablas Monte Carlo reutilizables
# ------------------------------------------------------------

def tabla_mc_por_nodo(df_mc, nodos=None, patron_metrica=None, mostrar="ic", decimales=3):
    """
    Construye una tabla MC en formato métrica x nodo.

    Requiere que df_mc tenga columnas:
        metrica, nodo, media, error_mc, ic_inf, ic_sup

    Reutiliza formatear_estimacion_mc y estilizar_tabla_mc si ya existen
    en el cuaderno común.
    """
    df = df_mc.copy()

    if nodos is not None:
        df = df[df["nodo"].isin(nodos)].copy()

    if patron_metrica is not None:
        df = df[df["metrica"].str.contains(patron_metrica, regex=False)].copy()

    if df.empty:
        return pd.DataFrame()

    tabla = df.pivot(index="metrica", columns="nodo", values=["media", "error_mc", "ic_inf", "ic_sup"])
    salida = pd.DataFrame(index=tabla.index)

    for nodo in df["nodo"].drop_duplicates():
        salida[nodo] = [
            formatear_estimacion_mc(
                {
                    "media":    tabla.loc[m, ("media",    nodo)],
                    "error_mc": tabla.loc[m, ("error_mc", nodo)],
                    "ic_inf":   tabla.loc[m, ("ic_inf",   nodo)],
                    "ic_sup":   tabla.loc[m, ("ic_sup",   nodo)],
                },
                decimales=decimales,
                mostrar=mostrar,
            )
            for m in tabla.index
        ]

    return estilizar_tabla_mc(salida)


# ------------------------------------------------------------
# 4. Gráficos reutilizables compactos
# ------------------------------------------------------------

def graficar_metricas_en_grid_pastel(df_plot, metricas, titulo, n_cols=4):
    """
    Representa varias métricas en una malla compacta.

    df_plot debe contener:
        metrica, x, grupo, media
    """
    import matplotlib.pyplot as plt

    metricas = list(metricas)
    if len(metricas) == 0:
        print("No hay métricas para representar.")
        return

    n_cols = min(int(n_cols), 4)
    n_rows = math.ceil(len(metricas) / n_cols)
    colores = ["#8DB6CD", "#F4A7A1", "#B8D8BA", "#D7BDE2", "#F7D794", "#A3D2CA"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.1 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, metrica in zip(axes, metricas):
        datos = df_plot[df_plot["metrica"] == metrica]
        for h, (grupo, sub) in enumerate(datos.groupby("grupo", sort=False)):
            ax.plot(
                sub["x"],
                sub["media"],
                marker="o",
                linewidth=2,
                label=grupo,
                color=colores[h % len(colores)],
            )
        ax.set_title(metrica, fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes[len(metricas):]:
        ax.axis("off")

    fig.suptitle(titulo, fontsize=13)
    fig.tight_layout()
    plt.show()


# ============================================================
# FUNCIONES COMUNES AMPLIADAS PARA EL MODELO 7
# Actividades no mecánicas: Restauración, Descanso, Espectáculo
# ============================================================

# ------------------------------------------------------------
# 1. Nodos extendidos (9 nodos: 6 atracciones + 3 actividades)
# ------------------------------------------------------------

NOMBRES_NODOS_M7 = [
    "Montaña Rusa",       # 0
    "Tiovivo",            # 1
    "Flotador Acuático",  # 2
    "Fantasía",           # 3
    "Montaña Fantasma",   # 4
    "Viaje a la Luna",    # 5
    "Restauración",       # 6
    "Descanso",           # 7
    "Espectáculo",        # 8
]

N_NODOS_M7        = len(NOMBRES_NODOS_M7)
N_ATRACCIONES_M7  = 6   # índices 0-5
IDX_RESTAURACION  = 6
IDX_DESCANSO      = 7
IDX_ESPECTACULO   = 8

# ------------------------------------------------------------
# 2. Coordenadas de los 9 nodos y matriz de distancias
# ------------------------------------------------------------

COORDENADAS_M7 = np.array([
    [0.0, 0.0],   # Montaña Rusa
    [2.0, 1.0],   # Tiovivo
    [5.0, 0.0],   # Flotador Acuático
    [1.0, 4.0],   # Fantasía
    [4.0, 4.0],   # Montaña Fantasma
    [6.0, 3.0],   # Viaje a la Luna
    [2.0, 1.5],   # Restauración
    [1.5, 3.5],   # Descanso
    [4.0, 0.0],   # Espectáculo
], dtype=float)

DISTANCIAS_CORREGIDAS_M7 = calcular_matriz_distancias_corregidas(COORDENADAS_M7)

# ------------------------------------------------------------
# 3. Popularidad extendida: 4 perfiles × 9 nodos
# ------------------------------------------------------------

# Columnas 0-5: igual que POPULARIDAD_PERFIL_MODELO6
# Columnas 6-8: Restauración, Descanso, Espectáculo (nuevas)
POPULARIDAD_PERFIL_M7 = np.array([
    # MR    TIV   FLOT  FAN   MF    VL    REST  DESC  ESP
    [0.80, 1.40, 1.10, 1.50, 0.70, 0.90, 1.40, 1.20, 1.50],  # Familias
    [1.60, 0.40, 1.20, 0.70, 1.50, 1.30, 0.80, 0.50, 0.90],  # Jóvenes
    [1.50, 0.50, 1.00, 0.80, 1.30, 1.20, 0.60, 0.40, 0.70],  # Intensivos
    [0.90, 1.00, 1.10, 1.30, 0.80, 0.90, 1.20, 1.50, 1.10],  # Relajados
], dtype=float)

# ------------------------------------------------------------
# 4. Parámetros de utilidad para nodos complementarios
# ------------------------------------------------------------

# delta_rest[k]: cómo crece el atractivo de Restauración por cada
#                atracción completada (v).
# delta_desc[k]: ídem para Descanso.
# phi[k]:        cómo crece el atractivo de Rest/Desc por cada minuto
#                de espera acumulada (w).
# psi[k]:        cómo crece el atractivo del Espectáculo cuando
#                falta poco para la siguiente función (kappa - dt_esp).

DELTA_REST_M7  = np.array([0.12, 0.04, 0.03, 0.10], dtype=float)
DELTA_DESC_M7  = np.array([0.10, 0.02, 0.02, 0.14], dtype=float)
PHI_M7         = np.array([0.04, 0.01, 0.02, 0.05], dtype=float)
PSI_M7         = np.array([0.08, 0.05, 0.04, 0.06], dtype=float)

VENTANA_MAX_ESPECTACULO_M7 = 45.0   # minutos; fuera de ventana → descartado

# ------------------------------------------------------------
# 5. Parámetros operativos de los nodos nuevos
# ------------------------------------------------------------

C_RESTAURACION_M7             = 8
TIEMPO_RESTAURACION_TRI_M7    = (4.0, 8.0, 15.0)   # min, Tri(a,m,b)

CAPACIDAD_DESCANSO_M7         = 80
TIEMPO_DESCANSO_TRI_M7        = (5.0, 10.0, 20.0)  # min, Tri(a,m,b)

HORARIOS_ESPECTACULO_M7       = [120, 240, 360, 480]  # minutos desde apertura
CAPACIDAD_ESPECTACULO_M7      = 150
DURACION_ESPECTACULO_M7       = 20.0                  # minutos

# ------------------------------------------------------------
# 6. Próxima función del espectáculo y tiempo de espera
# ------------------------------------------------------------

def proxima_funcion_espectaculo(t,
                                horarios=HORARIOS_ESPECTACULO_M7,
                                duracion=DURACION_ESPECTACULO_M7):
    """
    Devuelve (indice_funcion, inicio_funcion, minutos_hasta_inicio)
    para la próxima función que aún no ha terminado en el instante t.

    Si no hay función disponible, devuelve (None, None, np.inf).
    """
    for idx, h in enumerate(horarios):
        fin_funcion = h + duracion
        if t < fin_funcion:           # función no ha terminado aún
            return idx, h, max(0.0, h - t)
    return None, None, np.inf


def espectaculo_disponible(t,
                            horarios=HORARIOS_ESPECTACULO_M7,
                            duracion=DURACION_ESPECTACULO_M7,
                            ventana=VENTANA_MAX_ESPECTACULO_M7):
    """
    Devuelve (disponible, idx_funcion, inicio_funcion, dt)
    donde dt es el tiempo hasta la próxima función.

    disponible=True únicamente si 0 <= dt <= ventana.
    Si el visitante ya estaría dentro de la función (dt<0 pero
    fin > t) también se acepta (dt=0).
    """
    idx, inicio, dt = proxima_funcion_espectaculo(t, horarios, duracion)
    if idx is None:
        return False, None, None, np.inf
    disponible = (dt <= ventana)
    return disponible, idx, inicio, dt

# ------------------------------------------------------------
# 7. Utilidad extendida para los 9 nodos
# ------------------------------------------------------------

def calcular_utilidades_m7(
    i_actual,
    en_cola_total,          # array longitud 9; posiciones 6,7 = visitantes en nodo
    perfil,
    visitas_completadas,
    espera_acumulada,
    t_ahora,
    distancias=DISTANCIAS_CORREGIDAS_M7,
    popularidad_perfil=POPULARIDAD_PERFIL_M7,
    alpha_perfil=ALPHA_PERFIL_MODELO6,
    beta_perfil=BETA_PERFIL_MODELO6,
    theta_perfil=THETA_PERFIL_MODELO6,
    capacidad_media_atracciones=CAPACIDAD_MEDIA_APROXIMADA,
    horarios=HORARIOS_ESPECTACULO_M7,
    duracion_esp=DURACION_ESPECTACULO_M7,
    ventana=VENTANA_MAX_ESPECTACULO_M7,
):
    """
    Calcula el vector de utilidades para los 9 nodos desde i_actual.

    Atracciones (0-5):
        U = alpha*log(A) - beta*log(d) - theta*Wq_estimada

    Restauración (6):
        U = alpha*log(A) - beta*log(d)
            + delta_rest*v + phi*w

    Descanso (7):
        U = alpha*log(A) - beta*log(d)
            + delta_desc*v + phi*w

    Espectáculo (8):
        U = alpha*log(A) - beta*log(d)
            + psi * max(0, ventana - dt_esp)
        Si dt_esp > ventana  →  U = -inf (nodo descartado)

    El nodo actual (i_actual) siempre recibe U = -inf.
    """
    alpha  = alpha_perfil[perfil]
    beta   = beta_perfil[perfil]
    theta  = theta_perfil[perfil]
    pop    = popularidad_perfil[perfil]        # longitud 9

    # Espera estimada solo para atracciones (índices 0-5)
    wq_atr = espera_estimada_por_cola(
        en_cola_total[:N_ATRACCIONES_M7],
        capacidad_media_atracciones
    )

    utilidad = np.full(N_NODOS_M7, -np.inf, dtype=float)

    for j in range(N_NODOS_M7):
        if j == i_actual:
            continue

        base = alpha * np.log(max(pop[j], 1e-9)) - beta * np.log(distancias[i_actual, j])

        if j < N_ATRACCIONES_M7:
            # Atracción mecánica
            utilidad[j] = base - theta * wq_atr[j]

        elif j == IDX_RESTAURACION:
            utilidad[j] = (base
                           + DELTA_REST_M7[perfil] * visitas_completadas
                           + PHI_M7[perfil]        * espera_acumulada)

        elif j == IDX_DESCANSO:
            utilidad[j] = (base
                           + DELTA_DESC_M7[perfil] * visitas_completadas
                           + PHI_M7[perfil]        * espera_acumulada)

        elif j == IDX_ESPECTACULO:
            disp, _, _, dt = espectaculo_disponible(t_ahora, horarios, duracion_esp, ventana)
            if not disp:
                utilidad[j] = -np.inf
            else:
                utilidad[j] = base + PSI_M7[perfil] * max(0.0, ventana - dt)

    return utilidad


def elegir_destino_m7(
    i_actual,
    en_cola_total,
    perfil,
    tolerancia,
    visitas_completadas,
    espera_acumulada,
    t_ahora,
    ocupados_espectaculo,        # dict {idx_funcion: plazas_ocupadas}
    capacidad_espectaculo=CAPACIDAD_ESPECTACULO_M7,
    **kwargs,
):
    """
    Elige el destino para el visitante aplicando:
      1. Calcular utilidades de los 9 nodos.
      2. Descartar espectáculo si aforo lleno.
      3. Para atracciones: verificar tolerancia a la espera.
         Para Restauración: tolerancia ilimitada.
         Para Descanso: tolerancia ilimitada.
         Para Espectáculo: sin cola de espera (ya cubierto por ventana).
      4. Elegir el de mayor utilidad tolerable.
         Si ninguna atracción es tolerable pero hay nodo complementario
         disponible, se elige el mejor de estos.
         Si ninguno en absoluto, se devuelve el de menor espera estimada
         entre atracciones (fallback del Modelo 6).

    Devuelve
    --------
    destino          : int
    espera_estimada  : float  (0 para nodos complementarios)
    sin_tolerable    : bool   (True si hubo rechazo por espera en atracciones)
    espectaculo_lleno: bool   (True si se descartó espectáculo por aforo)
    utilidad         : np.array
    """
    utilidad = calcular_utilidades_m7(
        i_actual=i_actual,
        en_cola_total=en_cola_total,
        perfil=perfil,
        visitas_completadas=visitas_completadas,
        espera_acumulada=espera_acumulada,
        t_ahora=t_ahora,
        **kwargs,
    )

    # Verificar aforo espectáculo
    esp_lleno = False
    disp, idx_f, inicio_f, _ = espectaculo_disponible(t_ahora)
    if idx_f is not None:
        plazas_usadas = ocupados_espectaculo.get(idx_f, 0)
        if plazas_usadas >= capacidad_espectaculo:
            utilidad[IDX_ESPECTACULO] = -np.inf
            esp_lleno = True

    # Espera estimada para atracciones
    wq_atr = espera_estimada_por_cola(
        en_cola_total[:N_ATRACCIONES_M7],
        CAPACIDAD_MEDIA_APROXIMADA
    )

    # Candidatos ordenados por utilidad
    candidatos = [j for j in range(N_NODOS_M7)
                  if j != i_actual and np.isfinite(utilidad[j])]
    candidatos.sort(key=lambda j: utilidad[j], reverse=True)

    sin_tolerable = False

    for j in candidatos:
        if j < N_ATRACCIONES_M7:
            if wq_atr[j] <= tolerancia:
                return j, float(wq_atr[j]), False, esp_lleno, utilidad
        else:
            # Nodos complementarios: tolerancia ilimitada
            return j, 0.0, False, esp_lleno, utilidad

    # Ninguno tolerable: registrar frustración y elegir atracción de menor espera
    sin_tolerable = True
    atr_candidatos = [j for j in range(N_ATRACCIONES_M7) if j != i_actual]
    if atr_candidatos:
        fallback = min(atr_candidatos, key=lambda j: wq_atr[j])
        return fallback, float(wq_atr[fallback]), True, esp_lleno, utilidad

    # No debería ocurrir, pero por seguridad:
    return 0, 0.0, True, esp_lleno, utilidad



# ============================================================
# FUNCIONES GENÉRICAS DE VISUALIZACIÓN Y COMPARACIÓN
# Añadir al bloque "Funciones comunes"
# ============================================================

# ------------------------------------------------------------
# A. Gráfico de métricas globales filtradas por grupo de nodos
# ------------------------------------------------------------

def graficar_global_por_grupo(
    resultado_mc,
    nodos,
    metricas=None,
    titulo="Métricas globales",
    n_cols=4,
):
    """
    Representa métricas globales para un conjunto de nodos dado.

    Parámetros
    ----------
    resultado_mc : dict
        Resultado agregado de Monte Carlo con clave "globales".
    nodos : list of str
        Etiquetas de nodo a representar (franjas, perfiles,
        combinaciones perfil/franja, etc.).
    metricas : list of str, opcional
        Métricas a representar. Si None, se usan todas las
        presentes en resultado_mc["globales"] para esos nodos.
    titulo : str
        Título del gráfico.
    n_cols : int
        Número máximo de columnas en la cuadrícula.
    """
    df = resultado_mc["globales"].copy()
    df = df[df["nodo"].isin(nodos)].copy()

    if df.empty:
        print(f"graficar_global_por_grupo: sin datos para nodos={nodos}")
        return

    if metricas is None:
        metricas = sorted(df["metrica"].unique().tolist())
    else:
        metricas = [m for m in metricas if m in df["metrica"].values]

    if not metricas:
        print("graficar_global_por_grupo: ninguna métrica disponible.")
        return

    # Construir df_plot con columnas x, grupo, metrica, media
    df_plot = df[df["metrica"].isin(metricas)].copy()
    df_plot["x"]     = df_plot["nodo"]
    df_plot["grupo"] = titulo  # un solo grupo en este caso

    graficar_metricas_en_grid_pastel(
        df_plot, metricas, titulo=titulo, n_cols=n_cols
    )


# ------------------------------------------------------------
# B. Separación automática tamaños / tiempos
# ------------------------------------------------------------

# Patrones que identifican métricas de "tamaño" (L, Lq, N, tasas)
# y métricas de "tiempo" (W, Wq, permanencia, movimiento).
_PATRONES_TAMANO = [
    "numero_medio", "tasa_", "llegadas", "salidas",
    "rechazos", "atracciones_completadas", "funciones_",
    "ocupacion_", "porcentaje_"
]
_PATRONES_TIEMPO = [
    "tiempo_medio", "permanencia", "espera", "movimiento"
]


def _clasificar_metrica(nombre):
    """Devuelve 'tamano', 'tiempo' o 'otro'."""
    n = nombre.lower()
    for p in _PATRONES_TIEMPO:
        if p in n:
            return "tiempo"
    for p in _PATRONES_TAMANO:
        if p in n:
            return "tamano"
    return "otro"


def graficar_global_tamanos_tiempos(
    resultado_mc,
    nodos,
    titulo_base="Métricas globales",
    metricas=None,
    n_cols=4,
):
    """
    Separa automáticamente las métricas en dos grupos
    (tamaños/tasas y tiempos) y genera un gráfico por grupo.

    Parámetros
    ----------
    resultado_mc : dict
    nodos : list of str
        Etiquetas de nodo a representar.
    titulo_base : str
        Prefijo del título; se añade "— Tamaños" y "— Tiempos".
    metricas : list of str, opcional
        Si None, se usan todas las métricas disponibles.
    n_cols : int
    """
    df = resultado_mc["globales"].copy()
    df = df[df["nodo"].isin(nodos)].copy()

    if df.empty:
        print(f"graficar_global_tamanos_tiempos: sin datos para nodos={nodos}")
        return

    if metricas is None:
        metricas = sorted(df["metrica"].unique().tolist())

    metricas_tamano = [m for m in metricas if _clasificar_metrica(m) in ("tamano", "otro")]
    metricas_tiempo = [m for m in metricas if _clasificar_metrica(m) == "tiempo"]

    for grupo, lista in [("Tamaños y tasas", metricas_tamano),
                         ("Tiempos",         metricas_tiempo)]:
        lista = [m for m in lista if m in df["metrica"].values]
        if not lista:
            continue
        df_plot = df[df["metrica"].isin(lista)].copy()
        df_plot["x"]     = df_plot["nodo"]
        df_plot["grupo"] = grupo
        graficar_metricas_en_grid_pastel(
            df_plot, lista,
            titulo=f"{titulo_base} — {grupo}",
            n_cols=n_cols,
        )


# ------------------------------------------------------------
# C. Comparación genérica entre dos modelos
# ------------------------------------------------------------
EQUIVALENCIAS_METRICAS = {
    "tasa_efectiva_llegada":        ["lambda_efectiva"],
    "utilizacion":                  ["rho"],
    "numero_medio_en_atraccion":    ["L"],
    "numero_medio_en_cola":         ["Lq"],
    "tiempo_medio_total_atraccion": ["W"],
    "tiempo_medio_espera_cola":     ["Wq"],
    "numero_esperado_visitas":      ["visitas_por_cliente_externo"],
    "numero_medio_visitantes_parque":       ["L_global"],
    "numero_medio_visitantes_colas_parque": ["Lq_global"],
    "tiempo_medio_permanencia_parque":      ["W_global"],
    "tiempo_medio_espera_colas_parque":     ["Wq_global"],
}

def diagnosticar_metricas_comunes(
    resultado_a,
    nombre_a,
    resultado_b,
    nombre_b,
    normalizar=True,
    incluir_franjas=True,
    incluir_perfiles=False,
):
    """
    Imprime un diagnóstico detallado de las métricas y nodos
    comunes entre dos modelos, y las exclusivas de cada uno.

    Útil para decidir qué se puede comparar antes de llamar
    a comparar_dos_modelos() o comparar_modelos().

    Parámetros
    ----------
    resultado_a, resultado_b : dict
        Resultados MC con claves "locales" y "globales".
    nombre_a, nombre_b : str
    normalizar : bool
        Si True aplica EQUIVALENCIAS_METRICAS antes de comparar.
    incluir_franjas : bool
        Si True incluye métricas con sufijo _franja_ en el análisis.
    incluir_perfiles : bool
        Si False (por defecto) excluye métricas con sufijo de perfil.

    Devuelve
    --------
    dict con claves:
        "locales_comunes", "locales_solo_a", "locales_solo_b",
        "globales_comunes", "globales_solo_a", "globales_solo_b",
        "nodos_comunes", "nodos_solo_a", "nodos_solo_b"
    """
    def preparar(df):
        if not normalizar:
            return df
        df = df.copy()
        alias_a_canonico = {
            alias: canonico
            for canonico, aliases in EQUIVALENCIAS_METRICAS.items()
            for alias in aliases
        }
        def norm(nombre):
            if "_franja_" in nombre:
                base, suf = nombre.split("_franja_", 1)
                return f"{alias_a_canonico.get(base, base)}_franja_{suf}"
            return alias_a_canonico.get(nombre, nombre)
        df["metrica"] = df["metrica"].apply(norm)
        return df

    def filtrar_metricas(df):
        metricas = set(df["metrica"].unique())
        if not incluir_franjas:
            metricas = {m for m in metricas if "_franja_" not in m}
        if not incluir_perfiles:
            metricas = {
                m for m in metricas
                if not any(
                    id_texto(p) in m
                    for p in ["Familias", "Jóvenes", "Intensivos", "Relajados"]
                )
            }
        return metricas

    loc_a = preparar(resultado_a["locales"])
    loc_b = preparar(resultado_b["locales"])
    glo_a = preparar(resultado_a["globales"])
    glo_b = preparar(resultado_b["globales"])

    met_loc_a = filtrar_metricas(loc_a)
    met_loc_b = filtrar_metricas(loc_b)
    met_glo_a = filtrar_metricas(glo_a)
    met_glo_b = filtrar_metricas(glo_b)

    nodos_a = set(loc_a["nodo"].unique())
    nodos_b = set(loc_b["nodo"].unique())

    loc_comunes  = sorted(met_loc_a & met_loc_b)
    loc_solo_a   = sorted(met_loc_a - met_loc_b)
    loc_solo_b   = sorted(met_loc_b - met_loc_a)
    glo_comunes  = sorted(met_glo_a & met_glo_b)
    glo_solo_a   = sorted(met_glo_a - met_glo_b)
    glo_solo_b   = sorted(met_glo_b - met_glo_a)
    nodos_comunes = sorted(nodos_a & nodos_b)
    nodos_solo_a  = sorted(nodos_a - nodos_b)
    nodos_solo_b  = sorted(nodos_b - nodos_a)

    sep = "─" * 60

    print(sep)
    print(f"DIAGNÓSTICO: {nombre_a}  vs  {nombre_b}")
    print(f"  normalización aplicada : {normalizar}")
    print(f"  incluir franjas        : {incluir_franjas}")
    print(f"  incluir perfiles       : {incluir_perfiles}")
    print(sep)

    print(f"\n{'NODOS (locales)'}")
    print(f"  Comunes ({len(nodos_comunes)}): {nodos_comunes}")
    print(f"  Solo en {nombre_a} ({len(nodos_solo_a)}): {nodos_solo_a}")
    print(f"  Solo en {nombre_b} ({len(nodos_solo_b)}): {nodos_solo_b}")

    print(f"\n{'MÉTRICAS LOCALES'}")
    print(f"  Comunes ({len(loc_comunes)}):")
    for m in loc_comunes:
        print(f"    ✓ {m}")
    if loc_solo_a:
        print(f"  Solo en {nombre_a} ({len(loc_solo_a)}):")
        for m in loc_solo_a:
            print(f"    · {m}")
    if loc_solo_b:
        print(f"  Solo en {nombre_b} ({len(loc_solo_b)}):")
        for m in loc_solo_b:
            print(f"    · {m}")

    print(f"\n{'MÉTRICAS GLOBALES'}")
    print(f"  Comunes ({len(glo_comunes)}):")
    for m in glo_comunes:
        print(f"    ✓ {m}")
    if glo_solo_a:
        print(f"  Solo en {nombre_a} ({len(glo_solo_a)}):")
        for m in glo_solo_a:
            print(f"    · {m}")
    if glo_solo_b:
        print(f"  Solo en {nombre_b} ({len(glo_solo_b)}):")
        for m in glo_solo_b:
            print(f"    · {m}")

    print(sep)

    return {
        "locales_comunes":  loc_comunes,
        "locales_solo_a":   loc_solo_a,
        "locales_solo_b":   loc_solo_b,
        "globales_comunes": glo_comunes,
        "globales_solo_a":  glo_solo_a,
        "globales_solo_b":  glo_solo_b,
        "nodos_comunes":    nodos_comunes,
        "nodos_solo_a":     nodos_solo_a,
        "nodos_solo_b":     nodos_solo_b,
    }


def comparar_dos_modelos(
    resultado_a,
    nombre_a,
    resultado_b,
    nombre_b,
    nodos_locales=None,
    metricas_locales=None,
    metricas_globales=None,
    nodo_global="Global",
    n_cols=4,
):
    """
    Compara dos modelos sobre métricas y nodos comunes.

    Detección automática (Opción A):
    - Nodos locales comunes: intersección de los nodos presentes
      en ambos df["locales"] para las métricas seleccionadas.
    - Métricas locales comunes: intersección de nombres de métrica
      en ambos df["locales"].
    - Métricas globales comunes: intersección en df["globales"]
      para nodo_global.

    Los parámetros nodos_locales, metricas_locales y
    metricas_globales permiten sobreescribir la detección
    automática cuando se desee.

    Parámetros
    ----------
    resultado_a, resultado_b : dict
        Resultados MC de los dos modelos.
    nombre_a, nombre_b : str
        Etiquetas para leyenda.
    nodos_locales : list of str, opcional
        Si None, se calcula la intersección automáticamente.
    metricas_locales : list of str, opcional
        Si None, se calcula la intersección automáticamente.
    metricas_globales : list of str, opcional
        Si None, se calcula la intersección automáticamente.
    nodo_global : str
        Etiqueta del nodo global (por defecto "Global").
    n_cols : int
        Columnas en la cuadrícula de gráficos locales.
    """
    import matplotlib.pyplot as plt

    loc_a = resultado_a["locales"]
    loc_b = resultado_b["locales"]
    glo_a = resultado_a["globales"]
    glo_b = resultado_b["globales"]

    colores = ["#8DB6CD", "#F4A7A1"]

    # ── Métricas locales comunes ─────────────────────────────
    if metricas_locales is None:
        metricas_locales = sorted(
            set(loc_a["metrica"].unique()) &
            set(loc_b["metrica"].unique())
        )
    # Filtrar solo métricas "base" (sin sufijos de franja/perfil)
    # para que la comparación sea legible
    metricas_locales = [
        m for m in metricas_locales
        if "franja" not in m and not any(
            m.endswith(f"_{id_texto(p)}")
            for p in ["Familias", "Jóvenes", "Intensivos", "Relajados",
                      "familias", "jovenes", "intensivos", "relajados"]
        )
    ]

    # ── Nodos locales comunes ────────────────────────────────
    if nodos_locales is None:
        nodos_a = set(loc_a[loc_a["metrica"].isin(metricas_locales)]["nodo"].unique())
        nodos_b = set(loc_b[loc_b["metrica"].isin(metricas_locales)]["nodo"].unique())
        nodos_locales = sorted(nodos_a & nodos_b)

    # ── Métricas globales comunes ────────────────────────────
    if metricas_globales is None:
        glo_a_g = glo_a[glo_a["nodo"] == nodo_global]
        glo_b_g = glo_b[glo_b["nodo"] == nodo_global]
        metricas_globales = sorted(
            set(glo_a_g["metrica"].unique()) &
            set(glo_b_g["metrica"].unique())
        )

    # ── Gráfico local ────────────────────────────────────────
    if metricas_locales and nodos_locales:
        n_m   = len(metricas_locales)
        n_c   = min(n_cols, n_m)
        n_r   = math.ceil(n_m / n_c)
        fig, axes = plt.subplots(n_r, n_c,
                                 figsize=(4.2 * n_c, 3.2 * n_r),
                                 squeeze=False)
        axes = axes.flatten()
        x    = np.arange(len(nodos_locales))
        etq  = [nd.replace(" ", "\n") for nd in nodos_locales]

        for idx, metrica in enumerate(metricas_locales):
            ax = axes[idx]
            for color, nombre, resultado in [
                (colores[0], nombre_a, resultado_a),
                (colores[1], nombre_b, resultado_b),
            ]:
                vals = (
                    resultado["locales"][resultado["locales"]["metrica"] == metrica]
                    .set_index("nodo")
                    .reindex(nodos_locales)["media"]
                )
                ax.plot(x, vals, marker="o", linewidth=2,
                        color=color, label=nombre)
            ax.set_title(metrica, fontsize=9, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(etq, fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)

        for j in range(n_m, len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            f"Comparación local: {nombre_a} vs {nombre_b}",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        plt.show()
    else:
        print("comparar_dos_modelos: sin métricas o nodos locales comunes.")

    # ── Gráfico global ───────────────────────────────────────
    if metricas_globales:
        # Separar tamaños y tiempos automáticamente
        met_tam = [m for m in metricas_globales
                   if _clasificar_metrica(m) in ("tamano", "otro")]
        met_tpo = [m for m in metricas_globales
                   if _clasificar_metrica(m) == "tiempo"]

        for subtitulo, lista in [("Tamaños y tasas", met_tam),
                                  ("Tiempos",        met_tpo)]:
            if not lista:
                continue
            fig, ax = plt.subplots(
                figsize=(max(8, len(lista) * 1.5), 4.5)
            )
            x = np.arange(len(lista))
            for i, (color, nombre, resultado) in enumerate([
                (colores[0], nombre_a, resultado_a),
                (colores[1], nombre_b, resultado_b),
            ]):
                vals = (
                    resultado["globales"][
                        (resultado["globales"]["metrica"].isin(lista)) &
                        (resultado["globales"]["nodo"] == nodo_global)
                    ]
                    .set_index("metrica")
                    .reindex(lista)["media"]
                )
                ax.bar(x + (i - 0.5) * 0.35, vals,
                       width=0.35, color=color, label=nombre)
            ax.set_xticks(x)
            ax.set_xticklabels(lista, rotation=30, ha="right", fontsize=9)
            ax.set_title(
                f"Comparación global — {subtitulo}: {nombre_a} vs {nombre_b}",
                fontsize=12, fontweight="bold"
            )
            ax.legend()
            ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.show()
    else:
        print("comparar_dos_modelos: sin métricas globales comunes.")


def graficar_global_perfil_x_franja(
    resultado_mc,
    perfiles,
    franjas,
    metricas=None,
    titulo="Métricas globales por perfil y franja",
    n_cols=4,
):
    """
    Representa métricas globales con franjas en el eje X
    y un color/línea por perfil de visitante.
 
    Los nodos en df["globales"] deben tener la forma "Perfil/franja"
    (p.ej. "Familias/60-180").
 
    Parámetros
    ----------
    resultado_mc : dict
    perfiles : list of str   — etiquetas de perfil (definen los colores)
    franjas  : list of str   — etiquetas de franja  (definen el eje X)
    metricas : list of str, opcional
    titulo   : str
    n_cols   : int
    """
    import matplotlib.pyplot as plt
 
    df = resultado_mc["globales"].copy()
 
    # Construir el conjunto de nodos "Perfil/franja" esperados
    nodos_pf = [f"{p}/{fr}" for p in perfiles for fr in franjas]
    df = df[df["nodo"].isin(nodos_pf)].copy()
 
    if df.empty:
        print("graficar_global_perfil_x_franja: sin datos.")
        return
 
    # Separar perfil y franja desde la columna nodo
    df[["perfil_etiq", "franja_etiq"]] = df["nodo"].str.split("/", n=1, expand=True)
 
    if metricas is None:
        metricas = sorted(df["metrica"].unique().tolist())
    else:
        metricas = [m for m in metricas if m in df["metrica"].values]
 
    if not metricas:
        print("graficar_global_perfil_x_franja: ninguna métrica disponible.")
        return
 
    n_cols  = min(int(n_cols), len(metricas))
    n_rows  = math.ceil(len(metricas) / n_cols)
    colores = ["#8DB6CD", "#F4A7A1", "#B8D8BA", "#D7BDE2",
               "#F7D794", "#A3D2CA", "#F0B8A0", "#C5B4E3"]
 
    x_pos   = {fr: i for i, fr in enumerate(franjas)}
    x_ticks = list(range(len(franjas)))
 
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.4 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    axes = axes.flatten()
 
    for idx, metrica in enumerate(metricas):
        ax  = axes[idx]
        df_m = df[df["metrica"] == metrica]
 
        for k, perfil in enumerate(perfiles):
            df_p = (
                df_m[df_m["perfil_etiq"] == perfil]
                .set_index("franja_etiq")
                .reindex(franjas)
            )
            ax.plot(
                x_ticks,
                df_p["media"].values,
                marker="o",
                linewidth=2,
                color=colores[k % len(colores)],
                label=perfil,
            )
 
        ax.set_title(metrica, fontsize=9, fontweight="bold")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(franjas, fontsize=8, rotation=30, ha="right")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
 
    for j in range(len(metricas), len(axes)):
        axes[j].axis("off")
 
    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
