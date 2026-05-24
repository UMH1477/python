from urllib.request import urlretrieve
url = "https://raw.githubusercontent.com/UMH1477/python/refs/heads/main/parque_atracciones/parque_comunes.py"
urlretrieve(url, 'parque_comunes.py')
from parque_comunes import *      

# ============================================================
# MODELO 1. RED DE JACKSON M/M/s
# Versión adaptada a parámetros y funciones genéricas
# ============================================================

import simpy
import numpy as np
import pandas as pd
import math


# ------------------------------------------------------------
# 1. Parámetros específicos del Modelo 1
# ------------------------------------------------------------

# En el Modelo 1 cada atracción se aproxima como una cola M/M/s.
# La capacidad de servidores s_i coincide con la capacidad total
# simultánea de la atracción.
S_MODELO1 = CAPACIDAD_TOTAL.copy()

# Servicios exponenciales con media igual a la duración base
# de la atracción.
MU_MODELO1 = 1 / DURACION_ATRACCION


# ------------------------------------------------------------
# 2. Métricas teóricas del Modelo 1
# ------------------------------------------------------------

def calcular_teoricas_modelo1(gamma_hora=600):
    """
    Calcula las métricas teóricas del Modelo 1.

    Hipótesis:
        - Red abierta de Jackson.
        - Llegadas externas Poisson.
        - Cada nodo es M/M/s.
        - Servicios exponenciales.
        - Encaminamiento uniforme:
              tras cada atracción, el visitante elige entre
              las otras 5 atracciones y la salida.
    """

    gamma_total, gamma_vec, P = parametros_encaminamiento_uniforme(
        gamma_hora=gamma_hora
    )

    lambdas = resolver_ecuaciones_trafico(
        gamma_vec=gamma_vec,
        P=P
    )

    filas_locales = []

    for i, nombre in enumerate(NOMBRES_NODOS):

        metricas = metricas_mm_s(
            lambd=lambdas[i],
            mu=MU_MODELO1[i],
            s=int(S_MODELO1[i])
        )

        metricas["visitas_por_cliente_externo"] = (
            lambdas[i] / gamma_total
            if gamma_total > 0
            else np.nan
        )

        for metrica, valor in metricas.items():
            filas_locales.append({
                "metrica": metrica,
                "nodo": nombre,
                "valor_teorico": valor
            })

    teoricas_locales = pd.DataFrame(filas_locales)

    L_global = teoricas_locales.loc[
        teoricas_locales["metrica"] == "L",
        "valor_teorico"
    ].sum()

    Lq_global = teoricas_locales.loc[
        teoricas_locales["metrica"] == "Lq",
        "valor_teorico"
    ].sum()

    W_global = L_global / gamma_total if gamma_total > 0 else np.nan
    Wq_global = Lq_global / gamma_total if gamma_total > 0 else np.nan

    teoricas_globales = pd.DataFrame({
        "metrica": [
            "L_global",
            "Lq_global",
            "W_global",
            "Wq_global"
        ],
        "nodo": ["Global"] * 4,
        "valor_teorico": [
            L_global,
            Lq_global,
            W_global,
            Wq_global
        ]
    })

    return {
        "locales": teoricas_locales,
        "globales": teoricas_globales,
        "lambdas": lambdas,
        "P": P,
        "gamma_vec": gamma_vec,
        "gamma_total": gamma_total
    }


# ------------------------------------------------------------
# 3. Simulación de una réplica del Modelo 1
# ------------------------------------------------------------

def simular_modelo1(
    tiempo_limite=600,
    semilla=123,
    gamma_hora=600
):
    """
    Simula una réplica del Modelo 1.

    Características:
        - red abierta de Jackson simulada con SimPy;
        - llegadas externas Poisson;
        - servicios exponenciales individuales;
        - cada nodo se modela como M/M/s;
        - no hay servicio por lotes;
        - no hay embarque/desembarque explícito;
        - no hay desplazamientos;
        - no hay abandonos.
    """

    rng = np.random.default_rng(semilla)

    env = simpy.Environment()

    nombres = NOMBRES_NODOS
    n = N_NODOS

    gamma_total, gamma_vec, P = parametros_encaminamiento_uniforme(
        gamma_hora=gamma_hora
    )

    recursos = [
        simpy.Resource(env, capacity=int(S_MODELO1[i]))
        for i in range(n)
    ]

    # --------------------------------------------------------
    # Contadores locales
    # --------------------------------------------------------

    llegadas = np.zeros(n, dtype=int)
    salidas = np.zeros(n, dtype=int)
    llegadas_externas = np.zeros(n, dtype=int)

    suma_W = np.zeros(n)
    suma_Wq = np.zeros(n)

    area_L = np.zeros(n)
    area_Lq = np.zeros(n)

    ultimo_t = np.zeros(n)

    num_sistema = np.zeros(n, dtype=int)
    num_cola = np.zeros(n, dtype=int)

    ocupacion = np.zeros(n)
    ultimo_t_ocupacion = np.zeros(n)

    # --------------------------------------------------------
    # Contadores globales
    # --------------------------------------------------------

    total_entran_parque = 0
    total_salen_parque = 0

    suma_permanencia_parque = 0.0
    suma_espera_parque = 0.0

    # --------------------------------------------------------
    # Actualización de áreas para L y Lq
    # --------------------------------------------------------

    def actualizar_areas(i):
        """
        Actualiza las áreas bajo las curvas:
            - número en sistema;
            - número en cola.

        Se llama justo antes de cambiar el estado del nodo.
        """

        ahora = env.now
        dt = ahora - ultimo_t[i]

        if dt > 0:
            area_L[i] += num_sistema[i] * dt
            area_Lq[i] += num_cola[i] * dt
            ultimo_t[i] = ahora

    # --------------------------------------------------------
    # Actualización de ocupación para rho
    # --------------------------------------------------------

    def actualizar_ocupacion(i):
        """
        Actualiza el área de servidores ocupados.

        La utilización se estima como:

            área de servidores ocupados / (s_i * tiempo_limite)
        """

        ahora = env.now
        dt = ahora - ultimo_t_ocupacion[i]

        if dt > 0:
            ocupacion[i] += recursos[i].count * dt
            ultimo_t_ocupacion[i] = ahora

    # --------------------------------------------------------
    # Proceso visitante
    # --------------------------------------------------------

    def visitante(i, t_entrada_parque, espera_acumulada):
        """
        Proceso que representa a un visitante en una atracción.
        """

        nonlocal total_salen_parque
        nonlocal suma_permanencia_parque
        nonlocal suma_espera_parque

        # ----------------------------------------------------
        # Llegada al nodo
        # ----------------------------------------------------

        actualizar_areas(i)

        num_sistema[i] += 1
        llegadas[i] += 1

        t_llegada_nodo = env.now

        recurso = recursos[i]

        # Si todos los servidores están ocupados,
        # el visitante entra en cola.
        en_cola = recurso.count >= recurso.capacity

        if en_cola:
            num_cola[i] += 1

        with recurso.request() as req:

            yield req

            # ------------------------------------------------
            # Inicio de servicio
            # ------------------------------------------------

            actualizar_areas(i)
            actualizar_ocupacion(i)

            if en_cola:
                num_cola[i] -= 1

            espera = env.now - t_llegada_nodo

            suma_Wq[i] += espera
            espera_acumulada += espera

            # Servicio exponencial M/M/s
            tiempo_servicio = rng.exponential(
                1 / MU_MODELO1[i]
            )

            yield env.timeout(tiempo_servicio)

            # ------------------------------------------------
            # Fin de servicio
            # ------------------------------------------------

            actualizar_ocupacion(i)

            salidas[i] += 1
            suma_W[i] += env.now - t_llegada_nodo

            actualizar_areas(i)
            num_sistema[i] -= 1

        # ----------------------------------------------------
        # Encaminamiento uniforme común
        # ----------------------------------------------------

        siguiente = elegir_siguiente_nodo_uniforme(
            i_actual=i,
            rng=rng
        )

        if siguiente == -1:

            total_salen_parque += 1

            suma_permanencia_parque += (
                env.now - t_entrada_parque
            )

            suma_espera_parque += espera_acumulada

        else:

            env.process(
                visitante(
                    i=siguiente,
                    t_entrada_parque=t_entrada_parque,
                    espera_acumulada=espera_acumulada
                )
            )

    # --------------------------------------------------------
    # Generadores de llegadas externas
    # --------------------------------------------------------

    def generador_externo(i):
        """
        Genera llegadas externas al nodo i.
        """

        nonlocal total_entran_parque

        while True:

            tiempo_entre_llegadas = rng.exponential(
                1 / gamma_vec[i]
            )

            yield env.timeout(tiempo_entre_llegadas)

            if env.now > tiempo_limite:
                break

            llegadas_externas[i] += 1
            total_entran_parque += 1

            env.process(
                visitante(
                    i=i,
                    t_entrada_parque=env.now,
                    espera_acumulada=0.0
                )
            )

    for i in range(n):
        env.process(generador_externo(i))

    # --------------------------------------------------------
    # Ejecución de la réplica
    # --------------------------------------------------------

    env.run(until=tiempo_limite)

    # Cerramos las áreas hasta el horizonte de simulación.
    for i in range(n):
        actualizar_areas(i)
        actualizar_ocupacion(i)

    # --------------------------------------------------------
    # Métricas locales
    # --------------------------------------------------------

    filas_locales = []

    for i, nombre in enumerate(nombres):

        metricas = {
            "lambda_efectiva": llegadas[i] / tiempo_limite,

            "tasa_salida": salidas[i] / tiempo_limite,

            "rho": (
                ocupacion[i] / (S_MODELO1[i] * tiempo_limite)
                if S_MODELO1[i] > 0
                else np.nan
            ),

            "L": area_L[i] / tiempo_limite,

            "Lq": area_Lq[i] / tiempo_limite,

            "W": (
                suma_W[i] / salidas[i]
                if salidas[i] > 0
                else np.nan
            ),

            "Wq": (
                suma_Wq[i] / salidas[i]
                if salidas[i] > 0
                else np.nan
            ),

            "visitas_por_cliente_externo": (
                llegadas[i] / total_entran_parque
                if total_entran_parque > 0
                else np.nan
            )
        }

        for metrica, valor in metricas.items():
            filas_locales.append({
                "metrica": metrica,
                "nodo": nombre,
                "valor": valor
            })

    metricas_locales = pd.DataFrame(filas_locales)

    # --------------------------------------------------------
    # Métricas globales
    # --------------------------------------------------------

    metricas_globales = pd.DataFrame({
        "metrica": [
            "L_global",
            "Lq_global",
            "W_global",
            "Wq_global"
        ],
        "nodo": ["Global"] * 4,
        "valor": [
            area_L.sum() / tiempo_limite,

            area_Lq.sum() / tiempo_limite,

            (
                suma_permanencia_parque / total_salen_parque
                if total_salen_parque > 0
                else np.nan
            ),

            (
                suma_espera_parque / total_salen_parque
                if total_salen_parque > 0
                else np.nan
            )
        ]
    })

    return {
        "locales": metricas_locales,
        "globales": metricas_globales,
        "entradas_parque": total_entran_parque,
        "salidas_parque": total_salen_parque
    }


# ------------------------------------------------------------
# 4. Simulación Monte Carlo del Modelo 1
# ------------------------------------------------------------

def ejecutar_monte_carlo_modelo1(
    n_replicas=30,
    tiempo_limite=600,
    gamma_hora=600,
    semilla_inicial=1000,
    confianza=0.95
):
    """
    Ejecuta varias réplicas independientes del Modelo 1
    y agrega las métricas mediante estimación Monte Carlo.
    """

    resultados_replicas = []

    for r in range(n_replicas):

        resultado_r = simular_modelo1(
            tiempo_limite=tiempo_limite,
            semilla=semilla_inicial + r,
            gamma_hora=gamma_hora
        )

        resultados_replicas.append(resultado_r)

    resultado_mc = agregar_replicas(
        resultados_replicas=resultados_replicas,
        confianza=confianza
    )

    return resultado_mc


# ------------------------------------------------------------
# 5. Comparación de una réplica con resultados teóricos
# ------------------------------------------------------------

def comparar_una_replica_con_teoria_modelo1(
    tiempo_limite=600,
    semilla=123,
    gamma_hora=600
):
    """
    Compara una réplica simulada con las métricas teóricas M/M/s.
    """

    simulacion = simular_modelo1(
        tiempo_limite=tiempo_limite,
        semilla=semilla,
        gamma_hora=gamma_hora
    )

    teoricas = calcular_teoricas_modelo1(
        gamma_hora=gamma_hora
    )

    locales = simulacion["locales"].merge(
        teoricas["locales"],
        on=["metrica", "nodo"],
        how="left"
    )

    locales = locales.rename(
        columns={"valor": "valor_simulado"}
    )

    globales = simulacion["globales"].merge(
        teoricas["globales"],
        on=["metrica", "nodo"],
        how="left"
    )

    globales = globales.rename(
        columns={"valor": "valor_simulado"}
    )

    return {
        "locales": locales,
        "globales": globales,
        "simulacion": simulacion,
        "teoricas": teoricas
    }
