# ============================================================
# MODELO 6. MODELO 5 + PERFILES + TOLERANCIA A LA ESPERA
#           + MÉTRICAS POR FRANJA, PERFIL Y PERFIL/FRANJA
# ============================================================
from urllib.request import urlretrieve
url = "https://raw.githubusercontent.com/UMH1477/python/refs/heads/main/parque_atracciones/parque_comunes.py"
urlretrieve(url, 'parque_comunes.py')
from parque_comunes import *  

import importlib
from urllib.request import urlretrieve

for i in range(1,6):
  model_name = f'parque_modelo{i}'
  url = f"https://raw.githubusercontent.com/UMH1477/python/refs/heads/main/parque_atracciones/{model_name}.py"
  urlretrieve(url, f'{model_name}.py')
  # Dynamically import all contents of the module into the current namespace
  # This is equivalent to 'from module_name import *'
  exec(f"from {model_name} import *", globals())
    
import simpy
import numpy as np
import pandas as pd

# Requiere tener ejecutado previamente:
#   1. Bloque común hasta Modelo 5.
#   2. Ampliación común: modelo6_bloque_comun_ampliacion.py


# ------------------------------------------------------------
# 1. Parámetros específicos del Modelo 6
# ------------------------------------------------------------

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
# 2. Funciones específicas del Modelo 6
# ------------------------------------------------------------

def etiquetas_franjas_modelo6(franjas=FRANJAS_LLEGADAS_MODELO5):
    return etiquetas_franjas(franjas)


def indice_franja_modelo6(t, franjas=FRANJAS_LLEGADAS_MODELO5):
    return indice_franja(t, franjas)


def elegir_perfil_modelo6(rng):
    return elegir_indice_ponderado(rng, PROB_PERFILES_MODELO6)


def tolerancia_espera_modelo6(perfil, rng):
    return generar_triangular(rng, TOLERANCIA_TRIANGULAR_MODELO6[perfil])


def elegir_entrada_inicial_modelo6(perfil, rng):
    return elegir_indice_ponderado(rng, POPULARIDAD_PERFIL_MODELO6[perfil])


def probabilidad_salida_modelo6(perfil, visitas_completadas, espera_acumulada):
    return probabilidad_salida_lineal_capada(
        p0=P0_SALIDA_PERFIL_MODELO6[perfil],
        a=A_SALIDA_PERFIL_MODELO6[perfil],
        b=B_SALIDA_PERFIL_MODELO6[perfil],
        pmax=PMAX_SALIDA_PERFIL_MODELO6[perfil],
        visitas_completadas=visitas_completadas,
        espera_acumulada=espera_acumulada,
    )


def elegir_destino_modelo6(
    i_actual,
    en_cola,
    perfil,
    tolerancia,
    rng=None,
    distancias=DISTANCIAS_CORREGIDAS_MODELO5,
    capacidad_media=CAPACIDAD_MEDIA_APROXIMADA,
):
    return elegir_destino_por_utilidad_tolerancia(
        i_actual=i_actual,
        en_cola=en_cola,
        perfil=perfil,
        tolerancia=tolerancia,
        popularidad_perfil=POPULARIDAD_PERFIL_MODELO6,
        alpha_perfil=ALPHA_PERFIL_MODELO6,
        beta_perfil=BETA_PERFIL_MODELO6,
        theta_perfil=THETA_PERFIL_MODELO6,
        distancias=distancias,
        capacidad_media=capacidad_media,
        espera_estimada_func=espera_estimada_por_cola,
    )


# ------------------------------------------------------------
# 3. Estadísticas del Modelo 6
# ------------------------------------------------------------

class EstadisticasModelo6:
    def __init__(self, env, nombres, capacidad_total, unidades_paralelas,
                 perfiles=PERFILES_MODELO6,
                 franjas=FRANJAS_LLEGADAS_MODELO5):

        self.env = env
        self.nombres = list(nombres)
        self.capacidad_total = np.asarray(capacidad_total, dtype=float)
        self.unidades_paralelas = np.asarray(unidades_paralelas, dtype=int)
        self.perfiles = list(perfiles)
        self.franjas = list(franjas)
        self.etiquetas_franja = etiquetas_franjas(franjas)

        self.n = len(self.nombres)
        self.k = len(self.perfiles)
        self.f = len(self.franjas)

        # Estado instantáneo.
        self.en_nodo = np.zeros(self.n, dtype=int)
        self.en_cola = np.zeros(self.n, dtype=int)
        self.ocupados = np.zeros(self.n, dtype=int)
        self.en_nodo_perfil = np.zeros((self.k, self.n), dtype=int)
        self.en_cola_perfil = np.zeros((self.k, self.n), dtype=int)
        self.en_parque_perfil = np.zeros(self.k, dtype=int)

        self.entradas_parque = 0
        self.salidas_parque = 0
        self.salidas_parque_perfil = np.zeros(self.k, dtype=int)
        self.entradas_parque_franja = np.zeros(self.f, dtype=int)
        self.salidas_parque_franja = np.zeros(self.f, dtype=int)
        self.entradas_parque_perfil = np.zeros(self.k, dtype=int)
        self.entradas_parque_perfil_franja = np.zeros((self.k, self.f), dtype=int)
        self.salidas_parque_perfil_franja = np.zeros((self.k, self.f), dtype=int)

        # Últimos tiempos de actualización.
        self.ultimo_global = 0.0
        self.ultimo_nodo = np.zeros(self.n, dtype=float)

        # Integrales temporales globales.
        self.area_parque = 0.0
        self.area_colas_parque = 0.0
        self.area_parque_franja = np.zeros(self.f)
        self.area_colas_parque_franja = np.zeros(self.f)
        self.area_parque_perfil = np.zeros(self.k)
        self.area_colas_perfil = np.zeros(self.k)
        self.area_parque_perfil_franja = np.zeros((self.k, self.f))
        self.area_colas_perfil_franja = np.zeros((self.k, self.f))

        # Integrales temporales locales.
        self.area_nodo = np.zeros(self.n)
        self.area_cola = np.zeros(self.n)
        self.area_ocupacion = np.zeros(self.n)
        self.area_nodo_franja = np.zeros((self.f, self.n))
        self.area_cola_franja = np.zeros((self.f, self.n))
        self.area_ocupacion_franja = np.zeros((self.f, self.n))
        self.area_nodo_perfil = np.zeros((self.k, self.n))
        self.area_cola_perfil = np.zeros((self.k, self.n))
        self.area_nodo_perfil_franja = np.zeros((self.k, self.f, self.n))
        self.area_cola_perfil_franja = np.zeros((self.k, self.f, self.n))

        # Contadores locales.
        self.llegadas_nodo = np.zeros(self.n, dtype=int)
        self.salidas_nodo = np.zeros(self.n, dtype=int)
        self.suma_espera_nodo = np.zeros(self.n)
        self.suma_tiempo_nodo = np.zeros(self.n)

        self.llegadas_nodo_franja = np.zeros((self.f, self.n), dtype=int)
        self.salidas_nodo_franja = np.zeros((self.f, self.n), dtype=int)
        self.suma_espera_nodo_franja = np.zeros((self.f, self.n))
        self.suma_tiempo_nodo_franja = np.zeros((self.f, self.n))

        self.llegadas_nodo_perfil = np.zeros((self.k, self.n), dtype=int)
        self.salidas_nodo_perfil = np.zeros((self.k, self.n), dtype=int)
        self.suma_espera_nodo_perfil = np.zeros((self.k, self.n))
        self.suma_tiempo_nodo_perfil = np.zeros((self.k, self.n))

        self.llegadas_nodo_perfil_franja = np.zeros((self.k, self.f, self.n), dtype=int)
        self.salidas_nodo_perfil_franja = np.zeros((self.k, self.f, self.n), dtype=int)
        self.suma_espera_nodo_perfil_franja = np.zeros((self.k, self.f, self.n))
        self.suma_tiempo_nodo_perfil_franja = np.zeros((self.k, self.f, self.n))

        # Métricas globales por visitante que abandona el parque.
        self.suma_permanencia = 0.0
        self.suma_espera_total = 0.0
        self.suma_movimiento_total = 0.0
        self.suma_visitas_total = 0.0
        self.satisfechos = 0
        self.rechazos_espera = 0

        self.suma_permanencia_perfil = np.zeros(self.k)
        self.suma_espera_total_perfil = np.zeros(self.k)
        self.suma_movimiento_total_perfil = np.zeros(self.k)
        self.suma_visitas_total_perfil = np.zeros(self.k)
        self.satisfechos_perfil = np.zeros(self.k, dtype=int)
        self.rechazos_espera_perfil = np.zeros(self.k, dtype=int)

        self.suma_permanencia_franja = np.zeros(self.f)
        self.suma_espera_total_franja = np.zeros(self.f)
        self.suma_movimiento_total_franja = np.zeros(self.f)
        self.suma_visitas_total_franja = np.zeros(self.f)
        self.satisfechos_franja = np.zeros(self.f, dtype=int)
        self.rechazos_espera_franja = np.zeros(self.f, dtype=int)

        self.suma_permanencia_perfil_franja = np.zeros((self.k, self.f))
        self.suma_espera_total_perfil_franja = np.zeros((self.k, self.f))
        self.suma_movimiento_total_perfil_franja = np.zeros((self.k, self.f))
        self.suma_visitas_total_perfil_franja = np.zeros((self.k, self.f))
        self.satisfechos_perfil_franja = np.zeros((self.k, self.f), dtype=int)
        self.rechazos_espera_perfil_franja = np.zeros((self.k, self.f), dtype=int)

    def _tiempo_franja(self, r):
        return duracion_franja_en_horizonte(r, self.tiempo_limite, self.franjas)

    @staticmethod
    def _ratio(num, den):
        return num / den if den > 0 else np.nan

    def actualizar_global(self):
        ahora = self.env.now
        t0 = self.ultimo_global
        if ahora <= t0:
            return

        n_parque = int(self.en_parque_perfil.sum())
        n_colas = int(self.en_cola.sum())
        colas_perfil = self.en_cola_perfil.sum(axis=1)

        dt = ahora - t0
        self.area_parque += n_parque * dt
        self.area_colas_parque += n_colas * dt
        self.area_parque_perfil += self.en_parque_perfil * dt
        self.area_colas_perfil += colas_perfil * dt

        for r, dur in recorrer_intervalos_franja(t0, ahora, self.franjas):
            self.area_parque_franja[r] += n_parque * dur
            self.area_colas_parque_franja[r] += n_colas * dur
            self.area_parque_perfil_franja[:, r] += self.en_parque_perfil * dur
            self.area_colas_perfil_franja[:, r] += colas_perfil * dur

        self.ultimo_global = ahora

    def actualizar_nodo(self, j):
        ahora = self.env.now
        t0 = self.ultimo_nodo[j]
        if ahora <= t0:
            return

        dt = ahora - t0
        self.area_nodo[j] += self.en_nodo[j] * dt
        self.area_cola[j] += self.en_cola[j] * dt
        self.area_ocupacion[j] += self.ocupados[j] * dt
        self.area_nodo_perfil[:, j] += self.en_nodo_perfil[:, j] * dt
        self.area_cola_perfil[:, j] += self.en_cola_perfil[:, j] * dt

        for r, dur in recorrer_intervalos_franja(t0, ahora, self.franjas):
            self.area_nodo_franja[r, j] += self.en_nodo[j] * dur
            self.area_cola_franja[r, j] += self.en_cola[j] * dur
            self.area_ocupacion_franja[r, j] += self.ocupados[j] * dur
            self.area_nodo_perfil_franja[:, r, j] += self.en_nodo_perfil[:, j] * dur
            self.area_cola_perfil_franja[:, r, j] += self.en_cola_perfil[:, j] * dur

        self.ultimo_nodo[j] = ahora

    def registrar_entrada_parque(self, perfil):
        self.actualizar_global()
        r = indice_franja(self.env.now, self.franjas)
        self.entradas_parque += 1
        self.entradas_parque_perfil[perfil] += 1
        self.en_parque_perfil[perfil] += 1
        if r is not None:
            self.entradas_parque_franja[r] += 1
            self.entradas_parque_perfil_franja[perfil, r] += 1

    def registrar_entrada_nodo(self, j, perfil):
        self.actualizar_global()
        self.actualizar_nodo(j)
        r = indice_franja(self.env.now, self.franjas)

        self.en_nodo[j] += 1
        self.en_cola[j] += 1
        self.en_nodo_perfil[perfil, j] += 1
        self.en_cola_perfil[perfil, j] += 1

        self.llegadas_nodo[j] += 1
        self.llegadas_nodo_perfil[perfil, j] += 1
        if r is not None:
            self.llegadas_nodo_franja[r, j] += 1
            self.llegadas_nodo_perfil_franja[perfil, r, j] += 1

    def registrar_inicio_servicio(self, j, perfil, n_lote):
        self.actualizar_global()
        self.actualizar_nodo(j)
        self.en_cola[j] -= 1
        self.en_cola_perfil[perfil, j] -= 1
        self.ocupados[j] += 1

    def registrar_fin_servicio(self, j, perfil, espera, tiempo_total):
        self.actualizar_global()
        self.actualizar_nodo(j)
        r = indice_franja(self.env.now, self.franjas)

        self.en_nodo[j] -= 1
        self.en_nodo_perfil[perfil, j] -= 1
        self.ocupados[j] -= 1

        self.salidas_nodo[j] += 1
        self.suma_espera_nodo[j] += espera
        self.suma_tiempo_nodo[j] += tiempo_total

        self.salidas_nodo_perfil[perfil, j] += 1
        self.suma_espera_nodo_perfil[perfil, j] += espera
        self.suma_tiempo_nodo_perfil[perfil, j] += tiempo_total

        if r is not None:
            self.salidas_nodo_franja[r, j] += 1
            self.suma_espera_nodo_franja[r, j] += espera
            self.suma_tiempo_nodo_franja[r, j] += tiempo_total
            self.salidas_nodo_perfil_franja[perfil, r, j] += 1
            self.suma_espera_nodo_perfil_franja[perfil, r, j] += espera
            self.suma_tiempo_nodo_perfil_franja[perfil, r, j] += tiempo_total

    def registrar_rechazo_espera(self, perfil):
        r = indice_franja(self.env.now, self.franjas)
        self.rechazos_espera += 1
        self.rechazos_espera_perfil[perfil] += 1
        if r is not None:
            self.rechazos_espera_franja[r] += 1
            self.rechazos_espera_perfil_franja[perfil, r] += 1

    def registrar_salida_parque(self, visitante):
        self.actualizar_global()
        perfil = visitante["perfil"]
        r = indice_franja(self.env.now, self.franjas)

        permanencia = self.env.now - visitante["t_entrada_parque"]
        espera = visitante["espera_total"]
        movimiento = visitante["movimiento_total"]
        visitas = visitante["visitas"]
        satisfecho = int(visitas >= visitante["objetivo"])

        self.en_parque_perfil[perfil] -= 1
        self.salidas_parque += 1
        self.salidas_parque_perfil[perfil] += 1

        self.suma_permanencia += permanencia
        self.suma_espera_total += espera
        self.suma_movimiento_total += movimiento
        self.suma_visitas_total += visitas
        self.satisfechos += satisfecho

        self.suma_permanencia_perfil[perfil] += permanencia
        self.suma_espera_total_perfil[perfil] += espera
        self.suma_movimiento_total_perfil[perfil] += movimiento
        self.suma_visitas_total_perfil[perfil] += visitas
        self.satisfechos_perfil[perfil] += satisfecho

        if r is not None:
            self.salidas_parque_franja[r] += 1
            self.salidas_parque_perfil_franja[perfil, r] += 1
            self.suma_permanencia_franja[r] += permanencia
            self.suma_espera_total_franja[r] += espera
            self.suma_movimiento_total_franja[r] += movimiento
            self.suma_visitas_total_franja[r] += visitas
            self.satisfechos_franja[r] += satisfecho

            self.suma_permanencia_perfil_franja[perfil, r] += permanencia
            self.suma_espera_total_perfil_franja[perfil, r] += espera
            self.suma_movimiento_total_perfil_franja[perfil, r] += movimiento
            self.suma_visitas_total_perfil_franja[perfil, r] += visitas
            self.satisfechos_perfil_franja[perfil, r] += satisfecho

    def cerrar(self, tiempo_limite):
        self.tiempo_limite = tiempo_limite
        self.actualizar_global()
        for j in range(self.n):
            self.actualizar_nodo(j)

    def construir_metricas(self, tiempo_limite):
        self.cerrar(tiempo_limite)
        filas_locales = []
        filas_globales = []

        # Métricas locales base.
        for j, nombre in enumerate(self.nombres):
            filas_locales.extend([
                {"metrica": "tasa_efectiva_llegada", "nodo": nombre, "valor": self.llegadas_nodo[j] / tiempo_limite},
                {"metrica": "tasa_salida", "nodo": nombre, "valor": self.salidas_nodo[j] / tiempo_limite},
                {"metrica": "utilizacion", "nodo": nombre, "valor": self.area_ocupacion[j] / (self.capacidad_total[j] * tiempo_limite)},
                {"metrica": "numero_medio_en_atraccion", "nodo": nombre, "valor": self.area_nodo[j] / tiempo_limite},
                {"metrica": "numero_medio_en_cola", "nodo": nombre, "valor": self.area_cola[j] / tiempo_limite},
                {"metrica": "tiempo_medio_total_atraccion", "nodo": nombre, "valor": self._ratio(self.suma_tiempo_nodo[j], self.salidas_nodo[j])},
                {"metrica": "tiempo_medio_espera_cola", "nodo": nombre, "valor": self._ratio(self.suma_espera_nodo[j], self.salidas_nodo[j])},
                {"metrica": "numero_esperado_visitas", "nodo": nombre, "valor": self._ratio(self.llegadas_nodo[j], self.entradas_parque)},
            ])

        # Métricas locales por franja.
        for r, etiqueta in enumerate(self.etiquetas_franja):
            dur = self._tiempo_franja(r)
            den_entradas = self.entradas_parque_franja[r]
            for j, nombre in enumerate(self.nombres):
                suf = f"franja_{etiqueta}"
                filas_locales.extend([
                    {"metrica": f"tasa_efectiva_llegada_{suf}", "nodo": nombre, "valor": self.llegadas_nodo_franja[r, j] / dur if dur > 0 else np.nan},
                    {"metrica": f"tasa_salida_{suf}", "nodo": nombre, "valor": self.salidas_nodo_franja[r, j] / dur if dur > 0 else np.nan},
                    {"metrica": f"utilizacion_{suf}", "nodo": nombre, "valor": self.area_ocupacion_franja[r, j] / (self.capacidad_total[j] * dur) if dur > 0 else np.nan},
                    {"metrica": f"numero_medio_en_atraccion_{suf}", "nodo": nombre, "valor": self.area_nodo_franja[r, j] / dur if dur > 0 else np.nan},
                    {"metrica": f"numero_medio_en_cola_{suf}", "nodo": nombre, "valor": self.area_cola_franja[r, j] / dur if dur > 0 else np.nan},
                    {"metrica": f"tiempo_medio_total_atraccion_{suf}", "nodo": nombre, "valor": self._ratio(self.suma_tiempo_nodo_franja[r, j], self.salidas_nodo_franja[r, j])},
                    {"metrica": f"tiempo_medio_espera_cola_{suf}", "nodo": nombre, "valor": self._ratio(self.suma_espera_nodo_franja[r, j], self.salidas_nodo_franja[r, j])},
                    {"metrica": f"numero_esperado_visitas_{suf}", "nodo": nombre, "valor": self._ratio(self.llegadas_nodo_franja[r, j], den_entradas)},
                ])

        # Métricas locales por perfil y por perfil/franja.
        for k, perfil in enumerate(self.perfiles):
            perfil_id = id_texto(perfil)
            den_entradas_perfil = self.entradas_parque_perfil[k]
            for j, nombre in enumerate(self.nombres):
                filas_locales.extend([
                    {"metrica": f"numero_medio_en_atraccion_{perfil_id}", "nodo": nombre, "valor": self.area_nodo_perfil[k, j] / tiempo_limite},
                    {"metrica": f"numero_medio_en_cola_{perfil_id}", "nodo": nombre, "valor": self.area_cola_perfil[k, j] / tiempo_limite},
                    {"metrica": f"tiempo_medio_total_atraccion_{perfil_id}", "nodo": nombre, "valor": self._ratio(self.suma_tiempo_nodo_perfil[k, j], self.salidas_nodo_perfil[k, j])},
                    {"metrica": f"tiempo_medio_espera_cola_{perfil_id}", "nodo": nombre, "valor": self._ratio(self.suma_espera_nodo_perfil[k, j], self.salidas_nodo_perfil[k, j])},
                    {"metrica": f"numero_esperado_visitas_{perfil_id}", "nodo": nombre, "valor": self._ratio(self.llegadas_nodo_perfil[k, j], den_entradas_perfil)},
                ])

            for r, etiqueta in enumerate(self.etiquetas_franja):
                dur = self._tiempo_franja(r)
                den_pf = self.entradas_parque_perfil_franja[k, r]
                for j, nombre in enumerate(self.nombres):
                    suf = f"{perfil_id}_franja_{etiqueta}"
                    filas_locales.extend([
                        {"metrica": f"numero_medio_en_atraccion_{suf}", "nodo": nombre, "valor": self.area_nodo_perfil_franja[k, r, j] / dur if dur > 0 else np.nan},
                        {"metrica": f"numero_medio_en_cola_{suf}", "nodo": nombre, "valor": self.area_cola_perfil_franja[k, r, j] / dur if dur > 0 else np.nan},
                        {"metrica": f"tiempo_medio_total_atraccion_{suf}", "nodo": nombre, "valor": self._ratio(self.suma_tiempo_nodo_perfil_franja[k, r, j], self.salidas_nodo_perfil_franja[k, r, j])},
                        {"metrica": f"tiempo_medio_espera_cola_{suf}", "nodo": nombre, "valor": self._ratio(self.suma_espera_nodo_perfil_franja[k, r, j], self.salidas_nodo_perfil_franja[k, r, j])},
                        {"metrica": f"numero_esperado_visitas_{suf}", "nodo": nombre, "valor": self._ratio(self.llegadas_nodo_perfil_franja[k, r, j], den_pf)},
                    ])

        # Métricas globales base.
        filas_globales.extend([
            {"metrica": "numero_medio_visitantes_parque", "nodo": "Global", "valor": self.area_parque / tiempo_limite},
            {"metrica": "numero_medio_visitantes_colas_parque", "nodo": "Global", "valor": self.area_colas_parque / tiempo_limite},
            {"metrica": "tiempo_medio_permanencia_parque", "nodo": "Global", "valor": self._ratio(self.suma_permanencia, self.salidas_parque)},
            {"metrica": "tiempo_medio_espera_colas_parque", "nodo": "Global", "valor": self._ratio(self.suma_espera_total, self.salidas_parque)},
            {"metrica": "tiempo_medio_movimiento_parque", "nodo": "Global", "valor": self._ratio(self.suma_movimiento_total, self.salidas_parque)},
            {"metrica": "atracciones_completadas_por_visitante", "nodo": "Global", "valor": self._ratio(self.suma_visitas_total, self.salidas_parque)},
            {"metrica": "porcentaje_visitantes_satisfechos", "nodo": "Global", "valor": self._ratio(self.satisfechos, self.salidas_parque)},
            {"metrica": "rechazos_por_espera", "nodo": "Global", "valor": self.rechazos_espera},
            {"metrica": "tasa_rechazos_por_espera", "nodo": "Global", "valor": self.rechazos_espera / tiempo_limite},
        ])

        # Globales por perfil.
        for k, perfil in enumerate(self.perfiles):
            den = self.salidas_parque_perfil[k]
            filas_globales.extend([
                {"metrica": "numero_medio_visitantes_parque", "nodo": perfil, "valor": self.area_parque_perfil[k] / tiempo_limite},
                {"metrica": "numero_medio_visitantes_colas_parque", "nodo": perfil, "valor": self.area_colas_perfil[k] / tiempo_limite},
                {"metrica": "tiempo_medio_permanencia_parque", "nodo": perfil, "valor": self._ratio(self.suma_permanencia_perfil[k], den)},
                {"metrica": "tiempo_medio_espera_colas_parque", "nodo": perfil, "valor": self._ratio(self.suma_espera_total_perfil[k], den)},
                {"metrica": "tiempo_medio_movimiento_parque", "nodo": perfil, "valor": self._ratio(self.suma_movimiento_total_perfil[k], den)},
                {"metrica": "atracciones_completadas_por_visitante", "nodo": perfil, "valor": self._ratio(self.suma_visitas_total_perfil[k], den)},
                {"metrica": "porcentaje_visitantes_satisfechos", "nodo": perfil, "valor": self._ratio(self.satisfechos_perfil[k], den)},
                {"metrica": "rechazos_por_espera", "nodo": perfil, "valor": self.rechazos_espera_perfil[k]},
            ])

        # Globales por franja.
        for r, etiqueta in enumerate(self.etiquetas_franja):
            dur = self._tiempo_franja(r)
            den = self.salidas_parque_franja[r]
            filas_globales.extend([
                {"metrica": "numero_medio_visitantes_parque", "nodo": etiqueta, "valor": self.area_parque_franja[r] / dur if dur > 0 else np.nan},
                {"metrica": "numero_medio_visitantes_colas_parque", "nodo": etiqueta, "valor": self.area_colas_parque_franja[r] / dur if dur > 0 else np.nan},
                {"metrica": "tiempo_medio_permanencia_parque", "nodo": etiqueta, "valor": self._ratio(self.suma_permanencia_franja[r], den)},
                {"metrica": "tiempo_medio_espera_colas_parque", "nodo": etiqueta, "valor": self._ratio(self.suma_espera_total_franja[r], den)},
                {"metrica": "tiempo_medio_movimiento_parque", "nodo": etiqueta, "valor": self._ratio(self.suma_movimiento_total_franja[r], den)},
                {"metrica": "atracciones_completadas_por_visitante", "nodo": etiqueta, "valor": self._ratio(self.suma_visitas_total_franja[r], den)},
                {"metrica": "porcentaje_visitantes_satisfechos", "nodo": etiqueta, "valor": self._ratio(self.satisfechos_franja[r], den)},
                {"metrica": "rechazos_por_espera", "nodo": etiqueta, "valor": self.rechazos_espera_franja[r]},
            ])

        # Globales por perfil/franja.
        for k, perfil in enumerate(self.perfiles):
            for r, etiqueta in enumerate(self.etiquetas_franja):
                nodo_pf = f"{perfil}/{etiqueta}"
                dur = self._tiempo_franja(r)
                den = self.salidas_parque_perfil_franja[k, r]
                filas_globales.extend([
                    {"metrica": "numero_medio_visitantes_parque", "nodo": nodo_pf, "valor": self.area_parque_perfil_franja[k, r] / dur if dur > 0 else np.nan},
                    {"metrica": "numero_medio_visitantes_colas_parque", "nodo": nodo_pf, "valor": self.area_colas_perfil_franja[k, r] / dur if dur > 0 else np.nan},
                    {"metrica": "tiempo_medio_permanencia_parque", "nodo": nodo_pf, "valor": self._ratio(self.suma_permanencia_perfil_franja[k, r], den)},
                    {"metrica": "tiempo_medio_espera_colas_parque", "nodo": nodo_pf, "valor": self._ratio(self.suma_espera_total_perfil_franja[k, r], den)},
                    {"metrica": "tiempo_medio_movimiento_parque", "nodo": nodo_pf, "valor": self._ratio(self.suma_movimiento_total_perfil_franja[k, r], den)},
                    {"metrica": "atracciones_completadas_por_visitante", "nodo": nodo_pf, "valor": self._ratio(self.suma_visitas_total_perfil_franja[k, r], den)},
                    {"metrica": "porcentaje_visitantes_satisfechos", "nodo": nodo_pf, "valor": self._ratio(self.satisfechos_perfil_franja[k, r], den)},
                    {"metrica": "rechazos_por_espera", "nodo": nodo_pf, "valor": self.rechazos_espera_perfil_franja[k, r]},
                ])

        return pd.DataFrame(filas_locales), pd.DataFrame(filas_globales)


# ------------------------------------------------------------
# 4. Nodo por lotes del Modelo 6
# ------------------------------------------------------------

class NodoLotesModelo6:
    def __init__(self, env, indice, stats, rng):
        self.env = env
        self.indice = indice
        self.stats = stats
        self.rng = rng
        self.cola = simpy.Store(env)

        for k in range(NUM_LOTES_PARALELOS[indice]):
            env.process(self.proceso_lote(k))

    def visitar(self, visitante):
        llegada = self.env.now
        perfil = visitante["perfil"]
        fin = self.env.event()

        self.stats.registrar_entrada_nodo(self.indice, perfil)
        yield self.cola.put({"visitante": visitante, "llegada": llegada, "fin": fin})

        resultado = yield fin
        return resultado

    def proceso_lote(self, k):
        capacidad = int(CAPACIDAD_LOTE[self.indice])
        espera_max = float(TIEMPO_MAX_ESPERA_LOTE[self.indice])

        while True:
            primero = yield self.cola.get()
            lote = [primero]
            inicio_espera_lote = self.env.now

            while len(lote) < capacidad:
                restante = espera_max - (self.env.now - inicio_espera_lote)
                if restante <= 0:
                    break

                evento_get = self.cola.get()
                resultado = yield evento_get | self.env.timeout(restante)

                if evento_get in resultado:
                    lote.append(resultado[evento_get])
                else:
                    evento_get.cancel()
                    break

            n_lote = len(lote)

            esperas = []
            for item in lote:
                visitante = item["visitante"]
                perfil = visitante["perfil"]
                espera = self.env.now - item["llegada"]
                esperas.append(espera)
                self.stats.registrar_inicio_servicio(self.indice, perfil, n_lote)

            ciclo = tiempo_ciclo_atraccion(self.indice, n_lote)
            yield self.env.timeout(ciclo)

            for item, espera in zip(lote, esperas):
                visitante = item["visitante"]
                perfil = visitante["perfil"]
                tiempo_total = self.env.now - item["llegada"]

                visitante["visitas"] += 1
                visitante["espera_total"] += espera

                self.stats.registrar_fin_servicio(self.indice, perfil, espera, tiempo_total)

                item["fin"].succeed({
                    "nodo": self.indice,
                    "espera": espera,
                    "tiempo_total": tiempo_total,
                })


# ------------------------------------------------------------
# 5. Simulador de una réplica del Modelo 6
# ------------------------------------------------------------

def simular_modelo6(tiempo_limite=600, semilla=123,
                    franjas=FRANJAS_LLEGADAS_MODELO5):
    rng = np.random.default_rng(semilla)
    env = simpy.Environment()

    stats = EstadisticasModelo6(
        env=env,
        nombres=NOMBRES_NODOS,
        capacidad_total=CAPACIDAD_TOTAL,
        unidades_paralelas=NUM_LOTES_PARALELOS,
        franjas=franjas,
    )

    nodos = [NodoLotesModelo6(env, j, stats, rng) for j in range(N_NODOS)]

    def proceso_visitante(id_visitante):
        perfil = elegir_perfil_modelo6(rng)
        tolerancia = tolerancia_espera_modelo6(perfil, rng)
        objetivo = int(OBJETIVO_PERFIL_MODELO6[perfil])
        nodo_actual = elegir_entrada_inicial_modelo6(perfil, rng)

        visitante = {
            "id": id_visitante,
            "perfil": perfil,
            "tolerancia": tolerancia,
            "objetivo": objetivo,
            "t_entrada_parque": env.now,
            "visitas": 0,
            "espera_total": 0.0,
            "movimiento_total": 0.0,
        }

        stats.registrar_entrada_parque(perfil)

        while env.now < tiempo_limite:
            yield env.process(nodos[nodo_actual].visitar(visitante))

            p_salir = probabilidad_salida_modelo6(
                perfil=perfil,
                visitas_completadas=visitante["visitas"],
                espera_acumulada=visitante["espera_total"],
            )

            if rng.random() < p_salir:
                stats.registrar_salida_parque(visitante)
                return

            destino, espera_est, sin_destino_tolerable, _ = elegir_destino_modelo6(
                i_actual=nodo_actual,
                en_cola=stats.en_cola.copy(),
                perfil=perfil,
                tolerancia=tolerancia,
            )

            if sin_destino_tolerable:
                stats.registrar_rechazo_espera(perfil)
                if rng.random() < p_salir:
                    stats.registrar_salida_parque(visitante)
                    return

            t_mov = tiempo_desplazamiento_triangular(rng)
            visitante["movimiento_total"] += t_mov
            yield env.timeout(t_mov)
            nodo_actual = destino

    def generador_llegadas():
        id_visitante = 0
        while env.now < tiempo_limite:
            gamma = gamma_minuto_t(env.now, franjas=franjas)

            if gamma <= 0:
                siguiente_fin = fin_franja_actual(env.now, franjas=franjas)
                if np.isfinite(siguiente_fin) and siguiente_fin > env.now:
                    yield env.timeout(min(siguiente_fin - env.now, tiempo_limite - env.now))
                else:
                    break
                continue

            tiempo_hasta_llegada = rng.exponential(1.0 / gamma)
            fin_franja = fin_franja_actual(env.now, franjas=franjas)

            if env.now + tiempo_hasta_llegada > fin_franja:
                yield env.timeout(min(fin_franja - env.now, tiempo_limite - env.now))
                continue

            yield env.timeout(tiempo_hasta_llegada)

            if env.now < tiempo_limite:
                id_visitante += 1
                env.process(proceso_visitante(id_visitante))

    env.process(generador_llegadas())
    env.run(until=tiempo_limite)

    locales, globales = stats.construir_metricas(tiempo_limite)

    return {
        "locales": locales,
        "globales": globales,
        "entradas_parque": stats.entradas_parque,
        "salidas_parque": stats.salidas_parque,
        "stats": stats,
    }


# ------------------------------------------------------------
# 6. Simulación Monte Carlo del Modelo 6
# ------------------------------------------------------------

def ejecutar_monte_carlo_modelo6(n_replicas=30, tiempo_limite=600,
                                 semilla_inicial=6000,
                                 franjas=FRANJAS_LLEGADAS_MODELO5):
    replicas = []

    for r in range(n_replicas):
        replicas.append(
            simular_modelo6(
                tiempo_limite=tiempo_limite,
                semilla=semilla_inicial + r,
                franjas=franjas,
            )
        )

    return agregar_replicas(replicas)


def prueba_rapida_modelo6(tiempo_limite=120, semilla=123):
    resultado = simular_modelo6(tiempo_limite=tiempo_limite, semilla=semilla)

    print("Prueba rápida Modelo 6")
    print("Entradas al parque:", resultado["entradas_parque"])
    print("Salidas del parque:", resultado["salidas_parque"])
    print("Métricas locales:")
    display(resultado["locales"].head(12))
    print("Métricas globales:")
    display(resultado["globales"].head(12))

    return resultado


# ------------------------------------------------------------
# 7. Visualización específica del Modelo 6
# ------------------------------------------------------------

def tabla_local_franja_modelo6(
    resultado_mc,
    franja,
    metricas_base=None,
    mostrar="ic",
    decimales=3,
):
    """
    Tabla local métrica × atracción para una franja concreta del Modelo 6.

    franja : str etiqueta (p.ej. "0-120") o int índice.
    """
    etiquetas = etiquetas_franjas(FRANJAS_LLEGADAS_MODELO5)
    etiqueta  = etiquetas[franja] if isinstance(franja, int) else str(franja)

    if metricas_base is None:
        metricas_base = [
            "tasa_efectiva_llegada",
            "tasa_salida",
            "utilizacion",
            "numero_medio_en_atraccion",
            "numero_medio_en_cola",
            "tiempo_medio_total_atraccion",
            "tiempo_medio_espera_cola",
            "numero_esperado_visitas",
        ]

    suf    = f"_franja_{etiqueta}"
    df     = resultado_mc["locales"].copy()
    mets_f = [m + suf for m in metricas_base]
    df     = df[df["metrica"].isin(mets_f)].copy()
    df["metrica"] = df["metrica"].replace({m + suf: m for m in metricas_base})

    return tabla_mc_por_nodo(df, nodos=NOMBRES_NODOS, mostrar=mostrar, decimales=decimales)


def tabla_local_perfil_modelo6(
    resultado_mc,
    perfil,
    metricas_base=None,
    mostrar="ic",
    decimales=3,
):
    """
    Tabla local métrica × atracción para un perfil concreto del Modelo 6.

    perfil : str etiqueta (p.ej. "Familias") o int índice.
    """
    if isinstance(perfil, int):
        perfil = PERFILES_MODELO6[perfil]

    pid = id_texto(perfil)

    if metricas_base is None:
        metricas_base = [
            "tasa_efectiva_llegada",
            "tasa_salida",
            "utilizacion",
            "numero_medio_en_atraccion",
            "numero_medio_en_cola",
            "tiempo_medio_total_atraccion",
            "tiempo_medio_espera_cola",
            "numero_esperado_visitas",
        ]

    suf    = f"_{pid}"
    df     = resultado_mc["locales"].copy()
    mets_p = [m + suf for m in metricas_base]
    df     = df[df["metrica"].isin(mets_p)].copy()
    df["metrica"] = df["metrica"].replace({m + suf: m for m in metricas_base})

    return tabla_mc_por_nodo(df, nodos=NOMBRES_NODOS, mostrar=mostrar, decimales=decimales)


def mostrar_resumen_modelo6(resultado_mc, mostrar="ic", decimales=3):
    """
    Muestra todas las tablas MC del Modelo 6:

        1. Métricas locales base (agregadas)
        2. Métricas locales por franja (una tabla por franja)
        3. Métricas locales por perfil (una tabla por perfil)
        4. Métricas globales base
        5. Métricas globales por perfil
        6. Métricas globales por franja
        7. Métricas globales por perfil/franja
    """

    metricas_base_local = [
        "tasa_efectiva_llegada",
        "tasa_salida",
        "utilizacion",
        "numero_medio_en_atraccion",
        "numero_medio_en_cola",
        "tiempo_medio_total_atraccion",
        "tiempo_medio_espera_cola",
        "numero_esperado_visitas",
    ]

    etiquetas = etiquetas_franjas(FRANJAS_LLEGADAS_MODELO5)
    sep = "=" * 70

    # ------------------------------------------------------------------
    # 1. Locales base
    # ------------------------------------------------------------------
    print(sep)
    print("MODELO 6. Métricas locales base")
    print(sep)
    df_base = resultado_mc["locales"][
        resultado_mc["locales"]["metrica"].isin(metricas_base_local)
    ]
    display(tabla_mc_por_nodo(df_base, mostrar=mostrar, decimales=decimales))

    # ------------------------------------------------------------------
    # 2. Locales por franja
    # ------------------------------------------------------------------
    for etiqueta in etiquetas:
        print(sep)
        print(f"MODELO 6. Métricas locales por atracción — franja {etiqueta}")
        print(sep)
        display(tabla_local_franja_modelo6(
            resultado_mc, franja=etiqueta,
            metricas_base=metricas_base_local,
            mostrar=mostrar, decimales=decimales,
        ))

    # ------------------------------------------------------------------
    # 3. Locales por perfil
    # ------------------------------------------------------------------
    for perfil in PERFILES_MODELO6:
        print(sep)
        print(f"MODELO 6. Métricas locales por atracción — perfil {perfil}")
        print(sep)
        display(tabla_local_perfil_modelo6(
            resultado_mc, perfil=perfil,
            metricas_base=metricas_base_local,
            mostrar=mostrar, decimales=decimales,
        ))

    # ------------------------------------------------------------------
    # 4. Globales base
    # ------------------------------------------------------------------
    print(sep)
    print("MODELO 6. Métricas globales base")
    print(sep)
    display(tabla_global_modelo6(
        resultado_mc, nodos=["Global"],
        mostrar=mostrar, decimales=decimales,
    ))

    # ------------------------------------------------------------------
    # 5. Globales por perfil
    # ------------------------------------------------------------------
    print(sep)
    print("MODELO 6. Métricas globales por perfil")
    print(sep)
    display(tabla_global_modelo6(
        resultado_mc, nodos=PERFILES_MODELO6,
        mostrar=mostrar, decimales=decimales,
    ))

    # ------------------------------------------------------------------
    # 6. Globales por franja
    # ------------------------------------------------------------------
    print(sep)
    print("MODELO 6. Métricas globales por franja")
    print(sep)
    display(tabla_global_modelo6(
        resultado_mc, nodos=etiquetas,
        mostrar=mostrar, decimales=decimales,
    ))

    # ------------------------------------------------------------------
    # 7. Globales por perfil/franja
    # ------------------------------------------------------------------
    nodos_pf = [f"{p}/{fr}" for p in PERFILES_MODELO6 for fr in etiquetas]
    print(sep)
    print("MODELO 6. Métricas globales por perfil y franja")
    print(sep)
    display(tabla_global_modelo6(
        resultado_mc, nodos=nodos_pf,
        mostrar=mostrar, decimales=decimales,
    ))

def graficar_global_franjas_modelo6(resultado_mc, metricas=None):
    graficar_global_por_grupo(
        resultado_mc,
        nodos=etiquetas_franjas(FRANJAS_LLEGADAS_MODELO5),
        metricas=metricas,
        titulo="Modelo 6. Métricas globales por franja",
    )
 
def graficar_global_perfil_franja_modelo6(resultado_mc, metricas=None):
    graficar_global_perfil_x_franja(
        resultado_mc,
        perfiles=PERFILES_MODELO6,
        franjas=etiquetas_franjas(FRANJAS_LLEGADAS_MODELO5),
        metricas=metricas,
        titulo="Modelo 6. Métricas globales por perfil y franja",
    )
 
def graficar_global_modelo6_tamanos_y_tiempos(resultado_mc):
    graficar_global_tamanos_tiempos(
        resultado_mc,
        nodos=etiquetas_franjas(FRANJAS_LLEGADAS_MODELO5),
        titulo_base="Modelo 6. Métricas globales por franja",
    )

#=====================================================================
# ALMACENAR COMO PARQUET LAS SIMULACIONES DE MODELO 6 Y 7 

import os
import pandas as pd


def guardar_resultado_mc(resultado_mc, carpeta, nombre_modelo):
    """
    Guarda los DataFrames locales y globales de un resultado MC
    en formato Parquet dentro de la carpeta indicada.

    Crea dos ficheros:
        <carpeta>/<nombre_modelo>_locales.parquet
        <carpeta>/<nombre_modelo>_globales.parquet

    Parámetros
    ----------
    resultado_mc : dict
        Resultado de agregar_replicas(), con claves "locales" y "globales".
    carpeta : str
        Ruta de la carpeta destino (se crea si no existe).
    nombre_modelo : str
        Prefijo del nombre de fichero, p.ej. "modelo6".
    """
    os.makedirs(carpeta, exist_ok=True)

    path_loc = os.path.join(carpeta, f"{nombre_modelo}_locales.parquet")
    path_glo = os.path.join(carpeta, f"{nombre_modelo}_globales.parquet")

    resultado_mc["locales"].to_parquet(path_loc, index=False)
    resultado_mc["globales"].to_parquet(path_glo, index=False)

    print(f"Guardado: {path_loc}")
    print(f"Guardado: {path_glo}")


def cargar_resultado_mc(carpeta, nombre_modelo):
    """
    Carga los DataFrames locales y globales guardados por
    guardar_resultado_mc() y devuelve un dict con la misma
    estructura que agregar_replicas().

    Parámetros
    ----------
    carpeta : str
    nombre_modelo : str

    Devuelve
    --------
    dict con claves "locales" y "globales".
    """
    path_loc = os.path.join(carpeta, f"{nombre_modelo}_locales.parquet")
    path_glo = os.path.join(carpeta, f"{nombre_modelo}_globales.parquet")

    locales  = pd.read_parquet(path_loc)
    globales = pd.read_parquet(path_glo)

    print(f"Cargado: {path_loc}  ({len(locales)} filas)")
    print(f"Cargado: {path_glo}  ({len(globales)} filas)")

    return {"locales": locales, "globales": globales}
