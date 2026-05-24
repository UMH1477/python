# ============================================================
# MODELO 7. MODELO 6 + ACTIVIDADES NO MECÁNICAS
#           Restauración (M/M/c), Descanso (capacidad finita),
#           Espectáculo (funciones programadas)
# ============================================================
from urllib.request import urlretrieve
url = "https://raw.githubusercontent.com/UMH1477/python/refs/heads/main/parque_atracciones/parque_comunes.py"
urlretrieve(url, 'parque_comunes.py')
from parque_comunes import *  

import importlib
from urllib.request import urlretrieve

for i in range(1,7):
  model_name = f'parque_modelo{i}'
  url = f"https://raw.githubusercontent.com/UMH1477/python/refs/heads/main/parque_atracciones/{model_name}.py"
  urlretrieve(url, f'{model_name}.py')
  # Dynamically import all contents of the module into the current namespace
  # This is equivalent to 'from module_name import *'
  exec(f"from {model_name} import *", globals())
    
import simpy
import numpy as np
import pandas as pd


# ============================================================
# 1. ESTADÍSTICAS DEL MODELO 7
# ============================================================

class EstadisticasModelo7:
    """
    Registra todas las métricas del Modelo 7.

    Nodos 0-5  : atracciones mecánicas (igual que Modelo 6).
    Nodo 6     : Restauración (cola M/M/c).
    Nodo 7     : Descanso (capacidad finita, sin cola de espera real).
    Nodo 8     : Espectáculo (funciones programadas).

    Las métricas del Espectáculo se recogen por función (índice 0-3)
    y por perfil, además de las métricas temporales habituales.
    """

    def __init__(self, env, nombres=NOMBRES_NODOS_M7,
                 franjas=FRANJAS_LLEGADAS_MODELO5,
                 perfiles=PERFILES_MODELO6,
                 horarios=HORARIOS_ESPECTACULO_M7,
                 capacidad_espectaculo=CAPACIDAD_ESPECTACULO_M7,
                 c_restauracion=C_RESTAURACION_M7,
                 capacidad_descanso=CAPACIDAD_DESCANSO_M7):

        self.env      = env
        self.nombres  = list(nombres)
        self.franjas  = list(franjas)
        self.perfiles = list(perfiles)
        self.horarios = list(horarios)
        self.cap_esp  = int(capacidad_espectaculo)
        self.c_rest   = int(c_restauracion)
        self.cap_desc = int(capacidad_descanso)

        self.n  = len(self.nombres)           # 9
        self.k  = len(self.perfiles)          # 4
        self.f  = len(self.franjas)           # 5
        self.nf = len(self.horarios)          # 4 funciones

        self.etiquetas_franja = etiquetas_franjas(self.franjas)

        # ── Estado instantáneo ──────────────────────────────
        self.en_nodo          = np.zeros(self.n, dtype=int)
        self.en_cola          = np.zeros(self.n, dtype=int)
        self.ocupados         = np.zeros(self.n, dtype=int)
        self.en_nodo_perfil   = np.zeros((self.k, self.n), dtype=int)
        self.en_cola_perfil   = np.zeros((self.k, self.n), dtype=int)
        self.en_parque_perfil = np.zeros(self.k, dtype=int)

        # ── Contadores de entradas/salidas ───────────────────
        self.entradas_parque                   = 0
        self.salidas_parque                    = 0
        self.entradas_parque_perfil            = np.zeros(self.k, dtype=int)
        self.salidas_parque_perfil             = np.zeros(self.k, dtype=int)
        self.entradas_parque_franja            = np.zeros(self.f, dtype=int)
        self.salidas_parque_franja             = np.zeros(self.f, dtype=int)
        self.entradas_parque_perfil_franja     = np.zeros((self.k, self.f), dtype=int)
        self.salidas_parque_perfil_franja      = np.zeros((self.k, self.f), dtype=int)

        # ── Integrales temporales globales ───────────────────
        self.ultimo_global             = 0.0
        self.area_parque               = 0.0
        self.area_colas_parque         = 0.0
        self.area_parque_franja        = np.zeros(self.f)
        self.area_colas_parque_franja  = np.zeros(self.f)
        self.area_parque_perfil        = np.zeros(self.k)
        self.area_colas_perfil         = np.zeros(self.k)

        # ── Integrales temporales locales ────────────────────
        self.ultimo_nodo        = np.zeros(self.n, dtype=float)
        self.area_nodo          = np.zeros(self.n)
        self.area_cola          = np.zeros(self.n)
        self.area_ocupacion     = np.zeros(self.n)
        self.area_nodo_franja   = np.zeros((self.f, self.n))
        self.area_cola_franja   = np.zeros((self.f, self.n))
        self.area_ocup_franja   = np.zeros((self.f, self.n))

        # ── Contadores locales ───────────────────────────────
        self.llegadas_nodo           = np.zeros(self.n, dtype=int)
        self.salidas_nodo            = np.zeros(self.n, dtype=int)
        self.suma_espera_nodo        = np.zeros(self.n)
        self.suma_tiempo_nodo        = np.zeros(self.n)
        self.llegadas_nodo_franja    = np.zeros((self.f, self.n), dtype=int)
        self.salidas_nodo_franja     = np.zeros((self.f, self.n), dtype=int)
        self.suma_espera_nodo_franja = np.zeros((self.f, self.n))
        self.suma_tiempo_nodo_franja = np.zeros((self.f, self.n))
        self.llegadas_nodo_perfil    = np.zeros((self.k, self.n), dtype=int)
        self.salidas_nodo_perfil     = np.zeros((self.k, self.n), dtype=int)
        self.suma_espera_nodo_perfil = np.zeros((self.k, self.n))
        self.suma_tiempo_nodo_perfil = np.zeros((self.k, self.n))

        # ── Métricas globales por visitante ──────────────────
        self.suma_permanencia        = 0.0
        self.suma_espera_total       = 0.0
        self.suma_movimiento_total   = 0.0
        self.suma_visitas_total      = 0.0
        self.rechazos_espera         = 0
        self.rechazos_espectaculo    = 0   # descartado por aforo lleno

        self.suma_permanencia_perfil      = np.zeros(self.k)
        self.suma_espera_total_perfil     = np.zeros(self.k)
        self.suma_movimiento_total_perfil = np.zeros(self.k)
        self.suma_visitas_total_perfil    = np.zeros(self.k)
        self.rechazos_espera_perfil       = np.zeros(self.k, dtype=int)
        self.rechazos_espectaculo_perfil  = np.zeros(self.k, dtype=int)

        self.suma_permanencia_franja      = np.zeros(self.f)
        self.suma_espera_total_franja     = np.zeros(self.f)
        self.suma_movimiento_total_franja = np.zeros(self.f)
        self.suma_visitas_total_franja    = np.zeros(self.f)

        # ── Arrays perfil × franja ────────────────────────────
        self.salidas_parque_perfil_franja       = np.zeros((self.k, self.f), dtype=int)
        self.suma_permanencia_perfil_franja      = np.zeros((self.k, self.f))
        self.suma_espera_total_perfil_franja     = np.zeros((self.k, self.f))
        self.suma_movimiento_total_perfil_franja = np.zeros((self.k, self.f))
        self.suma_visitas_total_perfil_franja    = np.zeros((self.k, self.f))
        self.rechazos_espera_perfil_franja       = np.zeros((self.k, self.f), dtype=int)

        # ── Métricas específicas del Espectáculo ─────────────
        # ocupacion_por_funcion[idx_f] = número de visitantes que entraron
        self.ocupacion_por_funcion           = np.zeros(self.nf, dtype=int)
        # ocupacion_por_funcion_perfil[k, idx_f]
        self.ocupacion_por_funcion_perfil    = np.zeros((self.k, self.nf), dtype=int)
        # visitantes que no cabieron en una función concreta
        self.rechazados_por_funcion          = np.zeros(self.nf, dtype=int)

    # ── Helpers ─────────────────────────────────────────────

    def _tiempo_franja(self, r):
        return duracion_franja_en_horizonte(r, self.tiempo_limite, self.franjas)

    @staticmethod
    def _ratio(num, den):
        return num / den if den > 0 else np.nan

    # ── Actualizadores de integrales ─────────────────────────

    def actualizar_global(self):
        ahora = self.env.now
        t0    = self.ultimo_global
        if ahora <= t0:
            return
        dt = ahora - t0

        n_parque = int(self.en_parque_perfil.sum())
        n_colas  = int(self.en_cola.sum())
        colas_p  = self.en_cola_perfil.sum(axis=1)

        self.area_parque       += n_parque * dt
        self.area_colas_parque += n_colas  * dt
        self.area_parque_perfil       += self.en_parque_perfil * dt
        self.area_colas_perfil        += colas_p * dt

        for r, dur in recorrer_intervalos_franja(t0, ahora, self.franjas):
            self.area_parque_franja[r]       += n_parque * dur
            self.area_colas_parque_franja[r] += n_colas  * dur

        self.ultimo_global = ahora

    def actualizar_nodo(self, j):
        ahora = self.env.now
        t0    = self.ultimo_nodo[j]
        if ahora <= t0:
            return
        dt = ahora - t0

        self.area_nodo[j]      += self.en_nodo[j]  * dt
        self.area_cola[j]      += self.en_cola[j]  * dt
        self.area_ocupacion[j] += self.ocupados[j] * dt

        for r, dur in recorrer_intervalos_franja(t0, ahora, self.franjas):
            self.area_nodo_franja[r, j] += self.en_nodo[j]  * dur
            self.area_cola_franja[r, j] += self.en_cola[j]  * dur
            self.area_ocup_franja[r, j] += self.ocupados[j] * dur

        self.ultimo_nodo[j] = ahora

    # ── Registros de eventos ─────────────────────────────────

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
        self.en_nodo[j]          += 1
        self.en_cola[j]          += 1
        self.en_nodo_perfil[perfil, j] += 1
        self.en_cola_perfil[perfil, j] += 1
        self.llegadas_nodo[j]    += 1
        self.llegadas_nodo_perfil[perfil, j] += 1
        if r is not None:
            self.llegadas_nodo_franja[r, j]    += 1

    def registrar_inicio_servicio(self, j, perfil, n_lote=1):
        self.actualizar_global()
        self.actualizar_nodo(j)
        self.en_cola[j]          -= 1
        self.en_cola_perfil[perfil, j] -= 1
        self.ocupados[j]         += 1

    def registrar_fin_servicio(self, j, perfil, espera, tiempo_total):
        self.actualizar_global()
        self.actualizar_nodo(j)
        r = indice_franja(self.env.now, self.franjas)
        self.en_nodo[j]          -= 1
        self.en_nodo_perfil[perfil, j] -= 1
        self.ocupados[j]         -= 1
        self.salidas_nodo[j]     += 1
        self.suma_espera_nodo[j] += espera
        self.suma_tiempo_nodo[j] += tiempo_total
        self.salidas_nodo_perfil[perfil, j]     += 1
        self.suma_espera_nodo_perfil[perfil, j] += espera
        self.suma_tiempo_nodo_perfil[perfil, j] += tiempo_total
        if r is not None:
            self.salidas_nodo_franja[r, j]    += 1
            self.suma_espera_nodo_franja[r, j] += espera
            self.suma_tiempo_nodo_franja[r, j] += tiempo_total

    def registrar_rechazo_espera(self, perfil):
        r = indice_franja(self.env.now, self.franjas)
        self.rechazos_espera += 1
        self.rechazos_espera_perfil[perfil] += 1
        if r is not None:
            self.rechazos_espera_perfil_franja[perfil, r] += 1

    def registrar_rechazo_espectaculo(self, perfil):
        self.rechazos_espectaculo += 1
        self.rechazos_espectaculo_perfil[perfil] += 1

    def registrar_entrada_espectaculo(self, perfil, idx_funcion):
        """Cuenta visitante que consigue plaza en una función."""
        self.ocupacion_por_funcion[idx_funcion]           += 1
        self.ocupacion_por_funcion_perfil[perfil, idx_funcion] += 1

    def registrar_salida_parque(self, visitante):
        self.actualizar_global()
        perfil = visitante["perfil"]
        r      = indice_franja(self.env.now, self.franjas)

        permanencia = self.env.now - visitante["t_entrada_parque"]
        espera      = visitante["espera_total"]
        movimiento  = visitante["movimiento_total"]
        visitas     = visitante["visitas"]

        self.en_parque_perfil[perfil]        -= 1
        self.salidas_parque                  += 1
        self.salidas_parque_perfil[perfil]   += 1

        self.suma_permanencia      += permanencia
        self.suma_espera_total     += espera
        self.suma_movimiento_total += movimiento
        self.suma_visitas_total    += visitas

        self.suma_permanencia_perfil[perfil]      += permanencia
        self.suma_espera_total_perfil[perfil]     += espera
        self.suma_movimiento_total_perfil[perfil] += movimiento
        self.suma_visitas_total_perfil[perfil]    += visitas

        if r is not None:
            self.salidas_parque_franja[r]                      += 1
            self.salidas_parque_perfil_franja[perfil, r]       += 1
            self.suma_permanencia_franja[r]                    += permanencia
            self.suma_espera_total_franja[r]                   += espera
            self.suma_movimiento_total_franja[r]               += movimiento
            self.suma_visitas_total_franja[r]                  += visitas
            # perfil × franja
            self.suma_permanencia_perfil_franja[perfil, r]     += permanencia
            self.suma_espera_total_perfil_franja[perfil, r]    += espera
            self.suma_movimiento_total_perfil_franja[perfil, r]+= movimiento
            self.suma_visitas_total_perfil_franja[perfil, r]   += visitas

    def cerrar(self, tiempo_limite):
        self.tiempo_limite = tiempo_limite
        self.actualizar_global()
        for j in range(self.n):
            self.actualizar_nodo(j)

    # ── Construcción de métricas ─────────────────────────────

    def construir_metricas(self, tiempo_limite):
        self.cerrar(tiempo_limite)
        filas_locales  = []
        filas_globales = []

        # ── Métricas locales base (todos los nodos) ──────────
        for j, nombre in enumerate(self.nombres):
            cap_j = (CAPACIDAD_TOTAL[j] if j < N_ATRACCIONES_M7
                     else (self.c_rest if j == IDX_RESTAURACION
                           else (self.cap_desc if j == IDX_DESCANSO
                                 else self.cap_esp)))
            filas_locales.extend([
                {"metrica": "tasa_efectiva_llegada",   "nodo": nombre, "valor": self.llegadas_nodo[j] / tiempo_limite},
                {"metrica": "tasa_salida",             "nodo": nombre, "valor": self.salidas_nodo[j] / tiempo_limite},
                {"metrica": "utilizacion",             "nodo": nombre, "valor": self.area_ocupacion[j] / (cap_j * tiempo_limite)},
                {"metrica": "numero_medio_en_atraccion","nodo": nombre, "valor": self.area_nodo[j] / tiempo_limite},
                {"metrica": "numero_medio_en_cola",    "nodo": nombre, "valor": self.area_cola[j] / tiempo_limite},
                {"metrica": "tiempo_medio_total_atraccion","nodo": nombre, "valor": self._ratio(self.suma_tiempo_nodo[j], self.salidas_nodo[j])},
                {"metrica": "tiempo_medio_espera_cola","nodo": nombre, "valor": self._ratio(self.suma_espera_nodo[j], self.salidas_nodo[j])},
                {"metrica": "numero_esperado_visitas", "nodo": nombre, "valor": self._ratio(self.llegadas_nodo[j], self.entradas_parque)},
            ])

        # ── Métricas locales por franja ───────────────────────
        for r, etiqueta in enumerate(self.etiquetas_franja):
            dur = self._tiempo_franja(r)
            for j, nombre in enumerate(self.nombres):
                cap_j = (CAPACIDAD_TOTAL[j] if j < N_ATRACCIONES_M7
                         else (self.c_rest if j == IDX_RESTAURACION
                               else (self.cap_desc if j == IDX_DESCANSO
                                     else self.cap_esp)))
                suf = f"franja_{etiqueta}"
                filas_locales.extend([
                    {"metrica": f"tasa_efectiva_llegada_{suf}",    "nodo": nombre, "valor": self.llegadas_nodo_franja[r, j] / dur if dur > 0 else np.nan},
                    {"metrica": f"tasa_salida_{suf}",              "nodo": nombre, "valor": self.salidas_nodo_franja[r, j] / dur if dur > 0 else np.nan},
                    {"metrica": f"utilizacion_{suf}",              "nodo": nombre, "valor": self.area_ocup_franja[r, j] / (cap_j * dur) if dur > 0 else np.nan},
                    {"metrica": f"numero_medio_en_atraccion_{suf}","nodo": nombre, "valor": self.area_nodo_franja[r, j] / dur if dur > 0 else np.nan},
                    {"metrica": f"numero_medio_en_cola_{suf}",     "nodo": nombre, "valor": self.area_cola_franja[r, j] / dur if dur > 0 else np.nan},
                    {"metrica": f"tiempo_medio_total_atraccion_{suf}","nodo": nombre, "valor": self._ratio(self.suma_tiempo_nodo_franja[r, j], self.salidas_nodo_franja[r, j])},
                    {"metrica": f"tiempo_medio_espera_cola_{suf}", "nodo": nombre, "valor": self._ratio(self.suma_espera_nodo_franja[r, j], self.salidas_nodo_franja[r, j])},
                ])

        # ── Métricas locales por perfil ───────────────────────
        for k, perfil in enumerate(self.perfiles):
            pid = id_texto(perfil)
            den = self.entradas_parque_perfil[k]
            for j, nombre in enumerate(self.nombres):
                filas_locales.extend([
                    {"metrica": f"numero_medio_en_atraccion_{pid}","nodo": nombre, "valor": self.area_nodo_franja[:, j].sum() / tiempo_limite if False else self._ratio(self.suma_tiempo_nodo_perfil[k, j], self.salidas_nodo_perfil[k, j])},
                    {"metrica": f"tiempo_medio_espera_cola_{pid}", "nodo": nombre, "valor": self._ratio(self.suma_espera_nodo_perfil[k, j], self.salidas_nodo_perfil[k, j])},
                    {"metrica": f"numero_esperado_visitas_{pid}",  "nodo": nombre, "valor": self._ratio(self.llegadas_nodo_perfil[k, j], den)},
                ])

        # ── Métricas específicas del Espectáculo ─────────────
        etiq_funciones = [f"Funcion_{idx+1}_t{h}" for idx, h in enumerate(self.horarios)]
        for idx_f, etiq in enumerate(etiq_funciones):
            filas_locales.append({
                "metrica": "ocupacion_espectaculo",
                "nodo": etiq,
                "valor": float(self.ocupacion_por_funcion[idx_f])
            })
            for k, perfil in enumerate(self.perfiles):
                filas_locales.append({
                    "metrica": f"ocupacion_espectaculo_{id_texto(perfil)}",
                    "nodo": etiq,
                    "valor": float(self.ocupacion_por_funcion_perfil[k, idx_f])
                })

        # ── Métricas globales base ────────────────────────────
        filas_globales.extend([
            {"metrica": "numero_medio_visitantes_parque",        "nodo": "Global", "valor": self.area_parque / tiempo_limite},
            {"metrica": "numero_medio_visitantes_colas_parque",  "nodo": "Global", "valor": self.area_colas_parque / tiempo_limite},
            {"metrica": "tiempo_medio_permanencia_parque",       "nodo": "Global", "valor": self._ratio(self.suma_permanencia, self.salidas_parque)},
            {"metrica": "tiempo_medio_espera_colas_parque",      "nodo": "Global", "valor": self._ratio(self.suma_espera_total, self.salidas_parque)},
            {"metrica": "tiempo_medio_movimiento_parque",        "nodo": "Global", "valor": self._ratio(self.suma_movimiento_total, self.salidas_parque)},
            {"metrica": "atracciones_completadas_por_visitante", "nodo": "Global", "valor": self._ratio(self.suma_visitas_total, self.salidas_parque)},
            {"metrica": "rechazos_por_espera",                   "nodo": "Global", "valor": float(self.rechazos_espera)},
            {"metrica": "rechazos_espectaculo_aforo_lleno",      "nodo": "Global", "valor": float(self.rechazos_espectaculo)},
            {"metrica": "funciones_espectaculo_realizadas",      "nodo": "Global", "valor": float(np.sum(self.ocupacion_por_funcion > 0))},
        ])

        # ── Globales por perfil ───────────────────────────────
        for k, perfil in enumerate(self.perfiles):
            den = self.salidas_parque_perfil[k]
            filas_globales.extend([
                {"metrica": "numero_medio_visitantes_parque",        "nodo": perfil, "valor": self.area_parque_perfil[k] / tiempo_limite},
                {"metrica": "numero_medio_visitantes_colas_parque",  "nodo": perfil, "valor": self.area_colas_perfil[k] / tiempo_limite},
                {"metrica": "tiempo_medio_permanencia_parque",       "nodo": perfil, "valor": self._ratio(self.suma_permanencia_perfil[k], den)},
                {"metrica": "tiempo_medio_espera_colas_parque",      "nodo": perfil, "valor": self._ratio(self.suma_espera_total_perfil[k], den)},
                {"metrica": "tiempo_medio_movimiento_parque",        "nodo": perfil, "valor": self._ratio(self.suma_movimiento_total_perfil[k], den)},
                {"metrica": "atracciones_completadas_por_visitante", "nodo": perfil, "valor": self._ratio(self.suma_visitas_total_perfil[k], den)},
                {"metrica": "rechazos_por_espera",                   "nodo": perfil, "valor": float(self.rechazos_espera_perfil[k])},
                {"metrica": "rechazos_espectaculo_aforo_lleno",      "nodo": perfil, "valor": float(self.rechazos_espectaculo_perfil[k])},
            ])

        # ── Globales por franja ───────────────────────────────
        for r, etiqueta in enumerate(self.etiquetas_franja):
            dur = self._tiempo_franja(r)
            den = self.salidas_parque_franja[r]
            filas_globales.extend([
                {"metrica": "numero_medio_visitantes_parque",        "nodo": etiqueta, "valor": self.area_parque_franja[r] / dur if dur > 0 else np.nan},
                {"metrica": "numero_medio_visitantes_colas_parque",  "nodo": etiqueta, "valor": self.area_colas_parque_franja[r] / dur if dur > 0 else np.nan},
                {"metrica": "tiempo_medio_permanencia_parque",       "nodo": etiqueta, "valor": self._ratio(self.suma_permanencia_franja[r], den)},
                {"metrica": "tiempo_medio_espera_colas_parque",      "nodo": etiqueta, "valor": self._ratio(self.suma_espera_total_franja[r], den)},
                {"metrica": "tiempo_medio_movimiento_parque",        "nodo": etiqueta, "valor": self._ratio(self.suma_movimiento_total_franja[r], den)},
                {"metrica": "atracciones_completadas_por_visitante", "nodo": etiqueta, "valor": self._ratio(self.suma_visitas_total_franja[r], den)},
            ])

        # ── Globales por perfil/franja ─────────────────────
        for k, perfil in enumerate(self.perfiles):
            for r, etiqueta in enumerate(self.etiquetas_franja):
                nodo_pf = f"{perfil}/{etiqueta}"
                dur     = self._tiempo_franja(r)
                den     = self.salidas_parque_perfil_franja[k, r]
                filas_globales.extend([
                    {"metrica": "numero_medio_visitantes_parque",
                     "nodo": nodo_pf,
                     "valor": self.area_parque_perfil[k] / dur if dur > 0 else np.nan},
                    {"metrica": "numero_medio_visitantes_colas_parque",
                     "nodo": nodo_pf,
                     "valor": self.area_colas_perfil[k] / dur if dur > 0 else np.nan},
                    {"metrica": "tiempo_medio_permanencia_parque",
                     "nodo": nodo_pf,
                     "valor": self._ratio(self.suma_permanencia_perfil_franja[k, r], den)},
                    {"metrica": "tiempo_medio_espera_colas_parque",
                     "nodo": nodo_pf,
                     "valor": self._ratio(self.suma_espera_total_perfil_franja[k, r], den)},
                    {"metrica": "tiempo_medio_movimiento_parque",
                     "nodo": nodo_pf,
                     "valor": self._ratio(self.suma_movimiento_total_perfil_franja[k, r], den)},
                    {"metrica": "atracciones_completadas_por_visitante",
                     "nodo": nodo_pf,
                     "valor": self._ratio(self.suma_visitas_total_perfil_franja[k, r], den)},
                    {"metrica": "rechazos_por_espera",
                     "nodo": nodo_pf,
                     "valor": float(self.rechazos_espera_perfil_franja[k, r])},
                ])

        return pd.DataFrame(filas_locales), pd.DataFrame(filas_globales)


# ============================================================
# 2. NODOS DE SERVICIO
# ============================================================

# ── 2a. Atracción mecánica por lotes (igual que Modelo 6) ───

class NodoLotesModelo7:
    """Atracción mecánica por lotes. Idéntica lógica al Modelo 6."""

    def __init__(self, env, indice, stats, rng):
        self.env    = env
        self.indice = indice
        self.stats  = stats
        self.rng    = rng
        self.cola   = simpy.Store(env)
        for k in range(NUM_LOTES_PARALELOS[indice]):
            env.process(self._proceso_lote())

    def visitar(self, visitante):
        llegada = self.env.now
        perfil  = visitante["perfil"]
        fin     = self.env.event()
        self.stats.registrar_entrada_nodo(self.indice, perfil)
        yield self.cola.put({"visitante": visitante, "llegada": llegada, "fin": fin})
        resultado = yield fin
        return resultado

    def _proceso_lote(self):
        capacidad  = int(CAPACIDAD_LOTE[self.indice])
        espera_max = float(TIEMPO_MAX_ESPERA_LOTE[self.indice])
        while True:
            primero = yield self.cola.get()
            lote    = [primero]
            t_ini   = self.env.now
            while len(lote) < capacidad:
                restante = espera_max - (self.env.now - t_ini)
                if restante <= 0:
                    break
                ev = self.cola.get()
                res = yield ev | self.env.timeout(restante)
                if ev in res:
                    lote.append(res[ev])
                else:
                    ev.cancel()
                    break
            n_lote = len(lote)
            for item in lote:
                self.stats.registrar_inicio_servicio(self.indice, item["visitante"]["perfil"], n_lote)
            ciclo = tiempo_ciclo_atraccion(self.indice, n_lote)
            yield self.env.timeout(ciclo)
            for item in lote:
                v      = item["visitante"]
                espera = self.env.now - item["llegada"] - ciclo
                espera = max(0.0, self.env.now - item["llegada"] - ciclo + ciclo - (ciclo))
                espera = self.env.now - item["llegada"]
                # espera real = tiempo desde llegada hasta inicio del servicio
                # = (tiempo actual) - (llegada) - ciclo
                espera_real = (self.env.now - ciclo) - item["llegada"]
                espera_real = max(0.0, espera_real)
                tiempo_total = self.env.now - item["llegada"]
                v["visitas"] += 1
                v["espera_total"] += espera_real
                self.stats.registrar_fin_servicio(
                    self.indice, v["perfil"], espera_real, tiempo_total)
                item["fin"].succeed({"nodo": self.indice,
                                     "espera": espera_real,
                                     "tiempo_total": tiempo_total})


# ── 2b. Nodo Restauración (M/M/c con simpy.Resource) ────────

class NodoRestauracion:
    """
    Cola M/M/c con c_restauracion mostradores.
    Tolerancia de espera ilimitada: el visitante siempre espera.
    """

    def __init__(self, env, stats, rng,
                 c=C_RESTAURACION_M7,
                 tri=TIEMPO_RESTAURACION_TRI_M7):
        self.env    = env
        self.stats  = stats
        self.rng    = rng
        self.tri    = tri
        self.j      = IDX_RESTAURACION
        self.recurso = simpy.Resource(env, capacity=c)

    def visitar(self, visitante):
        llegada = self.env.now
        perfil  = visitante["perfil"]
        self.stats.registrar_entrada_nodo(self.j, perfil)

        with self.recurso.request() as req:
            yield req
            espera = self.env.now - llegada
            self.stats.registrar_inicio_servicio(self.j, perfil)
            t_servicio = generar_triangular(self.rng, self.tri)
            yield self.env.timeout(t_servicio)

        tiempo_total = self.env.now - llegada
        visitante["espera_total"] += espera
        self.stats.registrar_fin_servicio(self.j, perfil, espera, tiempo_total)
        return {"nodo": self.j, "espera": espera, "tiempo_total": tiempo_total}


# ── 2c. Nodo Descanso (capacidad finita, sin cola) ───────────

class NodoDescanso:
    """
    Zona de descanso con capacidad máxima simultánea.
    Si no hay plazas libres el visitante no entra (se trata como
    destino no disponible en elegir_destino_m7; si aun así llega,
    se reencamina inmediatamente).
    """

    def __init__(self, env, stats, rng,
                 capacidad=CAPACIDAD_DESCANSO_M7,
                 tri=TIEMPO_DESCANSO_TRI_M7):
        self.env      = env
        self.stats    = stats
        self.rng      = rng
        self.tri      = tri
        self.j        = IDX_DESCANSO
        self.capacidad = capacidad
        self.recurso  = simpy.Resource(env, capacity=capacidad)

    @property
    def plazas_libres(self):
        return self.capacidad - self.recurso.count

    def visitar(self, visitante):
        llegada = self.env.now
        perfil  = visitante["perfil"]

        if self.plazas_libres <= 0:
            # No hay plazas: salida inmediata sin registrar
            return {"nodo": self.j, "espera": 0.0, "tiempo_total": 0.0, "rechazado": True}

        self.stats.registrar_entrada_nodo(self.j, perfil)
        with self.recurso.request() as req:
            yield req
            self.stats.registrar_inicio_servicio(self.j, perfil)
            t_descanso = generar_triangular(self.rng, self.tri)
            yield self.env.timeout(t_descanso)

        tiempo_total = self.env.now - llegada
        self.stats.registrar_fin_servicio(self.j, perfil, 0.0, tiempo_total)
        return {"nodo": self.j, "espera": 0.0, "tiempo_total": tiempo_total, "rechazado": False}


# ── 2d. Nodo Espectáculo (funciones programadas) ─────────────

class NodoEspectaculo:
    """
    Espectáculo con funciones programadas.

    Cada función es un evento que se activa en su horario.
    Los visitantes esperan dentro del nodo hasta que empieza la función.
    Su "espera" = tiempo desde que llegan hasta el inicio de la función.
    La "duración total" = espera + duración del espectáculo.

    Cuando una función está llena (cap_espectaculo plazas) los nuevos
    visitantes son rechazados.
    """

    def __init__(self, env, stats, rng,
                 horarios=HORARIOS_ESPECTACULO_M7,
                 capacidad=CAPACIDAD_ESPECTACULO_M7,
                 duracion=DURACION_ESPECTACULO_M7):
        self.env        = env
        self.stats      = stats
        self.rng        = rng
        self.horarios   = list(horarios)
        self.capacidad  = int(capacidad)
        self.duracion   = float(duracion)
        self.j          = IDX_ESPECTACULO

        # Plazas ocupadas por función {idx_f: count}
        self.ocupados_por_funcion = {i: 0 for i in range(len(horarios))}

        # Eventos de inicio por función (se activan en su horario)
        self.eventos_inicio = [env.event() for _ in horarios]

        # Lanzar proceso que activa los eventos en su momento
        env.process(self._gestor_funciones())

    def _gestor_funciones(self):
        for idx_f, h in enumerate(self.horarios):
            espera = h - self.env.now
            if espera > 0:
                yield self.env.timeout(espera)
            # Activar el evento de inicio (despierta a todos los que esperan)
            if not self.eventos_inicio[idx_f].triggered:
                self.eventos_inicio[idx_f].succeed()
            # Esperar duración del espectáculo antes de la siguiente función
            yield self.env.timeout(self.duracion)

    def visitar(self, visitante):
        """
        El visitante llega habiendo comprobado que la función está disponible
        (ventana no superada). Aquí verificamos aforo.
        """
        llegada = self.env.now
        perfil  = visitante["perfil"]

        disp, idx_f, inicio_f, dt = espectaculo_disponible(llegada)

        if idx_f is None:
            return {"nodo": self.j, "espera": 0.0, "tiempo_total": 0.0, "rechazado": True}

        if self.ocupados_por_funcion[idx_f] >= self.capacidad:
            self.stats.registrar_rechazo_espectaculo(perfil)
            return {"nodo": self.j, "espera": 0.0, "tiempo_total": 0.0, "rechazado": True}

        # Reservar plaza
        self.ocupados_por_funcion[idx_f] += 1
        self.stats.registrar_entrada_nodo(self.j, perfil)
        self.stats.registrar_inicio_servicio(self.j, perfil)
        self.stats.registrar_entrada_espectaculo(perfil, idx_f)

        # Esperar inicio de la función si aún no ha llegado
        if not self.eventos_inicio[idx_f].triggered:
            yield self.eventos_inicio[idx_f]

        espera = max(0.0, inicio_f - llegada)

        # Disfrutar el espectáculo
        # Calcular cuánto queda del espectáculo desde ahora
        fin_esp = inicio_f + self.duracion
        restante = max(0.0, fin_esp - self.env.now)
        if restante > 0:
            yield self.env.timeout(restante)

        tiempo_total = self.env.now - llegada
        self.stats.registrar_fin_servicio(self.j, perfil, espera, tiempo_total)
        return {"nodo": self.j, "espera": espera, "tiempo_total": tiempo_total, "rechazado": False}


# ============================================================
# 3. SIMULADOR DE UNA RÉPLICA
# ============================================================

def simular_modelo7(tiempo_limite=600, semilla=123,
                    franjas=FRANJAS_LLEGADAS_MODELO5):

    rng = np.random.default_rng(semilla)
    env = simpy.Environment()

    stats = EstadisticasModelo7(env=env, franjas=franjas)

    # Nodos 0-5: atracciones mecánicas
    nodos_atr = [NodoLotesModelo7(env, j, stats, rng)
                 for j in range(N_ATRACCIONES_M7)]

    # Nodos complementarios
    nodo_rest = NodoRestauracion(env, stats, rng)
    nodo_desc = NodoDescanso(env, stats, rng)
    nodo_esp  = NodoEspectaculo(env, stats, rng)

    def get_nodo(j):
        if j < N_ATRACCIONES_M7:
            return nodos_atr[j]
        elif j == IDX_RESTAURACION:
            return nodo_rest
        elif j == IDX_DESCANSO:
            return nodo_desc
        else:
            return nodo_esp

    def proceso_visitante(id_visitante):
        perfil    = elegir_perfil_modelo6(rng)
        tolerancia = tolerancia_espera_modelo6(perfil, rng)
        objetivo   = int(OBJETIVO_PERFIL_MODELO6[perfil])
        nodo_actual = elegir_entrada_inicial_modelo6(perfil, rng)

        visitante = {
            "id":              id_visitante,
            "perfil":          perfil,
            "tolerancia":      tolerancia,
            "objetivo":        objetivo,
            "t_entrada_parque": env.now,
            "visitas":         0,
            "espera_total":    0.0,
            "movimiento_total": 0.0,
        }

        stats.registrar_entrada_parque(perfil)

        while env.now < tiempo_limite:
            nodo = get_nodo(nodo_actual)

            # Visita al nodo
            resultado = yield env.process(nodo.visitar(visitante))

            # Si fue rechazado (descanso lleno o espectáculo lleno),
            # elegir otro destino directamente sin registrar salida
            if resultado.get("rechazado", False):
                # Fallback: elegir atracción de menor espera
                en_cola_total = np.concatenate([
                    stats.en_cola[:N_ATRACCIONES_M7],
                    [stats.en_cola[IDX_RESTAURACION],
                     stats.en_cola[IDX_DESCANSO],
                     stats.en_cola[IDX_ESPECTACULO]]
                ])
                atr_cands = [j for j in range(N_ATRACCIONES_M7) if j != nodo_actual]
                wq = espera_estimada_por_cola(en_cola_total[:N_ATRACCIONES_M7])
                nodo_actual = min(atr_cands, key=lambda j: wq[j])
                continue

            # Probabilidad de salida
            p_salir = probabilidad_salida_modelo6(
                perfil=perfil,
                visitas_completadas=visitante["visitas"],
                espera_acumulada=visitante["espera_total"],
            )
            if rng.random() < p_salir:
                stats.registrar_salida_parque(visitante)
                return

            # Elegir siguiente destino
            en_cola_total = np.concatenate([
                stats.en_cola[:N_ATRACCIONES_M7],
                [stats.en_cola[IDX_RESTAURACION],
                 stats.en_cola[IDX_DESCANSO],
                 stats.en_cola[IDX_ESPECTACULO]]
            ])

            destino, espera_est, sin_tolerable, esp_lleno, _ = elegir_destino_m7(
                i_actual=nodo_actual,
                en_cola_total=en_cola_total,
                perfil=perfil,
                tolerancia=tolerancia,
                visitas_completadas=visitante["visitas"],
                espera_acumulada=visitante["espera_total"],
                t_ahora=env.now,
                ocupados_espectaculo=nodo_esp.ocupados_por_funcion,
            )

            if sin_tolerable:
                stats.registrar_rechazo_espera(perfil)
                if rng.random() < p_salir:
                    stats.registrar_salida_parque(visitante)
                    return

            # Desplazamiento
            t_mov = tiempo_desplazamiento_triangular(rng)
            visitante["movimiento_total"] += t_mov
            yield env.timeout(t_mov)
            nodo_actual = destino

        # Fin del horizonte: registrar si aún en parque
        if visitante["visitas"] > 0:
            stats.registrar_salida_parque(visitante)

    def generador_llegadas():
        id_v = 0
        while env.now < tiempo_limite:
            gamma = gamma_minuto_t(env.now, franjas=franjas)
            if gamma <= 0:
                sig_fin = fin_franja_actual(env.now, franjas=franjas)
                if np.isfinite(sig_fin) and sig_fin > env.now:
                    yield env.timeout(min(sig_fin - env.now, tiempo_limite - env.now))
                else:
                    break
                continue
            t_llegada = rng.exponential(1.0 / gamma)
            fin_fr    = fin_franja_actual(env.now, franjas=franjas)
            if env.now + t_llegada > fin_fr:
                yield env.timeout(min(fin_fr - env.now, tiempo_limite - env.now))
                continue
            yield env.timeout(t_llegada)
            if env.now < tiempo_limite:
                id_v += 1
                env.process(proceso_visitante(id_v))

    env.process(generador_llegadas())
    env.run(until=tiempo_limite)

    locales, globales = stats.construir_metricas(tiempo_limite)

    return {
        "locales":         locales,
        "globales":        globales,
        "entradas_parque": stats.entradas_parque,
        "salidas_parque":  stats.salidas_parque,
        "stats":           stats,
    }


# ============================================================
# 4. SIMULACIÓN MONTE CARLO
# ============================================================

def ejecutar_monte_carlo_modelo7(n_replicas=30, tiempo_limite=600,
                                  semilla_inicial=7000,
                                  franjas=FRANJAS_LLEGADAS_MODELO5):
    replicas = []
    for r in range(n_replicas):
        replicas.append(
            simular_modelo7(
                tiempo_limite=tiempo_limite,
                semilla=semilla_inicial + r,
                franjas=franjas,
            )
        )
    return agregar_replicas(replicas)


def prueba_rapida_modelo7(tiempo_limite=120, semilla=123):
    resultado = simular_modelo7(tiempo_limite=tiempo_limite, semilla=semilla)
    print("Prueba rápida Modelo 7")
    print("Entradas al parque:", resultado["entradas_parque"])
    print("Salidas del parque:", resultado["salidas_parque"])
    print("Métricas locales (primeras filas):")
    display(resultado["locales"].head(15))
    print("Métricas globales (primeras filas):")
    display(resultado["globales"].head(12))
    return resultado


# ============================================================
# 5. VISUALIZACIÓN ESPECÍFICA DEL MODELO 7
# ============================================================

def mostrar_resumen_modelo7(resultado_mc, mostrar="ic", decimales=3):
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

    print("=" * 70)
    print("MODELO 7. Métricas locales base — Atracciones")
    print("=" * 70)
    df_atr = resultado_mc["locales"][
        resultado_mc["locales"]["metrica"].isin(metricas_base) &
        resultado_mc["locales"]["nodo"].isin(NOMBRES_NODOS_M7[:N_ATRACCIONES_M7])
    ]
    display(tabla_mc_por_nodo(df_atr, mostrar=mostrar, decimales=decimales))

    print("=" * 70)
    print("MODELO 7. Métricas locales base — Actividades no mecánicas")
    print("=" * 70)
    nodos_comp = [NOMBRES_NODOS_M7[IDX_RESTAURACION],
                  NOMBRES_NODOS_M7[IDX_DESCANSO],
                  NOMBRES_NODOS_M7[IDX_ESPECTACULO]]
    df_comp = resultado_mc["locales"][
        resultado_mc["locales"]["metrica"].isin(metricas_base) &
        resultado_mc["locales"]["nodo"].isin(nodos_comp)
    ]
    display(tabla_mc_por_nodo(df_comp, mostrar=mostrar, decimales=decimales))

    print("=" * 70)
    print("MODELO 7. Ocupación del Espectáculo por función")
    print("=" * 70)
    df_esp = resultado_mc["locales"][
        resultado_mc["locales"]["metrica"] == "ocupacion_espectaculo"
    ]
    display(tabla_mc_por_nodo(df_esp, mostrar=mostrar, decimales=decimales))

    print("=" * 70)
    print("MODELO 7. Ocupación del Espectáculo por función y perfil")
    print("=" * 70)
    for k, perfil in enumerate(PERFILES_MODELO6):
        pid = id_texto(perfil)
        df_ep = resultado_mc["locales"][
            resultado_mc["locales"]["metrica"] == f"ocupacion_espectaculo_{pid}"
        ]
        if not df_ep.empty:
            print(f"  Perfil: {perfil}")
            display(tabla_mc_por_nodo(df_ep, mostrar=mostrar, decimales=decimales))

    print("=" * 70)
    print("MODELO 7. Métricas globales")
    print("=" * 70)
    display(tabla_mc_por_nodo(
        resultado_mc["globales"],
        nodos=["Global"],
        mostrar=mostrar, decimales=decimales
    ))

    print("=" * 70)
    print("MODELO 7. Métricas globales por perfil")
    print("=" * 70)
    display(tabla_mc_por_nodo(
        resultado_mc["globales"],
        nodos=PERFILES_MODELO6,
        mostrar=mostrar, decimales=decimales
    ))

    print("=" * 70)
    print("MODELO 7. Métricas globales por franja")
    print("=" * 70)
    display(tabla_mc_por_nodo(
        resultado_mc["globales"],
        nodos=etiquetas_franjas(FRANJAS_LLEGADAS_MODELO5),
        mostrar=mostrar, decimales=decimales
    ))


def graficar_modelo7_actividades(resultado_mc):
    """
    Gráfico comparativo de métricas locales base para los 3 nodos
    de actividades no mecánicas, separando tamaños y tiempos.
    """
    metricas_tamano = [
        "numero_medio_en_atraccion",
        "numero_medio_en_cola",
        "tasa_efectiva_llegada",
    ]
    metricas_tiempo = [
        "tiempo_medio_total_atraccion",
        "tiempo_medio_espera_cola",
    ]

    nodos_comp = [NOMBRES_NODOS_M7[IDX_RESTAURACION],
                  NOMBRES_NODOS_M7[IDX_DESCANSO],
                  NOMBRES_NODOS_M7[IDX_ESPECTACULO]]

    for titulo, metricas in [
        ("Actividades no mecánicas — Tamaños", metricas_tamano),
        ("Actividades no mecánicas — Tiempos", metricas_tiempo),
    ]:
        df = resultado_mc["locales"].copy()
        df = df[df["metrica"].isin(metricas) & df["nodo"].isin(nodos_comp)]
        df["x"]     = df["nodo"]
        df["grupo"] = df["metrica"]
        graficar_metricas_en_grid_pastel(df, metricas, titulo=f"Modelo 7. {titulo}")


def graficar_espectaculo_por_funcion(resultado_mc):
    """
    Barras con ocupación media de cada función del espectáculo,
    desagregadas por perfil.
    """
    import matplotlib.pyplot as plt

    etiq_f = [f"Funcion_{i+1}_t{h}"
              for i, h in enumerate(HORARIOS_ESPECTACULO_M7)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # ── Global ──────────────────────────────────────────────
    ax = axes[0]
    df_g = resultado_mc["locales"][
        resultado_mc["locales"]["metrica"] == "ocupacion_espectaculo"
    ].set_index("nodo").reindex(etiq_f)
    x = np.arange(len(etiq_f))
    ax.bar(x, df_g["media"].fillna(0), color="#8DB6CD", width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{i+1}\nt={h}" for i, h in enumerate(HORARIOS_ESPECTACULO_M7)], fontsize=9)
    ax.set_title("Ocupación media por función (Global)", fontweight="bold")
    ax.set_ylabel("Visitantes")
    ax.set_ylim(0, CAPACIDAD_ESPECTACULO_M7 * 1.15)
    ax.axhline(CAPACIDAD_ESPECTACULO_M7, color="red", linestyle="--",
               linewidth=1, label=f"Capacidad ({CAPACIDAD_ESPECTACULO_M7})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Por perfil ───────────────────────────────────────────
    ax = axes[1]
    colores_perfil = ["#8DB6CD", "#F4A7A1", "#B8D8BA", "#D7BDE2"]
    n_perfiles = len(PERFILES_MODELO6)
    ancho = 0.18
    for k, perfil in enumerate(PERFILES_MODELO6):
        pid = id_texto(perfil)
        df_p = resultado_mc["locales"][
            resultado_mc["locales"]["metrica"] == f"ocupacion_espectaculo_{pid}"
        ].set_index("nodo").reindex(etiq_f)
        offset = (k - n_perfiles / 2 + 0.5) * ancho
        ax.bar(x + offset, df_p["media"].fillna(0),
               width=ancho, color=colores_perfil[k], label=perfil)
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{i+1}\nt={h}" for i, h in enumerate(HORARIOS_ESPECTACULO_M7)], fontsize=9)
    ax.set_title("Ocupación media por función y perfil", fontweight="bold")
    ax.set_ylabel("Visitantes")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Modelo 7. Espectáculo", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def graficar_global_perfil_franja_modelo7(resultado_mc, metricas=None):
    graficar_global_perfil_x_franja(
        resultado_mc,
        perfiles=PERFILES_MODELO6,
        franjas=etiquetas_franjas(FRANJAS_LLEGADAS_MODELO5),
        metricas=metricas,
        titulo="Modelo 7. Métricas globales por perfil y franja",
    )

def comparar_modelo6_modelo7(resultado_m6, resultado_m7,
                              metricas_locales=None, metricas_globales=None):
    comparar_dos_modelos(
        resultado_m6, "Modelo 6",
        resultado_m7, "Modelo 7",
        metricas_locales=metricas_locales,
        metricas_globales=metricas_globales,
    )
