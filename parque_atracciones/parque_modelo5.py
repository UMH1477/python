# ============================================================
# MODELO 5 ACTUALIZADO. MODELO 4 + LLEGADAS NO ESTACIONARIAS
#           + ELECCIÓN ADAPTATIVA DE DESTINO
#           + MÉTRICAS LOCALES Y GLOBALES POR FRANJA HORARIA
# ============================================================

import simpy
import numpy as np
import pandas as pd

# Requiere tener ejecutado previamente:
#   1. Bloque común existente hasta Modelo 5.
#   2. Ampliación común reutilizable creada para Modelo 6:
#      - etiquetas_franjas
#      - indice_franja
#      - duracion_franja_en_horizonte
#      - recorrer_intervalos_franja
#      - tabla_mc_por_nodo
#      - graficar_metricas_en_grid_pastel


# ------------------------------------------------------------
# 1. Estadísticas del Modelo 5 con métricas por franja
# ------------------------------------------------------------

class EstadisticasModelo5(EstadisticasModelo4):
    """
    Estadísticas de una réplica del Modelo 5.

    Mantiene las métricas del Modelo 4 y añade:
        - métricas de decisión adaptativa;
        - atracciones completadas por visitante;
        - tiempo medio en movimiento por visitante;
        - métricas locales por franja horaria;
        - métricas globales por franja horaria.

    La franja se asigna así:
        - L, Lq y utilización: integración temporal restringida a cada franja;
        - tasas de llegada/salida: eventos ocurridos dentro de cada franja;
        - W y Wq locales: visitantes que terminan el nodo dentro de cada franja;
        - W y Wq globales: visitantes que salen del parque dentro de cada franja.
    """

    def __init__(
        self,
        env,
        nombres,
        capacidad_total,
        max_visitas_tabla=10,
        franjas=FRANJAS_LLEGADAS_MODELO5
    ):

        super().__init__(
            env=env,
            nombres=nombres,
            capacidad_total=capacidad_total
        )

        self.max_visitas_tabla = int(max_visitas_tabla)
        self.franjas = list(franjas)
        self.etiquetas_franja = etiquetas_franjas(self.franjas)
        self.f = len(self.franjas)

        # Métricas de salida por número de visitas completadas.
        self.oportunidades_salida_por_v = np.zeros(
            self.max_visitas_tabla,
            dtype=int
        )
        self.salidas_por_v = np.zeros(
            self.max_visitas_tabla,
            dtype=int
        )

        # Métricas de intensidad de uso.
        self.suma_visitas_completadas_salida = 0.0

        # Métricas de movimiento por visitante.
        self.suma_tiempo_movimiento_visitantes_salida = 0.0

        # Métricas de elección adaptativa.
        self.destinos_elegidos = np.zeros(self.n, dtype=int)
        self.suma_espera_estimada_destino = np.zeros(self.n)
        self.suma_espera_estimada_eleccion = 0.0
        self.num_decisiones_continuar = 0

        # Áreas temporales por franja.
        self.area_L_franja = np.zeros((self.f, self.n))
        self.area_Lq_franja = np.zeros((self.f, self.n))
        self.area_ocupacion_franja = np.zeros((self.f, self.n))
        self.area_desplazamiento_franja = np.zeros(self.f)

        # Conteos locales por franja.
        self.entradas_nodo_franja = np.zeros((self.f, self.n), dtype=int)
        self.salidas_nodo_franja = np.zeros((self.f, self.n), dtype=int)

        # Tiempos locales por franja.
        self.suma_W_franja = np.zeros((self.f, self.n))
        self.suma_Wq_franja = np.zeros((self.f, self.n))

        # Lotes por franja.
        self.num_lotes_iniciados_franja = np.zeros((self.f, self.n), dtype=int)
        self.suma_ocupacion_lotes_franja = np.zeros((self.f, self.n))
        self.suma_tiempo_embarque_franja = np.zeros((self.f, self.n))
        self.suma_tiempo_desembarque_franja = np.zeros((self.f, self.n))
        self.suma_tiempo_ciclo_franja = np.zeros((self.f, self.n))

        # Conteos globales por franja.
        self.entradas_parque_franja = np.zeros(self.f, dtype=int)
        self.salidas_parque_franja = np.zeros(self.f, dtype=int)

        # Tiempos globales por visitante que sale en cada franja.
        self.suma_tiempo_parque_franja = np.zeros(self.f)
        self.suma_espera_parque_franja = np.zeros(self.f)
        self.suma_tiempo_movimiento_salida_franja = np.zeros(self.f)
        self.suma_visitas_completadas_salida_franja = np.zeros(self.f)

        # Decisiones adaptativas por franja.
        self.destinos_elegidos_franja = np.zeros((self.f, self.n), dtype=int)
        self.suma_espera_estimada_destino_franja = np.zeros((self.f, self.n))
        self.suma_espera_estimada_eleccion_franja = np.zeros(self.f)
        self.num_decisiones_continuar_franja = np.zeros(self.f, dtype=int)

    @staticmethod
    def _ratio(num, den):
        return num / den if den > 0 else np.nan

    def _franja_actual(self):
        return indice_franja(self.env.now, self.franjas)

    def _duracion_franja(self, r, tiempo_limite):
        return duracion_franja_en_horizonte(r, tiempo_limite, self.franjas)

    def actualizar_areas(self):
        """
        Actualiza las áreas globales heredadas y, además, reparte esas áreas
        entre franjas horarias.
        """

        ahora = self.env.now
        t0 = self.ultimo_t
        dt = ahora - t0

        if dt <= 0:
            return

        L_inst = self.en_cola + self.en_servicio
        Lq_inst = self.en_cola

        # Áreas heredadas.
        self.area_L += L_inst * dt
        self.area_Lq += Lq_inst * dt
        self.area_ocupacion += self.en_servicio * dt
        self.area_desplazamiento += self.en_desplazamiento * dt

        # Áreas por franja.
        for r, dur in recorrer_intervalos_franja(t0, ahora, self.franjas):
            self.area_L_franja[r, :] += L_inst * dur
            self.area_Lq_franja[r, :] += Lq_inst * dur
            self.area_ocupacion_franja[r, :] += self.en_servicio * dur
            self.area_desplazamiento_franja[r] += self.en_desplazamiento * dur

        self.ultimo_t = ahora

    def registrar_llegada_nodo(self, i):
        r = self._franja_actual()
        super().registrar_llegada_nodo(i)
        if r is not None:
            self.entradas_nodo_franja[r, i] += 1

    def terminar_servicio_lote(self, i, n_visitantes):
        r = self._franja_actual()
        super().terminar_servicio_lote(i, n_visitantes)
        if r is not None:
            self.salidas_nodo_franja[r, i] += n_visitantes

    def registrar_tiempos_nodo(self, i, espera, tiempo_total):
        r = self._franja_actual()
        super().registrar_tiempos_nodo(i, espera, tiempo_total)
        if r is not None:
            self.suma_Wq_franja[r, i] += espera
            self.suma_W_franja[r, i] += tiempo_total

    def registrar_lote(self, i, n_visitantes, capacidad_lote,
                       tiempo_embarque=None, tiempo_desembarque=None, tiempo_ciclo=None):
        """
        Compatible con Modelo 3/4/5.

        En el cuaderno, Modelo 4 llama a registrar_lote con tiempos de
        embarque/desembarque/ciclo. Modelos anteriores pueden llamarlo solo
        con i, n_visitantes, capacidad_lote.
        """

        r = self._franja_actual()

        # Estadísticas base de ocupación de lote.
        self.num_lotes_iniciados[i] += 1
        self.suma_ocupacion_lotes[i] += n_visitantes / capacidad_lote

        # Si el objeto hereda las métricas operativas del Modelo 3/4, se acumulan.
        if tiempo_embarque is not None:
            self.suma_tiempo_embarque[i] += tiempo_embarque
        if tiempo_desembarque is not None:
            self.suma_tiempo_desembarque[i] += tiempo_desembarque
        if tiempo_ciclo is not None:
            self.suma_tiempo_ciclo[i] += tiempo_ciclo

        if r is not None:
            self.num_lotes_iniciados_franja[r, i] += 1
            self.suma_ocupacion_lotes_franja[r, i] += n_visitantes / capacidad_lote
            if tiempo_embarque is not None:
                self.suma_tiempo_embarque_franja[r, i] += tiempo_embarque
            if tiempo_desembarque is not None:
                self.suma_tiempo_desembarque_franja[r, i] += tiempo_desembarque
            if tiempo_ciclo is not None:
                self.suma_tiempo_ciclo_franja[r, i] += tiempo_ciclo

    def registrar_entrada_parque(self):
        r = self._franja_actual()
        super().registrar_entrada_parque()
        if r is not None:
            self.entradas_parque_franja[r] += 1

    def registrar_salida_parque(self, tiempo_parque, espera_total):
        r = self._franja_actual()
        super().registrar_salida_parque(
            tiempo_parque=tiempo_parque,
            espera_total=espera_total
        )
        if r is not None:
            self.salidas_parque_franja[r] += 1
            self.suma_tiempo_parque_franja[r] += tiempo_parque
            self.suma_espera_parque_franja[r] += espera_total

    def indice_visitas_tabla(self, visitas_completadas):
        v = int(visitas_completadas)
        if v <= 1:
            return 0
        if v >= self.max_visitas_tabla:
            return self.max_visitas_tabla - 1
        return v - 1

    def registrar_oportunidad_salida(self, visitas_completadas):
        k = self.indice_visitas_tabla(visitas_completadas)
        self.oportunidades_salida_por_v[k] += 1

    def registrar_salida_por_visitas(self, visitas_completadas):
        k = self.indice_visitas_tabla(visitas_completadas)
        self.salidas_por_v[k] += 1

    def registrar_eleccion_destino(self, destino, espera_estimada_destino):
        r = self._franja_actual()

        self.destinos_elegidos[destino] += 1
        self.suma_espera_estimada_destino[destino] += espera_estimada_destino
        self.suma_espera_estimada_eleccion += espera_estimada_destino
        self.num_decisiones_continuar += 1

        if r is not None:
            self.destinos_elegidos_franja[r, destino] += 1
            self.suma_espera_estimada_destino_franja[r, destino] += espera_estimada_destino
            self.suma_espera_estimada_eleccion_franja[r] += espera_estimada_destino
            self.num_decisiones_continuar_franja[r] += 1

    def registrar_salida_parque_modelo5(
        self,
        tiempo_parque,
        espera_total,
        visitas_completadas,
        tiempo_movimiento_total
    ):
        r = self._franja_actual()

        self.registrar_salida_parque(
            tiempo_parque=tiempo_parque,
            espera_total=espera_total
        )

        self.suma_visitas_completadas_salida += visitas_completadas
        self.suma_tiempo_movimiento_visitantes_salida += tiempo_movimiento_total

        if r is not None:
            self.suma_visitas_completadas_salida_franja[r] += visitas_completadas
            self.suma_tiempo_movimiento_salida_franja[r] += tiempo_movimiento_total

    def construir_metricas(self, tiempo_limite):
        """
        Construye métricas locales y globales del Modelo 5 actualizado.
        """

        self.actualizar_areas()

        filas_locales = []
        filas_globales = []

        # ----------------------------------------------------
        # Métricas locales agregadas sobre todo el horizonte.
        # ----------------------------------------------------
        total_destinos = self.destinos_elegidos.sum()

        for i, nombre in enumerate(self.nombres):
            salidas_i = self.salidas_nodo[i]
            lotes_i = self.num_lotes_iniciados[i]

            metricas_i = {
                "lambda_efectiva": self.entradas_nodo[i] / tiempo_limite,
                "tasa_salida": self.salidas_nodo[i] / tiempo_limite,
                "rho": self.area_ocupacion[i] / (self.capacidad_total[i] * tiempo_limite) if self.capacidad_total[i] > 0 else np.nan,
                "L": self.area_L[i] / tiempo_limite,
                "Lq": self.area_Lq[i] / tiempo_limite,
                "W": self._ratio(self.suma_W[i], salidas_i),
                "Wq": self._ratio(self.suma_Wq[i], salidas_i),
                "visitas_por_cliente_externo": self._ratio(self.entradas_nodo[i], self.entradas_parque),
                "factor_ocupacion_lote": self._ratio(self.suma_ocupacion_lotes[i], lotes_i),
                "tiempo_medio_embarque": self._ratio(self.suma_tiempo_embarque[i], lotes_i),
                "tiempo_medio_desembarque": self._ratio(self.suma_tiempo_desembarque[i], lotes_i),
                "tiempo_medio_ciclo": self._ratio(self.suma_tiempo_ciclo[i], lotes_i),
                "proporcion_destinos_elegidos": self._ratio(self.destinos_elegidos[i], total_destinos),
                "espera_estimada_media_al_ser_elegida": self._ratio(self.suma_espera_estimada_destino[i], self.destinos_elegidos[i]),
            }

            for metrica, valor in metricas_i.items():
                filas_locales.append({
                    "metrica": metrica,
                    "nodo": nombre,
                    "valor": valor
                })

        # ----------------------------------------------------
        # Métricas locales por franja horaria.
        # ----------------------------------------------------
        for r, etiqueta in enumerate(self.etiquetas_franja):
            dur = self._duracion_franja(r, tiempo_limite)
            den_entradas_parque = self.entradas_parque_franja[r]
            total_destinos_r = self.destinos_elegidos_franja[r, :].sum()
            suf = f"franja_{etiqueta}"

            for i, nombre in enumerate(self.nombres):
                salidas_ri = self.salidas_nodo_franja[r, i]
                lotes_ri = self.num_lotes_iniciados_franja[r, i]

                metricas_ri = {
                    f"lambda_efectiva_{suf}": self.entradas_nodo_franja[r, i] / dur if dur > 0 else np.nan,
                    f"tasa_salida_{suf}": self.salidas_nodo_franja[r, i] / dur if dur > 0 else np.nan,
                    f"rho_{suf}": self.area_ocupacion_franja[r, i] / (self.capacidad_total[i] * dur) if dur > 0 and self.capacidad_total[i] > 0 else np.nan,
                    f"L_{suf}": self.area_L_franja[r, i] / dur if dur > 0 else np.nan,
                    f"Lq_{suf}": self.area_Lq_franja[r, i] / dur if dur > 0 else np.nan,
                    f"W_{suf}": self._ratio(self.suma_W_franja[r, i], salidas_ri),
                    f"Wq_{suf}": self._ratio(self.suma_Wq_franja[r, i], salidas_ri),
                    f"visitas_por_cliente_externo_{suf}": self._ratio(self.entradas_nodo_franja[r, i], den_entradas_parque),
                    f"factor_ocupacion_lote_{suf}": self._ratio(self.suma_ocupacion_lotes_franja[r, i], lotes_ri),
                    f"tiempo_medio_embarque_{suf}": self._ratio(self.suma_tiempo_embarque_franja[r, i], lotes_ri),
                    f"tiempo_medio_desembarque_{suf}": self._ratio(self.suma_tiempo_desembarque_franja[r, i], lotes_ri),
                    f"tiempo_medio_ciclo_{suf}": self._ratio(self.suma_tiempo_ciclo_franja[r, i], lotes_ri),
                    f"proporcion_destinos_elegidos_{suf}": self._ratio(self.destinos_elegidos_franja[r, i], total_destinos_r),
                    f"espera_estimada_media_al_ser_elegida_{suf}": self._ratio(self.suma_espera_estimada_destino_franja[r, i], self.destinos_elegidos_franja[r, i]),
                }

                for metrica, valor in metricas_ri.items():
                    filas_locales.append({
                        "metrica": metrica,
                        "nodo": nombre,
                        "valor": valor
                    })

        # ----------------------------------------------------
        # Métricas globales agregadas sobre todo el horizonte.
        # ----------------------------------------------------
        area_nodos = self.area_L.sum()
        area_colas = self.area_Lq.sum()

        filas_globales.extend([
            {"metrica": "L_global", "nodo": "Global", "valor": (area_nodos + self.area_desplazamiento) / tiempo_limite},
            {"metrica": "Lq_global", "nodo": "Global", "valor": area_colas / tiempo_limite},
            {"metrica": "W_global", "nodo": "Global", "valor": self._ratio(self.suma_tiempo_parque, self.salidas_parque)},
            {"metrica": "Wq_global", "nodo": "Global", "valor": self._ratio(self.suma_espera_parque, self.salidas_parque)},
            {"metrica": "L_desplazamiento", "nodo": "Global", "valor": self.area_desplazamiento / tiempo_limite},
            {"metrica": "tiempo_desplazamiento_medio", "nodo": "Global", "valor": self._ratio(self.suma_tiempo_desplazamiento, self.num_desplazamientos)},
            {"metrica": "desplazamientos_por_cliente_externo", "nodo": "Global", "valor": self._ratio(self.num_desplazamientos, self.entradas_parque)},
            {"metrica": "atracciones_completadas_por_visitante", "nodo": "Global", "valor": self._ratio(self.suma_visitas_completadas_salida, self.salidas_parque)},
            {"metrica": "tiempo_medio_movimiento_por_visitante", "nodo": "Global", "valor": self._ratio(self.suma_tiempo_movimiento_visitantes_salida, self.salidas_parque)},
            {"metrica": "espera_estimada_media_al_elegir_destino", "nodo": "Global", "valor": self._ratio(self.suma_espera_estimada_eleccion, self.num_decisiones_continuar)},
            {"metrica": "destinos_elegidos_por_cliente_externo", "nodo": "Global", "valor": self._ratio(self.num_decisiones_continuar, self.entradas_parque)},
        ])

        for k in range(self.max_visitas_tabla):
            if k < self.max_visitas_tabla - 1:
                etiqueta = f"p_salida_empirica_tras_{k + 1}_visitas"
            else:
                etiqueta = f"p_salida_empirica_tras_{self.max_visitas_tabla}_o_mas_visitas"

            filas_globales.append({
                "metrica": etiqueta,
                "nodo": "Global",
                "valor": self._ratio(self.salidas_por_v[k], self.oportunidades_salida_por_v[k])
            })

        # ----------------------------------------------------
        # Métricas globales por franja horaria.
        # ----------------------------------------------------
        for r, etiqueta in enumerate(self.etiquetas_franja):
            dur = self._duracion_franja(r, tiempo_limite)
            den = self.salidas_parque_franja[r]
            suf_nodo = etiqueta

            filas_globales.extend([
                {"metrica": "L_global", "nodo": suf_nodo, "valor": (self.area_L_franja[r, :].sum() + self.area_desplazamiento_franja[r]) / dur if dur > 0 else np.nan},
                {"metrica": "Lq_global", "nodo": suf_nodo, "valor": self.area_Lq_franja[r, :].sum() / dur if dur > 0 else np.nan},
                {"metrica": "W_global", "nodo": suf_nodo, "valor": self._ratio(self.suma_tiempo_parque_franja[r], den)},
                {"metrica": "Wq_global", "nodo": suf_nodo, "valor": self._ratio(self.suma_espera_parque_franja[r], den)},
                {"metrica": "L_desplazamiento", "nodo": suf_nodo, "valor": self.area_desplazamiento_franja[r] / dur if dur > 0 else np.nan},
                {"metrica": "atracciones_completadas_por_visitante", "nodo": suf_nodo, "valor": self._ratio(self.suma_visitas_completadas_salida_franja[r], den)},
                {"metrica": "tiempo_medio_movimiento_por_visitante", "nodo": suf_nodo, "valor": self._ratio(self.suma_tiempo_movimiento_salida_franja[r], den)},
                {"metrica": "espera_estimada_media_al_elegir_destino", "nodo": suf_nodo, "valor": self._ratio(self.suma_espera_estimada_eleccion_franja[r], self.num_decisiones_continuar_franja[r])},
                {"metrica": "destinos_elegidos_por_cliente_externo", "nodo": suf_nodo, "valor": self._ratio(self.num_decisiones_continuar_franja[r], self.entradas_parque_franja[r])},
            ])

        return pd.DataFrame(filas_locales), pd.DataFrame(filas_globales)


# ------------------------------------------------------------
# 2. Nodo del Modelo 5
# ------------------------------------------------------------

class NodoLotesModelo5(NodoLotesModelo4):
    """
    Atracción del Modelo 5.

    Mantiene:
        - servicio por lotes;
        - embarque/desembarque;
        - desplazamientos triangulares del Modelo 4.

    Cambia:
        - la salida depende de las visitas completadas;
        - el siguiente destino se elige de forma adaptativa.
    """

    def proceso_desplazamiento(self, visitante, siguiente):
        """
        Proceso individual de desplazamiento hacia el siguiente nodo.

        No se añade desplazamiento cuando el visitante sale del parque.
        """

        self.stats.iniciar_desplazamiento()

        t_mov = tiempo_desplazamiento_triangular(self.rng)

        yield self.env.timeout(t_mov)

        self.stats.terminar_desplazamiento(t_mov)

        visitante["tiempo_movimiento_acumulado"] += t_mov
        visitante["t_llegada_nodo"] = self.env.now
        visitante["espera_nodo"] = 0.0

        self.nodos[siguiente].recibir_visitante(visitante)

    def proceso_lote(self, k):
        """
        Proceso de un ciclo/lote de la atracción.
        """

        while True:

            while len(self.cola) == 0:
                yield self.evento_cambio

            primero = self.cola[0]
            t_primero = primero["t_llegada_nodo"]
            t_limite_arranque = t_primero + self.tiempo_max_espera

            while (
                len(self.cola) < self.capacidad_lote
                and self.env.now < t_limite_arranque
            ):
                tiempo_restante = t_limite_arranque - self.env.now
                yield self.evento_cambio | self.env.timeout(tiempo_restante)

            lote = self.extraer_lote()
            n_visitantes = len(lote)

            if n_visitantes == 0:
                continue

            self.stats.mover_cola_a_servicio(
                self.indice,
                n_visitantes
            )

            t_inicio_ciclo = self.env.now

            for visitante in lote:
                espera = t_inicio_ciclo - visitante["t_llegada_nodo"]
                visitante["espera_nodo"] = espera
                visitante["espera_acumulada"] += espera

            t_embarque = tiempo_embarque(self.indice, n_visitantes)
            t_desembarque = tiempo_desembarque(self.indice, n_visitantes)
            t_ciclo = t_embarque + self.duracion_atraccion + t_desembarque

            self.stats.registrar_lote(
                i=self.indice,
                n_visitantes=n_visitantes,
                capacidad_lote=self.capacidad_lote,
                tiempo_embarque=t_embarque,
                tiempo_desembarque=t_desembarque,
                tiempo_ciclo=t_ciclo
            )

            yield self.env.timeout(t_embarque)
            yield self.env.timeout(self.duracion_atraccion)
            yield self.env.timeout(t_desembarque)

            t_fin_ciclo = self.env.now

            self.stats.terminar_servicio_lote(
                self.indice,
                n_visitantes
            )

            for visitante in lote:
                tiempo_total_nodo = t_fin_ciclo - visitante["t_llegada_nodo"]

                self.stats.registrar_tiempos_nodo(
                    self.indice,
                    espera=visitante["espera_nodo"],
                    tiempo_total=tiempo_total_nodo
                )

                visitante["visitas_completadas"] += 1
                v = visitante["visitas_completadas"]

                self.stats.registrar_oportunidad_salida(
                    visitas_completadas=v
                )

                p_salir = probabilidad_salida_por_visitas(v)

                if self.rng.random() < p_salir:
                    self.stats.registrar_salida_por_visitas(
                        visitas_completadas=v
                    )

                    tiempo_parque = t_fin_ciclo - visitante["t_entrada_parque"]

                    self.stats.registrar_salida_parque_modelo5(
                        tiempo_parque=tiempo_parque,
                        espera_total=visitante["espera_acumulada"],
                        visitas_completadas=v,
                        tiempo_movimiento_total=visitante["tiempo_movimiento_acumulado"]
                    )

                else:
                    siguiente, espera_estimada_destino, probabilidades = elegir_destino_adaptativo(
                        i_actual=self.indice,
                        en_cola=self.stats.en_cola,
                        rng=self.rng
                    )

                    self.stats.registrar_eleccion_destino(
                        destino=siguiente,
                        espera_estimada_destino=espera_estimada_destino
                    )

                    self.env.process(
                        self.proceso_desplazamiento(
                            visitante=visitante,
                            siguiente=siguiente
                        )
                    )


# ------------------------------------------------------------
# 3. Simulación de una réplica del Modelo 5 actualizado
# ------------------------------------------------------------

def simular_modelo5(
    tiempo_limite=600,
    semilla=123,
    franjas_llegadas=FRANJAS_LLEGADAS_MODELO5
):
    """
    Simula una réplica del Modelo 5 con métricas por franja horaria.
    """

    rng = np.random.default_rng(semilla)
    env = simpy.Environment()

    stats = EstadisticasModelo5(
        env=env,
        nombres=NOMBRES_NODOS,
        capacidad_total=CAPACIDAD_TOTAL,
        franjas=franjas_llegadas
    )

    nodos = []

    for i, nombre in enumerate(NOMBRES_NODOS):
        nodo = NodoLotesModelo5(
            env=env,
            indice=i,
            nombre=nombre,
            capacidad_lote=CAPACIDAD_LOTE[i],
            num_lotes_paralelos=NUM_LOTES_PARALELOS[i],
            duracion_atraccion=DURACION_ATRACCION[i],
            tiempo_max_espera=TIEMPO_MAX_ESPERA_LOTE[i],
            stats=stats,
            rng=rng
        )
        nodos.append(nodo)

    for nodo in nodos:
        nodo.nodos = nodos

    def generador_llegadas_externas_no_estacionarias(i):
        while env.now < tiempo_limite:
            tasa_i = gamma_nodo_minuto_t(
                t=env.now,
                i=i,
                franjas=franjas_llegadas
            )

            fin_franja = fin_franja_actual(
                t=env.now,
                franjas=franjas_llegadas
            )

            proximo_corte = min(fin_franja, tiempo_limite)

            if tasa_i <= 0:
                yield env.timeout(proximo_corte - env.now)
                continue

            tiempo_entre_llegadas = rng.exponential(1 / tasa_i)

            if env.now + tiempo_entre_llegadas < proximo_corte:
                yield env.timeout(tiempo_entre_llegadas)

                if env.now > tiempo_limite:
                    break

                stats.registrar_entrada_parque()

                visitante = {
                    "t_entrada_parque": env.now,
                    "t_llegada_nodo": env.now,
                    "espera_acumulada": 0.0,
                    "espera_nodo": 0.0,
                    "visitas_completadas": 0,
                    "tiempo_movimiento_acumulado": 0.0
                }

                nodos[i].recibir_visitante(visitante)

            else:
                yield env.timeout(proximo_corte - env.now)

    for i in range(N_NODOS):
        env.process(generador_llegadas_externas_no_estacionarias(i))

    env.run(until=tiempo_limite)

    metricas_locales, metricas_globales = stats.construir_metricas(
        tiempo_limite=tiempo_limite
    )

    return {
        "locales": metricas_locales,
        "globales": metricas_globales,
        "entradas_parque": stats.entradas_parque,
        "salidas_parque": stats.salidas_parque
    }


# ------------------------------------------------------------
# 4. Simulación Monte Carlo del Modelo 5 actualizado
# ------------------------------------------------------------

def ejecutar_monte_carlo_modelo5(
    n_replicas=30,
    tiempo_limite=600,
    semilla_inicial=5000,
    confianza=0.95,
    franjas_llegadas=FRANJAS_LLEGADAS_MODELO5
):
    """
    Ejecuta varias réplicas independientes del Modelo 5 actualizado
    y agrega las métricas mediante estimación Monte Carlo.
    """

    resultados_replicas = []

    for r in range(n_replicas):
        resultado_r = simular_modelo5(
            tiempo_limite=tiempo_limite,
            semilla=semilla_inicial + r,
            franjas_llegadas=franjas_llegadas
        )
        resultados_replicas.append(resultado_r)

    return agregar_replicas(
        resultados_replicas=resultados_replicas,
        confianza=confianza
    )


# ------------------------------------------------------------
# 5. Prueba rápida de funcionamiento
# ------------------------------------------------------------

def prueba_rapida_modelo5(tiempo_limite=120, semilla=123):
    """
    Ejecuta una réplica corta para comprobar que el simulador funciona.
    """

    resultado = simular_modelo5(
        tiempo_limite=tiempo_limite,
        semilla=semilla
    )

    print("Métricas locales de una réplica corta")
    display(resultado["locales"])

    print("Métricas globales de una réplica corta")
    display(resultado["globales"])

    print("Entradas al parque:", resultado["entradas_parque"])
    print("Salidas del parque:", resultado["salidas_parque"])

    return resultado


# ------------------------------------------------------------
# 6. Tablas de métricas por franja
# ------------------------------------------------------------

def tabla_local_franja_modelo5(
    resultado_mc,
    franja,
    metricas_base=None,
    mostrar="ic",
    decimales=4
):
    """
    Tabla local métrica x atracción para una franja concreta.

    franja puede ser:
        - etiqueta, por ejemplo "0-120";
        - índice entero de franja.
    """

    etiquetas = etiquetas_franjas(FRANJAS_LLEGADAS_MODELO5)

    if isinstance(franja, int):
        etiqueta = etiquetas[franja]
    else:
        etiqueta = str(franja)

    if metricas_base is None:
        metricas_base = [
            "lambda_efectiva",
            "tasa_salida",
            "rho",
            "L",
            "Lq",
            "W",
            "Wq",
            "visitas_por_cliente_externo",
            "proporcion_destinos_elegidos",
            "espera_estimada_media_al_ser_elegida"
        ]

    df = resultado_mc["locales"].copy()
    suf = f"_franja_{etiqueta}"
    metricas_franja = [m + suf for m in metricas_base]
    df = df[df["metrica"].isin(metricas_franja)].copy()

    mapa = {m + suf: m for m in metricas_base}
    df["metrica"] = df["metrica"].replace(mapa)

    return tabla_mc_por_nodo(
        df_mc=df,
        nodos=NOMBRES_NODOS,
        mostrar=mostrar,
        decimales=decimales
    )


def tabla_global_franjas_modelo5(
    resultado_mc,
    metricas=None,
    mostrar="ic",
    decimales=4
):
    """
    Tabla global métrica x franja horaria.
    """

    etiquetas = etiquetas_franjas(FRANJAS_LLEGADAS_MODELO5)
    df = resultado_mc["globales"].copy()
    df = df[df["nodo"].isin(etiquetas)].copy()

    if metricas is not None:
        df = df[df["metrica"].isin(metricas)].copy()

    return tabla_mc_por_nodo(
        df_mc=df,
        nodos=etiquetas,
        mostrar=mostrar,
        decimales=decimales
    )


def mostrar_tablas_franjas_modelo5(
    resultado_mc,
    mostrar="ic",
    decimales=4,
    metricas_globales=None,
    metricas_locales=None
):
    """
    Muestra una tabla global por franjas y una tabla local por cada franja.
    """

    print("=" * 70)
    print("MODELO 5. MÉTRICAS GLOBALES POR FRANJA HORARIA")
    print("=" * 70)
    display(tabla_global_franjas_modelo5(
        resultado_mc=resultado_mc,
        metricas=metricas_globales,
        mostrar=mostrar,
        decimales=decimales
    ))

    etiquetas = etiquetas_franjas(FRANJAS_LLEGADAS_MODELO5)

    for etiqueta in etiquetas:
        print("=" * 70)
        print(f"MODELO 5. MÉTRICAS LOCALES POR ATRACCIÓN. FRANJA {etiqueta}")
        print("=" * 70)
        display(tabla_local_franja_modelo5(
            resultado_mc=resultado_mc,
            franja=etiqueta,
            metricas_base=metricas_locales,
            mostrar=mostrar,
            decimales=decimales
        ))


# ------------------------------------------------------------
# 7. Visualizaciones por franja
# ------------------------------------------------------------

def graficar_global_franjas_modelo5(
    resultado_mc,
    metricas=None,
    titulo="Modelo 5. Métricas globales por franja horaria"
):
    """
    Gráfico compacto de métricas globales por franja.
    """

    etiquetas = etiquetas_franjas(FRANJAS_LLEGADAS_MODELO5)
    df = resultado_mc["globales"].copy()
    df = df[df["nodo"].isin(etiquetas)].copy()

    if metricas is None:
        metricas = [
            "L_global",
            "Lq_global",
            "W_global",
            "Wq_global",
            "L_desplazamiento",
            "atracciones_completadas_por_visitante",
            "tiempo_medio_movimiento_por_visitante",
            "espera_estimada_media_al_elegir_destino",
            "destinos_elegidos_por_cliente_externo"
        ]

    df = df[df["metrica"].isin(metricas)].copy()
    df["x"] = pd.Categorical(df["nodo"], categories=etiquetas, ordered=True)
    df["grupo"] = "Modelo 5"

    graficar_metricas_en_grid_pastel(
        df_plot=df,
        metricas=metricas,
        titulo=titulo,
        n_cols=3
    )


def graficar_local_franjas_modelo5(
    resultado_mc,
    metricas_base=None,
    titulo="Modelo 5. Métricas locales por franja horaria"
):
    """
    Para cada métrica base, representa las atracciones en el eje X
    y una línea por franja horaria.
    """

    etiquetas = etiquetas_franjas(FRANJAS_LLEGADAS_MODELO5)

    if metricas_base is None:
        metricas_base = [
            "lambda_efectiva",
            "tasa_salida",
            "rho",
            "L",
            "Lq",
            "W",
            "Wq",
            "visitas_por_cliente_externo",
            "proporcion_destinos_elegidos",
            "espera_estimada_media_al_ser_elegida"
        ]

    df = resultado_mc["locales"].copy()
    filas = []

    for metrica_base in metricas_base:
        for etiqueta in etiquetas:
            nombre_metrica = f"{metrica_base}_franja_{etiqueta}"
            aux = df[df["metrica"] == nombre_metrica].copy()
            if aux.empty:
                continue
            aux["metrica"] = metrica_base
            aux["x"] = pd.Categorical(aux["nodo"], categories=NOMBRES_NODOS, ordered=True)
            aux["grupo"] = etiqueta
            filas.append(aux)

    if not filas:
        print("No hay métricas locales por franja para representar.")
        return

    df_plot = pd.concat(filas, ignore_index=True)

    graficar_metricas_en_grid_pastel(
        df_plot=df_plot,
        metricas=metricas_base,
        titulo=titulo,
        n_cols=3
    )

