# ============================================================
# MODELO 4. MODELO 3 + DESPLAZAMIENTOS ENTRE ATRACCIONES
# ============================================================
from urllib.request import urlretrieve
url = "https://raw.githubusercontent.com/UMH1477/python/refs/heads/main/parque_atracciones/parque_comunes.py"
urlretrieve(url, 'parque_comunes.py')
from parque_comunes import *  

import importlib
from urllib.request import urlretrieve

for i in range(1,4):
  model_name = f'parque_modelo{i}'
  url = f"https://raw.githubusercontent.com/UMH1477/python/refs/heads/main/parque_atracciones/{model_name}.py"
  urlretrieve(url, f'{model_name}.py')
  # Dynamically import all contents of the module into the current namespace
  # This is equivalent to 'from module_name import *'
  exec(f"from {model_name} import *", globals())
    
import simpy
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. Estadísticas del Modelo 4
# ------------------------------------------------------------

class EstadisticasModelo4(EstadisticasModelo3):
    """
    Estadísticas de una réplica del Modelo 4.

    Hereda las estadísticas locales del Modelo 3 y añade:
        - visitantes en desplazamiento;
        - área temporal de visitantes desplazándose;
        - tiempos medios de desplazamiento;
        - L_global incluyendo atracciones, colas y desplazamientos.

    Las métricas locales por nodo se mantienen comparables con el Modelo 3:
        L, Lq, W y Wq se calculan desde la llegada efectiva al nodo.
    """

    def __init__(self, env, nombres, capacidad_total):

        super().__init__(env, nombres, capacidad_total)

        # Estado y área de visitantes caminando entre atracciones.
        self.en_desplazamiento = 0
        self.area_desplazamiento = 0.0

        # Conteos y tiempos de desplazamiento.
        self.num_desplazamientos = 0
        self.suma_tiempo_desplazamiento = 0.0

    def actualizar_areas(self):
        """
        Actualiza áreas de nodos y área global de desplazamiento.
        """

        ahora = self.env.now
        dt = ahora - self.ultimo_t

        if dt > 0:

            L_inst = self.en_cola + self.en_servicio
            Lq_inst = self.en_cola

            self.area_L += L_inst * dt
            self.area_Lq += Lq_inst * dt
            self.area_ocupacion += self.en_servicio * dt
            self.area_desplazamiento += self.en_desplazamiento * dt

            self.ultimo_t = ahora

    def iniciar_desplazamiento(self):
        """
        Registra el inicio de un desplazamiento entre dos atracciones.
        """

        self.actualizar_areas()
        self.en_desplazamiento += 1

    def terminar_desplazamiento(self, tiempo_desplazamiento):
        """
        Registra el final de un desplazamiento entre dos atracciones.
        """

        self.actualizar_areas()
        self.en_desplazamiento -= 1
        self.num_desplazamientos += 1
        self.suma_tiempo_desplazamiento += tiempo_desplazamiento

    def construir_metricas(self, tiempo_limite):
        """
        Construye métricas locales y globales del Modelo 4.

        A nivel global:
            L_global incluye visitantes en nodos y en desplazamiento.
            Lq_global solo incluye visitantes esperando en colas.
            W_global incluye permanencia total en el parque, con desplazamientos.
            Wq_global solo incluye esperas en colas.
        """

        metricas_locales, metricas_globales = super().construir_metricas(
            tiempo_limite=tiempo_limite
        )

        area_nodos = self.area_L.sum()

        metricas_globales = pd.DataFrame({
            "metrica": [
                "L_global",
                "Lq_global",
                "W_global",
                "Wq_global",
                "L_desplazamiento",
                "tiempo_desplazamiento_medio",
                "desplazamientos_por_cliente_externo"
            ],
            "nodo": ["Global"] * 7,
            "valor": [
                (area_nodos + self.area_desplazamiento) / tiempo_limite,

                self.area_Lq.sum() / tiempo_limite,

                (
                    self.suma_tiempo_parque / self.salidas_parque
                    if self.salidas_parque > 0
                    else np.nan
                ),

                (
                    self.suma_espera_parque / self.salidas_parque
                    if self.salidas_parque > 0
                    else np.nan
                ),

                self.area_desplazamiento / tiempo_limite,

                (
                    self.suma_tiempo_desplazamiento / self.num_desplazamientos
                    if self.num_desplazamientos > 0
                    else np.nan
                ),

                (
                    self.num_desplazamientos / self.entradas_parque
                    if self.entradas_parque > 0
                    else np.nan
                )
            ]
        })

        return metricas_locales, metricas_globales


# ------------------------------------------------------------
# 2. Nodo del Modelo 4
# ------------------------------------------------------------

class NodoLotesModelo4(NodoLotesModelo3):
    """
    Atracción del Modelo 4.

    Mantiene el funcionamiento del Modelo 3 y, tras finalizar una atracción,
    si el visitante no sale del parque, genera un proceso de desplazamiento
    hasta la siguiente atracción.
    """

    def proceso_desplazamiento(self, visitante, siguiente):
        """
        Proceso individual de desplazamiento hacia el siguiente nodo.
        """

        self.stats.iniciar_desplazamiento()

        t_mov = tiempo_desplazamiento_triangular(self.rng)

        yield self.env.timeout(t_mov)

        self.stats.terminar_desplazamiento(t_mov)

        visitante["t_llegada_nodo"] = self.env.now
        visitante["espera_nodo"] = 0.0

        self.nodos[siguiente].recibir_visitante(visitante)

    def proceso_lote(self, k):
        """
        Proceso de un ciclo/lote de la atracción.
        """

        while True:

            # ------------------------------------------------
            # Esperar hasta que haya al menos un visitante
            # ------------------------------------------------

            while len(self.cola) == 0:
                yield self.evento_cambio

            # ------------------------------------------------
            # Esperar hasta llenar el lote o hasta el tiempo
            # máximo del primer visitante.
            # ------------------------------------------------

            primero = self.cola[0]

            t_primero = primero["t_llegada_nodo"]
            t_limite_arranque = t_primero + self.tiempo_max_espera

            while (
                len(self.cola) < self.capacidad_lote
                and self.env.now < t_limite_arranque
            ):

                tiempo_restante = t_limite_arranque - self.env.now

                yield (
                    self.evento_cambio
                    | self.env.timeout(tiempo_restante)
                )

            # ------------------------------------------------
            # Arranque del lote
            # ------------------------------------------------

            lote = self.extraer_lote()
            n_visitantes = len(lote)

            if n_visitantes == 0:
                continue

            self.stats.mover_cola_a_servicio(
                self.indice,
                n_visitantes
            )

            t_inicio_ciclo = self.env.now

            # Esperas individuales de los visitantes del lote.
            for visitante in lote:

                espera = t_inicio_ciclo - visitante["t_llegada_nodo"]

                visitante["espera_nodo"] = espera
                visitante["espera_acumulada"] += espera

            # ------------------------------------------------
            # Tiempos dependientes del tamaño real del lote
            # ------------------------------------------------

            t_embarque = tiempo_embarque(
                self.indice,
                n_visitantes
            )

            t_desembarque = tiempo_desembarque(
                self.indice,
                n_visitantes
            )

            t_ciclo = (
                t_embarque
                + self.duracion_atraccion
                + t_desembarque
            )

            self.stats.registrar_lote(
                i=self.indice,
                n_visitantes=n_visitantes,
                capacidad_lote=self.capacidad_lote,
                tiempo_embarque=t_embarque,
                tiempo_desembarque=t_desembarque,
                tiempo_ciclo=t_ciclo
            )

            # Embarque.
            yield self.env.timeout(t_embarque)

            # Disfrute de la atracción.
            yield self.env.timeout(self.duracion_atraccion)

            # Desembarque.
            yield self.env.timeout(t_desembarque)

            t_fin_ciclo = self.env.now

            self.stats.terminar_servicio_lote(
                self.indice,
                n_visitantes
            )

            # ------------------------------------------------
            # Salida del nodo y encaminamiento visitante a visitante
            # ------------------------------------------------

            for visitante in lote:

                tiempo_total_nodo = (
                    t_fin_ciclo - visitante["t_llegada_nodo"]
                )

                self.stats.registrar_tiempos_nodo(
                    self.indice,
                    espera=visitante["espera_nodo"],
                    tiempo_total=tiempo_total_nodo
                )

                siguiente = elegir_siguiente_nodo_uniforme(
                    i_actual=self.indice,
                    rng=self.rng
                )

                if siguiente == -1:

                    tiempo_parque = (
                        t_fin_ciclo - visitante["t_entrada_parque"]
                    )

                    self.stats.registrar_salida_parque(
                        tiempo_parque=tiempo_parque,
                        espera_total=visitante["espera_acumulada"]
                    )

                else:

                    self.env.process(
                        self.proceso_desplazamiento(
                            visitante=visitante,
                            siguiente=siguiente
                        )
                    )


# ------------------------------------------------------------
# 3. Simulación de una réplica del Modelo 4
# ------------------------------------------------------------

def simular_modelo4(
    tiempo_limite=600,
    semilla=123,
    gamma_hora=600
):
    """
    Simula una réplica del Modelo 4.

    Características:
        - mantiene toda la modelización del Modelo 3;
        - añade desplazamiento aleatorio entre atracciones;
        - T_mov ~ Triangular(1, 2, 4), en minutos;
        - las llegadas externas entran directamente en su primera atracción;
        - los desplazamientos internos cuentan en la permanencia global
          y en L_global, pero no en las métricas locales de atracción.
    """

    rng = np.random.default_rng(semilla)

    env = simpy.Environment()

    gamma_total, gamma_vec, P = parametros_encaminamiento_uniforme(
        gamma_hora=gamma_hora
    )

    stats = EstadisticasModelo4(
        env=env,
        nombres=NOMBRES_NODOS,
        capacidad_total=CAPACIDAD_TOTAL
    )

    nodos = []

    for i, nombre in enumerate(NOMBRES_NODOS):

        nodo = NodoLotesModelo4(
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

    # --------------------------------------------------------
    # Generadores de llegadas externas
    # --------------------------------------------------------

    def generador_llegadas_externas(i):
        """
        Genera llegadas externas al nodo i.
        """

        while True:

            tiempo_entre_llegadas = rng.exponential(
                1 / gamma_vec[i]
            )

            yield env.timeout(tiempo_entre_llegadas)

            if env.now > tiempo_limite:
                break

            stats.registrar_entrada_parque()

            visitante = {
                "t_entrada_parque": env.now,
                "t_llegada_nodo": env.now,
                "espera_acumulada": 0.0,
                "espera_nodo": 0.0
            }

            nodos[i].recibir_visitante(visitante)

    for i in range(N_NODOS):
        env.process(generador_llegadas_externas(i))

    # --------------------------------------------------------
    # Ejecución de la réplica
    # --------------------------------------------------------

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
# 4. Simulación Monte Carlo del Modelo 4
# ------------------------------------------------------------

def ejecutar_monte_carlo_modelo4(
    n_replicas=30,
    tiempo_limite=600,
    gamma_hora=600,
    semilla_inicial=4000,
    confianza=0.95
):
    """
    Ejecuta varias réplicas independientes del Modelo 4
    y agrega las métricas mediante estimación Monte Carlo.
    """

    resultados_replicas = []

    for r in range(n_replicas):

        resultado_r = simular_modelo4(
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
# 5. Prueba rápida de funcionamiento
# ------------------------------------------------------------

def prueba_rapida_modelo4():
    """
    Ejecuta una réplica corta para comprobar que el simulador funciona.
    """

    resultado = simular_modelo4(
        tiempo_limite=60,
        semilla=123,
        gamma_hora=600
    )

    print("Métricas locales de una réplica corta")
    display(resultado["locales"])

    print("Métricas globales de una réplica corta")
    display(resultado["globales"])

    print("Entradas al parque:", resultado["entradas_parque"])
    print("Salidas del parque:", resultado["salidas_parque"])

    return resultado
