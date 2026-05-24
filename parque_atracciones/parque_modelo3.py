# ============================================================
# MODELO 3. SERVICIO POR LOTES + EMBARQUE/DESEMBARQUE
# Versión adaptada a parámetros y funciones genéricas
# ============================================================
from urllib.request import urlretrieve
url = "https://raw.githubusercontent.com/UMH1477/python/refs/heads/main/parque_atracciones/parque_comunes.py"
urlretrieve(url, 'parque_comunes.py')
from parque_comunes import *  

import simpy
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. Clase de estadísticas del Modelo 3
# ------------------------------------------------------------

class EstadisticasModelo3:
    """
    Acumula estadísticas de una réplica del Modelo 3.

    Registra:
        - métricas locales básicas;
        - métricas globales básicas;
        - ocupación de lotes;
        - tiempos medios de embarque, desembarque y ciclo.
    """

    def __init__(self, env, nombres, capacidad_total):

        self.env = env
        self.nombres = nombres
        self.n = len(nombres)
        self.capacidad_total = np.asarray(capacidad_total, dtype=float)

        # Estado instantáneo
        self.en_cola = np.zeros(self.n, dtype=int)
        self.en_servicio = np.zeros(self.n, dtype=int)

        # Áreas temporales
        self.ultimo_t = 0.0
        self.area_L = np.zeros(self.n)
        self.area_Lq = np.zeros(self.n)
        self.area_ocupacion = np.zeros(self.n)

        # Conteos locales
        self.entradas_nodo = np.zeros(self.n, dtype=int)
        self.salidas_nodo = np.zeros(self.n, dtype=int)

        # Tiempos individuales
        self.suma_W = np.zeros(self.n)
        self.suma_Wq = np.zeros(self.n)

        # Lotes
        self.num_lotes_iniciados = np.zeros(self.n, dtype=int)
        self.suma_ocupacion_lotes = np.zeros(self.n)

        # Tiempos operativos de lotes
        self.suma_tiempo_embarque = np.zeros(self.n)
        self.suma_tiempo_desembarque = np.zeros(self.n)
        self.suma_tiempo_ciclo = np.zeros(self.n)

        # Globales
        self.entradas_parque = 0
        self.salidas_parque = 0
        self.suma_tiempo_parque = 0.0
        self.suma_espera_parque = 0.0

    def actualizar_areas(self):
        """
        Actualiza las áreas bajo las curvas de:
            - L;
            - Lq;
            - ocupación.
        """

        ahora = self.env.now
        dt = ahora - self.ultimo_t

        if dt > 0:

            L_inst = self.en_cola + self.en_servicio
            Lq_inst = self.en_cola

            self.area_L += L_inst * dt
            self.area_Lq += Lq_inst * dt
            self.area_ocupacion += self.en_servicio * dt

            self.ultimo_t = ahora

    def registrar_llegada_nodo(self, i):
        """
        Registra llegada de un visitante al nodo i.
        """

        self.actualizar_areas()

        self.entradas_nodo[i] += 1
        self.en_cola[i] += 1

    def mover_cola_a_servicio(self, i, n_visitantes):
        """
        Mueve visitantes desde la cola al lote en servicio.
        """

        self.actualizar_areas()

        self.en_cola[i] -= n_visitantes
        self.en_servicio[i] += n_visitantes

    def terminar_servicio_lote(self, i, n_visitantes):
        """
        Finaliza el servicio de un lote.
        """

        self.actualizar_areas()

        self.en_servicio[i] -= n_visitantes
        self.salidas_nodo[i] += n_visitantes

    def registrar_tiempos_nodo(self, i, espera, tiempo_total):
        """
        Registra tiempos individuales en el nodo.
        """

        self.suma_Wq[i] += espera
        self.suma_W[i] += tiempo_total

    def registrar_lote(
        self,
        i,
        n_visitantes,
        capacidad_lote,
        tiempo_embarque,
        tiempo_desembarque,
        tiempo_ciclo
    ):
        """
        Registra información operativa del lote.
        """

        self.num_lotes_iniciados[i] += 1
        self.suma_ocupacion_lotes[i] += n_visitantes / capacidad_lote
        self.suma_tiempo_embarque[i] += tiempo_embarque
        self.suma_tiempo_desembarque[i] += tiempo_desembarque
        self.suma_tiempo_ciclo[i] += tiempo_ciclo

    def registrar_entrada_parque(self):
        """
        Registra llegada externa al parque.
        """

        self.entradas_parque += 1

    def registrar_salida_parque(self, tiempo_parque, espera_total):
        """
        Registra salida definitiva del parque.
        """

        self.salidas_parque += 1
        self.suma_tiempo_parque += tiempo_parque
        self.suma_espera_parque += espera_total

    def construir_metricas(self, tiempo_limite):
        """
        Construye los DataFrames de métricas locales y globales.
        """

        self.actualizar_areas()

        filas_locales = []

        for i, nombre in enumerate(self.nombres):

            salidas_i = self.salidas_nodo[i]
            lotes_i = self.num_lotes_iniciados[i]

            metricas_i = {
                "lambda_efectiva": self.entradas_nodo[i] / tiempo_limite,

                "tasa_salida": self.salidas_nodo[i] / tiempo_limite,

                "rho": (
                    self.area_ocupacion[i]
                    / (self.capacidad_total[i] * tiempo_limite)
                    if self.capacidad_total[i] > 0
                    else np.nan
                ),

                "L": self.area_L[i] / tiempo_limite,

                "Lq": self.area_Lq[i] / tiempo_limite,

                "W": (
                    self.suma_W[i] / salidas_i
                    if salidas_i > 0
                    else np.nan
                ),

                "Wq": (
                    self.suma_Wq[i] / salidas_i
                    if salidas_i > 0
                    else np.nan
                ),

                "visitas_por_cliente_externo": (
                    self.entradas_nodo[i] / self.entradas_parque
                    if self.entradas_parque > 0
                    else np.nan
                ),

                "factor_ocupacion_lote": (
                    self.suma_ocupacion_lotes[i] / lotes_i
                    if lotes_i > 0
                    else np.nan
                ),

                "tiempo_embarque_medio_lote": (
                    self.suma_tiempo_embarque[i] / lotes_i
                    if lotes_i > 0
                    else np.nan
                ),

                "tiempo_desembarque_medio_lote": (
                    self.suma_tiempo_desembarque[i] / lotes_i
                    if lotes_i > 0
                    else np.nan
                ),

                "tiempo_ciclo_medio_lote": (
                    self.suma_tiempo_ciclo[i] / lotes_i
                    if lotes_i > 0
                    else np.nan
                )
            }

            for metrica, valor in metricas_i.items():

                filas_locales.append({
                    "metrica": metrica,
                    "nodo": nombre,
                    "valor": valor
                })

        metricas_locales = pd.DataFrame(filas_locales)

        metricas_globales = pd.DataFrame({
            "metrica": [
                "L_global",
                "Lq_global",
                "W_global",
                "Wq_global"
            ],
            "nodo": ["Global"] * 4,
            "valor": [
                self.area_L.sum() / tiempo_limite,

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
                )
            ]
        })

        return metricas_locales, metricas_globales


# ------------------------------------------------------------
# 2. Nodo con servicio por lotes, embarque y desembarque
# ------------------------------------------------------------

class NodoLotesModelo3:
    """
    Atracción con servicio por lotes, embarque y desembarque.

    El lote arranca cuando:
        - hay suficientes visitantes para llenar el lote; o
        - el primer visitante del lote alcanza su tiempo máximo de espera.

    Una vez arrancado:
        - embarque;
        - disfrute de la atracción;
        - desembarque.
    """

    def __init__(
        self,
        env,
        indice,
        nombre,
        capacidad_lote,
        num_lotes_paralelos,
        duracion_atraccion,
        tiempo_max_espera,
        stats,
        rng
    ):

        self.env = env
        self.indice = indice
        self.nombre = nombre

        self.capacidad_lote = int(capacidad_lote)
        self.num_lotes_paralelos = int(num_lotes_paralelos)
        self.duracion_atraccion = float(duracion_atraccion)
        self.tiempo_max_espera = float(tiempo_max_espera)

        self.stats = stats
        self.rng = rng

        # Cola física de visitantes esperando.
        self.cola = []

        # Evento que despierta a los procesos de lote.
        self.evento_cambio = env.event()

        # Se lanzan tantos procesos de lote como unidades paralelas tenga el nodo.
        for k in range(self.num_lotes_paralelos):
            env.process(self.proceso_lote(k))

    def avisar_cambio(self):
        """
        Despierta a los procesos de lote cuando cambia la cola.
        """

        if not self.evento_cambio.triggered:
            self.evento_cambio.succeed()

        self.evento_cambio = self.env.event()

    def recibir_visitante(self, visitante):
        """
        Recibe un visitante y lo añade a la cola del nodo.
        """

        self.stats.registrar_llegada_nodo(self.indice)

        self.cola.append(visitante)

        self.avisar_cambio()

    def extraer_lote(self):
        """
        Extrae el siguiente lote de visitantes de la cola.
        """

        n_lote = min(
            self.capacidad_lote,
            len(self.cola)
        )

        lote = self.cola[:n_lote]
        self.cola = self.cola[n_lote:]

        return lote

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

                    visitante["t_llegada_nodo"] = t_fin_ciclo
                    visitante["espera_nodo"] = 0.0

                    self.nodos[siguiente].recibir_visitante(visitante)



# ------------------------------------------------------------
# 3. Simulación de una réplica del Modelo 3
# ------------------------------------------------------------

def simular_modelo3(
    tiempo_limite=600,
    semilla=123,
    gamma_hora=600
):
    """
    Simula una réplica del Modelo 3.

    Características:
        - llegadas externas Poisson;
        - encaminamiento uniforme;
        - servicio por lotes;
        - embarque dependiente del tamaño del lote;
        - disfrute de duración fija;
        - desembarque dependiente del tamaño del lote;
        - Flotador Acuático con 10 flotadores en paralelo.
    """

    rng = np.random.default_rng(semilla)

    env = simpy.Environment()

    gamma_total, gamma_vec, P = parametros_encaminamiento_uniforme(
        gamma_hora=gamma_hora
    )

    stats = EstadisticasModelo3(
        env=env,
        nombres=NOMBRES_NODOS,
        capacidad_total=CAPACIDAD_TOTAL
    )

    nodos = []

    for i, nombre in enumerate(NOMBRES_NODOS):

        nodo = NodoLotesModelo3(
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
# 3. Simulación de una réplica del Modelo 3
# ------------------------------------------------------------

def simular_modelo3(
    tiempo_limite=600,
    semilla=123,
    gamma_hora=600
):
    """
    Simula una réplica del Modelo 3.

    Características:
        - llegadas externas Poisson;
        - encaminamiento uniforme;
        - servicio por lotes;
        - embarque dependiente del tamaño del lote;
        - disfrute de duración fija;
        - desembarque dependiente del tamaño del lote;
        - Flotador Acuático con 10 flotadores en paralelo.
    """

    rng = np.random.default_rng(semilla)

    env = simpy.Environment()

    gamma_total, gamma_vec, P = parametros_encaminamiento_uniforme(
        gamma_hora=gamma_hora
    )

    stats = EstadisticasModelo3(
        env=env,
        nombres=NOMBRES_NODOS,
        capacidad_total=CAPACIDAD_TOTAL
    )

    nodos = []

    for i, nombre in enumerate(NOMBRES_NODOS):

        nodo = NodoLotesModelo3(
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
# 4. Simulación Monte Carlo del Modelo 3
# ------------------------------------------------------------

def ejecutar_monte_carlo_modelo3(
    n_replicas=30,
    tiempo_limite=600,
    gamma_hora=600,
    semilla_inicial=3000,
    confianza=0.95
):
    """
    Ejecuta varias réplicas independientes del Modelo 3
    y agrega las métricas mediante estimación Monte Carlo.
    """

    resultados_replicas = []

    for r in range(n_replicas):

        resultado_r = simular_modelo3(
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

def prueba_rapida_modelo3():
    """
    Ejecuta una réplica corta para comprobar que el simulador funciona.
    """

    resultado = simular_modelo3(
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
