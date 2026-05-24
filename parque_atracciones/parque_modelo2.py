# ============================================================
# MODELO 2. SERVICIO POR LOTES
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
# 1. Clase de estadísticas del Modelo 2
# ------------------------------------------------------------

class EstadisticasModelo2:
    """
    Acumula las estadísticas de una réplica del Modelo 2.

    En este modelo hay servicio por lotes, por lo que se registran:
        - número esperando en cola;
        - número en servicio dentro de lotes activos;
        - número total en el nodo = cola + servicio;
        - ocupación temporal de la capacidad;
        - ocupación media de los lotes al arrancar.
    """

    def __init__(self, env, nombres, capacidad_total):

        self.env = env
        self.nombres = nombres
        self.n = len(nombres)
        self.capacidad_total = np.asarray(capacidad_total, dtype=float)

        # Estado instantáneo por nodo
        self.en_cola = np.zeros(self.n, dtype=int)
        self.en_servicio = np.zeros(self.n, dtype=int)

        # Último instante de actualización de áreas
        self.ultimo_t = 0.0

        # Áreas temporales
        self.area_L = np.zeros(self.n)
        self.area_Lq = np.zeros(self.n)
        self.area_ocupacion = np.zeros(self.n)

        # Conteos locales
        self.entradas_nodo = np.zeros(self.n, dtype=int)
        self.salidas_nodo = np.zeros(self.n, dtype=int)

        # Tiempos locales
        self.suma_W = np.zeros(self.n)
        self.suma_Wq = np.zeros(self.n)

        # Lotes
        self.num_lotes_iniciados = np.zeros(self.n, dtype=int)
        self.suma_ocupacion_lotes = np.zeros(self.n)

        # Globales
        self.entradas_parque = 0
        self.salidas_parque = 0
        self.suma_tiempo_parque = 0.0
        self.suma_espera_parque = 0.0

    def actualizar_areas(self):
        """
        Actualiza las áreas bajo las curvas de:
            L, Lq y ocupación.
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
        Registra la llegada de un visitante al nodo i.
        """

        self.actualizar_areas()

        self.entradas_nodo[i] += 1
        self.en_cola[i] += 1

    def mover_cola_a_servicio(self, i, n_visitantes):
        """
        Mueve n visitantes desde la cola al servicio.
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

    def registrar_lote(self, i, n_visitantes, capacidad_lote):
        """
        Registra la ocupación de un lote al arrancar.
        """

        self.num_lotes_iniciados[i] += 1
        self.suma_ocupacion_lotes[i] += n_visitantes / capacidad_lote

    def registrar_entrada_parque(self):
        """
        Registra una llegada externa al parque.
        """

        self.entradas_parque += 1

    def registrar_salida_parque(self, tiempo_parque, espera_total):
        """
        Registra la salida definitiva de un visitante del parque.
        """

        self.salidas_parque += 1
        self.suma_tiempo_parque += tiempo_parque
        self.suma_espera_parque += espera_total

    def construir_metricas(self, tiempo_limite):
        """
        Construye las métricas locales y globales de la réplica.
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
# 2. Nodo con servicio por lotes de duración fija
# ------------------------------------------------------------

class NodoLotesModelo2:
    """
    Representa una atracción con servicio por lotes.

    La atracción arranca un lote cuando:
        1. hay suficientes visitantes para llenar el lote; o
        2. el primer visitante del lote lleva esperando al menos
           tiempo_max_espera.

    Puede haber varios lotes en paralelo, como ocurre en el
    Flotador Acuático.
    """

    def __init__(
        self,
        env,
        indice,
        nombre,
        capacidad_lote,
        num_lotes_paralelos,
        duracion_servicio,
        tiempo_max_espera,
        stats,
        rng
    ):

        self.env = env
        self.indice = indice
        self.nombre = nombre

        self.capacidad_lote = int(capacidad_lote)
        self.num_lotes_paralelos = int(num_lotes_paralelos)
        self.duracion_servicio = float(duracion_servicio)
        self.tiempo_max_espera = float(tiempo_max_espera)

        self.stats = stats
        self.rng = rng

        # Cola física de visitantes esperando.
        # Cada elemento es un diccionario con tiempos acumulados.
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
        Añade un visitante a la cola del nodo.
        """

        self.stats.registrar_llegada_nodo(self.indice)

        self.cola.append(visitante)

        self.avisar_cambio()

    def extraer_lote(self):
        """
        Extrae de la cola los visitantes que formarán el siguiente lote.
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
        Proceso de un lote/ciclo de la atracción.
        """

        while True:

            # ------------------------------------------------
            # Esperar a que haya al menos un visitante
            # ------------------------------------------------

            while len(self.cola) == 0:
                yield self.evento_cambio

            # ------------------------------------------------
            # Esperar hasta llenar el lote o hasta alcanzar
            # el tiempo máximo de espera del primero.
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

            self.stats.registrar_lote(
                self.indice,
                n_visitantes,
                self.capacidad_lote
            )

            t_inicio_servicio = self.env.now

            # Esperas individuales del lote.
            for visitante in lote:

                espera = t_inicio_servicio - visitante["t_llegada_nodo"]

                visitante["espera_nodo"] = espera
                visitante["espera_acumulada"] += espera

            # ------------------------------------------------
            # Servicio fijo de la atracción
            # ------------------------------------------------

            yield self.env.timeout(self.duracion_servicio)

            t_fin_servicio = self.env.now

            self.stats.terminar_servicio_lote(
                self.indice,
                n_visitantes
            )

            # ------------------------------------------------
            # Salida del nodo y encaminamiento visitante a visitante
            # ------------------------------------------------

            for visitante in lote:

                tiempo_total_nodo = (
                    t_fin_servicio - visitante["t_llegada_nodo"]
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
                        t_fin_servicio - visitante["t_entrada_parque"]
                    )

                    self.stats.registrar_salida_parque(
                        tiempo_parque=tiempo_parque,
                        espera_total=visitante["espera_acumulada"]
                    )

                else:

                    visitante["t_llegada_nodo"] = t_fin_servicio
                    visitante["espera_nodo"] = 0.0

                    self.nodos[siguiente].recibir_visitante(visitante)


# ------------------------------------------------------------
# 3. Simulación de una réplica del Modelo 2
# ------------------------------------------------------------

def simular_modelo2(
    tiempo_limite=600,
    semilla=123,
    gamma_hora=600
):
    """
    Simula una réplica del Modelo 2.

    Características:
        - llegadas externas Poisson;
        - encaminamiento uniforme;
        - servicio por lotes;
        - duración fija de cada atracción;
        - arranque por lote lleno o por tiempo máximo de espera;
        - Flotador Acuático con 10 flotadores en paralelo de capacidad 2.
    """

    rng = np.random.default_rng(semilla)

    env = simpy.Environment()

    gamma_total, gamma_vec, P = parametros_encaminamiento_uniforme(
        gamma_hora=gamma_hora
    )

    stats = EstadisticasModelo2(
        env=env,
        nombres=NOMBRES_NODOS,
        capacidad_total=CAPACIDAD_TOTAL
    )

    nodos = []

    for i, nombre in enumerate(NOMBRES_NODOS):

        nodo = NodoLotesModelo2(
            env=env,
            indice=i,
            nombre=nombre,
            capacidad_lote=CAPACIDAD_LOTE[i],
            num_lotes_paralelos=NUM_LOTES_PARALELOS[i],
            duracion_servicio=DURACION_ATRACCION[i],
            tiempo_max_espera=TIEMPO_MAX_ESPERA_LOTE[i],
            stats=stats,
            rng=rng
        )

        nodos.append(nodo)

    # Damos a cada nodo acceso a la lista completa de nodos
    # para poder encaminar visitantes después del servicio.
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
# 4. Simulación Monte Carlo del Modelo 2
# ------------------------------------------------------------

def ejecutar_monte_carlo_modelo2(
    n_replicas=30,
    tiempo_limite=600,
    gamma_hora=600,
    semilla_inicial=2000,
    confianza=0.95
):
    """
    Ejecuta varias réplicas independientes del Modelo 2
    y agrega las métricas mediante estimación Monte Carlo.
    """

    resultados_replicas = []

    for r in range(n_replicas):

        resultado_r = simular_modelo2(
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

def prueba_rapida_modelo2():
    """
    Ejecuta una réplica corta para comprobar que el simulador funciona.
    """

    resultado = simular_modelo2(
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
