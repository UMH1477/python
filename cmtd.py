import numpy as np
import pandas as pd
# Instalación e importación
! pip install pydtmc
! pip install Graphviz
! pip install pydot
import pydtmc
from pydtmc import MarkovChain

#------------------------------------------------------------------------------------

def cmtd_matrix_n(mc, n):
  """
  Función para obtener la matriz de transición de n pasos
  dada un proceso definido con MarkovChain()

  Parámetros de entrada:
    - mc: proceso definido con MarkovChain()
    - n: número de saltos.

  Parámetros de salida:
    - p_n: matriz de transición de n pasos.
  """
  
#------------------------------------------------------------------------------------
# Matriz de ocupación del proceso

def mat_ocupacion_proceso(mc, n):
  """
  Función para obtener la matriz de ocupación asocida al proceso mc en n transiciones

  Parámetros de entrada:
  - mc: proceso
  - n: número de transiciones

  Parámetros de salida:
  - mocupa: matriz de ocupacion
  """
  mocupa = np.zeros((len(mc.states), len(mc.states)))
  for i in range(n+1):
    mocupa += np.linalg.matrix_power(mc.p, i)

  return mocupa
