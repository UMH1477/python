# -*- coding: utf-8 -*-
# Funciones básicas para el bloque 1. Distribuciones y CMTD
import numpy as np          # importamos numpy como np
import pandas as pd         # importamos pandas como pd
import math
import random

from scipy import stats, optimize
from scipy.special import gamma

# Cargamos módulos de análisis gráficos
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set_theme(style = 'whitegrid')
# %config InlineBackend.figure_format = 'retina'


#============================================================================================
# DISTRIBUCIONES DISCRETAS
#============================================================================================

def graficar_discreta(x, fx):
  """
  Función para representar gráficamente la función de masa de probabilidad y la función de distribución de una variable discreta.

  Args:
    x: valores de la variable discreta
    fx: función de masa de probabilidad para cada valor de x

  Returns:
    Gráficos de la función de masa de probabilidad y la función de distribución.
  """
  # posiciones en el gráfico de los valores d ela variable discreta
  pos = np.arange(len(x))
  # Función de distribución
  fdist = [sum(fx[:(l+1)]) for l in range(len(fx))]

  # Entorno gráfico
  fig, ax = plt.subplots(1, 2, figsize=(7, 4))
  # Pintamos los puntos con x y la función de masa de probabilidad
  ax[0].plot(pos, fx, 'bo');
  # Dibujamos las líneas verticales correspondientes con sus caractarísticas
  ax[0].vlines(pos, 0, fx, colors='b', lw=5, alpha=0.5);
  # Ponemos un título
  ax[0].set_title('Función de masa de probabilidad')
  # Ponemos etiquetas a los ejes x e y
  ax[0].set_xticks(pos, labels=x)
  ax[0].set_ylabel('Probabilidad')
  ax[0].set_xlabel('Espacio muestral')

  #### Función de distribución
  # Pintamos los puntos con x y la función de distribución
  ax[1].plot(pos, fdist, 'bo');
  # Dibujamos las líneas verticales correspondientes con sus caractarísticas
  ax[1].vlines(pos, 0, fdist, colors='b', lw=5, alpha=0.5);
  # Ponemos un título
  ax[1].set_title('Función de distribución')
  # Ponemos etiquetas a los ejes x e y
  ax[1].set_xticks(pos, labels=x)
  ax[1].set_ylabel('Probabilidad')
  ax[1].set_xlabel('Espacio muestral')
  plt.tight_layout()
#---------------------------------------------------------------------------------------
# Función para obtener un dataframe con la función de masa de probabilidad y la función de distribución de una variable discreta.
def distr_discreta(x, fx):
  """
  Función para obtener un dataframe con la función de masa de probabilidad y la función de distribución de una variable discreta.

  Args:
    x: valores de la varaible discreta
    fx: función de masa de probabilidad

  Returns:
    pdDataFrame con los valores de la variable, la función de masa de probabilidad y la función de distribución.
  """
  # posiciones en el gráfico de los valores d ela variable discreta
  pos = np.arange(len(x))
  # Función de distribución
  fdist = [sum(fx[:(l+1)]) for l in range(len(fx))]
  return(pd.DataFrame({"x": x, "fmp":fx, "fdist":fdist}))

#---------------------------------------------------------------------------------------
# Simular una m.a. de una distribución discreta y devolver media y varianza
def simula_discreta(x, fx, n):
  """
  Función para simular una m.a. de una distribución discreta y devolver
  media y varianza de los datos simulados.
  Args:
    x: valores de la variable discreta
    fx: función de masa de probabilidad

  Returns
    Lista con el valor medio y desviación típica de la variable de interés en el periodo n de simulación
  """
  muestra = np.random.choice(x, size = n, replace = True, p = fx)
  resul = [round(muestra.mean(),2), round(muestra.var(),2)]
  return(resul)
#============================================================================================

# ESTIMACIÓN MONTE CARLO
# Función para obtener el estimador Monte Carlo de h(x) y un intervalo de confianza al 95%
def MC_estim(sims):
  """
  Función para obtener el estimador Monte Carlo de h(x) y un intervalo de confianza al 95%

  Args:
   sims: Si queremos un estimador de h(x) pasamos directamente als simulaciones,
          mientras que si deseamos una probabildiad debemos pasar el vector 1-0
          que cumple con las condiciones de la probabilidad buscada

  Returns: 
    Devuelve el estimador e intervalo de confianza por Monte Carlo
  """
  from scipy.stats import norm

  # Número de simulaciones cargadas
  size = len(sims)
  # Estimador MC
  estim = sims.mean()
  # Estimador MC del IC
  error = math.sqrt(sims.var())*math.sqrt(size-1)/size
  cuantil = norm.ppf(1-0.05/2)
  ic_low = estim - cuantil*error
  ic_up = estim + cuantil*error
  # Resultado
  return([round(estim,4), round(ic_low,4), round(ic_up,4)])

#============================================================================================
# AJUSTAR Y COMPARAR DISTRIBUCIONES DISCRETAS/CONTINUAS
#============================================================================================
import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------------------------
# 1. FUNCIÓN AUXILIAR  para chequear y comparar distribuciones discretas
# ------------------------------------------------------------------------------
def calculate_chi2_robust(data, dist_name, params, n_params_est):
    """
    Realiza el test Chi-Cuadrado con agrupación dinámica (binning) 
    y normalización de probabilidades.
    """
    observed_counts = pd.Series(data).value_counts().sort_index()
    total_n = len(data)
    k_values = np.arange(observed_counts.index.min(), observed_counts.index.max() + 1)
    
    # --- Selección de PMF teórica ---
    if dist_name == 'poisson':
        probs = stats.poisson.pmf(k_values, params[0])
    elif dist_name == 'geom':
        probs = stats.geom.pmf(k_values, params[0], loc=params[1])
    elif dist_name == 'binom':
        probs = stats.binom.pmf(k_values, params[0], params[1])
    elif dist_name == 'nbinom':
        probs = stats.nbinom.pmf(k_values, params[0], params[1])
    elif dist_name == 'hypergeom':
        # M, n, N = params
        probs = stats.hypergeom.pmf(k_values, params[0], params[1], params[2])
            
    expected_freqs = probs * total_n
    
    # Mapear observados
    obs_dict = observed_counts.to_dict()
    observed_freqs = np.array([obs_dict.get(k, 0) for k in k_values])
    
    # Agrupación (Binning) - Regla de Cochran
    obs_grouped, exp_grouped = [], []
    curr_obs, curr_exp = 0, 0
    
    for o, e in zip(observed_freqs, expected_freqs):
        curr_obs += o
        curr_exp += e
        if curr_exp >= 5:
            obs_grouped.append(curr_obs)
            exp_grouped.append(curr_exp)
            curr_obs, curr_exp = 0, 0
            
    if curr_exp > 0:
        if len(exp_grouped) > 0:
            exp_grouped[-1] += curr_exp
            obs_grouped[-1] += curr_obs
        else:
            exp_grouped.append(curr_exp)
            obs_grouped.append(curr_obs)

    obs_final = np.array(obs_grouped)
    exp_final = np.array(exp_grouped)
    
    if np.sum(exp_final) > 0:
        exp_final = exp_final * (np.sum(obs_final) / np.sum(exp_final))

    n_bins = len(exp_final)
    dof = n_bins - 1 - n_params_est
    
    if dof <= 0:
        return np.nan, np.nan
        
    chi2_stat, p_val = stats.chisquare(f_obs=obs_final, f_exp=exp_final, ddof=n_params_est)
    return chi2_stat, p_val

# ------------------------------------------------------------------------------
 
#============================================================================================
# 2. AJUSTE Y COMPARACIÓN DE DISTRIBUCIONES CONTINUAS GOF (Básico. Mejorada con best_fit_continuous
#============================================================================================

def gof_continuous(data):
    """
    Ajusta distribuciones continuas (MOM) y genera un reporte visual profesional.
    Devuelve un DataFrame con parámetros accesibles en 'Params_Dict'.
    """
    
    # 1. Preparación de datos
    x = np.array(data)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        print("❌ Error: No hay datos válidos.")
        return pd.DataFrame()

    # Estadísticos básicos
    mu = np.mean(x)
    var = np.var(x, ddof=1)
    std = np.std(x, ddof=1)
    x_min = np.min(x)
    x_max = np.max(x)
    
    results = []

    # ==============================================================================
    # 1. UNIFORME
    # ==============================================================================
    range_uni = np.sqrt(12 * var)
    uni_a = mu - (range_uni / 2)
    uni_scale = range_uni
    
    d, p = stats.kstest(x, 'uniform', args=(uni_a, uni_scale))
    results.append({
        'Distribución': 'Uniforme',
        'Parámetros_Txt': f'Min={uni_a:.2f}, Range={uni_scale:.2f}',
        'Params_Dict': {'loc': uni_a, 'scale': uni_scale},
        'KS Stat': d, 'P-Value': p
    })

    # ==============================================================================
    # 2. EXPONENCIAL
    # ==============================================================================
    exp_scale = mu
    d, p = stats.kstest(x, 'expon', args=(0, exp_scale))
    results.append({
        'Distribución': 'Exponencial',
        'Parámetros_Txt': f'Scale={exp_scale:.2f}',
        'Params_Dict': {'loc': 0, 'scale': exp_scale},
        'KS Stat': d, 'P-Value': p
    })

    # ==============================================================================
    # 3. NORMAL
    # ==============================================================================
    d, p = stats.kstest(x, 'norm', args=(mu, std))
    results.append({
        'Distribución': 'Normal',
        'Parámetros_Txt': f'Mu={mu:.2f}, Std={std:.2f}',
        'Params_Dict': {'loc': mu, 'scale': std},
        'KS Stat': d, 'P-Value': p
    })

    # ==============================================================================
    # 4. GAMMA
    # ==============================================================================
    if var > 0 and mu != 0:
        gam_scale = var / mu
        gam_a = (mu ** 2) / var
        d, p = stats.kstest(x, 'gamma', args=(gam_a, 0, gam_scale))
        results.append({
            'Distribución': 'Gamma',
            'Parámetros_Txt': f'Alpha={gam_a:.2f}, Beta={gam_scale:.2f}',
            'Params_Dict': {'a': gam_a, 'loc': 0, 'scale': gam_scale},
            'KS Stat': d, 'P-Value': p
        })

    # ==============================================================================
    # 5. ERLANG (Gamma con shape entero)
    # ==============================================================================
    if var > 0 and mu != 0:
        erl_k = max(1, round((mu ** 2) / var))
        erl_scale = mu / erl_k
        d, p = stats.kstest(x, 'gamma', args=(erl_k, 0, erl_scale))
        results.append({
            'Distribución': 'Erlang',
            'Parámetros_Txt': f'k={int(erl_k)}, Beta={erl_scale:.2f}',
            'Params_Dict': {'a': erl_k, 'loc': 0, 'scale': erl_scale},
            'KS Stat': d, 'P-Value': p
        })

    # ==============================================================================
    # 6. TRIANGULAR
    # ==============================================================================
    tri_loc = x_min
    tri_scale = x_max - x_min
    mode_est = 3 * mu - x_min - x_max
    mode_est = max(x_min, min(x_max, mode_est)) # Clamp
    
    if tri_scale > 0:
        tri_c = (mode_est - tri_loc) / tri_scale
        d, p = stats.kstest(x, 'triang', args=(tri_c, tri_loc, tri_scale))
        results.append({
            'Distribución': 'Triangular',
            'Parámetros_Txt': f'c={tri_c:.2f}, Loc={tri_loc:.2f}, Scale={tri_scale:.2f}',
            'Params_Dict': {'c': tri_c, 'loc': tri_loc, 'scale': tri_scale},
            'KS Stat': d, 'P-Value': p
        })

    # ==============================================================================
    # 7. WEIBULL
    # ==============================================================================
    if mu > 0 and std > 0:
        cv_sq = (std / mu) ** 2
        def weibull_eq(k):
            if k <= 0: return 100.0
            return (gamma(1 + 2/k) / (gamma(1 + 1/k)**2)) - 1 - cv_sq

        try:
            wei_k = optimize.fsolve(weibull_eq, 1.0)[0]
        except:
            wei_k = 1.0
        
        if wei_k > 0:
            wei_scale = mu / gamma(1 + 1/wei_k)
            d, p = stats.kstest(x, 'weibull_min', args=(wei_k, 0, wei_scale))
            results.append({
                'Distribución': 'Weibull',
                'Parámetros_Txt': f'Shape={wei_k:.2f}, Scale={wei_scale:.2f}',
                'Params_Dict': {'c': wei_k, 'loc': 0, 'scale': wei_scale},
                'KS Stat': d, 'P-Value': p
            })

    # ==============================================================================
    # 8. LOG-NORMAL
    # ==============================================================================
    if min_val := np.min(x) > 0: # Solo si todos son positivos
        phi = np.sqrt(var + mu**2)
        mu_log = np.log(mu**2 / phi)
        sigma_log = np.sqrt(np.log(phi**2 / mu**2))
        scale_log = np.exp(mu_log)
        
        d, p = stats.kstest(x, 'lognorm', args=(sigma_log, 0, scale_log))
        results.append({
            'Distribución': 'Log-Normal',
            'Parámetros_Txt': f's={sigma_log:.2f}, Scale={scale_log:.2f}',
            'Params_Dict': {'s': sigma_log, 'loc': 0, 'scale': scale_log},
            'KS Stat': d, 'P-Value': p
        })

    # --- PROCESAMIENTO FINAL ---
    df = pd.DataFrame(results)
    if df.empty: return df

    df['Decision'] = df['P-Value'].apply(lambda val: '✅ Aceptable' if val > 0.05 else '❌ Rechazado')
    df = df.sort_values(by='P-Value', ascending=False).reset_index(drop=True)

    # --- REPORTE VISUAL ---
    print("\n" + "═"*80)
    print("📊  RESULTADOS DEL AJUSTE (DISTRIBUCIONES CONTINUAS - KS TEST)")
    print("═"*80)
    
    cols_show = ['Distribución', 'Parámetros_Txt', 'KS Stat', 'P-Value', 'Decision']
    print(df[cols_show].to_string(index=False, formatters={
        'KS Stat': '{:.4f}'.format,
        'P-Value': '{:.4f}'.format
    }))
    print("─"*80)

    # Ganador
    best = df.iloc[0]
    print(f"\n🏆  MEJOR AJUSTE: \033[1m{best['Distribución']}\033[0m")
    print(f"    P-Value: {best['P-Value']:.4f}")
    if best['P-Value'] > 0.05:
        print("    ✅ No hay evidencia para rechazar esta distribución.")
    else:
        print("    ⚠️ Precaución: El ajuste no es ideal (P-Value < 0.05).")
    
    print(f"\n⚙️  PARÁMETROS TÉCNICOS (Para Scipy):")
    print(f"    {best['Params_Dict']}")
    print("═"*80 + "\n")
    
    # acceder al ganador  
    # ganador = df_ajuste.iloc[0]
    # modelo = ganador['Distribución']
    # params = ganador['Params_Dict']
    
    return df
# ==============================================================================
# ==============================================================================
# 3. FUNCIÓN PARA AJUSTAR Y COMPARAR DISTRIBUCIONES DISCRETAS
# ==============================================================================
def best_fit_discrete(data, tol_vmr=0.15):
    """
    Ajusta distribuciones discretas a los datos.
    Aplica el principio de parsimonia con dos reglas:
    1. Si Varianza/Media ≈ 1, da preferencia a Poisson frente a Binomial Negativa.
    2. Si tanto Binomial como Poisson superan el test (p > 0.05), prioriza Poisson.
    """
    x = np.array(data)
    x = x[~np.isnan(x)].astype(int)
    
    if len(x) == 0: 
        print("❌ Error: No hay datos válidos.")
        return pd.DataFrame(), np.nan
    
    mu = np.mean(x)
    var = np.var(x, ddof=1)
    min_val, max_val = np.min(x), np.max(x)
    if var == 0: var = 1e-6
    if mu == 0: mu = 1e-6
    
    results = []
    
    # 1. Poisson
    chi2, p = calculate_chi2_robust(x, 'poisson', [mu], 1)
    results.append({'Distribución': 'Poisson', 'Parámetros_Txt': f'λ={mu:.2f}', 'Chi2': chi2, 'P-Value': p, 'Params_Dict': {'mu': mu}})
    
    # 2. Geométrica
    if min_val == 0: p_geom = 1/(mu+1); loc_geom = -1; lbl = 'Geom (desde 0)'
    else: p_geom = 1/mu; loc_geom = 0; lbl = 'Geom (desde 1)'
    chi2, p = calculate_chi2_robust(x, 'geom', [p_geom, loc_geom], 1)
    results.append({'Distribución': lbl, 'Parámetros_Txt': f'p={p_geom:.3f}', 'Chi2': chi2, 'P-Value': p, 'Params_Dict': {'p': p_geom, 'loc': loc_geom}})
    
    # 3. Binomial
    if var < mu:
        p_bin = 1 - (var/mu)
        n_bin = max(int(round(mu/p_bin)), max_val)
        p_bin_adj = mu/n_bin
        chi2, p = calculate_chi2_robust(x, 'binom', [n_bin, p_bin_adj], 2)
        results.append({'Distribución': 'Binomial', 'Parámetros_Txt': f'n={n_bin}, p={p_bin_adj:.2f}', 'Chi2': chi2, 'P-Value': p, 'Params_Dict': {'n': n_bin, 'p': p_bin_adj}})
    
    # 4. Binomial Negativa
    if var > mu:
        p_nbin = mu/var
        n_val = (mu**2)/(var-mu)
        chi2, p = calculate_chi2_robust(x, 'nbinom', [n_val, p_nbin], 2)
        results.append({'Distribución': 'Binomial Negativa', 'Parámetros_Txt': f'r={n_val:.2f}, p={p_nbin:.2f}', 'Chi2': chi2, 'P-Value': p, 'Params_Dict': {'n': n_val, 'p': p_nbin}})

    # 5. Hipergeométrica 
    M_hyper = max_val * 10 
    n_hyper = int(M_hyper * (mu / max_val)) if max_val > 0 else 0
    N_hyper = max_val
    if M_hyper > n_hyper and n_hyper > 0:
        chi2, p = calculate_chi2_robust(x, 'hypergeom', [M_hyper, n_hyper, N_hyper], 3)
        results.append({'Distribución': 'Hipergeométrica', 'Parámetros_Txt': f'M={M_hyper}, n={n_hyper}, N={N_hyper}', 'Chi2': chi2, 'P-Value': p, 'Params_Dict': {'M': M_hyper, 'n': n_hyper, 'N': N_hyper}})

    # --- PROCESAMIENTO FINAL ---
    df = pd.DataFrame(results).dropna(subset=['Chi2'])
    if df.empty: return df, np.nan

    df['Decision'] = df['P-Value'].apply(lambda val: '✅ Aceptable' if val > 0.05 else '❌ Rechazado')
    df = df.sort_values(by='P-Value', ascending=False).reset_index(drop=True)

    # 🌟 LÓGICAS DE PREFERENCIA (PRINCIPIO DE PARSIMONIA) 🌟
    vmr = var / mu  
    mensajes_parsimonia = []
    
    # REGLA 1: Parsimonia por Índice de Dispersión (V/M ≈ 1)
    if (1 - tol_vmr) <= vmr <= (1 + tol_vmr):
        idx_poisson_lista = df.index[(df['Distribución'] == 'Poisson') & (df['Decision'] == '✅ Aceptable')].tolist()
        if idx_poisson_lista:
            idx = idx_poisson_lista[0]
            if idx != 0:
                row_poisson = df.iloc[[idx]]
                df = pd.concat([row_poisson, df.drop(idx)]).reset_index(drop=True)
                mensajes_parsimonia.append("Se ha forzado a Poisson al 1º puesto por Índice de Dispersión (V/M ≈ 1).")

    # REGLA 2: Si Binomial y Poisson son aceptables, Poisson debe ganar a Binomial
    idx_poisson_acc = df.index[(df['Distribución'] == 'Poisson') & (df['Decision'] == '✅ Aceptable')].tolist()
    idx_binom_acc = df.index[(df['Distribución'] == 'Binomial') & (df['Decision'] == '✅ Aceptable')].tolist()

    if idx_poisson_acc and idx_binom_acc:
        idx_p = idx_poisson_acc[0]
        idx_b = idx_binom_acc[0]
        
        # Si la Binomial está mejor posicionada que la Poisson, subimos la Poisson
        if idx_b < idx_p:
            row_p = df.iloc[[idx_p]]
            df_sin_p = df.drop(idx_p).reset_index(drop=True)
            # Insertamos Poisson justo antes de la Binomial
            df = pd.concat([df_sin_p.iloc[:idx_b], row_p, df_sin_p.iloc[idx_b:]]).reset_index(drop=True)
            
            msg = "Se ha priorizado Poisson sobre Binomial (ambas son aceptables, pero Poisson es un modelo más simple)."
            if msg not in mensajes_parsimonia:
                mensajes_parsimonia.append(msg)

    # Imprimir Reporte
    print("\n" + "═"*80)
    print("📊  RESULTADOS DEL AJUSTE (DISTRIBUCIONES DISCRETAS - CHI-CUADRADO)")
    print("═"*80)
    print(f"    Estadísticos de la muestra -> Media: {mu:.3f} | Varianza: {var:.3f} | Índice Dispersión (V/M): {vmr:.2f}")
    
    for msg in mensajes_parsimonia:
        print(f"    💡 INFO: {msg}")
        
    print("─"*80)
    print(df[['Distribución', 'Parámetros_Txt', 'Chi2', 'P-Value', 'Decision']].to_string(index=False, formatters={'Chi2': '{:,.4f}'.format, 'P-Value': '{:,.4f}'.format}))
    print("─"*80)
    
    best_row = df.iloc[0]
    print(f"\n🏆  MEJOR AJUSTE POSICIONADO: \033[1m{best_row['Distribución']}\033[0m")
    print(f"    P-Value: {best_row['P-Value']:.4f}")
    
    if best_row['P-Value'] > 0.05:
        print("    ✅ El ajuste es estadísticamente aceptable.")
        print(f"\n⚙️  PARÁMETROS TÉCNICOS:")
        print(f"    {best_row['Params_Dict']}")
        print("═"*80 + "\n")
        
        best_dict = {
            'Distribución': best_row['Distribución'],
            'params': best_row['Params_Dict']
        }
        return df, best_dict
    else:
        print("    ❌ ALERTA: La mejor distribución no supera el umbral del test Chi-Cuadrado.")
        print("    ⚠️  Los datos empíricos no se ajustan a ninguna de las distribuciones modeladas.")
        print("═"*80 + "\n")
        return df, np.nan


# ==============================================================================
# 4. FUNCIÓN PARA AJUSTAR Y COMPARAR DISTRIBUCIONES CONTINUAS
# ==============================================================================
import numpy as np
import pandas as pd
import warnings

from scipy import stats, optimize
from scipy.special import gamma, gammaln
from scipy.optimize import OptimizeWarning


def best_fit_continuous(data, alpha=0.05, verbose=True):
    """
    Ajusta varias distribuciones continuas por MOM (y algunas por fórmulas cerradas)
    y evalúa con KS-test. Devuelve (df_resultados, best) donde best es:
      - {'Distribución': str, 'params': dict} si el mejor pasa KS (p>alpha)
      - np.nan si ninguna pasa KS
    """
    # 1) Preparación robusta de datos
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        if verbose:
            print("❌ Error: No hay datos válidos.")
        return pd.DataFrame(), np.nan
    if n < 3:
        if verbose:
            print("❌ Error: Se necesitan al menos 3 datos válidos para un ajuste razonable.")
        return pd.DataFrame(), np.nan

    mu = float(np.mean(x))
    var = float(np.var(x, ddof=1))
    std = float(np.sqrt(var)) if var > 0 else 0.0
    x_min = float(np.min(x))
    x_max = float(np.max(x))

    # Protección: datos constantes o varianza no utilizable
    if not np.isfinite(std) or std <= 0:
        if verbose:
            print("❌ Error: Los datos no tienen varianza (son todos idénticos) o no es utilizable.")
        return pd.DataFrame(), np.nan

    results = []

    def _safe_kstest(dist, dist_name, params_txt, params_dict):
        """
        KS test robusto: usa CDF del objeto scipy.stats para evitar strings/args raros.
        Si falla o devuelve NaN, no añade resultado.
        """
        try:
            d, p = stats.kstest(x, dist.cdf)
            if np.isfinite(d) and np.isfinite(p):
                results.append({
                    "Distribución": dist_name,
                    "Parámetros_Txt": params_txt,
                    "Params_Dict": params_dict,
                    "KS Stat": float(d),
                    "P-Value": float(p),
                })
        except Exception:
            # Si una distro explota numéricamente, se ignora sin romper el proceso.
            return

    # Silenciar warnings típicos de SciPy (overflow/invalid/optimize)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=OptimizeWarning)

        # 1) UNIFORME (MOM)
        if var > 0:
            range_uni = np.sqrt(12.0 * var)
            if np.isfinite(range_uni) and range_uni > 0:
                uni_loc = mu - range_uni / 2.0
                uni_scale = range_uni
                dist = stats.uniform(loc=uni_loc, scale=uni_scale)
                _safe_kstest(
                    dist,
                    "Uniforme",
                    f"Min={uni_loc:.4f}, Range={uni_scale:.4f}",
                    {"loc": uni_loc, "scale": uni_scale},
                )

        # 2) EXPONENCIAL (MOM) — requiere soporte >=0 si fijamos loc=0
        if mu > 0 and x_min >= 0:
            exp_scale = mu
            dist = stats.expon(loc=0.0, scale=exp_scale)
            _safe_kstest(
                dist,
                "Exponencial",
                f"Scale={exp_scale:.4f}",
                {"loc": 0.0, "scale": exp_scale},
            )

        # 3) NORMAL (MOM)
        dist = stats.norm(loc=mu, scale=std)
        _safe_kstest(
            dist,
            "Normal",
            f"Mu={mu:.4f}, Std={std:.4f}",
            {"loc": mu, "scale": std},
        )

        # 4) GAMMA (MOM) — con loc=0 requiere x>=0
        if mu > 0 and var > 0 and x_min >= 0:
            gam_scale = var / mu
            gam_a = (mu * mu) / var
            if np.isfinite(gam_a) and np.isfinite(gam_scale) and gam_a > 0 and gam_scale > 0:
                dist = stats.gamma(a=gam_a, loc=0.0, scale=gam_scale)
                _safe_kstest(
                    dist,
                    "Gamma",
                    f"Alpha={gam_a:.4f}, Beta={gam_scale:.4f}",
                    {"a": gam_a, "loc": 0.0, "scale": gam_scale},
                )

        # 5) ERLANG (aprox. MOM) — gamma con k entero; requiere x>=0
        if mu > 0 and var > 0 and x_min >= 0:
            erl_k = int(max(1, round((mu * mu) / var)))
            erl_scale = mu / erl_k
            if np.isfinite(erl_scale) and erl_scale > 0:
                dist = stats.gamma(a=erl_k, loc=0.0, scale=erl_scale)
                _safe_kstest(
                    dist,
                    "Erlang",
                    f"k={erl_k}, Beta={erl_scale:.4f}",
                    {"a": erl_k, "loc": 0.0, "scale": erl_scale},
                )

        # 6) TRIANGULAR (MOM aprox con modo estimado)
        tri_loc = x_min
        tri_scale = x_max - x_min
        if np.isfinite(tri_scale) and tri_scale > 0:
            mode_est = 3.0 * mu - x_min - x_max
            mode_est = float(np.clip(mode_est, x_min, x_max))
            tri_c = (mode_est - tri_loc) / tri_scale
            tri_c = float(np.clip(tri_c, 1e-6, 1.0 - 1e-6))
            dist = stats.triang(c=tri_c, loc=tri_loc, scale=tri_scale)
            _safe_kstest(
                dist,
                "Triangular",
                f"c={tri_c:.4f}, Loc={tri_loc:.4f}, Scale={tri_scale:.4f}",
                {"c": tri_c, "loc": tri_loc, "scale": tri_scale},
            )

        # 7) WEIBULL (resolver shape k por CV con brentq, más estable que fsolve)
        # Requiere x>0 si fijamos loc=0
        if mu > 0 and std > 0 and x_min > 0:
            cv_sq = (std / mu) ** 2

            # E(k) = Gamma(1+2/k) / Gamma(1+1/k)^2 - 1 - cv^2
            # Usamos log-gamma para evitar overflows: exp(logGamma(..) - 2*logGamma(..))
            def weibull_eq(k):
                if k <= 0:
                    return np.inf
                lg1 = gammaln(1.0 + 2.0 / k)
                lg2 = gammaln(1.0 + 1.0 / k)
                ratio = np.exp(lg1 - 2.0 * lg2)
                return ratio - 1.0 - cv_sq

            wei_k = None
            # bracket típico razonable
            a, b = 1e-3, 1e3
            try:
                fa, fb = weibull_eq(a), weibull_eq(b)
                if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
                    wei_k = float(optimize.brentq(weibull_eq, a, b, maxiter=200))
            except Exception:
                wei_k = None

            if wei_k is not None and wei_k > 0:
                wei_scale = mu / gamma(1.0 + 1.0 / wei_k)
                if np.isfinite(wei_scale) and wei_scale > 0:
                    dist = stats.weibull_min(c=wei_k, loc=0.0, scale=wei_scale)
                    _safe_kstest(
                        dist,
                        "Weibull",
                        f"Shape={wei_k:.4f}, Scale={wei_scale:.4f}",
                        {"c": wei_k, "loc": 0.0, "scale": wei_scale},
                    )

        # 8) LOG-NORMAL (MOM) — requiere x>0
        if x_min > 0 and mu > 0 and var > 0:
            phi = np.sqrt(var + mu * mu)
            if np.isfinite(phi) and phi > 0:
                mu_log = np.log((mu * mu) / phi)
                sigma_log = np.sqrt(np.log((phi * phi) / (mu * mu)))
                scale_log = float(np.exp(mu_log))
                if np.isfinite(sigma_log) and sigma_log > 0 and np.isfinite(scale_log) and scale_log > 0:
                    dist = stats.lognorm(s=sigma_log, loc=0.0, scale=scale_log)
                    _safe_kstest(
                        dist,
                        "Log-Normal",
                        f"s={sigma_log:.4f}, Scale={scale_log:.4f}",
                        {"s": sigma_log, "loc": 0.0, "scale": scale_log},
                    )

    # 3) Procesamiento final
    df = pd.DataFrame(results)
    if df.empty:
        if verbose:
            print("❌ No se pudo evaluar ninguna distribución (datos fuera de soporte o problemas numéricos).")
        return df, np.nan

    df["Decision"] = np.where(df["P-Value"] > alpha, "✅ Aceptable", "❌ Rechazado")
    df = df.sort_values(by="P-Value", ascending=False).reset_index(drop=True)

    if verbose:
        print("\n" + "═" * 80)
        print("📊  RESULTADOS DEL AJUSTE (DISTRIBUCIONES CONTINUAS - KS TEST)")
        print("═" * 80)
        cols_show = ["Distribución", "Parámetros_Txt", "KS Stat", "P-Value", "Decision"]
        print(df[cols_show].to_string(
            index=False,
            formatters={"KS Stat": "{:.4f}".format, "P-Value": "{:.4f}".format}
        ))
        print("─" * 80)
        best_row = df.iloc[0]
        print(f"\n🏆  MEJOR AJUSTE POSICIONADO: {best_row['Distribución']}")
        print(f"    P-Value: {best_row['P-Value']:.4f}")

    best_row = df.iloc[0]
    if best_row["P-Value"] > alpha:
        best = {"Distribución": best_row["Distribución"], "params": best_row["Params_Dict"]}
        if verbose:
            print("    ✅ El ajuste es estadísticamente aceptable.")
            print("\n⚙️  PARÁMETROS TÉCNICOS (Para SciPy):")
            print(f"    {best_row['Params_Dict']}")
            print("═" * 80 + "\n")
        return df, best

    if verbose:
        print("    ❌ ALERTA: La mejor distribución no supera el umbral del test KS.")
        print("    ⚠️  No hay ajuste aceptable entre las distribuciones evaluadas.")
        print("═" * 80 + "\n")
    return df, np.nan

# ==============================================================================
# 5. FUNCIÓN PARA TRANSFORMAR EN UN OBJETO SCIPY.STATS LA DISTRIBUCIÓN GANADORA
# DE best_fit_discrete o best_fit_continuous
# ==============================================================================

def obtener_distribucion_ganadora(resultados):
    """
    Toma el output de best_fit_continuous() o best_fit_discrete() y genera
    un objeto de distribución 'congelado' de scipy.stats.
    Si la entrada es inválida o np.nan, cancela la operación de forma segura.
    Devuelve tres objetos (en una lista):
    [0] es la distribución de scipy.stats
    [1] es el nombre del modelo ajustado
    [2] son los parámetros de ese modelo
    """
    # 1. VALIDACIÓN LIMPIA Y ROBUSTA
    # Esperamos una tupla (df, diccionario_o_nan)
    if not isinstance(resultados, tuple) or len(resultados) != 2:
        print("❌ Error: Formato de entrada incorrecto. Se esperaba una tupla (df, dict/nan).")
        return None

    df, ganador = resultados

    # Comprobamos si el ajuste falló (ganador es np.nan o no es un diccionario)
    if not isinstance(ganador, dict):
        print("❌ Operación cancelada: No se ha detectado un ajuste estadísticamente válido.")
        print("⚠️ Imposible instanciar la variable aleatoria para simulación.")
        return np.nan

    # 2. EXTRACCIÓN DEL GANADOR
    nombre_dist = ganador.get('Distribución')
    parametros = ganador.get('params', {})

    # 3. MAPEO CON SCIPY.STATS (Continuas y Discretas)
    mapa_distribuciones = {
        # Continuas
        'Uniforme': stats.uniform,
        'Exponencial': stats.expon,
        'Normal': stats.norm,
        'Gamma': stats.gamma,
        'Erlang': stats.erlang,
        'Triangular': stats.triang,
        'Weibull': stats.weibull_min,
        'Log-Normal': stats.lognorm,

        # Discretas
        'Poisson': stats.poisson,
        'Geom (desde 1)': stats.geom,
        'Geom (desde 0)': stats.geom,
        'Binomial': stats.binom,
        'Binomial Negativa': stats.nbinom,
        'Hipergeométrica': stats.hypergeom
    }

    if nombre_dist not in mapa_distribuciones:
        print(f"❌ Error interno: La distribución '{nombre_dist}' no está en el mapa de scipy.stats.")
        return None

    # 4. INSTANCIACIÓN DE LA DISTRIBUCIÓN
    try:
        dist_class = mapa_distribuciones[nombre_dist]

        # El operador ** desempaqueta el diccionario como argumentos con nombre
        dist_stats = dist_class(**parametros)

        print("."*100)
        print(f"✅ Variable aleatoria '{nombre_dist}' instanciada correctamente.")
        print(f"con parámetros {parametros}")
        print("."*100)
      
        return dist_stats

    except Exception as e:
        print(f"❌ Error al intentar instanciar la distribución: {e}")
        return None
