# CAViaR con variables indicador (`caviar_i`) — Diseño

**Fecha:** 2026-06-25
**Módulo nuevo:** `auxi/caviar.py`
**Estado:** Aprobado, pendiente de implementación

## Contexto y motivación

Se quiere construir un modelo tipo CAViaR (Engle & Manganelli, 2004) en su
especificación más sencilla: dos variables indicador binarias que señalan si el
valor realizado cayó fuera de las fronteras cuantílicas de la especificación.

- `upside_breach`   = 1 si el realizado superó la frontera cuantílica alta, 0 en otro caso.
- `downside_breach` = 1 si el realizado quedó por debajo de la frontera cuantílica baja, 0 en otro caso.

Estos indicadores se incorporan como regresores adicionales en una regresión
cuantílica del estilo de las que ya usamos (`auxi/qreg.py`).

### Decisiones de diseño acordadas

1. **Breach in-sample contemporáneo, sin shift.** Los bounds se calculan con una
   regresión cuantílica in-sample `Q(y_t | x_t)` contemporánea. El indicador se
   ancla a la fila `t`. El desfase respecto al objetivo de pronóstico lo aporta
   automáticamente el *shift* del target en el direct forecasting: al predecir
   `y_{t+h}`, la fila `t` lleva implícitamente el breach de hace `h` periodos.
   No hay doble horizonte ni parámetro `h_breach`.

2. **No se contamina el panel.** Los indicadores dependen de la especificación
   (`vars_x` y cuantiles). Añadirlos al panel lo acoplaría a una especificación
   concreta. Por tanto los indicadores son **efímeros**: viven sólo dentro de la
   llamada, se usan como regresores y se descartan. El panel de entrada **nunca
   se muta**.

3. **Cuantiles extremos de la especificación.** Las fronteras se definen con
   `tau_low = min(breach_quantiles)` y `tau_high = max(breach_quantiles)`, donde
   `breach_quantiles` son los cuantiles de la especificación que se esté
   probando. Default `[0.05, 0.95]`. Así, al cambiar de especificación los
   breaches se recalculan solos.

4. **Los bounds usan la misma especificación (`vars_x`)** que la regresión
   CAViaR final. No se permite un conjunto de regresores distinto para definir el
   breach.

5. **`caviar_i` devuelve una tupla `(reg, indicators)`** para poder inspeccionar
   y graficar los breaches después de estimar.

6. **Nomenclatura `_i`:** indica la variante con *indicadores* (binaria), que no
   mide severidad del breach. Deja la puerta abierta a un futuro `caviar_s`
   (severidad).

## Arquitectura — tres capas (SoC)

El módulo replica la estructura por capas de `auxi/qreg.py`: un productor
numérico que alimenta tanto tablas como gráficos, sub-funciones de plot
atómicas, y un orquestador tipo `plot_quantile_results`. Donde una sub-función de
`qreg.py` ya es genérica, se **reutiliza** en vez de duplicar.

### Dependencias

`caviar.py` importa de `qreg.py`: `q_reg`, `plot_quantile_coefs`,
`plot_pseudo_r2`. También usa `numpy`, `pandas`, `matplotlib`, `statsmodels`.

---

### Capa 1 — Helpers privados (lógica pura, sin estado)

#### `_compute_quantile_bounds(df, vars_x, vars_y, tau_low, tau_high, **kwargs)`

- Ajusta dos regresiones cuantílicas in-sample reutilizando `q_reg`: una en
  `tau_low`, otra en `tau_high`.
- Genera las predicciones in-sample `Q(y_t | x_t)` de cada una.
- **Retorno:** `DataFrame` indexado igual que `df` con columnas
  `Bound_Low`, `Bound_High`.
- **Depende de:** `q_reg`.

#### `_compute_breaches(realized, bounds)`

- Lógica pura de comparación, sin saber de regresiones.
- `upside   = 1{realized > bounds["Bound_High"]}`
- `downside = 1{realized < bounds["Bound_Low"]}`
- **Retorno:** tupla `(upside, downside)` de dos `Series` `{0, 1}`.
- Donde el bound es NaN (fila sin features computables), el indicador es **NaN**,
  no 0 — para no confundir "no breach" con "no computable".
- **Depende de:** sólo numpy/pandas.

---

### Capa 2 — Estimación (hermanas de `q_reg` / `multiple_q_regs`)

#### `caviar_i(df, vars_x, vars_y, tau, breach_quantiles=None, errors="robust", **kwargs)`

1. `breach_quantiles` default `[0.05, 0.95]`. Deriva `tau_low = min(...)`,
   `tau_high = max(...)`. Error explícito si `tau_low == tau_high` (taus
   degenerados).
2. `_compute_quantile_bounds(df, vars_x, vars_y, tau_low, tau_high)` → bounds.
3. `_compute_breaches(df[vars_y], bounds)` → `(upside, downside)`.
4. Sobre una **copia interna** del panel, añade columnas `upside_breach` y
   `downside_breach`.
5. Corre la regresión cuantílica `y ~ vars_x + upside_breach + downside_breach`
   en el cuantil `tau`, reutilizando `q_reg` (que ya construye la fórmula y llama
   a `smf.quantreg`). Pasa `vcov=errors`.
6. **Retorno:** tupla `(reg, indicators)`, donde `reg` es el resultado de
   regresión ajustado (igual que `q_reg`) e `indicators` es un `DataFrame` con
   `upside_breach`, `downside_breach` (y opcionalmente los bounds) para
   inspección/plot.
- **No muta `df`.**

#### `multiple_caviar_i(data, vars_x, vars_y, quantiles=None, breach_quantiles=None, errors="robust")`

- Bucle sobre `quantiles` (default igual que `multiple_q_regs`,
  `[0.05, 0.25, 0.50, 0.75, 0.95]`).
- **Eficiencia clave:** los indicadores dependen sólo de `breach_quantiles`, no
  del `tau` de estimación. Se computan **una sola vez** antes del bucle y se
  reutilizan en todos los cuantiles.
- **Retorno:** `master_df` con el **mismo esquema** que `multiple_q_regs`:
  `Dependent Variable`, `Regressor`, `Tau`, `Coefficient`, `Significance`,
  `Pseudo R-Squared`. Incluye las filas de `upside_breach` y `downside_breach`
  junto a las de los regresores de la especificación.
- Este `master_df` es el output numérico que alimenta tanto las tablas como los
  gráficos.

---

### Capa 3 — Visualización (mirror de `qreg.py`, máximo reuso)

- **Reutiliza** `plot_quantile_coefs` y `plot_pseudo_r2` de `qreg.py` — su
  esquema de tabla es idéntico al de `master_df`.

#### `plot_breach_diagnostics(ax, df, vars_x, vars_y, breach_quantiles, ...)` (nueva, atómica)

- Serie temporal con `Bound_Low` / `Bound_High` y los puntos de breach marcados,
  al estilo de `forecasting.py` (panel de bounds & breaches).
- Recalcula bounds vía `_compute_quantile_bounds` y breaches vía
  `_compute_breaches`.
- Es la sub-función de plot propia de CAViaR.

#### `plot_caviar_i_results(data, vars_x, vars_y, breach_quantiles=None, quantiles=None, errors="robust")` (orquestador, mirror de `plot_quantile_results`)

Dashboard 2×2:
- `[0,0]` coefs de los regresores de la especificación (`plot_quantile_coefs`).
- `[0,1]` **coefs de `upside_breach` / `downside_breach` por tau**
  (`plot_quantile_coefs`) — el resultado estrella CAViaR.
- `[1,0]` Pseudo R² (`plot_pseudo_r2`).
- `[1,1]` `plot_breach_diagnostics` (bounds + breaches en el tiempo).
- **Retorno:** el `master_df` (como hace `plot_quantile_results`).

---

## Flujo de datos

```
df + especificación (vars_x, vars_y, breach_quantiles)
        │
        ▼
_compute_quantile_bounds  ──►  Bound_Low, Bound_High
        │
        ▼
_compute_breaches         ──►  upside_breach, downside_breach (efímeros)
        │
        ▼
caviar_i / multiple_caviar_i  ──►  reg ajustada / master_df  (número)
        │
        ▼
tablas  +  plot_caviar_i_results (dashboard)
```

Todo es efímero salvo el retorno. El panel de entrada nunca se altera.

## Manejo de errores

- `tau_low == tau_high` (taus degenerados) → `ValueError` explícito.
- Bound NaN en una fila → breach NaN (no 0).
- Si `q_reg` falla con `vcov` robusto, replicar el patrón de `multiple_q_regs`:
  fallback a `vcov="iid"`.

## Alineamiento

Sin shift. La fila `t` contiene el breach de `Q(y_t | x_t)`. El lag respecto al
target lo aporta luego el shift del direct forecasting.

## Fuera de alcance (YAGNI)

- Variante de severidad (`caviar_s`) — futura.
- Conjunto de regresores distinto para bounds vs regresión — descartado.
- Parámetro `h_breach` / doble horizonte — innecesario por el diseño in-sample.

## Referencias

Engle, R. F., & Manganelli, S. (2004). CAViaR: Conditional Autoregressive Value
at Risk by Regression Quantiles. *Journal of Business & Economic Statistics*,
22(4), 367–381.
