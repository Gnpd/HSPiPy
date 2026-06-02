[English](./README.md) | [中文](./README.zh-CN.md) | Español

[![PyPI](https://img.shields.io/pypi/v/HSPiPy.svg)](https://pypi.org/project/HSPiPy/) ![version](https://img.shields.io/badge/version-1.1.8-orange.svg)

> [!TIP]
> ¿Buscas una versión no-code? Obtenga [**HSP Desktop**](https://g-npd.com/es/hsp)

# HSPiPy

#### Parámetros de Solubilidad de Hansen en Python.

### Introducción
---------------

HSPiPy es una librería de Python diseñada para calcular y visualizar los Parámetros de Solubilidad de Hansen (HSP). Ofrece estimadores compatibles con machine learning, importación de datos conveniente y herramientas de visualización para analizar la compatibilidad de solventes en ciencias de materiales, polímeros y recubrimientos.

### Características
---------------

* Lectura de datos de solventes en múltiples formatos (CSV, HSD, HSDX)
* Cálculo de Parámetros de Solubilidad de Hansen mediante métodos de optimización robustos
* Soporte para uno o dos esferas de solubilidad
* Generación de visualizaciones 2D y 3D de las esferas de solubilidad
* Totalmente compatible con scikit-learn para flujos de trabajo de machine learning


### Documentación
---------------

- Referencia de clases:
    - [HSP (clase de alto nivel)](https://github.com/Gnpd/HSPiPy/blob/main/docs/HSP.md)
    - [HSPEstimator (compatible con scikit-learn)](https://github.com/Gnpd/HSPiPy/blob/main/docs/HSPEstimator.md)


### Instalación
---------------

Instala **HSPiPy** fácilmente con pip:

```
pip install HSPiPy
```

#### Dependencias

* numpy
* pandas
* matplotlib
* scipy
* scikit-learn

### Uso
--------

#### Leer datos HSP

Para leer datos HSP desde un archivo (CSV, HSD o HSDX), crea una instancia de la clase `HSP` y usa el método `read`:
```python
from hspipy import HSP

hsp = HSP()
hsp.read('ruta_a_tu_archivo.csv')
```


#### Calcular HSP

Usa el método `get` para calcular los Parámetros de Solubilidad de Hansen a partir de tus datos:
```python
# Modelo de esfera única (por defecto)
result = hsp.get(inside_limit=1)

# Modelo de doble esfera
result = hsp.get(inside_limit=1, n_spheres=2)
```

`get()` devuelve un objeto `HSPResult`. En un Jupyter Notebook se renderiza automáticamente como tabla. En un script usa `print(result)` o accede a los atributos directamente:

```python
print(result.hsp)             # coordenadas del centro D, P, H ajustadas
print(result.radius)          # radio de la esfera
print(result.accuracy)        # precisión de clasificación
print(result.datafit)         # valor DATAFIT
print(result.n_solvents_in)   # número de buenos solventes
print(result.n_solvents_out)  # número de malos solventes
print(result.n_total)         # total de solventes
print(result.n_wrong_in)      # malos solventes dentro de la esfera (falsos positivos)
print(result.n_wrong_out)     # buenos solventes fuera de la esfera (falsos negativos)
```

El parámetro `inside_limit` define el umbral de puntuación para considerar un solvente como "dentro" de la esfera de solubilidad (por defecto: `inside_limit=1`).

#### Visualizar HSP

Usa los métodos `plot_3d` y `plot_2d` para visualizar los datos HSP en formatos 3D y 2D respectivamente:
```python
# Generar gráficos individuales
hsp.plot_3d()
hsp.plot_2d()

# O generar ambos gráficos a la vez
hsp.plots()
```
![3dHSP](https://github.com/Gnpd/HSPiPy/blob/main/3dPlot.png)
![2dHSP](https://github.com/Gnpd/HSPiPy/blob/main/2dPlot.png)

### Métodos de la clase `HSP`
| Método | Descripción |
|--------|:-----------:|
| read(path) | Lee datos de solventes desde un archivo CSV, HSD o HSDX |
| get(inside_limit=1, n_spheres=1) | Ajusta las esferas HSP y devuelve un objeto `HSPResult` |
| plot_3d() | Grafica los datos HSP en 3D. Devuelve la figura |
| plot_2d() | Grafica los datos HSP en 2D. Devuelve la figura |
| plots() | Genera ambos gráficos 2D y 3D. Devuelve `(fig_3d, fig_2d)` |

`get()` devuelve un objeto `HSPResult`. Los siguientes atributos están disponibles en él y también se establecen en la instancia `HSP` tras llamar a `get()`:

| Atributo | Descripción |
|----------|-------------|
| `result.hsp` | Array Numpy — Coordenadas HSP ajustadas. Forma: `(3,)` para esfera única (D, P, H), o `(n_spheres, 3)` para múltiples esferas |
| `result.radius` | Float o array — Radio (o radios) de la(s) esfera(s) de solubilidad |
| `result.error` | Float — Valor de la función objetivo de la optimización (menor es mejor) |
| `result.accuracy` | Float — Precisión de clasificación del modelo ajustado sobre el conjunto de datos |
| `result.datafit` | Float — Valor DATAFIT (media geométrica de ajuste; 1.0 = clasificación perfecta) |
| `result.n_solvents_in` | Int — Número de buenos solventes (`0 < score <= inside_limit`) |
| `result.n_solvents_out` | Int — Número de malos solventes (`score == 0` o `score > inside_limit`) |
| `result.n_total` | Int — Total de solventes |
| `result.n_wrong_in` | Int — Malos solventes predichos dentro de la esfera (falsos positivos) |
| `result.n_wrong_out` | Int — Buenos solventes predichos fuera de la esfera (falsos negativos) |
| `hsp.d`, `hsp.p`, `hsp.h` | Float — Componentes individuales HSP (δD, δP, δH). Solo esfera única; `None` para multi-esfera |
| `hsp.inside` | Lista — Solventes clasificados como *dentro* de la(s) esfera(s), con sus valores HSP y puntuaciones |
| `hsp.outside` | Lista — Solventes clasificados como *fuera* de la(s) esfera(s), con sus valores HSP y puntuaciones |
| `hsp.grid` | Pandas DataFrame — Conjunto de datos de entrada completo, estandarizado con columnas: `Solvent`, `D`, `P`, `H` y `Score` |


### Uso con el estimador estilo scikit-learn
```python
import numpy as np
from hspipy import HSPEstimator

# Conjunto de datos de ejemplo (valores D, P, H con puntuaciones)
X = np.array([
    [16.0, 8.0, 5.0],
    [18.0, 7.5, 9.0],
    [20.0, 10.0, 12.0],
])
y = np.array([1, 1, 0])  # Etiquetas dentro/fuera o puntuaciones

est = HSPEstimator(n_spheres=1)
est.fit(X, y)

print("HSP ajustado:", est.hsp_)
print("Precisión:", est.score(X, y))
```

### Contribuir
----------------

¡Las contribuciones son bienvenidas! Si tienes sugerencias, solicitudes de funciones o informes de errores, abre un issue en el [repositorio de GitHub](https://github.com/Gnpd/HSPiPy/issues).

### Apoyar el proyecto
------------------------
Si HSPiPy es útil para tu trabajo, hay varias formas de apoyar su desarrollo continuo.

#### 🖥️ HSP Desktop (versión sin código)

HSP Desktop es una aplicación de escritorio asequible construida sobre el motor de HSPiPy, que proporciona una interfaz gráfica rápida e intuitiva para el análisis de Parámetros de Solubilidad de Hansen.

Es ideal para usuarios que:

- Prefieren una interfaz gráfica sobre scripts de Python
- Necesitan un flujo de trabajo rápido y de un solo clic para los cálculos HSP
- Desean herramientas avanzadas de visualización y exploración de solventes

La compra de HSP Desktop ayuda a financiar el desarrollo continuo de HSPiPy, incluyendo nuevas funciones, bases de datos, documentación y mantenimiento.

👉 https://g-npd.com/es/hsp

#### ❤️ GitHub Sponsors

También puedes apoyar el proyecto directamente a través de GitHub Sponsors:

👉 https://github.com/sponsors/Gnpd

Cada contribución ayuda a mantener HSPiPy gratuito, de código abierto y en desarrollo activo.

### Licencia
-----------

Esta librería está licenciada bajo la Licencia MIT. Consulta el archivo [LICENSE](https://github.com/Gnpd/HSPiPy/blob/main/LICENSE) para más detalles.

### Agradecimientos
----------------

HSPiPy fue inspirado por el conocido software HSP [Hansen Solubility Parameters in Practice (HSPiP)](https://www.hansen-solubility.com/HSPiP/) y por la comunidad HSP.
