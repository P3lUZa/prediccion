# Predicción
Predicción de series de tiempo financieras mediante estadística no paramétrica y bosques aleatorios

## Descripción
Este repositorio tiene módulos cuyo objetivo es predecir el valor 
que tendrá una serie de tiempo después de un periodo (llamado horizonte) 
a partir de un tiempo dado. La predicción la hacen a partir de los valores 
que haya tenido la serie hasta el tiempo que se les indique. 

Se cuenta con dos estrategias para realizar la estimación a futuro, una usa 
únicamente los valores de la serie y estadística no paramétrica mientras que 
la otra usa ciertos indicadores calculados a partir de la serie y el algoritmo 
de bosques aleatorios.

Ambas estrategias tienen la posibilidad de implementar la descompocisión 
empírica en funciones modales conocida como 
[CEEMDAN](https://www.researchgate.net/publication/220731876_Complete_ensemble_empirical_mode_decomposition_with_adaptive_noise) 
para mejorar la exactitud de la estimación.

## Contenido del repositorio

Los módulos con los que se cuenta son:

- `CD_Predictor` (Change Distribution Predictor) Programa que hace la predicción estimando la distribución de 
probabilidad de los cambios en la serie después del horizonte en fechas anteriores
y en valores cercanos al actual (es decir, el valor del tiempo dado).

- `CRF_Predictor` (CEEMDAN-Random Forest Predictor) Estimación hecha al procesar distintos indicadores calculados 
con los valores de la serie de tiempo y los de las funciones modales de la CEEMDAN. 
Este procesamiento es llevado a cabo por el algoritmo conocido como Bosque Aleatorio
tal y como está implementado en la paquetería scikit-learn.

`CRF_Predictor` no sólo hace una reconstrucción de la serie con las funciones 
modales que aportan más información, sino que usa como uno de los indicadores
la frecuencia instantánea de la reconstrucción y la de algunas funciones 
modales que se ha probado empíricamente que contribuyen a una predicción 
más exacta. Ésta frecuencia es calculada mediante la transformada de 
Huang-Hilbert.

Ambos métodos de predicción están pensados para series de tiempo financieras pero 
pueden ser usados para predicción de otro tipo de series. En general presentan un 
buen desempeño aún cuando la serie no es estacionaria ni muestra ningún tipo de 
tendencia a simple vista.

## Uso de los módulos

Para entender completamente la funcionalidad de estos programas, se incluirá en el
futuro la tesis para la cual fueron desarrollados. Por lo pronto, se explicarán los 
parámetros de los que dependen las clases incluidas en los módulos y cómo usarlas.

### Parámetros

La clase `CRPred` (contenida en el módulo `CRF_Predictor`) tiene tres parámetros:

- `ts` La serie de tiempo con la que se trabajará y la cual formará parte de la 
instancia de la clase. Ésta debe ser un objeto *Series* de la librería pandas.
- `horizon` El número de periodos en el futuro para el cual se quiere saber el valor 
de la serie. Si `ts` es una serie por días y se quiere predecir el valor 15 dias 
después de la fecha escogida, entonces éste parámetro debe pasarse como 15. Si no 
pasa algún valor entonces automáticamente valdrá 1.
- `tail` La cantidad de datos a partir del último valor de la serie que se quiere 
utilizar para alimentar al programa. El parámetro más influyente en el tiempo que 
se tarda en hacer la predicción. Si no se pasa algún valor entonces valdrá 2000, 
es decir, se usarán los últimos 2000 valores de la serie.

La clase `CDPred` (contenido en el módulo `CD_Predictor`) tiene cuatro parametros:

- `ts` Igual que en `CRPred`.
- `horizon` Igual que en `CRPred`.
- `tail` Igual que en `CRPred`.
- `min_data` El mínimo número de datos usados para estimar la distribución de cambios
en la serie al rededor del precio actual. No es recomendable pasar un valor menor a 50
o mayor a 200 ya que puede ser muy poco exacta la predicción y después de 200 no se 
obtiene casi ningún cambio en la predicción. Si no se pasa ningún valor entonces valdrá 120.

El parámetro `tail` es importante si se quiere usar la CEEMDAN ya que el tiempo que 
tarda en calcular las funciones intrínsecas aumenta considerablemente al aumentar 
los datos con los que se trabaja. 

Es mejor elegir valores menores para `min_data` si el horizonte es pequeño ya que 
se conserva una mayor localidad del precio y por lo tanto una mejor predicción. 
Empíricamente se ha observado que para horizontes de 5 o 10 periodos es mejor un 
valor alrededor de 70 y para 20 o más periodos es mejor al rededor de 120.

## Usando CDPred

Una vez que se ha creado una instancia de la clase CDPred, simplemente se usa el
método `predict` que sólo tiene dos parámetros: 

- `indexes` Un arreglo con los índices de la serie que se tomarán como el tiempo
actual para cada predicción.
- `interval` Una variable booleana que cuando se encuentra en `True` nos incluye 
un intervalo de confianza con cobertura del 95% para el valor futuro. Por 
defecto tiene un valor `True`.

Así, si `data` es un data frame de pandas que en la columna 'adj_close' contiene 
los precios ajustados de algún activo por día, las siguientes lineas nos darían la 
predicción después de 30 días a partir de la última y la sexagésima fecha 
de la serie

```
pred = CD_Predictor.CDPred(data['adj_close'], horizon = 30)
pred.predict([-1,60])
```

Si se quiere usar la CEEMDAN para efectuar una reconstrucción de la serie con las 
funciones modales más significativas y con ésta hacer una predicción, simplemente
se ejecuta el método `ceemdan` antes de hacer la predicción. El único parámetro de
este método es `imfs_omitted` el cual es el número de funciones modales que se 
quiere omitir en la reconstrucción, por defecto es 2 y siempre se omitirán las 
funciones con menor índice. Las siguientes lineas bastanrían para implementar
la CEEMDAN a los cálculos.

```
pred = CD_Predictor.CDPred(data['adj_close'], horizon = 30)
pred.ceemdan()
pred.predict([-1,60])
```

## Usando CRPred

### Modo simple

Después de declarar una instancia de la clase CRPred es necesario llamar al 
método `best_fit`. Luego, para predecir se usa el método `predict` de la misma manera
que se hace con la clase CDPred. Las siguientes lineas predicen las posiciones última
y sexagésima de la serie 10 posiciones después.

```
rf_t = CRPred.CRPred(data['adj_close'], horizon = 10)
rf_t.best_fit()
rf_t.predict([-1, 60])
```
Considere que `best_fit`toma un tiempo considerable en ejecutarse, pero una vez que lo
hace se pueden hacer predicciones sobre toda la serie sin demora.

### Modo avanzado

En el caso de la estimación por bosque aleatorio, el módulo está hecho de modo que 
se guarde la serie en una instancia de la clase `CRPred`, se calculen los indicadores 
mediante métodos de la clase, se afinen algunos hiperparámetros del algoritmo con 
la el método `train`, se entrene al algoritmo con los mejores parámetros encontrados 
mediante `fit` y finalmente se haga la predicción con el método `predict`. 

`best_fit` hace todo lo anterior a la predicción llamando a los métodos necesarios para 
aplicar la CEEMDAN y calcular los mejores indicadores con los que se entrena al algoritmo
que se han probado hasta ahora. Sin embargo, si se quiere añadir indicadores para hacer 
la predicción, sólo hace falta añadir una columna al dataframe `prep` y añadir el nombre 
de la columna a la lista `feature names`, ambos son atributos de la clase. Esta columna
debe contener el indicador correspondiente a cada dato de la serie.

Tras añadir las columnas necesarias se pueden ejecutar los métodos `train` y `fit` en este 
orden y el objeto de la clase estará listo para ejecutar `predict` con los indicadores
añadidos.

## Requerimientos
Además de pandas, numpy y scikitlearn, es necesario contar con las librerías PYEMD y talib.

