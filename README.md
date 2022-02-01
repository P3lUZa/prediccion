# Predicción
Predicción de series de tiempo financieras mediante estadística y bosques aleatorios

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

- `
