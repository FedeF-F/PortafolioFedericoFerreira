Title: UT2 PD1
Date: 2023-03-10 00:27
Category: 3. PDS



El objetivo de
utilizar este dataset es el de conseguir un modelo que pueda predecir a que
clase pertenece el vino dado sus atributos. Y el dataset fue creado simplemente
para hacer pruebas.

Estos datos son
el resultado de un análisis químico de vinos cultivados en la misma región de
Italia pero derivados de tres cultivares diferentes. El análisis determinó las
cantidades de 13 componentes que se encuentran en cada uno de los tres tipos de
vinos. ([http://archive.ics.uci.edu/dataset/109/wine](http://archive.ics.uci.edu/dataset/109/wine))

Atributos:

1) Alcohol
2) Malic acid
3) Ash
4) Alcalinity of ash
5) Magnesium
6) Total phenols
7) Flavanoids
8) Nonflavanoid phenols
9) Proanthocyanins

10)Color intensity

11)Hue

12)OD280/OD315 of diluted wines

13)Proline

No existen
valores faltantes, existe un valor outlier en la categoría Proline, siendo que
el promedio es de 746.9 y existe una columna con el valor 1680

Luego de ejecutar
el modelo en un set normalizado y un set sin procesar, podemos ver que en este
caso el set sin procesar tiene un mayor porcentaje de aciertos
