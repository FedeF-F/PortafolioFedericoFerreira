Title: Austin Animal Shelter
Date: 2024-03-10 00:27
Category: 4. Casos de Estudio

## Austin Animal Shelter

### Dataset y Analisis del Problema

Este caso de estudio contiene dos datasets, uno con la información que el refugio tiene de los animales al momento de ingresarlos, llamado intake, y otro conteniendo los datos de los animales al momento de irse del refugio, ya sea por adopción, hogar temporal o eutanasia.

En el caso del dataset de intake, se tienen los siguientes datos: animal_id, name, datetime, monthyear, found_location, intake_type,  intake_condition, animal_type, sex_upon_intake, breed y color

Del dataset outake, tenemos datos repetidos, los datos que difieren son los siguientes: date_of_birth, outcome_type, outcome_subtype, sex_upon_outcome, age_upon_outcome.

El objetivo del caso de estudio es determinar si el animal será adoptado o no dadas  sus caracteristicas

### Preparacion de datos

Primero pasamos los datasets por turboprep de rapidminer, las operaciones realizadas fueron las siguientes:

Sex_Upon_Intake/Outcome fue separado en dos columnas distintas, en Gender y Neutered, dado que estas columnas contenian sus datos en el formato "Female/Male Neutered/Intact"

Cambiamos el formato de la fecha a algo mas manejable

Añadimos un atributo "adoptado" basado en outcome_type, definido como binomial, true and false, tomando el subtipo foster como false

Luego de que la columna adoptado fuera creada se eliminan ambas columnas outcome.

Monthyear fue eliminado dado que era igual a datetime

animal_id se asigna como id y adopted como label

Se eliminan elementos duplicados

Se elimina la columna name dado que el nombre del animal no influye en si fue adoptado o no

Finalmente mergeamos intake y outcome con un left join para terminar la preparacion de datos.

### RapidMiner

El proceso final es este:

![Pelican](.\images\RapidMiner.png)

El primer paso luego de importar el dataset es eliminar los casos en los que adopted no tiene un valor, para eso usamos Filter Examples, luego, debemos hacer que la columna adopted, nuestra columna objetivo, tenga el rol de label, luego utilizamos multiply para poder enviar el dataset a varios algoritmos distintos para determinar cual es el mejor para este caso.

Los resultados de Naive Bayes:

![Pelican](.\images\NaiveBayes.png)

Los resultados de K-NN:

![Pelican](.\images\knn.png)

Los resultados de Logistic Regression:

![Pelican](.\images\Logisticregression.png)

Viendo que Logistic Regression es el algoritmo que da mejores resultados, utilizamos Foward Selection con Logistic Regression para ver si podemos conseguir mejores resultados, la matriz de confusión obtenida es la siguiente:

![Pelican](.\images\Foward Selection.png)
