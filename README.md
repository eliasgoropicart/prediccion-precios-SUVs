# Predicción de precios de SUVs en el mercado argentino

Proyecto final de *Machine Learning and Deep Learning* centrado en la estimación del precio de venta de vehículos SUV publicados en el mercado argentino. El trabajo combina limpieza de datos, ingeniería de variables, imputación de faltantes, reducción de dimensionalidad y modelos supervisados de distinta complejidad para construir predictores cada vez más precisos.

El repositorio reúne tanto el informe final como los notebooks, scripts auxiliares, datasets intermedios y artefactos usados para entrenar, comparar y aplicar modelos sobre un conjunto de test enmascarado.

## Objetivo del proyecto

Construir modelos capaces de inferir el precio de una SUV a partir de variables observables como:

- marca y modelo
- año
- kilómetros
- transmisión
- tipo de combustible
- cantidad de puertas
- color
- tipo de vendedor
- presencia de cámara de retroceso
- información extraída de la descripción del motor, en particular cilindrada y turbo

Además de la predicción de precios, el proyecto busca:

- entender la estructura del dataset y sus faltantes
- evaluar si las relaciones entre variables son lineales o no lineales
- comparar familias de modelos
- medir robustez con RMSE y R²
- explorar aplicaciones prácticas, como detección de vehículos potencialmente subvaluados

## Resumen del enfoque

El trabajo parte de un baseline lineal y avanza hacia modelos más expresivos:

1. análisis exploratorio del dataset bruto extraído de Mercado Libre
2. limpieza, corrección de categorías y tratamiento de valores faltantes
3. imputación de variables complejas, especialmente `Cilindrada` y `Con cámara de retroceso`
4. codificación de variables categóricas y escalado Min-Max
5. evaluación de un regresor lineal como baseline
6. reducción de dimensionalidad y agrupamiento de categorías para estudiar si la linealidad alcanza
7. modelado no lineal con KNN y métricas de distancia adaptadas a datos esparsos/binarios
8. entrenamiento de árboles, Random Forest y XGBoost
9. experimentación con autoencoders y red neuronal densa
10. validación final y procesamiento de un conjunto de test enmascarado

## Resultados principales

Según el informe final incluido en el repositorio:

- el baseline lineal mostró desempeño insuficiente y sugirió subajuste
- KNN con distancia de Jaccard mejoró fuertemente al baseline y confirmó la presencia de no linealidades
- XGBoost logró `RMSE = 5065.09 USD` y `R² = 0.9086`
- la red neuronal profunda superó levemente a XGBoost con `RMSE = 4912.23 USD` y `R² = 0.9140`
- el mejor enfoque fue un ensamble entre red neuronal y XGBoost con `RMSE = 4451.59 USD` y `R² = 0.9294`
- en validación cruzada 5-fold, el ensamble obtuvo en promedio `RMSE = 6774.74 ± 1491.58 USD` y `R² = 0.8918 ± 0.0319`

## Dataset

El informe indica que se trabajó con **18.254 publicaciones de SUVs** extraídas de Mercado Libre entre el **13 y el 30 de mayo de 2025**.

Características relevantes del dataset:

- variables numéricas: año, kilómetros, puertas, cilindrada, precio
- variables categóricas: marca, modelo, color, combustible, transmisión, vendedor
- variables derivadas: turbo, cámara de retroceso, edad del vehículo
- moneda unificada a USD según el tipo de cambio promedio del período

También se identificaron:

- faltantes importantes en `Con cámara de retroceso`
- faltantes menores en `Color`, `Motor` y `Transmisión`
- alta cardinalidad en `Marca` y `Modelo`
- fuerte esparsidad luego de one-hot encoding

## Pipeline realizado

### 1. Exploración y diagnóstico inicial

Se parte del CSV bruto y se inspeccionan:

- duplicados
- valores faltantes
- distribuciones por variable
- calidad del texto en `Motor`, `Versión` y `Descripción`

Esta etapa está apoyada por:

- notebooks de preprocesamiento
- scripts para analizar NaNs
- archivos de conteo por variable

### 2. Limpieza y estandarización

Se realizaron tareas como:

- eliminación o revisión de filas problemáticas
- unificación de categorías equivalentes
- agrupación de colores similares
- reorganización de combustibles y transmisiones en grupos más representativos
- extracción de información útil del campo `Motor`

De esa extracción surgen variables clave:

- `Cilindrada`
- `EsTurbo`

### 3. Imputación de faltantes

Para evitar perder demasiadas observaciones, se imputaron variables importantes:

- `Cilindrada`: mediante un regressor Random Forest
- `Con cámara de retroceso`: mediante un clasificador Random Forest

Los modelos y escaladores usados para esta etapa quedaron guardados en la carpeta de testing.

### 4. Codificación y escalado

Una vez estabilizado el dataset:

- las variables categóricas se codificaron principalmente con one-hot encoding
- las variables numéricas se escalaron con Min-Max scaling
- se armó un dataset listo para entrenamiento supervisado

### 5. Baseline lineal

Se entrenó un regresor lineal para:

- fijar un punto de referencia
- medir la complejidad real del problema
- estudiar si la linealidad alcanzaba para capturar la relación entre atributos y precio

La conclusión fue que no.

### 6. Reducción de dimensionalidad y agrupamientos

Se exploraron dos líneas:

- **heurística del codo** para decidir qué marcas y modelos conservar explícitamente y cuáles agrupar
- **FAMD** para proyectar el espacio mixto y ver cómo se comporta el baseline con menos componentes

Esto permitió confirmar que reducir linealmente el espacio no resolvía el problema y que la pérdida de señal no lineal perjudicaba el desempeño.

### 7. Captura de no linealidades

Se probó KNN con distintas nociones de distancia:

- Euclidean
- Manhattan
- Hamming
- Cosine
- Jaccard

Para Jaccard se preparó un dataset binarizado especial, discretizando variables numéricas y adecuando la representación del espacio.

### 8. Modelos de mayor complejidad

Luego se entrenaron modelos basados en árboles y redes:

- árbol de decisión
- Random Forest
- XGBoost
- autoencoders
- red neuronal densa

Finalmente, el mejor resultado reportado se obtuvo con un **ensamble entre XGBoost y red neuronal**.

### 9. Aplicación al dataset de testing

Sobre el archivo `SUVS_2025-test-masked.csv` se replicó el procesamiento:

- limpieza y normalización
- extracción de cilindrada
- imputación de cámara y cilindrada con modelos guardados
- alineación de features
- escalado final

Eso genera datasets listos para inferencia sobre observaciones sin precio visible.

## Estructura del repositorio

### Archivo principal

- `Galliano_GoróPicart_Informe_PF.pdf`: informe final del proyecto. Resume motivación, metodología, resultados, validación y conclusiones.

### `Primeros procesamientos y modelos baseline/`

Contiene la primera etapa del proyecto: análisis exploratorio, limpieza inicial, construcción del dataset tabular y evaluación del baseline.

- `preprocesamiento.ipynb`: notebook principal de limpieza, transformación de variables y pruebas del baseline.
- `preprocesar_dataset.py`: función auxiliar para separar train/validación, escalar variables y devolver conjuntos listos para modelar.
- `analizar_nans.py`: utilidad para contar y desglosar filas con NaNs.
- `entrenar_graficar_errores_por_fraccion.py`: entrena un modelo con distintas fracciones del set de entrenamiento y grafica MSE, RMSE y R².
- `codo_cobertura.py`: implementación de la heurística del codo sobre cobertura acumulada de categorías.
- `pf_suvs_i302_1s2025.csv`: dataset original bruto de publicaciones.
- `dataset_procesado.csv`: dataset procesado completo con codificación de variables y precio listo para modelado.
- `dataset_reducido_con_agrupamiento.csv`: versión reducida tras agrupar categorías de alta cardinalidad.
- `dataset_limpio_reconstruido_sin_onehot.csv`: versión tabular limpia sin expansión one-hot.
- `df_predict_cilindrada.csv`: dataset orientado a la predicción/imputación de cilindrada.
- `df_predict_cilindrada_camera.csv`: dataset con variables necesarias para imputar cilindrada y cámara.
- `nombre_del_archivo.csv`: archivo auxiliar/intermedio generado durante el trabajo.
- `limpio1_conteo_Motor.txt`: conteos y depuración del campo motor.
- `limpio2_conteo_Cilindrada.txt`: conteos de cilindrada.
- `limpio3_conteo_EsTurbo.txt`: conteos de la variable turbo.
- `limpio4_conteo_Cilindros.txt`: conteos relacionados con cilindros.
- `limpio5_conteo_TipoInyeccion.txt`: conteos de tipo de inyección.
- `limpio6_conteo_Potencia.txt`: conteos de potencia.
- `limpio7_conteo_Torque.txt`: conteos de torque.
- `limpio8_conteo_ConfiguracionMotor.txt`: conteos de configuración de motor.

### `Distribuciones/`

Agrupa archivos de conteo y distribución univariada de variables del dataset, útiles para exploración y toma de decisiones de limpieza.

- `conteo_Año.txt`
- `conteo_Color.txt`
- `conteo_Con cámara de retroceso.txt`
- `conteo_Kilómetros.txt`
- `conteo_Cilindrada.txt`
- `conteo_Marca.txt`
- `conteo_Modelo.txt`
- `conteo_Moneda.txt`
- `conteo_Motor.txt`
- `conteo_Precio.txt`
- `conteo_Puertas.txt`
- `conteo_Tipo de carrocería.txt`
- `conteo_Tipo de combustible.txt`
- `conteo_Tipo de vendedor.txt`
- `conteo_Transmisión.txt`

En conjunto, estos archivos sirven para entender sesgos, cardinalidades, errores frecuentes y desbalances.

### `Capturación de no linealidades/`

Reúne la etapa en la que se abandona el paradigma lineal y se estudian modelos sensibles a la estructura local del espacio.

- `modelos_no_lineales.ipynb`: notebook de KNN y comparación de métricas de distancia.
- `dataset_binarizado_jaccard.csv`: dataset binarizado especialmente preparado para usar distancia de Jaccard.
- `dataset_procesado.csv`: copia del dataset procesado usada como base en esta etapa.
- `ranking_mejores_metrica.csv`: ranking de las mejores configuraciones por métrica.

### `Modelos de mayor complejidad/`

Contiene la fase de árboles, boosting y redes neuronales.

- `redes.ipynb`: notebook con modelos basados en árboles, XGBoost y redes neuronales.
- `df_predict_cilindrada_camera.csv`: dataset de entrada usado en esta etapa para entrenar modelos más complejos.

### `Procesamiento del dataset de testing/`

Carpeta dedicada a transformar el conjunto de test enmascarado y aplicar modelos de imputación/normalización compatibles con entrenamiento.

- `procesamiento_dataset_testing.ipynb`: notebook principal para el pipeline de test.
- `SUVS_2025-test-masked.csv`: dataset de testing sin precio visible.
- `dataset_testing_procesado.csv`: versión procesada y escalada del test.
- `dataset_testing_procesado_sin_escalar.csv`: versión procesada antes del escalado final.
- `dataset_procesado.csv`: referencia del dataset de entrenamiento usado para alinear columnas/features.
- `random_forest_cilindrada.pkl`: modelo para imputar cilindrada.
- `random_forest_camera.pkl`: modelo para imputar cámara de retroceso.
- `scaler_cilindrada.pkl`: scaler asociado a la imputación de cilindrada.
- `scaler_camera.pkl`: scaler asociado a la imputación de cámara.
- `scaler_minmax.pkl`: scaler final para llevar el dataset de test a la misma escala del entrenamiento.
- `features_cilindrada.pkl`: lista de features esperadas por el modelo de cilindrada.
- `features_camera.pkl`: lista de features esperadas por el modelo de cámara.

### `__pycache__/`

Archivos compilados automáticamente por Python para acelerar imports. No contienen lógica del proyecto.

### Archivo temporal de oficina

- `.~lock.SUVS_2025-test-masked.csv#`: archivo de bloqueo generado por LibreOffice u otra herramienta de edición. No forma parte del pipeline.

## Flujo recomendado para recorrer el proyecto

Si se quiere entender el trabajo de punta a punta, el orden sugerido es:

1. leer `Galliano_GoróPicart_Informe_PF.pdf`
2. revisar `Primeros procesamientos y modelos baseline/preprocesamiento.ipynb`
3. consultar los scripts auxiliares de limpieza y análisis de faltantes
4. mirar `Capturación de no linealidades/modelos_no_lineales.ipynb`
5. continuar con `Modelos de mayor complejidad/redes.ipynb`
6. cerrar con `Procesamiento del dataset de testing/procesamiento_dataset_testing.ipynb`

## Metodología de evaluación

Las métricas principales usadas en el proyecto fueron:

- **RMSE**: error cuadrático medio en dólares
- **R²**: proporción de varianza explicada

También se utilizaron:

- curvas de error según fracción de entrenamiento
- comparación entre reales y predichos
- validación cruzada
- análisis visual de proyecciones latentes

## Hallazgos y conclusiones

Las principales conclusiones del trabajo fueron:

- la relación entre las variables y el precio no puede capturarse adecuadamente con un modelo lineal simple
- la estructura del dataset presenta alta dimensionalidad, esparsidad y dependencias no lineales
- KNN con una noción de distancia apropiada ya produce una mejora fuerte sobre el baseline
- los modelos basados en árboles, especialmente XGBoost, capturan mejor estas interacciones
- una red neuronal bien ajustada logra superar levemente a XGBoost
- el ensamble entre red neuronal y XGBoost ofrece el mejor equilibrio entre precisión y robustez
- el sistema tiene potencial para extenderse a casos reales de pricing y detección de oportunidades de compra

## Requisitos aproximados

El repositorio no incluye un archivo de dependencias congeladas, pero por el código utilizado se infiere que se trabajó con bibliotecas como:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `joblib`
- `tensorflow / keras`
- `xgboost`

## Notas

- Algunos resultados finales del informe mencionan un modelo ensamble y experimentos adicionales que no están empaquetados como scripts separados en esta carpeta raíz; parte de la lógica vive dentro de notebooks o en materiales externos enlazados desde el PDF.
- El apéndice del informe referencia una carpeta externa de Google Drive con códigos fuente y predicciones adicionales.

