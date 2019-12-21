# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:53:24 2019

@author: solis

parámetros de un módulo de funciones de interpolación de series temporales
    en puntos sin datos
implementa idw y las funciones de base radiales implementadas en scipy
"""

"""
PATHIN: directorio donde se ubica en fichero de puntos a interpolar (de la
    forma r'path2dis'; r es un indicador para que se interprete \ como un
    separados de carpetas)
PATHOUT: directorio donde se grabarán los datos interpolados
FPOINTS. Fichero con los puntos a interpolar y de resultados. El fichero debe
    tener al menos 1 fila y 3 o 4 columnas de datos:
        columna 1 el código del punto
        columnas 2 y 3: coordenadas x e y
        columna 4 (opcional): coordenada z
    El separador decimal es el punto, por ej. 1.2 y no debe haber separadores
        de miles, por ej. 1000.3. El separador de columnas es el tabulador.
    skip_lines: número de lineas en el fichero de puntos a interpolar que no
        se leen (cabecera, normalmente 1)
SHORT_NAME_4_VARIABLE: nombre muy corto que describe la variable a interpolar
El nombre del fichero de resultados se forma: nombre de FPOINTS +
   SHORT_NAME_4_VARIABLE + método de interpolación utilizado + extensión.
   También se crea un fichero de metadatos de la interpolación
"""
PATHIN = r'C:\Users\solis\Documents\work\manantiales'
PATHOUT = PATHIN
FPOINTS = 'masub_centroids.txt'
skip_lines: int = 1
SHORT_NAME_4_VARIABLE = 'pmd'

"""
DB con los datos.
    dbMeteoro: db con los datos; implementada accees, otro tipo puede ser
    select1: select para extraer los datos de dbMeteoro. Las columnas deben
        tener los siguientes nombres:
        columna 1:ID (identificador)
        columna 2: X, coord. X
        columna 3: Y
        columna 4: Z (puede faltar)
        columna siguiente: Fecha
        columna siguiente: v (valor del dato)
"""
dbMeteoro = r'C:\Users\solis\Documents\DB\aemet\EstacionesEspanya\ptexistencias.accdb'
# =============================================================================
# SELECT1 = "SELECT Estaciones.ID, Estaciones.X, Estaciones.Y, " +\
#               "Estaciones.ALTITUD as Z, PD.Fecha, PD.VALUE as v " +\
#           "FROM Estaciones INNER JOIN PD ON Estaciones.ID = PD.ID " +\
#           "WHERE PD.Fecha=?;"
# =============================================================================
SELECT1 = "SELECT Pexistencias.ind, Pexistencias.utmx as X, " +\
              "Pexistencias.utmy as Y, 0 AS Z, pmes.fecha, pmes.prec as v " +\
          "FROM Pexistencias INNER JOIN pmes ON Pexistencias.ind = pmes.indic "\
          "WHERE pmes.fecha=?;"

""" paso de tiempo de los datos
day. dia a día desde la fecha inicial hasta la fecha final
month. los datos mensuales están asociados a fechas; cada dato está
    asociado a la última fecha del mes; el programa está preparado para tratar
    estos saltos diariso variables entre fechas. Si la bdd los datos mensuales
    no están almacenados de esta manera no encontrará datos para hacer las
    interpolaciones
"""
time_step = 'month'

""" fechas entre las que se interpola, ambas incluidas
    fecha1(day1, month1, year1), fecha2(day2...
    Si time step=='month' las fechas inicial y final es aconsejable que sea
        el último días del mes. ¿Que pasaría si day1==15; en esa fecha no
        encontraría ningún dato y no interpolaría, pero la siguiente fecha
        ya sería el último día del mes siguiente
"""
day1, month1, year1 = 31, 1, 1950
day2, month2, year2 = 31, 10, 2019

""" METODOS DE INTERPOLACION """
""" Para idw
    poweridw: potencia en el método idw (1/dist**powerid); frecuentemente 2.
    kidw: número de puntos con los que se realiza la interpolación (min 2)
    epsidw: si la distancia entre un punto con dato y un punto a interpolar es
        menor que eps, la distancia se aumenta a mindist
    mindistidw: ver eps """
poweridw: float = 2.0
kidw: int = 8
epsidw: float = 0.001
mindistidw: float = 1.0

""" Para rbf
    krbf: número de puntos que intervienen en la interpolación de un punto;
        para un número grande y según la variable a interpolar el método
        da error -matriz no singular- (min 2)
    epsrbf: si la suma de los valores de los krbf puntos más próximos <= eps,
        el valor intepolado se asigna 0; es una condición pensada para trabajar
        con variables positivas con muchos ceros; si no se quiere que
        intervenga, asignarle un valor negativo muy bajo, por ej. -999999.
    mindistrbf: si la distancia entre el punto a interpolar y el más próximo
        con dato es <= mindistrbf se le asigna el valor de ese en el punto más
        próximo
    smooth: 0. no smooth >0. smooth (ver doc en scipy)
    force0: si 1 los valores interpolados < 0 se igualen a cero 0
"""
krbf: int = 4
epsrbf: float = 1.
mindistrbf: float = 10.
smooth: float = 0.
force0: int = 1
