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
PATHIN = r'C:\Users\solis\Documents\work\interpol'
PATHOUT = PATHIN
FPOINTS = 'manantialesh1.txt'
skip_lines: int = 1
SHORT_NAME_4_VARIABLE = 'pd'

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
dbMeteoro = r'C:\Users\solis\Documents\DB\AEMET_CHS.accdb'
SELECT1 = "SELECT Estaciones.ID, Estaciones.X, Estaciones.Y, " +\
              "Estaciones.ALTITUD as Z, PD.Fecha, PD.VALUE as v " +\
          "FROM Estaciones INNER JOIN PD ON Estaciones.ID = PD.ID " +\
          "WHERE PD.Fecha=?;"

""" paso de tiempo de los datos """
time_step = 'day'

""" fechas entre las que se interpola, ambas incluidas
    fecha1(day1, month1, year1), fecha2(day2... """
day1, month1, year1 = 1, 1, 1985
day2, month2, year2 = 31, 12, 2000

""" METODOS DE INTERPOLACION """
""" Para idw
    poweridw: potencia en el método idw (1/dist**powerid); frecuentemente 2.
    kidw: número de puntos con los que se realiza la interpolación (min 2)
    epsidw: si la distancia entre un punto con dato y un punto a interpolar es
        menor que eps, la distancia se aumenta a mindist
    mindistidw: ver eps """
poweridw: float = 2.0
kidw: int = 4
epsidw: float = 0.001
mindistidw: float = 1.0

""" Para rbf
    krbf: número de puntos que intervienen en la interpolación de un punto;
        para un número grande y según la variable a interpolar el método
        da error -matriz no singular-
    epsrbf: si la suma de los valores de los krbf puntos más próximos <= eps,
        el valor intepolado se asigna 0; es una condición pensada para trabajar
        con variables positivas con muchos ceros; si no se quiere que
        intervenga, asignarle un valor negativo muy bajo, por ej. -999999.
    mindistrbf: si la distancia entre el punto a interpolar y el más próximo
        con dato es <= mindistrbf se le asigna el valor de ese en el punto más
        próximo
    smooth: 0. no smooth >0. smooth (ver doc en scipy)
    force0: si 1 los valores interpolados < 0 se igualen a cero 0
    krbf: número de puntos con los que se realiza la interpolación (min 2) """
krbf: int = 4
epsrbf: float = 1.
mindistrbf: float = 10.
smooth: float = 0.
force0: int = 1
