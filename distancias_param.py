# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:32:22 2019

@author: solis

parámetros de la función distancias_puntos. Calcula la ditancia entre
    los puntos que devuelve select3 con respecto a los que devuelve el select1
"""
# param relacionados con los sondeos de la red la demarcación
dbAemet = r'C:\Users\solis\Documents\DB\AEMET_CHS.accdb'
dbIpa1 = r'C:\Users\solis\Documents\DB\Ipasub97.mdb'

PATHOUT = r'C:\Users\solis\Documents\work\interpol'
FOUT = 'distancias_ipa1_aemet.txt'

# select set 1
SELECT1 = "SELECT ID, X, Y FROM Estaciones;"

# select set 2
SELECT3 = "SELECT COD, X_UTM as X, Y_UTM as Y FROM IPA1 " +\
          "WHERE X_UTM>300000 and Y_UTM>4000000;"

# numero de puntos de select1 que se devuelven por cada punto de select3
k = 2
