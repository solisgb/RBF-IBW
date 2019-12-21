# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:26:19 2019

@author: solis

implementa la interpolación de series temporales en puntos sin datos utilizando
    el método idw y varias funciones de interpolación de base radial
    implementadas en scipy
"""
import numpy as np
from scipy import spatial, interpolate
import pyodbc
import littleLogging as logging

# Mensajes que aparecen más de una vez
MSG_NO_OPTION = 'Salir'
MSG_NO_VALID_OPTION = 'No es una selección válida'
MSGWRITE1OPTION = 'Sitúa el cursor al final de la línea ' + \
                  'y teclea un número entre los mostrados: '
MSG_FIN_PROCESO = 'Pulse una tecla para continuar'


def selectOption(options: (list, tuple), headers: (list, tuple)) -> int:
    """
    Función genérica para seleccionar una opción
    args
    options: strings que permiten identicar las opciones
    headers: : strings que explican las opciones
    """
    while True:
        for header in headers:
            print(header)
        extendedOptions = [MSG_NO_OPTION] + list(options)

        for i, option in enumerate(extendedOptions):
            print('{:d}.- {}'.format(i, option))
        selection = input(MSGWRITE1OPTION)
        try:
            selection = int(selection)
        except ValueError:
            print(MSG_NO_VALID_OPTION)
            continue
        if selection < 0 or selection > len(extendedOptions) - 1:
            print(MSG_NO_VALID_OPTION)
            continue
        else:
            return selection


def menu():
    """
    llama a una de las funciones de interolación
    el contenido de options[1:] debe coincidir en el de funcs en la función
        rbflocal_serie_temporal
    """
    options = ('idw', 'multiquadric', 'inverse', 'gaussian', 'linear',
               'cubic', 'quintic', 'thin_plate')
    headers = ('Programa de interpolación', 'Selecciona un método')

    iop = selectOption(options, headers)

    if iop == 0:
        return
    if iop == 1:
        idw_serie_temporal()
    else:
        irbf = iop - 1
        rbflocal_serie_temporal(options[irbf])


def rowsGet(conStr, select):
    con = pyodbc.connect(conStr)
    cur = con.cursor()
    cur.execute(select)
    colNames = [column[0] for column in cur.description]
    rows = [row for row in cur]
    con.close()
    return rows, colNames


def distancias_puntos():
    """
    para 2 sets de puntos, calcula los puntos de select2 a cada punto de
        select1
    las db de cada select puedenser distintas
    """
    from os.path import join
    import distancias_param as par

    cstrAemet = r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)}; ' +\
                f'DBQ={par.dbAemet};'
    cstrIpa1 = r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)}; ' +\
               f'DBQ={par.dbIpa1};'

    stations, _ = rowsGet(cstrAemet, par.SELECT1)
    xSt = np.array([[row[1], row[2]] for row in stations])

    ipas, _ = rowsGet(cstrIpa1, par.SELECT3)
    xIpas = np.array([[row[1], row[2]] for row in ipas])

    treeSt = spatial.cKDTree(xSt)
    if par.k > len(stations):
        par.k = len(stations)
    dd, ii = treeSt.query(xIpas, k=par.k)

    with open(join(par.PATHOUT, par.FOUT), 'w') as f:
        for j, (d, i) in enumerate(zip(dd, ii)):
            if par.k == 1:
                f.write(f'{ipas[j][0]}\t{stations[i][0]}\t{d:0.2f}\n')
            else:
                for d1, i1 in zip(d, i):
                    f.write(f'{ipas[j][0]}\t{stations[i1][0]}\t{d1:0.2f}\n')


def shapeRBF():
    """
    evalúa la forma de las RBF
    'multiquadric': np.sqrt((r/epsilon)**2 + 1),
    'inverse': 1.0/np.sqrt((r/epsilon)**2 + 1),
    'gaussian': np.exp(-(r/epsilon)**2),
    'linear': r,
    'cubic': r**3,
    'quintic': r**5,
    'thin_plate': r**2 * np.log(r)
    También evalúa idw
    """
    from os.path import join
    import matplotlibInterface as MPLI
    import shapeRBF_param as par

    funcs = ('multiquadric', 'inverse', 'gaussian', 'linear',
             'cubic', 'quintic', 'thin_plate')
    smooths = (0., 0.25)
    powers = (1.5, 2.0, 3.0)
    # funcion sin evaluada en ndata
    ndata = 201
    xdata0 = np.linspace(-np.pi, np.pi, ndata, np.float32)
    zdata0 = np.sin(xdata0)

    # funcion sin evaluada en mdata
    mdata = 5
    xdata = np.linspace(-np.pi, np.pi, mdata, np.float32)
    ydata = np.zeros(mdata, np.float32)
    zdata = np.sin(xdata)

    # nip puntos donde se va a evaluar la funcion sin con mdata
    nip = 25
    xi = np.linspace(-np.pi, np.pi, nip, np.float32)
    yi = np.zeros(nip, np.float32)

    fig = MPLI.MatplotlibInterface()
    fig.title = f'fun. seno'
    fig.append(xdata0, zdata0, f'seno {ndata} puntos')
    fig.append(xdata, zdata, f'seno {mdata} puntos',
               rc_params={'lines.marker':'.'})
    fig.plot(join(par.PATHOUT, 'func_seno.png'))

    # rbf distintos valores de smooth
    for func in funcs:
        print(func)
        fig = MPLI.MatplotlibInterface()
        fig.title = f'fun. seno y rbf {func}'
        fig.append(xdata0, zdata0, f'sin {ndata} puntos')
#        fig.append(xdata, zdata, f'seno {mdata} puntos')

        for smooth1 in smooths:
            rbfi = interpolate.Rbf(xdata, ydata, zdata, function=func,
                                   smooth=smooth1)
            zi = rbfi(xi, yi)
            leg = f'{func} {smooth1:0.2f}'
            fig.append(xi, zi, leg)
        fig.plot(join(par.PATHOUT, f'{func}.png'))

    # idw distintos valores de la potencia
    fig = MPLI.MatplotlibInterface()
    fig.title = f'fun. seno e idw'
    fig.append(xdata0, zdata0, f'sin {ndata} puntos')
#    fig.append(xdata, zdata, f'seno {mdata} puntos')
    for power in powers:
        zi = idw(xdata, ydata, zdata, xi, yi, power=power, n=mdata)
        leg = f'idw pow {power:0.2f}'
        fig.append(xi, zi, leg)
    fig.plot(join(par.PATHOUT, 'idw.png'))

    # todas las funciones en una figura sin smooth
    fig = MPLI.MatplotlibInterface()
    fig.title = f'fun. seno y rbf'
    fig.append(xdata, zdata, f'seno {ndata} puntos')
    for func in funcs:
        print(func)
        rbfi = interpolate.Rbf(xdata, ydata, zdata, function=func, smooth=0.0)
        zi = rbfi(xi, yi)
        leg = f'{func}'
        fig.append(xi, zi, leg)
    power = 2.0
    zi = idw(xdata, ydata, zdata, xi, yi, power=power, n=mdata)
    leg = f'idw pow {power:0.2f}'
    fig.append(xi, zi, leg)
    fig.plot(join(par.PATHOUT, f'rbfs.png'))


def points2interpolate_get(org, skip_lines):
    """
    lee los puntos a interpolar de un fichero de texto y devuelve un
        numpy array
    """
    with open(org, 'r') as f:
        rows = np.array([row.split('\t')
                         for i, row in enumerate(f.readlines())
                         if i >=skip_lines])
    return rows


def idw(xdata, ydata, zdata, xi, yi, power: float = 2.0, n: int = 4,
        mindist: float = 0.0001):
    """
    inverse distance weighted elevated to power in 2D
    input
    xdata, ydata, zdata: np.arrays shape (1,) equal length with x, y
        coordinates and value of the variable
    xi, yi: idem interpolation points
    n: nearest number of points considered in the interpolation
    mindist: si el punto a interpolar está a una distancia igual o menor
        de mindist el valor en el punto interpolado es igual al dato
    """
    xydata = np.column_stack((xdata, ydata))
    xyi = np.column_stack((xi, yi))
    zi = np.empty(xi.size)
    if n < 2:
        n = 4
    if n > xdata.size:
        n = xdata.size
    z = np.empty(n)

    tree = spatial.cKDTree(xydata)
    dd, ii = tree.query(xyi, k=n)
    tree = None
    for i, (d1, ii1) in enumerate(zip(dd, ii)):
        if d1[0] <= mindist:
            zi[i] = zdata[ii1[0]]
        else:
            for j, k in enumerate(ii1):
                z[j] = zdata[k]
            weights = 1./d1**power
            z = z * weights
            zi[i] = np.sum(z) / np.sum(weights)
    return zi


def idwcore(dat, coldat, dist, ii, power, z, zi):
    """
    interpolación idw elevado a una potencia
    input
        dat: array de datos dim (n, m)
            n: número de puntos
            m: x, y, (z), dato en el punto
        coldat: número de columna en dat del dato
        dist: array de distancias dim (n, k) k<=m
        ii: índices de los datos de dist en dat
        power: potencia
    output
        z: array dim (n, k), ya viene dimensionado
        zi: array de valores interpolados dim (n), ya viene dimensionado
    """
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            z[i,j] = dat[ii[i, j], coldat]
    weights = 1/dist**power
    z = z * weights
    for i in range(zi.shape[0]):
        zi[i] = np.sum(z[i, :]) / np.sum(weights[i, :])


def idw_serie_temporal():
    """
    Método de interpolación inverse distance weighted con distance elevada a
        una potencia, normalmente 2
    Interpola la variable en una serie de puntos dados en 3D. Si no tienes
        valor de Z pon en todos los puntos un valor cte, por ej 0
    Los datos son una serie temporal; los puntos con datos varían en el
        tiempo
    """
    from datetime import date, timedelta
    from os.path import join
    from time import time
    import interpolation_param as par

    time_steps = ('diaria', 'day', 'mensual', 'month', 'anual', 'year')
    cstrAemet = r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)}; ' +\
                f'DBQ={par.dbMeteoro};'
    PRINT_INTERVAL = 5.

    if par.time_step in time_steps[:2]:
        tstep = 1
        tstep_type = 1
        time_step = timedelta(days=1)
        datei = date(par.year1, par.month1, par.day1)
        datefin = date(par.year2, par.month2, par.day2)
    elif par.time_step in time_steps[2:4]:
        tstep = 1
        tstep_type = 2
        datei = date(par.year1, par.month1, par.day1)
        datefin = date(par.year2, par.month2, par.day2)
    else:
        raise ValueError(f'valor de time_step {par.time_step} ' +\
                         'no implementado')

    start_time = time()

    # datos donde hay que interpolar
    rows = points2interpolate_get(join(par.PATHIN, par.FPOINTS),
                                  par.skip_lines)
    fidi = rows[:, 0]  # array con los id de los puntos
    if rows.shape[1] == 3:  # 2D
        xi = rows[:, [1, 2]].astype(np.float32)  # array con las coordenadas
    elif rows.shape[1] == 4:  #3D
        xi = rows[:, [1, 2, 3]].astype(np.float32)
    else:
        raise ValueError('El núm de columnas en el fichero de puntos debe ' +\
                         '3 o 4: id del punto, x, y, (z)')
    rows = None
    z = np.empty((len(xi), par.kidw), np.float32)
    zi = np.empty((len(xi)), np.float32)  # array para los valores interpolados

    # datos para hacer las interpolaciones
    con = pyodbc.connect(cstrAemet)
    cur = con.cursor()

    # fichero de salida
    dst = file_name_out(par.PATHOUT, par.FPOINTS, par.SHORT_NAME_4_VARIABLE,
                        'idw')
    f = open(dst, 'w')
    f.write('fid\tfecha\tvalor\n')

    t0 = PRINT_INTERVAL
    while datei <= datefin:
        if tstep == 1:
            dateStr = datei.strftime('%d/%m/%Y')
            if time() - t0 > PRINT_INTERVAL:
                t0 = time()
                print(dateStr)
        cur.execute(par.SELECT1, (datei,))
        data = [row for row in cur]
        if xi.shape[1] == 2:
            data = np.array([[row.X, row.Y, row.v] for row in data])
            tree = spatial.cKDTree(data[:, [0, 1]])
        else:
            data = np.array([[row.X, row.Y, row.Z, row.v] for row in data])
            tree = spatial.cKDTree(data[:, [0, 1, 2]])
        dist, ii = tree.query(xi, k=par.kidw)
        dist = np.where(dist<=par.epsidw, par.mindistidw, dist)

        idwcore(data, xi.shape[1], dist, ii, par.poweridw, z, zi)

        for i in range(len(fidi)):
            f.write(f'{fidi[i]}\t{dateStr}\t{zi[i]:0.1f}\n')

        if tstep_type == 1:
            datei = datei + time_step
        elif tstep_type == 2:
            datei = addmonth_lastday(datei)
        else:
            raise ValueError(f'tstep_type {tstep_type} no implementado')

    elapsed_time = time() - start_time
    print(f'La interpolación tardó {elapsed_time:0.1f} s')

    # fichero de metadatos
    dst = file_name_out(par.PATHOUT, par.FPOINTS, par.SHORT_NAME_4_VARIABLE,
                        'idw_matadata')
    with open(dst, 'w') as f:
        f.write(f'idw\n')
        f.write(f'potencia a la que se eleva la dist, {par.poweridw:f}\n')
        f.write(f'núm datos por interpolación, {par.kidw:d}\n')
        f.write('distancia máx. a la que se considera que las ' +\
                'localizaciones de una estación y un punto coinciden, ' +\
                f'{par.epsidw:f}\n')
        f.write('corrección de la distancia de un punto para evitar ' +\
                f'infinitos en 1/dist, {par.mindistidw:f}\n')
        f.write(f'db de los datos, {par.dbMeteoro}\n')
        f.write(f'número de puntos interpolados {fidi.size:d}\n')
        f.write(f'tiempo transcurrido, {elapsed_time:0.1f} s\n')
        f.write(f'incidencias\n')
        a = logging.get_as_str()
        if a:
            f.write(f'{a}\n')
    _ = input(MSG_FIN_PROCESO)


def file_name_out(pathout: str, name_file: str, variable_name: str = None,
                  interpol_method: str = None):
    """
    forma el nombre del fichero de salida
    """
    from os.path import join, splitext
    name_file, extension = splitext(name_file)
    name_file = f'{name_file}_{variable_name}_{interpol_method}{extension}'
    return join(pathout, name_file)


def rbflocal_serie_temporal(rbf):
    """
    Interpola la variable en una serie de puntos dados en 3D. Si no tienes
        valor de Z pon en todos los puntos un valor cte, por ej 0
    Los datos son una serie temporal; los puntos con datos varían en el
        tiempo
    el contenido de options[1:] debe coincidir en el de funcs en la función
        rbflocal_serie_temporal
    """
    from datetime import date, timedelta
    from os.path import join
    from time import time
    import interpolation_param as par
    funcs = ('multiquadric', 'inverse', 'gaussian', 'linear',
             'cubic', 'quintic', 'thin_plate')
    time_steps = ('diaria', 'day', 'mensual', 'month', 'anual', 'year')
    PRINT_INTERVAL = 5.

    cstrAemet = r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)}; ' +\
                f'DBQ={par.dbMeteoro};'

    if par.time_step in time_steps[:2]:
        tstep = 1
        tstep_type = 1
        time_step = timedelta(days=1)
        datei = date(par.year1, par.month1, par.day1)
        datefin = date(par.year2, par.month2, par.day2)
    elif par.time_step in time_steps[2:4]:
        tstep = 1
        tstep_type = 2
        datei = date(par.year1, par.month1, par.day1)
        datefin = date(par.year2, par.month2, par.day2)
    else:
        raise ValueError(f'valor de time_step {par.time_step} ' +\
                         'no implementado')

    if rbf not in funcs:
        raise ValueError(f'{rbf} no es una función válida')

    start_time = time()

    # datos donde hay que interpolar
    rows = points2interpolate_get(join(par.PATHIN, par.FPOINTS),
                                  par.skip_lines)
    icols = [i for i in range(rows.shape[1]) if i > 0]
    if len(icols) < 2 or len(icols) > 3:
        raise ValueError('El núm de columnas en el fichero de puntos debe ' +\
                         '3 o 4: id del punto, x, y, (z)')
    ival = len(icols)
    fidi = rows[:, 0]  # array con los id de los puntos
    xi = rows[:, icols].astype(np.float32)  # array con sus coordenadas
    rows = None
    zi = np.empty((len(fidi)), np.float32)
    # el array de datos q intervienen en la interpolación tienen una columna
    # que el array de puntos, que es la del dato
    data_sl = np.empty((par.krbf, xi.shape[1] + 1), np.float32)

    # datos para hacer las interpolaciones
    con = pyodbc.connect(cstrAemet)
    cur = con.cursor()

    # fichero de salida
    dst = file_name_out(par.PATHOUT, par.FPOINTS, par.SHORT_NAME_4_VARIABLE,
                        rbf)
    f = open(dst, 'w')
    f.write('fid\tfecha\tvalor\n')

    t0 = PRINT_INTERVAL
    while datei <= datefin:
        if tstep == 1:
            dateStr = datei.strftime('%d/%m/%Y')
            if time() - t0 > PRINT_INTERVAL:
                t0 = time()
                print(dateStr)
        cur.execute(par.SELECT1, (datei,))
        data = [row for row in cur]
        if len(data) < par.krbf:
            logging.append(f'No datos en {dateStr}')
            continue
        if xi.shape[1] == 2:
            data = np.array([[row.X, row.Y, row.v] for row in data])
            tree = spatial.cKDTree(data[:, [0, 1]])
        else:
            data = np.array([[row.X, row.Y, row.Z, row.v] for row in data])
            tree = spatial.cKDTree(data[:, [0, 1, 2]])

        # local interpolation
        for i in range(len(fidi)):
            dist, ii = tree.query(xi[i,:], k=par.krbf)
            if dist[0] <= par.mindistrbf:
                zi[i] = data[ii[0], ival]
                continue
            for j in range(par.krbf):
                data_sl[j] = data[ii[j]]
            if np.sum(data_sl[:, ival]) <= par.epsrbf:
                zi[i] = 0.
                continue
            if xi.shape[1] == 2:
                try:
                    rbfi = interpolate.Rbf(data_sl[:, 0], data_sl[:, 1],
                                           data_sl[:, 2], function=rbf,
                                           smooth=par.smooth)
                    do_interpolation = True
                except np.linalg.LinAlgError as er:
                    logging.append(f'punto\t{i:d}\tfecha {dateStr} {er}',
                                   toScreen = False)
                    do_interpolation = False
                if do_interpolation:
                    zi[i] = rbfi(xi[i, 0], xi[i, 1])
                else:
                    zi[i] = data[ii[0], ival]
            else:
                try:
                    rbfi = interpolate.Rbf(data_sl[:, 0], data_sl[:, 1],
                                           data_sl[:, 2], data_sl[:, 3],
                                           function=rbf, smooth=par.smooth)
                    do_interpolation = True
                except np.linalg.LinAlgError as er:
                    logging.append(f'punto\t{i:d}\tfecha {dateStr} {er}',
                                   toScreen = False)
                    do_interpolation = False
                if do_interpolation:
                    zi[i] = rbfi(xi[i, 0], xi[i, 1], xi[i, 2])
                else:
                    zi[i] = data[ii[0], ival]
        if par.force0 == 1:
            zi = np.where(zi < 0., 0., zi)
        for i in range(len(fidi)):
            f.write(f'{fidi[i]}\t{dateStr}\t{zi[i]:0.1f}\n')
        if tstep_type == 1:
            datei = datei + time_step
        elif tstep_type == 2:
            datei = addmonth_lastday(datei)
        else:
            raise ValueError(f'tstep_type {tstep_type} no implementado')

    elapsed_time = time() - start_time
    print(f'La interpolación tardó {elapsed_time:0.1f} s')

    # fichero de metadatos
    dst = file_name_out(par.PATHOUT, par.FPOINTS, par.SHORT_NAME_4_VARIABLE,
                        rbf+'_metadata')
    with open(dst, 'w') as f:
        f.write(f'radial basis function, {rbf}\n')
        f.write(f'núm datos por interpolación, {par.krbf:d}\n')
        f.write('valor de suma de datos para que el valor interpolado se ' +\
                f'considere 0 sin interpolar, {par.epsrbf:f}\n')
        f.write('distancia máx.a la que se asigna el valor más próximo, ' +\
                f'{par.mindistrbf:f}\n')
        f.write(f'smooth, {par.smooth:f}\n')
        f.write(f'valores interpolados <0 se igualan a 0, {par.force0:d}\n')
        f.write(f'db de los datos, {par.dbMeteoro}\n')
        f.write(f'número de puntos interpolados {fidi.size:d}\n')
        f.write(f'tiempo transcurrido, {elapsed_time:0.1f} s\n')
        f.write(f'incidencias\n')
        a = logging.get_as_str()
        if a:
            f.write(f'{a}\n')

    _ = input(MSG_FIN_PROCESO)


def rbfInterpolation3Da(rbf):
    """
    Interpola la variable en una serie de puntos dados en 3D. Si no tienes
        valor de Z pon en todos los puntos un valor cte, por ej 0
    Para cada conjunto de datos, los utiliza todos para formar el interpolador
        de base radial; con datos de precipitación diaria esta aproximación
        da lugar a matrices no singulares y a errores de interpolación
    Esta función no se llama desde la interfaz
    """
    from datetime import date, timedelta
    from os.path import join
    import interpolation_param as par

    time_steps = ('diaria', 'day', 'mensual', 'month', 'anual', 'year')
    cstrAemet = r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)}; ' +\
                f'DBQ={par.dbMeteoro};'
    funcs = ('multiquadric', 'inverse', 'gaussian', 'linear', 'thin_plate')

    if par.time_step in time_steps[:2]:
        tstep = 1
        time_step = timedelta(days=1)
        datei = date(par.year1, par.month1, par.day1)
        datefin = date(par.year2, par.month2, par.day2)
    else:
        raise ValueError(f'valor de time_step {par.time_step} ' +\
                         'no implementado')

    if rbf not in funcs:
        raise ValueError(f'{rbf} no es una función válida')

    con = pyodbc.connect(cstrAemet)
    cur = con.cursor()

    with open(join(par.PATHIN, par.FPOINTS)) as f:
        rows = np.array([row.split('\t')
                         for i, row in enumerate(f.readlines())
                         if i >= par.skip_lines])

    fidi = rows[:, 0]
    xyzi = rows[:, [1, 2, 3]].astype(np.float32)
    rows = None

    dst = file_name_out(par.PATHOUT, par.FPOINTS, par.SHORT_NAME_4_VARIABLE,
                        rbf)
    f = open(dst, 'w')

    while datei <= datefin:
        if tstep == 1:
            dateStr = datei.strftime('%d/%m/%Y')
            print(dateStr)
        cur.execute(par.SELECT1, (datei,))
        data = [row for row in cur]
        xyzv = np.array([[row.X, row.Y, row.Z, row.v] for row in data])

        if par.rbf in funcs[0:3]:
            if par.epsrbf is None:
                op = 0
            else:
                op = 1
        else:
            op = 1

        if op == 0:
            rbfi = interpolate.Rbf(xyzv[:, 0], xyzv[:, 1], xyzv[:, 2],
                                   xyzv[:, 3], function=rbf)
        else:
            rbfi = interpolate.Rbf(xyzv[:, 0], xyzv[:, 1], xyzv[:, 2],
                                   xyzv[:, 3], epsilon=par.epsrbf,
                                   function=rbf)

        values = rbfi(xyzi[:, 0], xyzi[:, 1], xyzi[:, 2])
        if par.force0 == 1:
            values = np.where(values < 0., 0., values)
        for i in range(len(values)):
            f.write(f'{fidi[i]}\t{dateStr}\t{values[i]:0.2f}\n')
        datei = datei + time_step
    _ = input(MSG_FIN_PROCESO)


def addmonth_lastday(fecha):
    """
    Añade un mesa la fecha y pone al día el último del mes
    ej.: date(1957,1,1:31) -> datedate(1957,2,28)
    input:
        fecha. date type
    output:
        date type
    """
    from datetime import date
    from calendar import monthrange
    year = fecha.year
    month = fecha.month
    if month < 12:
        month += 1
    else:
        year += 1
        month = 1
    day = monthrange(year, month)[1]
    return date(year, month, day)
