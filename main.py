# -*- coding: utf-8 -*-
"""
Created on 06/09/2019

@author: Luis Solís

driver módulo interpol
"""


if __name__ == "__main__":

    try:
        from time import time
        import traceback
        import littleLogging as logging
        import interpol

        startTime = time()

#        interpol.menu()
        interpol.shapeRBF()
#        interpol.distancias_puntos()

        xtime = time() - startTime
        print(f'El script tardó {xtime:0.1f} s')

    except ValueError:
        msg = traceback.format_exc()
        logging.append(f'ValueError exception\n{msg}')
    except ImportError:
        msg = traceback.format_exc()
        print (f'ImportError exception\n{msg}')
    except Exception:
        msg = traceback.format_exc()
        logging.append(f'Exception\n{msg}')
    finally:
        logging.dump()
#        _ = input('Pulse una tecla para finalizar')

