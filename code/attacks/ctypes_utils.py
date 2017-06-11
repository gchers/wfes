# Wrappers around ctypes functions to convert
# lists of various types to/from ctypes.
import math
import ctypes

def c_int(x):
    """Wrapper around ctypes.c_int() to avoid importing both
    this module and ctypes.
    """
    return ctypes.c_int(x)

def c_list_to_list(ct_a):
    """Returns a list of floats from a ctypes array of floats.
    """
    return ct_a[:]

def list_to_c_int_list(a):
    """Converts a list of int into a
    ctype list of int.
    """
    N = len(a)
    ret = (ctypes.c_int * N)(*a)

    return ret

def list_to_c_float_list(a):
    """Converts a list of float into a
    ctype list of float.
    """
    N = len(a)
    ret = (ctypes.c_float * N)(*a)

    return ret

def list_list_to_c_float_list_list(a):
    """Converts a list of list of float into a
    ctype array of arrays of float.
    """
    rows = len(a)
    cols = len(a[0])
    PFLOAT = ctypes.POINTER(ctypes.c_float)  # Pointer to array of float

    ret = (PFLOAT * rows)()
    for i, x in enumerate(a):
        ret[i] = (ctypes.c_float * cols)(*x)

    return ret
