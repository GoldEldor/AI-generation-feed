# native.py
import ctypes
from pathlib import Path

_LIB_PATH = Path(__file__).resolve().parent / "native" / "liblivevision_native.so"
_lib = ctypes.CDLL(str(_LIB_PATH))
_lib.lv_strip_metadata.argtypes = [
    ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
]
_lib.lv_strip_metadata.restype = ctypes.c_longlong


def strip_metadata(data: bytes) -> bytes:
    out = ctypes.create_string_buffer(len(data))
    written = _lib.lv_strip_metadata(data, len(data), out, len(data))
    if written < 0:
        raise ValueError("Could not parse image for metadata removal.")
    return out.raw[:written]
  
