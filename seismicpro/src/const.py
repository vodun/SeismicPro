"""Package-level constants"""

EPS = 1e-10

# Size of trace headers in a SEG-Y file in bytes
TRACE_HEADER_SIZE = 240


# Valid aliases for file endianness and corresponding format string prefixes to be passed to struct.unpack
ENDIANNESS = {
    "big": ">",
    "msb": ">",
    "little": "<",
    "lsb": "<",
}


# Custom headers optionally added by SeismicPro to those from SEG-Y file
HDR_DEAD_TRACE = 'DeadTrace'
HDR_FIRST_BREAK = 'FirstBreak'


