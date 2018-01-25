from ctypes import LittleEndianStructure, c_uint32, c_uint8

class VDIF_Header(LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        # word 0
        ("seconds", c_uint32, 30),
        ("legacy", c_uint32, 1),
        ("invalid", c_uint32, 1),

        # word 1
        ("frame_number", c_uint32, 24),
        ("epoch", c_uint32, 6),
        ("unassigned", c_uint32, 2),

        # word 2
        ("frame_length_8bytes", c_uint32, 24),
        ("log2_channels", c_uint32, 5),
        ("version", c_uint32, 3),
        
        # word 3
        ("station_id", c_uint32, 16),
        ("thread_id", c_uint32, 10),
        ("bits_per_sample_minus_1", c_uint32, 5),
        ("complex", c_uint32, 1),

        # extended header
        ("word4", c_uint32),
        ("word5", c_uint32),
        ("word6", c_uint32),
        ("word7", c_uint32)
    ]

if __name__ == "__main__":
    h = VDIF_Header(seconds=10,
                    legacy=0,
                    invalid=0,
                    frame_number=42,
                    epoch=5,
                    unassigned=0,
                    version=0,
                    thread_id=0,
                    complex=0,
                    frame_length_8bytes=8032/8,
                    bits_per_sample_minus_1=1,
                    log2_channels=4,
                    word4=0,
                    word5=0,
                    word6=0,
                    word7=0)
    Data = c_uint8 * 8000
    with open('test.vdif', 'wb') as f:
        f.write(h)
        f.write(Data())
        
    
