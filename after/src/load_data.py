#--------------------------------------------------------------------#
# Project: MNIST Digit Recognizer
# Created by: ArjanCodes - https://youtu.be/ka70COItN40
# Changed by: Thiago Piovesan
# Objective: Learning about code refactoring nad code optimization
#--------------------------------------------------------------------#
# Github repo: https://github.com/ThiagoPiovesan/Refactoring-DS-Project_MNIST
# Github profile: https://github.com/ThiagoPiovesan 
#--------------------------------------------------------------------#
# This is to avoid the dict and tuple type hints erros
from __future__ import annotations  

# Libs importation:
import gzip
import struct
from pathlib import Path
import numpy as np

# Let's change how data is loaded:
# The main function has some hparameters that change the way the code,
# worsk, but also has some hidden parameters that are not exposed to the user.
# So, we gonna change it.

#====================================================================#

ALLOWED_TYPES = {
    "UNSIGNED_BYTE": b"\x08",
    "SIGNED_BYTE": b"\x09",
    "SHORT": b"\x0B",
    "INT": b"\x0C",
    "SINGLE": b"\x0D",
    "DOUBLE": b"\x0E",
}

#====================================================================#

def load_image_data(file_path: Path) -> np.ndarray:
    
    with gzip.open(file_path, "rb") as fp:
        _ = struct.unpack(">H", fp.read(2))         # dump padding bytes

        (data_type,) = struct.unpack(">c", fp.read(1))
        assert data_type == ALLOWED_TYPES["UNSIGNED_BYTE"]

        number_of_dimensions = ord(struct.unpack(">c", fp.read(1))[0])
        assert number_of_dimensions == 3

        (num_images,) = struct.unpack(">I", fp.read(4))
        (num_rows,) = struct.unpack(">I", fp.read(4))
        (num_cols,) = struct.unpack(">I", fp.read(4))

        raw = fp.read()
        assert len(raw) == num_images * num_rows * num_cols

    data: np.ndarray = np.frombuffer(raw, dtype=np.dtype(np.uint8).newbyteorder(">"))
    data = data.reshape((num_images, num_rows, num_cols))
    return data
#====================================================================#

def load_label_data(file_path: Path) -> np.ndarray:

    with gzip.open(file_path, "rb") as fp:
        _ = struct.unpack(">H", fp.read(2))  # dump padding bytes

        (data_type,) = struct.unpack(">c", fp.read(1))
        assert data_type == ALLOWED_TYPES["UNSIGNED_BYTE"]

        number_of_dimensions = ord(struct.unpack(">c", fp.read(1))[0])
        assert number_of_dimensions == 1

        (num_images,) = struct.unpack(">I", fp.read(4))
        
        raw = fp.read()
        assert len(raw) == num_images

    data = np.frombuffer(raw, dtype=np.dtype(np.uint8).newbyteorder(">"))
    return data
#====================================================================#
