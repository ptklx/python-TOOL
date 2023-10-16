import os
import re
import numpy as np



path = r"D:\file\docker_ubuntu\tflite_learn\android_cart_compile_pt\third_part\tflite\lib"


files = os.listdir(path)

for fi in files:

    root_p ="${tflite_DIR}/lib/"+fi
    print(root_p)