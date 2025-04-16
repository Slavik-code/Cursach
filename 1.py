from openpiv import tools, pyprocess, validation, filters, scaling

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import imageio
import importlib.resources
import pathlib

path = importlib.resources.files('openpiv')
frame_a  = tools.imread( 'frame_1.png' )
frame_b  = tools.imread( 'frame_2.png' )

window_size = 32
overlap=17
search_area_size= 38
dt = 0.04
#----------------------------------------------------------------------crosscorrelate-----------------------------------
u0, v0, sig2noise = pyprocess.extended_search_area_piv(
    frame_a.astype(np.int32),
    frame_b.astype(np.int32),
    window_size=window_size,
    overlap=overlap,
    dt=dt,
    search_area_size=search_area_size,
    sig2noise_method='peak2peak',
)
# centers vectors
x, y = pyprocess.get_coordinates(
    image_size=frame_a.shape,
    search_area_size=search_area_size,
    overlap=overlap,
)

plt.hist(sig2noise.flatten())
plt.show()
#------------------------------------------------postworking------------------------------------------------------------
# mask
invalid_mask = validation.sig2noise_val(
    sig2noise,
    threshold = 0,
)

# delete very max or very min
u2, v2 = filters.replace_outliers(
    u0, v0,
    invalid_mask,
    method='localmean',
    max_iter=3,
    kernel_size=3,
)

# convert x,y to mm
# convert u,v to mm/sec
# x, y, u3, v3 = scaling.uniform(
#     x, y, u2, v2,
#     scaling_factor = 96.52,  # 96.52 pixels/millimeter
# )
# 0,0 shall be bottom left, positive rotation rate is counterclockwise
x, y, u3, v3 = tools.transform_coordinates(x, y, u2, v2)



# ---------------------------------------------------------save--------------------------------------------------------
# print('X:', x)
# print('Y:', y)
# print('u:', u3)
# print('v:', v3)
tools.save('exp1_001.txt' , x, y, u3, v3, mask= invalid_mask)

fig, ax = plt.subplots(figsize=(8,8))
tools.display_vector_field(
    pathlib.Path('exp1_001.txt'),
    ax=ax, scaling_factor=96.52,
    scale=50, # scale defines here the arrow length
    width=0.0035, # width is the thickness of the arrow
    on_img=True, # overlay on the image
    image_name= str(path / 'data'/'test1'/'exp1_001_a.bmp'),
)
plt.show()
