import numpy as np
from tlpipe.pipeline.pipeline import Manager

pipefile = './dish_sim_1000rec.pipe'
P = Manager(pipefile)
# print P.params
# print P.task_params

N = 1000
for i in range(N):
    P.task_params['mm_input_maps'] = [ './input_ps_maps/rotate_pointsource_512_750MHz_%04d.hdf5' % i ]
    P.task_params['mm_ts_name'] = 'ts_%04d' % i
    # print P.task_params
    P.run()