from datetime import datetime, timedelta
import numpy as np

# Cell distance [m] from the sensor head, looking up.
# These values are specific to the Reefline experiments;
# They may need to be modified for other applications.
cell_distance = ([
  0.038,
  0.045,
  0.052,
  0.059,
  0.066,
  0.073,                        
  0.080,
  0.087,
  0.094,
  0.101,
  0.108,
  0.115,
  0.122,                        
  0.129,
  0.136,
  0.143,                        
  0.150,
  0.157,
  0.164
])

def read_aquadopp_time(path: str):
    data = [line.strip() for line in open(path).readlines()]
    num_records = len(data)
    time = []
    for line in data:
        line = line.split()
        month, day, year, hour, minute = [int(x) for x in line[:5]]
        seconds = float(line[5])
        ms = int((seconds - int(seconds)) * 1e6)
        time.append(datetime(year, month, day, hour, minute, int(seconds), ms))
    return np.array(time)

def read_aquadopp_velocity(path: str):
    data = [line.strip() for line in open(path).readlines()]
    num_records = len(data)
    num_cells = len(data[0].split()) - 2
    vel = np.zeros((num_records, num_cells))
    for n in range(num_records):
        vel[n,:] = np.array([float(x) for x in data[n].split()[2:]])
    return vel
