import time
from fid import calculate_fid
from cdc import calculate_cdc


res_dir = '/path/to/your/own/results'


t1 = time.time()

print(f'Calculating FID...')
fid = calculate_fid(res_dir)
t2 = time.time()

print('Calculating CDC...')
cdc = calculate_cdc(res_dir)
t3 = time.time()

print('FID evaluation time:', t2-t1)
print('CDC evaluation time:', t3-t2)
print('Total evaluation time:', t3-t1)

print(f'FID: {fid}, CDC: {cdc}')
