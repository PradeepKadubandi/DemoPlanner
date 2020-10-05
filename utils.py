from io import StringIO
import glob
import torch
import numpy as np

def writeline(builder, line, out_to_console=False):
    builder.write(line)
    builder.write('\n')
    if out_to_console:
        print (line)

def enumerate_files(rootdir='runs', extension='.tar'):
    for filename in sorted(glob.iglob(rootdir + '/**/*' + extension, recursive=True)):
        yield filename

def save_array_to_file(tensor_or_array, rel_or_abs_file_path):
    arr = tensor_or_array
    if isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.cpu().numpy()
    with open(rel_or_abs_file_path, 'w') as f:
        np.savetxt(f, arr, fmt='%5.5f', delimiter=',')
