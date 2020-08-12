from io import StringIO
import glob

def writeline(builder, line, out_to_console=False):
    builder.write(line)
    builder.write('\n')
    if out_to_console:
        print (line)

def enumerate_files(rootdir='runs', extension='tar'):
    for filename in sorted(glob.iglob(rootdir + '/**/*.' + extension, recursive=True)):
        yield filename
