from io import StringIO
import glob

def writeline(builder, line, debug=False):
    builder.write(line)
    builder.write('\n')
    if debug:
        print (line)

def enumerate_files(rootdir='runs', extension='tar'):
    for filename in sorted(glob.iglob(rootdir + '/**/*.' + extension, recursive=True)):
        yield filename
