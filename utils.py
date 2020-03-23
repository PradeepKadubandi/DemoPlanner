from io import StringIO

def writeline(builder, line, debug=False):
    builder.write(line)
    builder.write('\n')
    if debug:
        print (line)
