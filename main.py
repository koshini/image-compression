# Image compression
#
# You'll need Python 2.7 and must install these packages:
#
#   scipy, numpy
#
# You can run this *only* on PNM images, which the netpbm library is used for.
#
# You can also display a PNM image using the netpbm library as, for example:
#
#   python netpbm.py images/cortex.pnm


import sys, os, math, time, netpbm
import numpy as np
import struct

headerText = 'my compressed image - v1.0'

# Compress an image


def compress( inputFile, outputFile ):

  # Read the input file into a numpy array of 8-bit values
  #
  # The img.shape is a 3-type with rows,columns,channels, where
  # channels is the number of component in each pixel.  The img.dtype
  # is 'uint8', meaning that each component is an 8-bit unsigned
  # integer.

  img = netpbm.imread( inputFile ).astype('uint8')
  
  # Compress the image
  #
  # REPLACE THIS WITH YOUR OWN CODE TO FILL THE 'outputBytes' ARRAY.
  #
  # Note that single-channel images will have a 'shape' with only two
  # components: the y dimensions and the x dimension.  So you will
  # have to detect this and set the number of channels accordingly.
  # Furthermore, single-channel images must be indexed as img[y,x]
  # instead of img[y,x,1].  You'll need two pieces of similar code:
  # one piece for the single-channel case and one piece for the
  # multi-channel case.

  startTime = time.time()
 
  # outputBytes = []
  outputBytes = bytearray()

  diff = []

  # it's a single-channel image
  if len(img.shape) == 2:
    for y in range(img.shape[0]):
      for x in range(img.shape[1]):
        # Predictive encoding
          prediction = int(img[y - 1, x]) + (int(img[y, x - 1]) - int(img[y - 1, x - 1]))/2
          diff.append(int(img[y, x]) - int(prediction))

  else: # it's a multi-channel image

    for y in range(img.shape[0]):
      for x in range(img.shape[1]):
        for c in range(img.shape[2]):
          # Predictive encoding
          prediction = int(img[y - 1, x, c]) + (int(img[y, x - 1, c]) - int(img[y - 1, x - 1, c]))/2
          diff.append(int(img[y, x, c]) - int(prediction))

  # print(diff)
  # construct initial dictionary
  dict_size = 256
  dictionary = {struct.pack("h", i) : i+256 for i in xrange(-dict_size, dict_size)}
  
  # LZW encode the diff array
  s = ''
  for i in range(len(diff)):
 
    x = struct.pack('h', diff[i])
    temp = s + x
    if temp in dictionary:
      s = temp
    else:
      s_in = dictionary[s]
      s1= s_in >> 8
      s2= s_in & 255
      outputBytes.append(s1)
      outputBytes.append(s2)
      if len(dictionary) < 65536:
        dictionary[temp] = dict_size
        dict_size += 1
      s = x

  # encode the last s
  s_in = dictionary[s]
  s1= s_in >> 8
  s2= s_in & 255
  outputBytes.append(s1)
  outputBytes.append(s2)
  # print(dictionary)

  endTime = time.time()

  # Output the bytes
  #
  # Include the 'headerText' to identify the type of file.  Include
  # the rows, columns, channels so that the image shape can be
  # reconstructed.

  outputFile.write('%s\n' % headerText)
  outputFile.write( '%d %d %d\n' % (img.shape[0], img.shape[1], img.shape[2]) )
  outputFile.write( str(outputBytes) )

  # Print information about the compression
  
  inSize  = img.shape[0] * img.shape[1] * img.shape[2]
  outSize = len(outputBytes)

  sys.stderr.write( 'Input size:         %d bytes\n' % inSize )
  sys.stderr.write( 'Output size:        %d bytes\n' % outSize )
  sys.stderr.write( 'Compression factor: %.2f\n' % (inSize/float(outSize)) )
  sys.stderr.write( 'Compression time:   %.2f seconds\n' % (endTime - startTime) )
  



# Uncompress an image


def uncompress( inputFile, outputFile ):

  # Check that it's a known file

  if inputFile.readline() != headerText + '\n':
    sys.stderr.write( "Input is not in the '%s' format.\n" % headerText )
    sys.exit(1)
    
  # Read the rows, columns, and channels.  
  rows, columns, channels = [ int(x) for x in inputFile.readline().split() ]

  # Read the raw bytes.
  inputBytes = bytearray(inputFile.read())

  # Build the image
  # REPLACE THIS WITH YOUR OWN CODE TO CONVERT THE 'inputBytes' ARRAY INTO AN IMAGE IN 'img'.


  startTime = time.time()

  img = np.empty( [rows,columns,channels], dtype=np.uint8 )

  '''
  byteIter = iter(inputBytes)
  for y in range(rows):
    for x in range(columns):
      for c in range(channels):
        img[y,x,c] = byteIter.next()
  '''

  # construct initial dictionary
  dict_size = 256
  dictionary = {i+256 : struct.pack("h", i) for i in xrange(-dict_size, dict_size)}

  # due to string concatenation in a loop
  diff = []

  byte = iter(inputBytes)
  
  # LZW uncompression
  # read two bytes from the bytearray and take the sum, this gives you the original key
  a = byte.next()
  b = byte.next()
  index = a + b # an integer
  s = dictionary[index] 
  diff.append(struct.unpack("h", s)[0])
  
  for i in xrange(len(inputBytes)/2-1):
    a = byte.next()
    b = byte.next()
    index = a + b  # an integer
    if index in dictionary:
      t = dictionary[index]
    else:
      stemp = struct.unpack('h',s)[0]
      t = s + struct.pack("h", stemp)
    diff.append(struct.unpack("h", t)[0])
    if len(dictionary) < 65536:
      ttemp = struct.unpack('h',t)[0]
      dictionary[dict_size+257] = s + struct.pack("h", ttemp)
      dict_size += 1
    s = t
    
  diffs = iter(diff)
  for y in range(rows):
    for x in range(columns):
      for c in range(channels):
        if i< len(diff):
          prediction = int(img[y - 1, x, c]) + (int(img[y, x - 1, c]) - int(img[y - 1, x - 1, c]))/2
          img[y, x, c] = diffs.next() + prediction
        i += 1

  sys.stdout.write( "Inputbyte length:" + str(len(inputBytes)/2-1 ))
  sys.stdout.write( "Diff: " + str(len(diff)))


  endTime = time.time()

  # Output the image

  netpbm.imsave( outputFile, img )

  sys.stderr.write( 'Uncompression time: %.2f seconds\n' % (endTime - startTime) )

  
# The command line is 
#
#   main.py {flag} {input image filename} {output image filename}
#
# where {flag} is one of 'c' or 'u' for compress or uncompress and
# either filename can be '-' for standard input or standard output.


if len(sys.argv) < 4:
  sys.stderr.write( 'Usage: main.py c|u {input image filename} {output image filename}\n' )
  sys.exit(1)

# Get input file
 
if sys.argv[2] == '-':
  inputFile = sys.stdin
else:
  try:
    inputFile = open( sys.argv[2], 'r' )
  except:
    sys.stderr.write( "Could not open input file '%s'.\n" % sys.argv[2] )
    sys.exit(1)

# Get output file

if sys.argv[3] == '-':
  outputFile = sys.stdout
else:
  try:
    outputFile = open( sys.argv[3], 'w' )
  except:
    sys.stderr.write( "Could not open output file '%s'.\n" % sys.argv[3] )
    sys.exit(1)

# Run the algorithm

if sys.argv[1] == 'c':
  compress( inputFile, outputFile )
elif sys.argv[1] == 'u':
  uncompress( inputFile, outputFile )
else:
  sys.stderr.write( 'Usage: main.py c|u {input image filename} {output image filename}\n' )
  sys.exit(1)
