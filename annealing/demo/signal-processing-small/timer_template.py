
# Start timer
import time
t1 = time.time()

# Stop timer & print
t2 = time.time()
timing = t2 - t1
print('\nTime to process %d cubes: %.6f [sec]' %(totalcubes, timing))
print('Throughput: %.6f [cubes/sec]' %(totalcubes / timing))
