'''
Listing 4.2: Checking for the double data type
Note: Does not provide fallback code if double is unavailable
This can easily be done by using the book kernel and passing -DFP_64 in options
'''

import pyopencl as cl
import numpy as np
import utility

# fp64 was made an optional feature instead of an optional extension in OpenCL 1.2
# This means that it (should) be enabled by default
# No harm is done by enabling it explicitly, however
kernel_src = '''
#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void double_test(__global float* a,
                          __global float* b,
                          __global double* out) {
   *out = (double)(*a / *b);
}
'''

# Get device and context, create command queue and program
platforms = cl.get_platforms()

context = cl.Context(
        dev_type=cl.device_type.ALL,
        properties=[(cl.context_properties.PLATFORM, platforms[0])])

queue = cl.CommandQueue(context, dev, properties=None)


# Check for double floating point support
dev_extensions = dev.extensions.strip().split(' ')
if 'cl_khr_fp64' not in dev_extensions:
    raise RuntimeError('Device does not support double precision float')

# Build program in the specified context using the kernel source code
prog = cl.Program(context, kernel_src)
try:
    prog.build(options=['-Werror'], devices=[dev], cache_dir=None)
except:
    print('Build log:')
    print(prog.get_build_info(dev, cl.program_build_info.LOG))
    raise

a = np.float32(6.0)
b = np.float32(2.0)
# Could not get it to work using a scalar as output
out = np.zeros(shape=(1,), dtype=np.float64)

# Create buffers
flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
buffer_a = cl.Buffer(context, flags, hostbuf=a)
buffer_b = cl.Buffer(context, flags, hostbuf=b)
buffer_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=out.itemsize)

# Set buffers as arguments to the kernel
# The arguments can also be specified by calling kernel(....) directly instead
kernel = prog.double_test
kernel.set_arg(0, buffer_a)
kernel.set_arg(1, buffer_b)
kernel.set_arg(2, buffer_out)

# Enqueue kernel (with arguments)
n_globals = (1,)
n_locals = None
cl.enqueue_nd_range_kernel(queue, kernel, n_globals, n_locals)

# Enqueue command to copy from buffer_out to host memory
cl.enqueue_copy(queue, dest=out, src=buffer_out, is_blocking=True)

print('Output: ' + str(out[0]))