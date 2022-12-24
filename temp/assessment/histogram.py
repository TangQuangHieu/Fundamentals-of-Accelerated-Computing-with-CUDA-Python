# Add your solution here
@cuda.jit
def cuda_histogram(x, xmin, xmax, histogram_out):
    '''Increment bin counts in histogram_out, given histogram range [xmin, xmax).'''
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins
    # This is a very slow way to do this with NumPy, but looks similar to what you will do on the GPU
    start = cuda.grid(1)
    # This calculation gives the total number of threads in the entire grid
    stride = cuda.gridsize(1)   # 1 = one dimensional thread grid, returns a single value.
                                # This Numba-provided convenience function is equivalent to
                                # `cuda.blockDim.x * cuda.gridDim.x`

    # This thread will start work at the data element index equal to that of its own
    # unique index in the grid, and then, will stride the number of threads in the grid each
    # iteration so long as it has not stepped out of the data's bounds. In this way, each
    # thread may work on more than one data element, and together, all threads will work on
    # every data element.
    for i in range(start, x.shape[0], stride):
        # Assuming x and y inputs are same length
        #out[i] = x[i] + y[i]
        bin_number = np.int32((x[i] - xmin)/bin_width)
        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            # only increment if in range
            #histogram_out[bin_number] += 1
            cuda.atomic.add(histogram_out, bin_number, 1)  # Safely add 1 to offset bin_number i