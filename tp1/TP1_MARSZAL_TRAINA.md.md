# TP1

MARSZAL Rémi
TRAINA Tanguy
INFO3

## Exercice 2

When executing the file demo1.cu, we get a result that looks like that : 

0, 1
1, 1
2, 1
3, 1
0, 13
1, 13
2, 13
3, 13
...
0, 6
1, 6
2, 6
3, 6

What the program does is launching a kernel() function with :
- a grid dimension of 16
- a block dimension of 4

That means that the grid will be composed of 16 blocks, and each
block will contain 4 threads.

The output of the program shows, on each line, the thread index followed
by its block index.

What we can deduce from this is that threads belonging to the same block 
are executed in order (here, it is always thread 0, then 1, then 2, and finally 3), and 
that the program always execute all the threads of a block before going to the next block.

However, we can see that there is no execution order for the blocks. Here we started with 
the block indexed 1, then the bock 13, and we finished with the block 6.

So to conclude, execution order is guaranteed for threads within a block, 
but not for the blocks themselves.


## Exercice 3

There is an error in the program. We try to start a computation using a certain
amount of threads per blocks, but we don't check for the device capabilities 
regarding the amount of thread per blocks. This result in an error for the 
second computation. This error is a cudaErrorInvalidConfiguration.

For:
    range = 1024
We get :
    grid Dimension   = 32
    blocks Dimension = 32

For:
    range = 16777216
We get:
    grid Dimension   = 4096
    blocks Dimension = 4096

And the capabilities of the device :
    maxGridSize: (2147483647, 65535, 65535)
    maxThreadsDim: (1024, 1024, 64)
    maxThreadPerBlock: 1024

So, in order to make the second computation work, we need to adjust the range
to the capabilities of the device. Here, the maximum range possible is 1048576 (1024 threads/blocks).


## Problem

At the end of the execution, the fractal become pixelised.
This is cause by the value of delta, which reduce at each frame.
Since delta is a double value, at one moment, it reaches the value of 0, and 
that's when the fractal become pixelised.