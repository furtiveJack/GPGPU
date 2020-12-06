# TP2

TRAINA Tanguy

ESIPE - INFO3 - Opt Logiciel

## starve1.exe 

Execution results : 

    First invocation
    144
    1
    135
    0
    370.3276ms

    Second invocation
    0
    1
    135
    144
    5.0852ms

In the first invocation, the kernel is started with each block having
 only one thread. 
This means that there is not enough active warps, and so we got
a low occupancy problem, which causes the first invocation to be much 
slower than the second.


## starve2.exe 	

Execution results :

    Copying data to device
    First method
    305.3349ms

    Second method
    166.2955ms

In the first method, there is a warp divergence caused by the 
if (tid % 2 == 0) condition : in each warp, a half of the threads will
take one branch, is the other half will take the other, which make some
threads to remain idle.

Whereas in the second method, the condition is if (tid * 2 < size), which means 
that threads within a warp will take the same branch (because if tid*2 is lower than size 
for one thread of a warp, it is almost sure that all other threads of the warp will respect
the same condition). So no warp divergence in this case, and thus, a greater performance.


## starve3.exe 

Execution results :

    Copying data to device
    First method
    39.9954ms

    Second method
    0.0020ms


In the second method, we use a cache stored in shared memory. Each thread will
read a portion of the matrix and initialize the cache with the matrix values.
Once all thread have done their initialization job, they can compute the square sum,
using the values of the cache instead of reading in the matrix.

The first method does the same job, but without using a cache in shared memory.
Since shared memory is much faster, the computation has a really huge improve
in performance.


## starve4.exe 

Execution results :

    Copying data to device
    First method
    14.2828ms

    Second method
    11.0695ms

In the first method, threads read data in the matrix, but not at closest locations.
Whereas in the second method, the memory read are done in a coalescing way, which is 
faster.  


## Problem

1. We choose to store the convolutive filter in constant memory because 
threads will only need to read into it and it will be identical for all threads.

