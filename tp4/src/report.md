# TP * 

MARSAL Rémi
TRAINA Tanguy
ESIPE - INFO 3


# Current state of the work
We didn't succeed to implement an efficiant version of the the radix sort running on GPU.
The best results we had were sorting a 2^10 in 4.080576ms, but the CPU does a lot better.

## Attempt 1

This first attempt purpose was to see if were able to make the algorithm run in
a dumb, naive, but human-friendly way.
Indeed we succeed, with only one thread of the gpu doing all the computation. 
We got to arrays, one for reading, the other for writing. 

At each iteration (for k = 0 to 64)
    We are gonne split the array into two buckets, so first we need to know
    the offset of the second bucket.
    Using the split_kernel() function, we compute this offset.

    Then we launch sort_kernel(), which iterates through the array, and moving each 
    to the good position in the array, using the offset (keeping order).

Finally the array is sorted ! (268.604950ms for n = 16)

Just to test it, we to tried to parallelize juste a little the execution
using now two threads : one for each bucket.
This made the execution a lot much slower... (515.914978ms for n = 16)

## Attempt 2

We now want to parallelize the execution which much more thread.
To do so, we tried to implement the scan reduction method.

The idea is that, for each value k from 0 to 64 (k representing the 
current bit we are testing)) :

1. each thread save the initial value of arr[tid]
2. then for each element of the array, we replace it with the value of the bit 
number k in the initial value
3. we perform a scan reduction on the array (each value is replaced the number of value
equal to 1 before its position (itself included))
4. For each element, we now know how many 1 and 0 were before it in the original array.
This also give us the beginning offset of the second bucket.
5. Use all these informations to compute the new offset of the current thread value, and put it there.

At the end of the iteration, the array should be sorted.

### Results

This implementation is able to sort an array until 2^10 elements.
For n>11, the program crashes.
But until this maximum value of 2^10, the algorithm does the job, 
and since its parallel, it does it much faster than the first one (permorms in 4.080576ms).


## Attempt 3
Since we got a lot of memory and threads synchronization issues, we tried to 
split each step of the algorithm into a kernel function. This was made trying to
ensure that all threads has finished a particular task before going to the next one, and so keeping the memory consistent at any time. 
This didn't work.
