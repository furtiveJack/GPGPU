# TP3

TRAINA Tanguy
MARSZAL Rémi
ESIPE - INFO3

## Exercice 1

- chaos1.cu :

    We try to compute the sum of integers from 0 to n-1.
    All threads try to add their tid to a variable res.
    But this is not an atomic operation, so we get a wrong result at the end.
    To prevent this, we should use the atomicAddd function.

- chaos2.cu : 

    All threads use a share memory. This cache is initialized by all threads, using their tid.
    But to ensure that the cache has been well initialized, we have to wait for all the threads
    to finish their initialization job.
    So we must use the __syncthreads() function in order to wait for all threads to finish initializing the cache.
    Then we can start the computation. 


## Exercice 2

- search1.cu :

    Each threads whose tid is less than the array size check if arr[tid] contains the value searched,
    and initialize a variable herecount with 1 if so, 0 otherwise.
    Then, we shuffle down all threads of the warp to reduce them to one value, adding the herecount values 
    transferred by each shuffled thread. At the end of this operation, the first thread of the warp
    contains the number of time val has been found in this portion of the array.
    Then threads go one to the nex portion of the array, incremented by gridsize, and repeat
    the same operation until they have iterated through all of the array.
    At the end, res contains the exact number of time  val is present in the array.

- search2.cu :

    Each thread checks if the value in arr[tid] is the searched one.
    If one of the threads found the value, then the first thread of the warp adds one to res.
    Then all threads continue to the next portion of the array, incrementing their tids by gridsize.
    The value of res at the end is the exact amount of time val is present in the array.

- search3.cu : 

    This method searches for the value 'val' in the array 'arr'. 
    The value returned in res is the number of elements equal to 'val' in 'arr'.
    To do this, all threads loops through the array, checking if the value in array at 
    their current tid is equal to 'val'. If so, 'res' is incremented (atomatically).

- search4.cu :

    All threads check if the value in arr[tid] is the searched one. If so, res is incremented.
    Otherwise, threads will iterate through the array, on portions of gridzize.
    All threads that enter the while loop after the first incremention of 'res' will automatically
    return without computing (because when loading the value of res, it will already be != 0). 
    Threads that were computing will stop as soon as they reload the 'res' value from memory 
    (so probably at their next iteration, but maybe later).
    The value in res at the end is the number of threads that found val (so it may not be the
    exact amount of time val appears in the array, but it means that there is at least one occurence).

## Problem

Execution output :

    Finished generating graph

    GPU version, runtime  0.7199s
    Deviation: 0.892290
    Rank 0, index 683, normalized pagerank 0.0033809
    Rank 1, index 97, normalized pagerank 0.0030121
    Rank 2, index 484, normalized pagerank 0.0029379
    Rank 3, index 236, normalized pagerank 0.0028048
    Rank 4, index 547, normalized pagerank 0.0028004
    Rank 5, index 891, normalized pagerank 0.0026470
    Rank 6, index 422, normalized pagerank 0.0024067
    Rank 7, index 167, normalized pagerank 0.0022166
    Rank 8, index 223, normalized pagerank 0.0021501
    Rank 9, index 939, normalized pagerank 0.0021087
    Rank 10, index 410, normalized pagerank 0.0020817
    Rank 11, index 7, normalized pagerank 0.0020729
    Rank 12, index 570, normalized pagerank 0.0020640
    Rank 13, index 49, normalized pagerank 0.0020566
    Rank 14, index 70, normalized pagerank 0.0020484
    Rank 15, index 278, normalized pagerank 0.0020215

    CPU version, runtime 24.5790s
    Deviation: 0.892290
    Rank 0, index 683, normalized pagerank 0.0033809
    Rank 1, index 97, normalized pagerank 0.0030121
    Rank 2, index 484, normalized pagerank 0.0029379
    Rank 3, index 236, normalized pagerank 0.0028048
    Rank 4, index 547, normalized pagerank 0.0028004
    Rank 5, index 891, normalized pagerank 0.0026470
    Rank 6, index 422, normalized pagerank 0.0024067
    Rank 7, index 167, normalized pagerank 0.0022166
    Rank 8, index 223, normalized pagerank 0.0021501
    Rank 9, index 939, normalized pagerank 0.0021087
    Rank 10, index 410, normalized pagerank 0.0020817
    Rank 11, index 7, normalized pagerank 0.0020729
    Rank 12, index 570, normalized pagerank 0.0020640
    Rank 13, index 49, normalized pagerank 0.0020566
    Rank 14, index 70, normalized pagerank 0.0020484
    Rank 15, index 278, normalized pagerank 0.0020215


### Question 6

The results are identical on both side (GPU and CPU). The time spent on GPU computation is 
(of course) really smaller than on the CPU side.


From the example output provided on elearning, it looks like the results should differ.

But the pagerank algorithm seems to be deterministic (ie. for the same inputs, different executions should produce the same outputs). 

If pagerank really is deterministic, then why the example output provided doesn't show that ? If it is not, then there is probably something wrong in the code, but we could not find what.