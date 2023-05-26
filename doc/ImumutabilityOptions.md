
## Immutability Options


When creating a dataset you can choose between the immutability options `copy` and `pickle`. If you are working with multiple processes in parallel each child process will share its entire memory space with the main process. But because of the large nested structure with many python objects accessing the dataset will create a copy of it in the RAM for the process accessing it ("copy-on-read"). Therefore if the processes access the dataset it will be loaded into the RAM multiple times. While 'copy' gives a complete copy of the dataset to each process pickle uses bytestreams and compresses the data so the memory usage is decreased compared to copy.

This was originally observed in  https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/ and the blog also contains more details about it.

For datasets created from a list you can also choose a third option 'wu' which uses numpy arrays instead of many python objects to store the dataset in the memory. By doing so the "copy-on-read" effect does not trigger and the child process can access the datset from the main process without copying the data.
Therefore by using the 'wu' option you can prevent each process to have its own copy of the dataset stored in the RAM and instead share the dataset from the main process. Because of that the dataset does not need to be loaded into the RAM multiple times and the RAM memory usage can be reduced as shown in the diagrams below.

![Copy](pickle.svg)
<img src="copy.svg" alt= “Memory Usage with copy” width="800" height="400"> 
<center> 
Immutable_warranty=copy<center> 

<img src="pickle.svg" alt= “Memory Usage with pickle" width="40%" height="40%">
<center> 
Immutable_warranty=pickle <center> 

<img src="wu.svg" alt= “Memory Usage with wu" width="40%" height="40%">
<center> 
Immutable_warranty=wu 
</center>

Where the displayed USS (Unique Set Size) is the sum of all single USSs of each worker. The USS of one worker is the amount of RAM that is unique to that process and not shared with other processes.

Shared in the diagram represents the mean RAM of the processes that is shared with other processes. 

The displayed PSS ("Proportional Set Size") is the sum of all single PSSs. The PSS of one process is all the memory the proccess holds in RAM, so the sum of USS and shared, but the shared memory is divided by the number of processes sharing it.


The pickle option delivers a good compression of the data and reduces the used memory for one worker.
For multiple processes the wu-option can be more stable since the data does not need to be loaded into the RAM multiple times and the memory usage stays roughly constant over time.











