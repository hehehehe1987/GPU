/*利用GPU对非常多的数据进行排序*/


//所有进行基数排序的数据都需要转化为无符号的32位整型。以方便之后的位计算。
typedef unsigned int ubit32;

#define BLOCKSIZE = 512


/*
GPU的基数排序分成两个步骤，
1、线程内的基数排序：每个线程负责自己部分的数据，然后进行基数排序，线程负责部分内部数据成为从小到大的有序序列。
2、线程间有序序列归并：将每个线程负责的已经有序的序列按照有序序列归并的方式，block中所有线程每次只都找到一个最小值，然后从小到大插入到结果序列中。
*/
__device__ void radix_sort(
	                          ubit32 *data,
	                          ubit32 *tmp,
	                          unsigned int sizeperthrd, 
	                          unsigned int blockdimx,
	                          unsigned int len, 
	                          unsigned int tid
	                      )
{
	//基数排序方法：
	//1、每次只取出序列中候选待比较数的当前位。
	//2、将此位为0的值放入表示序列前半部分的序列(bucket0)中。
	//3、将此位为1的值放入后半部分的序列中。
	//4、序列前半部分与后半部分合并，形成新的序列。
	//5、当前位比较完成，比较更高的位。
    for(unsigned int bit = 0; bit < 32; bit++)
    {
    	//bit_mask为取位掩模，它只会有在某一位上为1，那么任何二进数与之相与操作。只会将原数某一位上是0或1给提取出来。
    	//掩模提取候选数字相应位的作用是根据其是0还是1来判断此数当前循环下放入序列的前半部分还是后半部分。
    	ubit32 bit_mask = 1 << bit;
    	unsigned int cnt0 = 0;
    	unsigned int cnt1 = 0;
    	//申请共享内存空间，反复读写共享内存的速度比反复读取全局内存速度要快很多。
    	//这里考虑到片内共享内存空间的限制，每个block处理的数据不能太多，即len不能太大。
    	//假设一个block包含512个线程，每个线程处理sizeperthrd个数，那么len必须小于等于512*sizeperthrd。
    	__shared__ unsigned int bucket0[len];  //利用共享内存作为block内各线程都能访问的前半个序列。
    	__shared__ unsigned int bucket1[len];  //同样利用共享内存实现后半部分。
    	//每个线程负责一部分的数据，这里线程不负责相邻数据，原因是会造成线程非合并访问：
    	//      即当线程1需要取得指针0位置数据时，线程2需要取指针32位置的数据，那线程2只能等线程1取完后，才能取指针32处的数。造成速度过慢。
    	//      所以，线程1取0位置、32位置、64位置....线程2取1位置、33位置、65位置...线程3........
    	//      这样的取数方式会使得全局内存访问的效率大大提升。
    	for(unsigned int i = 0; (i < len) && (i + tid < len); i += sizeperthrd)
    	{
    		// 该循环每个线程只负责将自己负责的线程中的数进行基数排序。
    		// sizeperthrd如果为32，则表示，tid=0的线程每次循环只会去拿一个数，每次取得的数在地址上相差32，
    		// i表示第i次循环。
   
    		//将本线程待排序的候选值取出。   
    		ubit32 value = data[i+tid]; 
    		//候选值与掩模相与，只有保留value某一位的值（是0或者1），而这个值如果是0，将会被放入新序列的前半部分。
    		//否则将被放入到后半部分。
    		if((value&bit_mask)>0)
    		{
    			//每个线程都会把自己的候选值放入至相隔sizeperthrd的空间中。tid线程号与累加sizeperthrd的计数器cnt1负责指示结果存放位置。
    			bucket1[cnt1+tid] = value;
    			cnt1 += sizeperthrd; 
    		}
    		else
    		{
    			bucket0[cnt0+tid] = value;
    			cnt0 += sizeperthrd;
    		}
    	}
    	for(unsigned int i=0; i<cnt1; i+=sizeperthrd)
    	{
    		bucket0[cnt0 + i + tid] = bucket1[i+tid];
    	}
    }
    __syncthreads();
    for(unsigned int i=0;  (i < len) && (i + tid < len); i+=sizeperthrd)
    {
    	//将结果放入中间缓存的tmp中，供之后的并行归并操作。
    	tmp[i+tid] = bucket0[i+tid];
    }
}

__device__ void mergeparallel(
	                             //*data为并行归并的输出空间，并行归并操作最终将结果输出至*data所指空间中
                                 ubit32 *data,
                                 //*tmp为并行归并的输入空间，也是整个基数排序的中间缓存空间。
                                 ubit32 *tmp,
                                 //len为为当前block下处理的数据的大小。
                                 unsigned int len,
                                 //tid为当前的线程编号。
                                 unsigned int tid,
                                 //threadnum为一个线程需要负责排序的数据个数。
                                 unsigned int threadnum
	                         )
{
	// minValue用来存放当前block中所有线程中最小值。
	__shared__ ubit32 minValue;
	// mintid用来存放当前block中取得最小值的那个线程编号。
	__shared__ unsigned int mintid;
	// tidHeader用来存放thread负责候选数据的头指针偏移量，
	// 该偏移量也表示当前线程负责的数据部分中，之前已经有多少个值被上报为最小值，并被取走了。
	// 那么下次该thread就从该头指针开始取候选值，与其他线程进行最小值比较。
	__shared__ unsigned int tidHeader[threadnum];
	unsigned int value = 0;
	tidHeader[tid] = 0;
    __syncthreads();
    //循环i次，每次都会找到block内所有线程负责的数据中，最小的那个，而block内有len个数，那么一共要循环len次。
    for(unsigned int i=0; i<len; i++)
    {
    	//idx表示当前线程需要去取的候选值、其指针偏移的大小。候选值将与其他线程候选值共同筛选出最小值。
    	//每个线程负责的数据之间的间隔是threadnum，而不同线程所取的数应该是相邻的。
    	//如果当前线程已经上报了tidHeader[tid]-1个数，那么就应该去取该线程的第tidHeader[tid]个数，
    	//加之线程负责的两数之间相距threadnum个空间，则第tid个线程应该取第tidHeader[tid] * threadnum + tid个数。
    	unsigned int idx = tidHeader[tid] * threadnum + tid;
    	if(idx < len) 
    	{
    		value = tmp[idx]; //如果指针偏移量小于数据大小，表示没有下标越界，则候选值value从相应空间中取得。
    	}                     //否则在else分支中被设置为无穷大。
    	else
    	{
    		value = inf;
    	} 
        //最小值存放空间初始化。这里只有tid为1的线程初始化，其它线程阻塞至__syncthreads();
    	if(tid == 0)
    	{
    		minValue = inf;
    		mintid = inf;
    	}
    	__syncthreads();
    	//将所有线程的候选值进行比较，选出候选值。
    	//这里atomicMin为原子操作，会将所有线程的值一一串行的与minValue进行比较，留下最小值。
    	//由于minValue为共享内存，原子操作速度会非常快，反之如果是对全局内存利用原子操作，将会导致速度性能下降。
    	atomicMin(&minValue， value);
    	__syncthreads();
    	//判断最终的最小值是否为本线程提供的候选值。
    	if(minValue == value)
    	{
    		//如果是，且有多个线程共同提供了最小值，那么选择线程编号最小的那个线程编号作为结果，并上报存储。
    		atomicMin(&mintid, tid);
    	}
    	__syncthreads();
    	//判断是否当前线程就是那个被上报的线程，
    	if(min_tid == tid)
    	{
    		//如果是，代表该线程负责的数据部分有一个值成功归并，那么将指针偏移后移一个。
    		tidHeader[tid]++;
    		//同时把候选值插入输出的第i个偏移位置。
    		data[i] = value;
    	}
    }

}

__global__ void sortkernel(ubit32* dp_num, ubit32* dp_tmp, unsigned int len)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sizeperthrd = (len + blockDim.x - 1) / blockDim;
    radix_sort(dp_data, dp_tmp, sizeperthrd, blockDim.x, len, tid);
    mergeparallel(dp_data, dp_tmp, len, tid, blockDim.x);
}


// 主入口函数。
// hp_nums为输入数组的头指针
// len为数组的长度
void GPUSort(
                ubit32* hp_nums, 
                unsigned int len
            )
{

    ubit32* dp_nums = NULL;
    ubit32* dp_tmp = NULL;
    unsigned int gridsize = (size+block.x-1) / block.x;
    dim3 block(BLOCKSIZE,1);
    dim2 grid(gridsize,1);
    cudaMalloc(dp_nums, len*sizeof(ubit32));
    cudaMalloc(dp_tmp, len*sizeof(ubit32));
    //采用stream的方式，实现多流并发，重叠不同进程的显存拷贝与核函数运行的时间
    cudaMemcpyAsync(dp_nums, hp_nums, len*sizeof(ubit32), cudaMemcpyHostToDevice, stream);
    //基数排序
    sortkernel<<<grid,block,0,stream>>>(dp_nums,dp_tmp,add);
    cudaMemcpyAsync(hp_nums, dp_nums, len*sizeof(ubit32), cudaMemcpyDeviceToHost, stream);
    cudaDeviceSychronize();
    cudaFree(dp_nums);
    cudaFree(dp_tmp);
    //以下步骤进行如下操作：
    //1、基数排序的结果只是将每个block排序为了有序序列，但是block之间仍需要进行有序序列归并。
    //2、为了进行归并，采用一个偏移计数数组，分别记录每个block数据已经完成了归并的偏移量。
    //3、归并需要O(N)时间复杂度和O(N)的空间复杂度。
    vector<unsigned int> blkidHeader(gridsize,0);
    vector<unsigned int> temp;
    for(int i=0; i < len; i++)
    {
        ubit32 minval = MAX_UINT;
        for(int j = 0; j < gridsize; j++)
        {
            if(minval > hp_nums[j * gridsize + blkidHeader[j]])
            {
                minval = hp_nums[j * gridsize + blkidHeader[j]];
                minid  = j;
            }
        }
        blkidHeader[j]++;
        temp[i] = minval;
    }
    for(int i=0; i < len; i++)
    {
        hp_nums[i] = temp[i];
    }
}