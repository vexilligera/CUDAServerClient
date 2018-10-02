public class Test {
	public static void main(String[] args) {
		// 1GB data
		int maxSize = 1 * 1024 * 1024 * 1024;
		System.out.printf("Testing file size of %d\n", maxSize);
		SparkCUDAClient client = new SparkCUDAClient("127.0.0.1", 2333, "./", "test0", maxSize);
		byte[] array = new byte[maxSize];
		array[0] = 65;
		array[1] = 0;
		array[maxSize - 1] = 1;
		long begin = System.nanoTime();
		client.swapToGPU(array);
		long end = System.nanoTime();
		long time = end - begin;
		System.out.printf("Time elapsed: write %d. nanoseconds\n", time);
		client.close();
	}
}
