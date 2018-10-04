import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;  
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.lang.Long;

public class Test {
	public static void main(String[] args) {
		// 1GB data
		int maxSize = 1 * 1024 * 1024 * 1024;
		int threadNum = 16;
		System.out.printf("Testing file size of %d\n", maxSize);
		List<SparkCUDAClient> clients = new ArrayList<>();
		for (int i = 0; i < threadNum; ++i) {
			String fileName = "test" + Integer.toString(i);
			SparkCUDAClient client = new SparkCUDAClient("127.0.0.1", 2333, "./", fileName, maxSize);
			clients.add(client);
		}
		byte[] array = new byte[maxSize];
		array[0] = 65;
		array[1] = 0;
		array[maxSize - 1] = 1;

		ExecutorService executorService;
		ExecutorService cachedThreadPool = Executors.newCachedThreadPool();

		long begin = System.nanoTime();
		for (int i = 0; i < threadNum; ++i) {
			final int idx = i;
			cachedThreadPool.execute(new Runnable() {
				@Override
				public void run() {
					clients.get(idx).swapToGPU(array);
				}
			});
		}
		cachedThreadPool.shutdown();
		try {
			cachedThreadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			System.out.println("Execution failed.");
		}
		long end = System.nanoTime();
		long time = end - begin;

		for (int i = 0; i < threadNum; ++i)
			clients.get(i).close();
		System.out.printf("Time elapsed: write %d nanoseconds\n", time);
	}
}
