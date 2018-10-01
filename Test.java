public class Test {
	public static void main(String[] args) {
		SparkCUDAClient client = new SparkCUDAClient("127.0.0.1", 2333, "./", "test0", 1024);
		byte[] array = new String("Some data").getBytes();
		client.swapToGPU(array);
	}
}