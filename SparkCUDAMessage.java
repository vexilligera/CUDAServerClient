import java.io.Serializable;

public class SparkCUDAMessage implements Serializable {
	private String message;
	private int messageType;

	public static int Connect = 0;
	public static int Swap = 1;

	public SparkCUDAMessage(int messageType, String message) {
		super();
		this.messageType = messageType;
		this.message = message;
	}

	@Override
	public String toString() {
		return String.format("{\"messageType\": %d, \"message\": \"%s\"}", messageType, message);
	}
}