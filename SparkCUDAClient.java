import java.io.File;
import java.io.RandomAccessFile;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.BufferedReader;
import java.nio.channels.FileChannel;
import java.nio.MappedByteBuffer;
import java.nio.ByteBuffer;
import java.net.Socket;

public class SparkCUDAClient {
	private long maxLength;
	private String threadId;
	private File file;
	private String fileName;
	private MappedByteBuffer buffer;
	private RandomAccessFile accessFile;
	private FileChannel fileChannel;
	private Socket socket;
	private String serverIP;
	private int serverPort;

	private String notifyServer(SparkCUDAMessage messsage) {
		String response = "";
		try {
			socket = new Socket(serverIP, serverPort);
			OutputStream os = socket.getOutputStream();
			PrintWriter printWriter = new PrintWriter(os);
			printWriter.print(messsage.toString());
			printWriter.flush();
			socket.shutdownOutput();

			InputStream is = socket.getInputStream();
			BufferedReader br = new BufferedReader(new InputStreamReader(is));
			while ((response = br.readLine()) != null)
				System.out.println("SparkCUDAClient: response from server: " + response);
			socket.shutdownInput();
			os.close();
			is.close();
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println("SparkCUDAClient: socket IO failed.");
		} finally {
			return response;
		}
	}

	public SparkCUDAClient(String serverIP, int serverPort, String filePath, String threadId, long bufferSize) {
		try {
			// setup swap memory file
			this.threadId = threadId.replace("/", "").replace("\\", "");
			fileName = filePath.replaceAll("/+$", "") + "/" + this.threadId;
			file = new File(fileName);
			file.createNewFile();
			accessFile = new RandomAccessFile(file, "rw");
			fileChannel = accessFile.getChannel();
			maxLength = bufferSize;
			buffer = fileChannel.map(FileChannel.MapMode.READ_WRITE, 0, bufferSize);

			// connect to server
			this.serverIP = serverIP;
			this.serverPort = serverPort;
			SparkCUDAMessage msgConnect = new SparkCUDAMessage(SparkCUDAMessage.Connect, fileName);
			String response = notifyServer(msgConnect);
			assert response == this.threadId;
			System.out.println("SparkCUDAClient: Connected to SparkCUDAServer.");
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println("SparkCUDA: exception during initialization.");
			System.err.println("SparkCUDA: Failed to create swap buffer.");
		} finally {
			System.out.printf("SparkCUDA: memory buffer created as %s \n", fileName);
			System.out.printf("SparkCUDA: maximum buffer size %d bytes.\n", maxLength);
		}
	}

	public void close() {
		SparkCUDAMessage msgClose = new SparkCUDAMessage(SparkCUDAMessage.Close, fileName);
		notifyServer(msgClose);
	}

	// copy from memory to GPU memory
	public void swapToGPU(byte[] data) {
		buffer.put(data);
		SparkCUDAMessage msgSwap = new SparkCUDAMessage(SparkCUDAMessage.Swap, fileName);
		notifyServer(msgSwap);
	}
}
