#include <iostream>
#include <thread>
#include <vector>
#include <string>
#include <cstring>
#include <cuda_runtime.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

typedef struct _SwapBuffer {
	std::string fileName;
	int fd;
	unsigned long size;
	void *mapped;
} SwapBuffer, *pSwapBuffer;

class SparkCUDAServer {
private:
	bool initCUDA();
	bool setupServer(int port);
	void messageLoop(int listenfd);
	int deviceCount = 0;
	int activeDevice = -1;
	int port = 0;
	static const int MAX_LENGTH = 1024;
	static const int MAX_THREADS = 1024;
	std::vector<cudaDeviceProp> deviceProps;
	std::vector<std::thread> threadPool;
	std::vector<SwapBuffer> swapBuffer;
public:
	SparkCUDAServer(int port, int deviceNum);
	~SparkCUDAServer();
	int getGPUCount();
};

bool SparkCUDAServer::initCUDA() {
	int count;
	cudaGetDeviceCount(&count);
	deviceCount = count;

	// No CUDA device exists
	if (!count)
		return false;
	
	cudaDeviceProp prop;
	for (int i = 0; i < count; ++i) {
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
			deviceProps.push_back(prop);
	}

	return true;
}

void SparkCUDAServer::messageLoop(int listenfd) {
	using std::cout;
	using std::endl;
	int n, connfd, fd;
	void *ptr;
	SwapBuffer sb;
	struct stat st;
	char buffer[SparkCUDAServer::MAX_LENGTH];
	const std::string okResponse("OK");
	std::vector<SwapBuffer>::iterator idx;
	rapidjson::Document doc;
	cout << "SparkCUDAServer: listening on port " << port << endl;
	while (true) {
		connfd = accept(listenfd, nullptr, nullptr);
		if (connfd == -1)
			continue;
		n = recv(connfd, buffer, SparkCUDAServer::MAX_LENGTH, 0);
		buffer[n] = '\0';
		doc.Parse(buffer);
		std::string message =  doc["message"].GetString();
		int msgType = doc["messageType"].GetInt();

		switch (msgType) {
		case 0:	// establish connection
			fd = open(message.c_str(), O_RDWR);
			fstat(fd, &st);
			ptr = mmap(0, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
			cudaHostRegister(ptr, st.st_size, cudaHostRegisterDefault);
			if (ptr == MAP_FAILED) {
				send(connfd, "\0", 1, 0);
				cout << "SparkCUDAServer: failed to create swap." << endl;
			}
			else {
				send(connfd, message.c_str(), message.length(), 0);
				sb.fd = fd;
				sb.fileName = message;
				sb.mapped = ptr;
				sb.size = st.st_size;
				swapBuffer.push_back(sb);
				cout << "SparkCUDAServer: connection established for " << message << endl;
			}
			break;
		case 1:case 2:
			sb.fileName = "";
			for (auto i = swapBuffer.begin(); i != swapBuffer.end(); ++i) {
				if (i->fileName == message) {
					sb = *i;
					idx = i;
					break;
				}
			}
			if (sb.fileName == "") {
				cout << "SparkCUDAServer: swap file not found." << endl;
				break;
			}
			if (msgType == 1) {	// swap to GPU
				send(connfd, okResponse.c_str(), okResponse.length(), 0);
				cout << "SparkCUDAServer: copied data from " << message << " to GPU" << endl;
			}
			else {	// close connection
				munmap(sb.mapped, sb.size);
				close(sb.fd);
				cudaHostUnregister(sb.mapped);
				swapBuffer.erase(idx);
				send(connfd, okResponse.c_str(), okResponse.length(), 0);
				cout << "SparkCUDAServer: released " << sb.fileName << " resources." << endl;
			}
			break;
		default:
			break;
		}
		close(connfd);
	}
	close(listenfd);
}

bool SparkCUDAServer::setupServer(int port) {
	sockaddr_in sockaddr;
	int listenfd;
	this->port = port;
	memset(&sockaddr, 0, sizeof(sockaddr));
	sockaddr.sin_family = AF_INET;
	sockaddr.sin_port = htons((unsigned short)port);
	sockaddr.sin_addr.s_addr = htonl(INADDR_ANY);
	listenfd = socket(AF_INET, SOCK_STREAM, 0);
	if (listenfd == -1)
		return false;

	bind(listenfd, (struct sockaddr*)&sockaddr, sizeof(sockaddr));
	listen(listenfd, SparkCUDAServer::MAX_THREADS);
	threadPool.push_back(std::thread(&SparkCUDAServer::messageLoop, this, listenfd));
	return true;
}

SparkCUDAServer::SparkCUDAServer(int port=2333, int deviceNum=-1) {
	using std::cout;
	using std::endl;
	// init CUDA and set device
	if (initCUDA()) {
		if (deviceNum < 0) {
			for (auto prop = deviceProps.begin(); prop != deviceProps.end(); ++prop)
				if (prop->major >= 1) {
					deviceNum = prop->major;
					break;
				}
		}
		cudaSetDevice(deviceNum);
		activeDevice = deviceNum;
		cout << "SparkCUDAServer: initialized and set to device " << activeDevice << endl;
	}
	if (setupServer(port) == -1)
		cout << "SparkCUDAServer: Failed to establish server." << endl;
}

SparkCUDAServer::~SparkCUDAServer() {
	for (auto &i : swapBuffer) {
		close(i.fd);
		cudaHostUnregister(i.mapped);
		munmap(i.mapped, i.size);
	}
	for (auto i = threadPool.begin(); i != threadPool.end(); ++i)
		i->detach();
}

int SparkCUDAServer::getGPUCount() {
	return deviceCount;
}

int main() {
	using std::cout;
	using std::endl;
	SparkCUDAServer server;
	sleep(100);
	return 0;
}
