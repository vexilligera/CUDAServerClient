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
	static const int PAGE_SIZE = 4096;
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
	char buffer[SparkCUDAServer::MAX_LENGTH];
	cout << "SparkCUDAServer: listening on port " << port << endl;
	rapidjson::Document doc;
	while (true) {
		connfd = accept(listenfd, nullptr, nullptr);
		if (connfd == -1)
			continue;
		n = recv(connfd, buffer, SparkCUDAServer::MAX_LENGTH, 0);
		buffer[n] = '\0';
		doc.Parse(buffer);
		std::string message =  doc["message"].GetString();

		switch (doc["messageType"].GetInt()) {
		case 0:	// establish connection
			fd = open(message.c_str(), O_RDWR);
			ptr = mmap(0, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
			if (ptr == MAP_FAILED) {
				send(connfd, "\0", 1, 0);
				cout << "SparkCUDAServer: failed to create swap." << endl;
			}
			else {
				send(connfd, message.c_str(), message.length(), 0);
				sb.fd = fd;
				sb.fileName = message;
				sb.mapped = ptr;
				swapBuffer.push_back(sb);
				cout << "SparkCUDAServer: connection established for " << message << endl;
			}
			break;
		case 1: // swap to GPU
			sb.fileName = "";
			for (auto i = swapBuffer.begin(); i != swapBuffer.end(); ++i) {
				if (i->fileName == message) {
					sb = *i;
					break;
				}
			}
			if (sb.fileName == "") {
				cout << "SparkCUDAServer: swap file not found." << endl;
				break;
			}
			cout << (char*)sb.mapped << endl;
			cout << "SparkCUDAServer: copied data from " << message << " to GPU" << endl;
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
	struct stat st;
	for (auto &i : swapBuffer) {
		fstat(i.fd, &st);
		munmap(i.mapped, st.st_size);
		close(i.fd);
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
