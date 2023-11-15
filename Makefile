CC=g++ -O3 -std=c++14
SRCS=$(wildcard *.cpp */*.cpp)
OBJS=$(patsubst %.cpp, %.o, $(SRCS))

# for MacOs
# INCLUDE = -I/usr/local/include/libtorch/include -I/usr/local/include/libtorch/include/torch/csrc/api/include
# LIB +=-L/usr/local/include/libtorch/lib -ltorch -lc10 -lpthread 
# FLAG = -Xlinker -rpath -Xlinker /usr/local/include/libtorch/lib

TYPE = CPU
# TYPE = GPU

# for linux
ifeq ($(TYPE), GPU)
	INCLUDE = -I/home/liuguanli/Documents/libtorch_gpu/include -I/home/liuguanli/Documents/libtorch_gpu/include/torch/csrc/api/include
	LIB +=-L/home/liuguanli/Documents/libtorch_gpu/lib -ltorch -lc10 -lpthread
	FLAG = -Wl,-rpath=/home/liuguanli/Documents/libtorch_gpu/lib
else
	INCLUDE = -I/b/ProgramData/libtorch_2_10/include -I/b/ProgramData/libtorch_2_10/include/torch/csrc/api/include 
	LIB +=-L/b/ProgramData/libtorch_2_10/lib -ltorch -lc10 -lpthread
	FLAG = -Wl,-rpath=/b/ProgramData/libtorch_2_10/lib
endif

# B:\Program Files\libtorch_2_10
# /ProgramData/libtorch_2_10

# INCLUDE = -I/home/liuguanli/Documents/libtorch/include -I/home/liuguanli/Documents/libtorch/include/torch/csrc/api/include
# LIB +=-L/home/liuguanli/Documents/libtorch/lib -ltorch -lc10 -lpthread
# FLAG = -Wl,-rpath=/home/liuguanli/Documents/libtorch/lib

NAME=$(wildcard *.cpp)
TARGET=$(patsubst %.cpp, %, $(NAME))


$(TARGET):$(OBJS)
	$(CC) -o $@ $^ $(INCLUDE) $(LIB) $(FLAG)
%.o:%.cpp
	$(CC) -o $@ -c $< -g $(INCLUDE)

clean:
	rm -rf $(TARGET) $(OBJS)

# # g++ -std=c++11 Exp.cpp FileReader.o -ltensorflow -o Exp_tf
