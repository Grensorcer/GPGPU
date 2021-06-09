all:
	g++ -Wall -Wextra -pedantic -std=c++17 src/main.cc -lopencv_core -lopencv_video -lopencv_videoio -lopencv_highgui -o main

bench:
	g++ -Wall -Wextra -pedantic -std=c++17 -Isrc bench_src/bench.cc -lopencv_core -lopencv_imgproc  -lopencv_video -lopencv_videoio -lopencv_highgui -lbenchmark -pthread -o bench
