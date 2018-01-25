m.PHONY : clean

libquantize.so : quantize.o
	gcc -shared -o libquantize.so quantize.o
	#gcc -shared -Wl,-soname,libquantize.so -o libquantize.so quantize.o

quantize.o : quantize.c
	gcc -c -fPIC quantize.c -o quantize.o

clean :
	-rm -vf libquantize.so quantize.o
