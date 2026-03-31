CC = clang
CFLAGS = -O2 -Wall -fPIC -arch arm64
LDFLAGS = -framework Foundation -dynamiclib

TARGET = libfastcache.dylib
SRCS = fast_cache.m
HDRS = fast_cache.h

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)

install: $(TARGET)
	cp $(TARGET) ../mlx_flash_compress/

.PHONY: all clean install
