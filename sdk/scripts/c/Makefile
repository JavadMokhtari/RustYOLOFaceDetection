# Complete Makefile with build, copy, and run
CC = cl

CURDIR = $(MAKEDIR)
ROOT = $(MAKEDIR)\..\..

CFLAGS = /I "include"
LIBS = $(ROOT)\bin\windows\facedetector.dll.lib
SRC = main.c
TARGET = main.exe
DLL_SOURCE = $(ROOT)\bin\windows\facedetector.dll
DLL_DEST = $(CURDIR)\facedetector.dll

all: help

help:
	@echo.
	@echo ================= FaceDetector Example =================
	@echo nmake build   - Build executable and copy DLL
	@echo nmake run     - Build and run
	@echo nmake clean   - Remove build directory
	@echo ========================================================
	@echo.

build: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(LIBS) /Fe$(TARGET) /nologo
	@copy /Y $(DLL_SOURCE) $(DLL_DEST) >nul

run: build
	@$(TARGET)

clean:
	@if exist $(TARGET) del $(TARGET)
	@if exist *.obj del *.obj
	@if exist $(DLL_DEST) del $(DLL_DEST)
