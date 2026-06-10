# Complete Makefile with build, copy, and run
CC = cl
CFLAGS = /I "include"
LIBS = ..\..\target\release\facedetector.dll.lib
TARGET = main.exe
SRC = main.c
DLL_SOURCE = ..\..\target\release\facedetector.dll
DLL_DEST = .\facedetector.dll

all: help

help:
	@echo --------------- FACEDETECTOR C EXAMPLE - MAKE COMMANDS ---------------
	@echo ======================================================================
	@echo.
	@echo - (none) - - - - - - Show this help
	@echo - build - - - - - -- Build executable and copy DLL
	@echo - run - - - - - - -- Build, copy DLL, and run
	@echo - clean - - - - - -- Remove all generated files
	@echo - help - - - - - - - Show this help
	@echo.
	@echo ======================================================================


build: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(LIBS) /Fe$(TARGET) /nologo

	@if exist "$(DLL_SOURCE)" ( \
		copy /Y "$(DLL_SOURCE)" "$(DLL_DEST)" >nul
	) else ( \
		echo [ERROR] DLL not found: $(DLL_SOURCE) && \
		exit 1 \
	)

run: build
	@$(TARGET)

clean:
	@if exist $(TARGET) del $(TARGET)
	@if exist *.obj del *.obj
	@if exist $(DLL_DEST) del $(DLL_DEST)
