run: all
	./sdf-cad
all:
	go build
clean:
	go clean
	-rm *.stl
	-rm *.dxf
