# F90 compiler and flags
FC = gfortran
FFLAGS = -Wall -g -O3

################### (should not change anything below)

default: main.o
	$(FC) CommonData.o Interpolator.o Integrator.o \
              GrowthFactor.o CosmoFunc.o main.o -o amf.exe
	rm *.mod *.o

CommonData.o:
	$(FC) $(FFLAGS) -c CommonData.F90 -o CommonData.o
Interpolator.o:
	$(FC) $(FFLAGS) -c Interpolator.F90 -o Interpolator.o
Integrator.o:
	$(FC) $(FFLAGS) -c Integrator.F90 -o Integrator.o
GrowthFactor.o: CommonData.o
	$(FC) $(FFLAGS) -c GrowthFactor.F90 -o GrowthFactor.o
CosmoFunc.o: CommonData.o Interpolator.o
	$(FC) $(FFLAGS) -c CosmoFunc.F90 -o CosmoFunc.o
main.o: CommonData.o GrowthFactor.o CosmoFunc.o Integrator.o
	$(FC) $(FFLAGS) -c main.F90 -o main.o

PHONY: clean

clean:
	rm -f *.mod *.o amf.exe
