all: simu ode_solver

OBJ=random_generator_simulation.o
CFLAGS=-W -Wall -g3 -O2 

simu: simu.cc $(OBJ)
	g++ $(CFLAGS) simu.cc $(OBJ) -o simu 

ode_solver: ode_solver.cc $(OBJ)
	g++ $(CFLAGS) ode_solver.cc $(OBJ) -o ode_solver 

random_generator_simulation.o: random_generator_simulation.cc cache_simulation.hh
	g++ $(CFLAGS) random_generator_simulation.cc -c

clean:
	rm $(OBJ) simu ode_solver
