#include<iostream>
#include<cmath>

enum {FIFO=0, RAND, LRU};

class random_gene{
  double *cumulative_probability; // probability
  double *probability; // probability
  double *print_probability; // probability that we print (in order to weight the results)
  double alpha;
public:
  int N; // number of objects
  random_gene(int N, double alpha);
  //random_gene(random_gene &rng);
  ~random_gene();
  void new_fixedPopularity(double alpha);
  void new_zipf(double alpha);
  int next_object();
  int number_of_objects();
  double popularities(int n);
  double print_popularities(int n);
  void swap_distribution();
};

class cache_boxes{
protected:
  int h;
  int *size_boxes;
  int type; // FIFO or RAND
  int **boxes;
  int t; 
  random_gene *rng;
public:
  cache_boxes(int h, int *size_boxes, int type, random_gene *rng);
  void simulation_step();
  virtual double popularity_of_box(int i);
  virtual double print_popularity_of_box(int i);
  void print_popularities_boxes();
};

class cache_boxes_simu : public cache_boxes{
private:
  int **boxes;
  int get_position_in_box(int i, int n);
  int insert_oject(int i, int n) ;
  void move_object_to_next_box(int i, int pos);
  int hit(int n);
  int **registered_popularities;
  int total_registered;
public:
  cache_boxes_simu(int h, int *size_boxes, int type, random_gene *rng);
  virtual double popularity_of_box(int i);
  virtual double print_popularity_of_box(int i);
  void simulation_step();
  void register_popularities();
  void print_registered_popularities();
};

class cache_boxes_ode : public cache_boxes{
  double **boxes;
  double **d_boxes;
  double dt;
  int N; 
  // double * popularities;
  double *relative_box_size;
  void ode_step();
public:
  cache_boxes_ode(int h, int *size_boxes, int type, random_gene *rng, double dt=0.1);
  ~cache_boxes_ode();
  virtual double popularity_of_box(int i);
  virtual double print_popularity_of_box(int i);
  void simulation_step();
};

void usage();
