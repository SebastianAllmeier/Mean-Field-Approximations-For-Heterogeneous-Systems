// Copyright Nicolas Gast 2014-2015, Nicolas.Gast@inria.fr
#include"cache_simulation.hh"


cache_boxes_ode::cache_boxes_ode(int h, int *size_boxes, int type, random_gene *rng, double dt) 
  : cache_boxes(h, size_boxes,type, rng) , dt(dt)
{
  if (type!=RAND) {std::cerr << "FIFO or others not implemented for ODE\n";}
  N = rng->number_of_objects();
  // popularities = new double[N];
  // for(int k=0;k<N;k++) popularities[k] = rng.popularities(k);
  boxes = new double*[h+1];
  d_boxes = new double*[h+1];
  for(int i=0;i<h+1;i++) {
    boxes[i] = new double[N];
    d_boxes[i] = new double[N];
    for(int j=0;j<N;j++){
      if (i==0) boxes[i][j] = 1/(double)N;
      else      boxes[i][j] = 0;
    }
  }
  double size_tot = 0;
  relative_box_size = new double[h+1];
  for(int i=0;i<h;i++)
    size_tot += (relative_box_size[i+1] = size_boxes[i]/(double)N);
  relative_box_size[0] = 1-size_tot;
  //std::cout<< relative_box_size[2] << "\n"; exit(-1);
}

cache_boxes_ode::~cache_boxes_ode(){
  for(int i=0;i<h+1;i++) {
    delete boxes[i];
    delete d_boxes[i];
  }
  delete boxes;
  delete d_boxes;
  delete relative_box_size;
}
void cache_boxes_ode::ode_step() {
  dt = 0.5;
  for(int i=0;i<h+1;i++) for(int k=0;k<N;k++)  d_boxes[i][k] = 0;
  double rho[h+1]; 
  rho[0] = -1e10;
  for(int i=1;i<h+1;i++){
    rho[i] = 0;
    for(int k=0;k<N;k++){
      rho[i] += rng->popularities(k)*boxes[i-1][k]/relative_box_size[i];
    }
  }
  for(int i=0;i<h;i++){
    for(int k=0;k<N;k++){
      d_boxes[i][k] += -rng->popularities(k)*boxes[i][k] + boxes[i+1][k]*rho[i+1];
      d_boxes[i+1][k] += rng->popularities(k)*boxes[i][k] - boxes[i+1][k]*rho[i+1];
    }
  }
  //double s = 0;
  for(int i=0;i<h+1;i++)
    for(int k=0;k<N;k++){
      boxes[i][k] += dt*d_boxes[i][k];
      //s+=boxes[i][k];
    }
  //std::cout << "hop, tot=" << s << "\n";
}
double cache_boxes_ode::popularity_of_box(int i) {
  double s = 0;
  for(int j=0;j<N;j++){
    s+=boxes[i+1][j]*rng->popularities(j);
  }
  return N*s;
}
double cache_boxes_ode::print_popularity_of_box(int i) {
  double s = 0;
  for(int j=0;j<N;j++){
    s+=boxes[i+1][j]*rng->print_popularities(j);
  }
  return N*s;
}
void cache_boxes_ode::simulation_step() {
  for(double t=0;t<1;t+=dt) ode_step();
}

void usage_ode(){
  std::cerr << "usage : ./ode_simulator";
  usage();
}
int main(int argc, char** argv){
  int N = 10;
#define MAX_H 100
  int h = 1;
  int size_boxes[MAX_H];
  size_boxes[0] = 5;
  double alpha = 1;
  int type = RAND;
  int fillSizeBox = false;
  time_t t; time(&t); srand(t);
  bool do_swap = false;
  for(int i=1;i<argc;i++){
    switch(*(argv[i])){
    case 's': srand(atoi(argv[i]+1)); fillSizeBox = false; break;     
    case 'N': N = atoi(argv[i]+1);     fillSizeBox = false;  break;
    case 'a': alpha = atof(argv[i]+1); fillSizeBox = false;  break;
    case 'F': type = FIFO; fillSizeBox = false; break;
    case 'R': type = RAND; fillSizeBox = false; break;
    case 'M': fillSizeBox = true; h = 1; size_boxes[0]=atoi(argv[i]+1); break;
    case 'S': do_swap = true; break;
    default:
      if(fillSizeBox && h<MAX_H) {size_boxes[h]=atoi(argv[i]);h++;}
      else {usage_ode(); exit(1); }
    }
  }
  random_gene zipf_rng(N,alpha);
  cache_boxes_ode c(h,size_boxes,type, &zipf_rng);
  for(int t=0;t<20000;t++){
    c.simulation_step();
    if (t%10==0) {
      if (t%2000==0 && do_swap) zipf_rng.swap_distribution();
      std::cout << t << " ";
      c.print_popularities_boxes();
    }
  }
}
