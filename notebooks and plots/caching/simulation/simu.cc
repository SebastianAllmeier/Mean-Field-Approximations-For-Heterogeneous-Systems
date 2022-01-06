// Copyright Nicolas Gast 2014-2015, Nicolas.Gast@inria.fr

#include"cache_simulation.hh"

void usage_simu(){
  std::cerr << "usage: ./simu ";
  usage();
}
int main(int argc, char** argv) {
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
  bool hit_per_item = false;
  for(int i=1;i<argc;i++){
    switch(*(argv[i])){
    case 's': srand(atoi(argv[i]+1)); fillSizeBox = false; break;     
    case 'N': N = atoi(argv[i]+1);     fillSizeBox = false;  break;
    case 'a': alpha = atof(argv[i]+1); fillSizeBox = false;  break;
    case 'F': type = FIFO; fillSizeBox = false; break;
    case 'R': type = RAND; fillSizeBox = false; break;
    case 'L': type = LRU; fillSizeBox = false; break;
    case 'M': fillSizeBox = true; size_boxes[0]=atoi(argv[i]+1); break;
    case 'S': do_swap = true; break;
    case 'H': hit_per_item = true; break; 
    default:
      if(fillSizeBox && h<MAX_H) {size_boxes[h]=atoi(argv[i]);h++;}
      else { usage_simu(); exit(1);}
    }
  }
  random_gene zipf_rng(N,alpha);
  cache_boxes_simu c(h,size_boxes,type,&zipf_rng);
  if (!hit_per_item)
    {
      for(int t=0;t<1000000;t++) {
	c.simulation_step();
	if (t%10==0) {
	  if (t%2000==0 && do_swap) zipf_rng.swap_distribution();
	  std::cout << t << " ";
	  c.print_popularities_boxes();
	}
      }
    }
  else {
    for(int t=0;t<1e7;t++) c.simulation_step(); // To reach steady-state
    for(int t=0;t<1e8;t++) {
      c.simulation_step(); // To reach steady-state
      if (t%N==0) c.register_popularities();
    }
    c.print_registered_popularities(); 
  }
}
