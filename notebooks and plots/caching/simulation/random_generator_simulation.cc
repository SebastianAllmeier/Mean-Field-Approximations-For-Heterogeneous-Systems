//Copyright Nicolas Gast 2014-2015, Nicolas.Gast@inria.fr

#include "cache_simulation.hh"

random_gene::~random_gene(){ delete probability; delete cumulative_probability;}
random_gene::random_gene(int N, double alpha) :alpha(alpha),  N(N) {
  if (alpha>0)
    new_zipf(alpha);
  else
    new_fixedPopularity(alpha);
}

void random_gene::new_fixedPopularity(double alpha){
  probability = new double[N];
  cumulative_probability = new double[N];
  probability[0] = 2/((1-alpha)*N);
  cumulative_probability[0] = probability[0];
  for(int i=1;i<N;i++){
    if (i<N/2)
      probability[i] = 2/((1-alpha)*N);
    else
      probability[i] = -2*alpha/((1-alpha)*N);
    cumulative_probability[i] = cumulative_probability[i-1]+probability[i];
  }
  //print_probability = probability;
  print_probability = new double[N];
  for(int i=1;i<N;i++){
    if (i<N/2)
      print_probability[i] = 0;
    else
      print_probability[i] = 2/((double)N);
  }
  //std::cerr << "DEBUG ** total pop=" <<cumulative_probability[N-1] <<"\n";
}

void random_gene::new_zipf(double alpha){
  probability = new double[N];
  cumulative_probability = new double[N];
  probability[0] = 1;
  cumulative_probability[0] = 1;
  for(int i=1;i<N;i++){
    probability[i] = 1/pow((i+1),alpha);
      cumulative_probability[i] = cumulative_probability[i-1]+probability[i];
  }
  for(int i=0;i<N;i++) {
    probability[i]/=cumulative_probability[N-1];
    cumulative_probability[i]/=cumulative_probability[N-1];
  }
  print_probability = probability;
  //std::cerr << "DEBUG ** total pop=" <<cumulative_probability[N-1] <<"\n";
}

int random_gene::next_object()
{
  double u = rand()/(1.+RAND_MAX);
  int i=0;
  while(u>cumulative_probability[i] && i < N) i++;
  return i;
}
double random_gene::popularities(int n){
  if (n<0 || n>=N) return 0;
  return probability[n];
}
double random_gene::print_popularities(int n){
  if (n<0 || n>=N) return 0;
  return print_probability[n];
}
int random_gene::number_of_objects(){
  return N;
}
void random_gene::swap_distribution() {
  srand((int) (100000*alpha) );
  alpha = .25+( (rand()%10000) / 15000.);
  std::cerr << alpha << "\n";
  new_zipf( alpha );
  
  for(int i=N-1;i>1;i--)
    {
      double pop = probability[i];
      int j = rand()%(i+1);
      probability[i] = probability[j];
      probability[j] = pop;
    }
  
  // for(int i=0;i<N/2;i++){
  //   double pop = probability[i];
  //   probability[i] = probability[N-1-i];
  //   probability[N-1-i] = pop;
  // }
  double cum=0;
  for(int i=0;i<N;i++){
    cum += probability[i];
    cumulative_probability[i] = cum;
  }
}


/* CACHE_BOXES */

cache_boxes::cache_boxes(int h, int *size_boxes, int type, random_gene *rng) : h(h), type(type), rng(rng) {
  this->size_boxes = new int[h];
  for(int i=0;i<h;i++) this->size_boxes[i] = size_boxes[i];
}

double cache_boxes::print_popularity_of_box(int i){
  std::cerr << "not implemented\n";
  return -i;
}
double cache_boxes::popularity_of_box(int i){
  std::cerr << "not implemented\n";
  return -i;
}
void cache_boxes::simulation_step(){
  std::cerr << "not implemented\n";
}
void cache_boxes::print_popularities_boxes(){
  double tot=0;
  for(int i=0;i<h;i++) {
    double s=print_popularity_of_box(i);
    std::cout << s << "\t";
    tot+=s;
  }
  std::cout<<tot<<"\n";
}



/* CACHE_BOXES_SIMU */

cache_boxes_simu::cache_boxes_simu(int h, int *size_boxes, int type, random_gene*rng) 
  : cache_boxes(h, size_boxes, type, rng), total_registered(0)
{
  boxes = new int*[h];
  for(int i=0;i<h;i++) {
    boxes[i] = new int[size_boxes[i]];
    for(int j=0;j<size_boxes[i];j++){
      boxes[i][j] = -1;
    }
  }
  registered_popularities = (int**) malloc(sizeof(int*)*rng->N);
  for(int i=0;i<rng->N;i++){
    registered_popularities[i] = (int*) malloc(sizeof(int)*h);
    for(int j=0;j<h;j++) {
      registered_popularities[i][j] = 0;
    }
  }
}

void cache_boxes_simu::print_registered_popularities(){
  for(int i=0;i<rng->N;i++){
    for(int j=0;j<h;j++) std::cout << registered_popularities[i][j]/(double)total_registered <<" ";
    std::cout<<"\n";
  }
}
void cache_boxes_simu::register_popularities(){
  for(int i=0;i<h;i++) {
    for(int j=0;j<size_boxes[i];j++){
      if (boxes[i][j] != -1)
	registered_popularities[boxes[i][j]][i]++;
    }
  }
  total_registered++;
}

int cache_boxes_simu::get_position_in_box(int i, int n){
  // If obect n is in box i, returns its position. Otherwise, returns 0.
  for(int j=0;j<size_boxes[i];j++){
    if(boxes[i][j] == n) return j;
  }
  return -1;
}

int cache_boxes_simu::insert_oject(int i, int n) {
  int obj_returned, newPos;
  switch(type){
  case RAND:
    newPos = rand()%size_boxes[i];
    obj_returned = boxes[i][newPos];
    boxes[i][newPos] = n;
    break;
  case FIFO:
    obj_returned = boxes[i][size_boxes[i]-1];
    for(int j=size_boxes[i]-1;j>0;j--)
      boxes[i][j] = boxes[i][j-1];
    boxes[i][0] = n;
    break;
  default:
    std::cerr << "undefined policy" << type<<". Sould be FIFO or RAND\n";
    exit(-1);
  }
  return obj_returned;
}

void cache_boxes_simu::move_object_to_next_box(int i, int pos){
  if (i+1<h) boxes[i][pos] = insert_oject(i+1,boxes[i][pos]);
}

int cache_boxes_simu::hit(int n){ // performs the update of cache when object 1 is
  // hit.  Returns i if item "n" is in the box i, -1
  // if not in the cache
  for (int i=0;i<h;i++) {
    int pos=-1;
    if ( (pos = get_position_in_box(i,n) ) != -1 ) {
      if ( i<h ) move_object_to_next_box(i,pos);
      return i;
    }
  }
  insert_oject(0,n);
  return -1;
}


void cache_boxes_simu::simulation_step(){
  hit(rng->next_object());
}
double cache_boxes_simu::popularity_of_box(int i){
  double s = 0;
  for(int j=0;j<size_boxes[i];j++) s+= rng->popularities(boxes[i][j]);
  return s;
}
double cache_boxes_simu::print_popularity_of_box(int i){
  double s = 0;
  for(int j=0;j<size_boxes[i];j++) s+= rng->print_popularities(boxes[i][j]);
  return s;
}
    

void usage(){
  std::cerr << "[sX|Nn|aA|F|R|S|Mh m0 m1 m2 ... mh]\n"
	    << "  sX              initialize random to X (call srand(X))\n"
	    << "  Nn              n items\n"
	    << "  aA              parameter of the zipf distribution\n"
	    << "  F               FIFO(m) policy\n"
	    << "  R               RAND(m) policy\n"
	    << "  S               re-shuffle item popularities every 2000 requests\n"
	    << "  Mm1 ... mh      simulate FIFO(m) or RAND(m) with h lists of sizes m1..mh"
	    << "  H               hit_per_item"
	    << "\n";
}
