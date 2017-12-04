#include <stdio.h>
#include <string.h>

#define BIG 1000000000

#define MAXV 800
#define MAXC 500
#define MAXE 1450


int cows;
int v,e;


int cow_pos[MAXC];
int degree[MAXV];
int con[MAXV][MAXV];
int cost[MAXV][MAXV];

int dist[MAXC][MAXV];


int heapsize;
int heap_id[MAXV];
int heap_val[MAXV];
int heap_lookup[MAXV];


bool validheap(void){
  for(int i = 0; i < heapsize; ++i){
    if(!(0 <= heap_id[i] && heap_id[i] < v)){
      return(false);
    }
    if(heap_lookup[heap_id[i]] != i){
      return(false);
    }
  }
  return(true);
}


void heap_swap(int i, int j){
  int s;

  s = heap_val[i];
  heap_val[i] = heap_val[j];
  heap_val[j] = s;

  heap_lookup[heap_id[i]] = j;

  heap_lookup[heap_id[j]] = i;

  s = heap_id[i];
  heap_id[i] = heap_id[j];
  heap_id[j] = s;

}


void heap_up(int i){
  if(i > 0 && heap_val[(i-1) / 2] > heap_val[i]){
    heap_swap(i, (i-1)/2);
    heap_up((i-1)/2);
  }
}


void heap_down(int i){
  int a = 2*i+1;
  int b = 2*i+2;

  if(b < heapsize){
    if(heap_val[b] < heap_val[a] && heap_val[b] < heap_val[i]){
      heap_swap(i, b);
      heap_down(b);
      return;
    }
  }
  if(a < heapsize && heap_val[a] < heap_val[i]){
    heap_swap(i, a);
    heap_down(a);
  }
}




int main(){


  FILE *filein = fopen("butter.in9", "r");
  fscanf(filein, "%d %d %d", &cows, &v, &e);
  for(int i = 0; i < cows; ++i){
    fscanf(filein, "%d", &cow_pos[i]);
    --cow_pos[i];
  }
  for(int i = 0; i < v; ++i){
    degree[i] = 0;
  }
  for(int i = 0; i < e; ++i){
    int a,b,c;
    fscanf(filein, "%d %d %d", &a, &b, &c);
    --a;
    --b;

    con[a][degree[a]] = b;
    cost[a][degree[a]] = c;
    ++degree[a];

    con[b][degree[b]] = a;
    cost[b][degree[b]] = c;
    ++degree[b];

  }
  fclose(filein);


  for(int i = 0; i < cows; ++i){
    heapsize = v;
    for(int j = 0; j < v; ++j){
      heap_id[j] = j;
      heap_val[j] = BIG;
      heap_lookup[j] = j;
    }
    heap_val[cow_pos[i]] = 0;
    heap_up(cow_pos[i]);

    bool fixed[MAXV];
    memset(fixed, false, v);
    for(int j = 0; j < v; ++j){
      int p = heap_id[0];
      dist[i][p] = heap_val[0];
      fixed[p] = true;
      heap_swap(0, heapsize-1);
      --heapsize;
      heap_down(0);

      for(int k = 0; k < degree[p]; ++k){
	int q = con[p][k];
	if(!fixed[q]){
	  if(heap_val[heap_lookup[q]] > dist[i][p] + cost[p][k]){
	    heap_val[heap_lookup[q]] = dist[i][p] + cost[p][k];
	    heap_up(heap_lookup[q]);
	  }
	}
      }

    }
  }

  int best = BIG;
  for(int i = 0; i < v; ++i){
    int total = 0;
    for(int j = 0; j < cows; ++j){
      total += dist[j][i];
    }
    if (total < best) {
        best = total;
    }
  }


  FILE *fileout = fopen("butter.out", "w");
  fprintf(fileout, "%d\n", best);
  fclose(fileout);


  return(0);
}
