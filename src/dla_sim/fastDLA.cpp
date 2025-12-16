#include "fastDLA.hpp"
#include <iostream>
#include <stdlib.h>

int main(int argc, char** argv)
{
  int n, seed;
  char filename[1000];
  double noiseReductionFactor=1;

  seed=time(NULL);
  filename[0]='\0';

  switch(argc-1)
    {
    case 4:
      sscanf(argv[4],"%lf",&noiseReductionFactor);
    case 3:
      sscanf(argv[3],"%d",&seed);
    case 2:
      sscanf(argv[2],"%s",&filename);
    case 1:
      sscanf(argv[1],"%d",&n);
      break;
    default:
      fprintf(stderr,"First 2 arguments (Cluster Size and Filename) are mandatory.\nThird argument (seed) and fourth argument (noise reduction factor) are optional.\n");
      return 1;
    }

  FastCluster grid(n,seed,noiseReductionFactor);

  if (filename[0])
    {
      FILE* fp=fopen(filename, "w");
      if (!fp)
	{
	  fprintf(stderr, "Failed to open file to save\n");
	  return 1;
	}
      else
	{
	  grid.writePoints(fp);
	  fclose(fp);
	  printf("Written cluster to file\n");
	}
    }
  else
    {
      printf("Not saving points as no filename given\n");
    }

  return 0;
}