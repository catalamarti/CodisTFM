#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdlib.h>

void shuffle(int *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

double r2()
{
    return (double)rand() / (double)RAND_MAX ;
}

int main(){

	srand(time(NULL));
	
	int i;
	FILE *f;

    	f = fopen("./data/sample_MC.txt", "w");
	
	int Nsamples=1000;
	
	double S[Nsamples][12];

	double a1min=0.0001, a1max=0.005;
	double a2min=0.0001, a2max=0.4;
	double a3min=0.0001, a3max=0.1;
	double a4min=20.0, a4max=70.0;
	double a5min=15.0, a5max=50.0;
	double a6min=0.0001, a6max=0.1;
	double a7min=0.4, a7max=0.95;
	double a8min=0.4, a8max=0.95;
	double a9min=0.4, a9max=0.95;
	double a10min=0.4, a10max=0.95;
	double a11min=0.03, a11max=0.09;
	double a12min=0.9, a12max=1.4;


	double Ax_a1, Ax_a2, Ax_a3, Ax_a4, Ax_a5, Ax_a6;
	double Ax_a7, Ax_a8, Ax_a9, Ax_a10, Ax_a11, Ax_a12;

	Ax_a1=(a1max-a1min);
	Ax_a2=(a2max-a2min);
	Ax_a3=(a3max-a3min);
	Ax_a4=(a4max-a4min);
	Ax_a5=(a5max-a5min);
	Ax_a6=(a6max-a6min);
	Ax_a7=(a7max-a7min);
	Ax_a8=(a8max-a8min);
	Ax_a9=(a9max-a9min);
	Ax_a10=(a10max-a10min);
	Ax_a11=(a11max-a11min);
	Ax_a12=(a12max-a12min);
	
	for(i=0;i<Nsamples;i++){
		S[i][0]=a1min+Ax_a1*r2();
		S[i][1]=a2min+Ax_a2*r2();
		S[i][2]=a3min+Ax_a3*r2();
		S[i][3]=a4min+Ax_a4*r2();
		S[i][4]=a5min+Ax_a5*r2();
		S[i][5]=a6min+Ax_a6*r2();
		S[i][6]=a7min+Ax_a7*r2();
		S[i][7]=a8min+Ax_a8*r2();
		S[i][8]=a9min+Ax_a9*r2();
		S[i][9]=a10min+Ax_a10*r2();
		S[i][10]=a11min+Ax_a11*r2();
		S[i][11]=a12min+Ax_a12*r2();
	}

	fprintf(f,"%d\n",Nsamples);

	for(i=0;i<Nsamples;i++){
		fprintf(f,"%f %f %f %f %f %f ",S[i][0],S[i][1],S[i][2],S[i][3],S[i][4],S[i][5]);
		fprintf(f,"%f %f %f %f %f %f\n",S[i][6],S[i][7],S[i][8],S[i][9],S[i][10],S[i][11]);		
	}
	fclose(f);

}
