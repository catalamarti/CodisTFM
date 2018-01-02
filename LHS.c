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

    	f = fopen("./data/sample.txt", "w");
	
	int Nsamples=10;
	
	double a1min=9.25, a1max=37.0;
	double a2min=0.5, a2max=4.0;
	double a3min=1.0, a3max=2.4;
	double a4min=23.0, a4max=92.0;
	double a5min=16.5, a5max=66.0;
	double a6min=0.05, a6max=0.2;
	double a7min=0.03, a7max=0.12;
	double a8min=0.7, a8max=1.3;
	double a10min=8.0, a10max=32.0;
	double a11min=2.5, a11max=10.0;
	
	double A1[Nsamples+1], A2[Nsamples+1], A3[Nsamples+1], A4[Nsamples+1], A5[Nsamples+1], A6[Nsamples+1];
	double A7[Nsamples+1], A8[Nsamples+1], A10[Nsamples+1], A11[Nsamples+1];
	double S[Nsamples][12];
	int S8[Nsamples];
	int c1[Nsamples], c2[Nsamples],c3[Nsamples], c4[Nsamples],c5[Nsamples], c6[Nsamples];
	int c7[Nsamples], c8[Nsamples], c10[Nsamples],c11[Nsamples];


	double Ax_a1, Ax_a2, Ax_a3, Ax_a4, Ax_a5, Ax_a6;
	double Ax_a7, Ax_a8, Ax_a10, Ax_a11;

	Ax_a1=(a1max-a1min)/Nsamples;
	Ax_a2=(a2max-a2min)/Nsamples;
	Ax_a3=(a3max-a3min)/Nsamples;
	Ax_a4=(a4max-a4min)/Nsamples;
	Ax_a5=(a5max-a5min)/Nsamples;
	Ax_a6=(a6max-a6min)/Nsamples;
	Ax_a7=(a7max-a7min)/Nsamples;
	Ax_a8=(a8max-a8min)/Nsamples;
	Ax_a10=(a10max-a10min)/Nsamples;
	Ax_a11=(a11max-a11min)/Nsamples;

	for(i=0;i<Nsamples;i++){
		c1[i]=i;
		c2[i]=i;
		c3[i]=i;
		c4[i]=i;
		c5[i]=i;
		c6[i]=i;
		c7[i]=i;
		c8[i]=i;
		c10[i]=i;
		c11[i]=i;
	}

	for(i=0;i<Nsamples+1;i++){
	
		A1[i]=a1min+i*Ax_a1;
		A2[i]=a2min+i*Ax_a2;
		A3[i]=a3min+i*Ax_a3;
		A4[i]=a4min+i*Ax_a4;
		A5[i]=a5min+i*Ax_a5;
		A6[i]=a6min+i*Ax_a6;
		A7[i]=a7min+i*Ax_a7;
		A8[i]=a8min+i*Ax_a8;
		A10[i]=a10min+i*Ax_a10;
		A11[i]=a11min+i*Ax_a11;
	}
	
	shuffle(c1,Nsamples);
	shuffle(c2,Nsamples);
	shuffle(c3,Nsamples);
	shuffle(c4,Nsamples);
	shuffle(c5,Nsamples);
	shuffle(c6,Nsamples);
	shuffle(c7,Nsamples);
	shuffle(c8,Nsamples);
	shuffle(c10,Nsamples);
	shuffle(c11,Nsamples);
	
	for(i=0;i<Nsamples;i++){
		S[i][0]=A1[c1[i]]+(A1[c1[i]+1]-A1[c1[i]])*r2();
		S[i][1]=A2[c2[i]]+(A2[c2[i]+1]-A2[c2[i]])*r2();
		S[i][2]=A3[c3[i]]+(A3[c3[i]+1]-A3[c3[i]])*r2();
		S[i][3]=A4[c4[i]]+(A4[c4[i]+1]-A4[c4[i]])*r2();
		S[i][4]=A5[c5[i]]+(A5[c5[i]+1]-A5[c5[i]])*r2();
		S[i][5]=A6[c6[i]]+(A6[c6[i]+1]-A6[c6[i]])*r2();
		S[i][6]=A7[c7[i]]+(A7[c7[i]+1]-A7[c7[i]])*r2();
		S[i][7]=A8[c8[i]]+(A8[c8[i]+1]-A8[c8[i]])*r2();
		S8[i]= i % 5 +1;
		S[i][9]=A10[c10[i]]+(A10[c10[i]+1]-A10[c10[i]])*r2();
		S[i][10]=A11[c11[i]]+(A11[c11[i]+1]-A11[c11[i]])*r2();
	}

	fprintf(f,"%d\n",Nsamples);

	for(i=0;i<Nsamples;i++){
		fprintf(f,"%f %f %f %f %f %f ",S[i][0],S[i][1],S[i][2],S[i][3],S[i][4],S[i][5]);
		fprintf(f,"%f %f %d %f %f\n",S[i][6],S[i][7],S8[i],S[i][9],S[i][10]);		
	}
	fclose(f);

}
