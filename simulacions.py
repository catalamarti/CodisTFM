from subprocess import call
import numpy as np
import xlsxwriter

with open('./data/sample.txt') as f:
    Nsamples = int(f.readline());
    array = [[float(x) for x in line.split()] for line in f];
f.close();

Nsimulacions = 1;
Nyears = 10;


workbook = xlsxwriter.Workbook('./data/results.xlsx');
worksheet1 = workbook.add_worksheet('parameters');
worksheet2 = workbook.add_worksheet('n_sick');
worksheet3 = workbook.add_worksheet('n_new_infected');
worksheet4 = workbook.add_worksheet('n_heal_infected');
worksheet5 = workbook.add_worksheet('n_dead_infected');
worksheet6 = workbook.add_worksheet('n_new_treatment');
worksheet7 = workbook.add_worksheet('n_new_treated');
worksheet8 = workbook.add_worksheet('n_recovered');

worksheet1.write(0,0,'f_HIV');
worksheet1.write(0,1,'f_diabetes');
worksheet1.write(0,2,'f_smoking');
worksheet1.write(0,3,'diagnose_mean_authocton');
worksheet1.write(0,4,'diagnose_mean_foreign');
worksheet1.write(0,5,'p_abandon');
worksheet1.write(0,6,'p_infect');
worksheet1.write(0,7,'p_sicken_all');
worksheet1.write(0,8,'f_sicken');
worksheet1.write(0,9,'f_sicken_child');
worksheet1.write(0,10,'f_sicken_young');

for j in range(0,Nyears):
	worksheet2.write(0,j,'year: ' + str(j+1));
	worksheet3.write(0,j,'year: ' + str(j+1));
	worksheet4.write(0,j,'year: ' + str(j+1));
	worksheet5.write(0,j,'year: ' + str(j+1));
	worksheet6.write(0,j,'year: ' + str(j+1));
	worksheet7.write(0,j,'year: ' + str(j+1));
	worksheet8.write(0,j,'year: ' + str(j+1));

for nsamp in range(0,Nsamples):

	f2 = open('./data/input.txt','w');
	for i in range(0, 11):
		f2.write('{:f} '.format(array[nsamp][i]));
	f2.close(); 

	mitjanes = np.zeros((Nyears,7));

	for i in range(0,Nsimulacions):
	
		call(["./a.out"]);

		f3 = open('./data/output.txt','r');
		out = [[float(x) for x in line.split()] for line in f3];
		f3.close();

		mitjanes = mitjanes + out;

		print 'Finished simulation: ' + str(i+1) + ' of ' + str(Nsimulacions) + ', using parameter set: ' + str(nsamp+1) + ' of ' + str(Nsamples) 

	mitjanes = mitjanes / Nsimulacions;

	for i in range(0,11):
		worksheet1.write(nsamp+1,i,array[nsamp][i])
	for i in range(0,Nyears):
		worksheet2.write(nsamp+1,i,mitjanes[i][0])
		worksheet3.write(nsamp+1,i,mitjanes[i][1])
		worksheet4.write(nsamp+1,i,mitjanes[i][2])
		worksheet5.write(nsamp+1,i,mitjanes[i][3])
		worksheet6.write(nsamp+1,i,mitjanes[i][4])
		worksheet7.write(nsamp+1,i,mitjanes[i][5])
		worksheet8.write(nsamp+1,i,mitjanes[i][6])

workbook.close();
