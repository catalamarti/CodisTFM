/***********************************************************
 * SIMULATOR OF AN INDIVIDUAL BASED MODEL OF TUBERCULOSIS  *
 * 				DYNAMICS AT A CITY LEVEL  v1.0             *
 * Developed as a part of a Bachelor's thesis within a     *
 * joint research project of the BIOCOM-SC and inLAB FIB   *
 * departments of the Unversitat Politècnica de Catalunya  *
 * Developed by Bernat Puig Camps, June 2017               *
 * Contact: bernatpuig@gmail.com                           *
 ***********************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <math.h>

#include "IBMheader.h"
#include "list.h"

int grid[CELLS][CELLS] = {{0}};
/* Initial values */
int num_healthy = 100488;
int num_infected = 4500;
int num_sick = 3;
int num_treatment = 11;
int num_treated = 121;

int n_new_sick;
int n_new_infected;
int n_heal_infected;
int n_dead_infected;
int n_new_treatment;
int n_new_treated;
int n_recovered;

/* Defines age_groups considered: 0-5, 6-15, ...*/
int age_groups[11] = {-1,5,15,25,35,45,55,65,75,85,90};

/* Probability of a generic infected/sickened to be on a given age group */
float p_infected_age[10] = {0.0220,0.0558,0.1201,0.2521,0.2775,0.1827,0.0575,0.0186,0.0118,0.0019};
float p_sickened_age[10] = {0.0270,0.0315,0.1393,0.3000,0.2225,0.1169,0.0719,0.0382,0.0393,0.0134};


/* Probabilities of gender and origin */
float p_infected_male = 0.6802;
float p_infected_foreign = 0.7357;
float p_sickened_male = 0.6978;
float p_sickened_foreign = 0.7157;

/* Characteristic times in days */
int t_infected_max = 7*365;
int t_treatment_min = 15;
int t_treatment_max = 180;
int t_to_healthy = 720; // Time after diagnose to consider healthy.

/* Probabilities of risk factors */
float p_HIV = 0.0042; 
float p_diabetes = 0.241;
float p_smoking = 0.056;

/* Probability of smear positive case */
float p_smear = 0.22;

/* Diagnose delay mean and std.
 * Mean has two components [autochton,foreign]
 */
 int diagnose_mean[2];
 int diagnose_std = 4;

/* Probability to abandon treatment before finishing it */
float p_abandon;
float p_relapse_min = 0.01;

/* Probability to infect an close individual */
float p_infect;

/* Parameters related to age class for new infected.
 * youngster (0-20), adult (20-60), elder (60-) */

/* Boundaries in days between age classes */
int age_class_boundaries[2] = {20*365, 60*365};

/* Probability to belong to a certain age class and gender
 * according to characteristics of the infecting case.
 * 			   [youngster] [male adult] [female adult] [elder] (Case)
 * [youngster]
 * [adult]
 * [elder]
 * (Contact)
 */
float p_age_class[3][4] = {{0.30, 0.05, 0.12, 0.05},
						   {0.65, 0.94, 0.85, 0.45},
						   {0.05, 0.01, 0.03, 0.50}};

/* Age groups in each age class and probability of each one */
int young_age_groups[5] = {-1,5,10,15,20};
float p_young_age[4] = {0.1970, 0.2121, 0.2879, 0.3030};

int adult_age_groups[5] = {19, 30, 40, 50, 60};
float p_adult_age[4] = {0.2581, 0.3557, 0.2419, 0.1443};

int elder_age_groups[4] = {59, 70, 80, 90};
float p_elder_age[3] = {0.6667, 0.2121, 0.1212};

/* Probability of origin and gender of new infected depending
 * of origin and gender of the infector
 * 					[Native male] [Native female] [Foreign male] [Foreign female] (Case)
 * [Native male]
 * [Native female]
 * [Foreign male]
 * [Foreign female]
 * (Contact)
 */
float p_origin_gender[4][4] = {{0.40, 0.35, 0.11, 0.08},
							   {0.28, 0.33, 0.03, 0.06},
							   {0.19, 0.16, 0.65, 0.49},
							   {0.13, 0.16, 0.21, 0.37}};

/* Probability to become sick depending on time infected */
float p_sicken_all;
float p_sicken[7];
int f_sicken;

/* Factors that, if present, multiply p_sicken */
float f_sicken_child; // [0,5) years
float f_sicken_young;  // [5,15) years
float f_sicken_HIV;
float f_sicken_diabetes;
float f_sicken_smoking;

/* Superior value of age group and probability to die for each group */
int p_die_age[2] = {10*365, 65*365};
double p_die[3] = {6.877e-7, 4.65e-6, 1.2299e-4};
double p_die_sick = {2.192e-4};

void main(){

	/* Initialize the random seed to generate different simulations */
	clock_t begin = clock();
	srand(time(NULL));
	
	int ii;
	FILE *inputfile;
	inputfile = fopen("data/input.txt","r");
	fscanf(inputfile,"%f",&f_sicken_HIV);
	fscanf(inputfile,"%f",&f_sicken_diabetes);
	fscanf(inputfile,"%f",&f_sicken_smoking);
	float diag0;
	fscanf(inputfile,"%f",&diag0);
	diagnose_mean[0] = (int)diag0;
	float diag1;
	fscanf(inputfile,"%f",&diag1);
	diagnose_mean[1] = (int)diag1;
	fscanf(inputfile,"%f",&p_abandon);
	fscanf(inputfile,"%f",&p_infect);
	fscanf(inputfile,"%f",&p_sicken_all);
	fscanf(inputfile,"%d",&f_sicken);
	fscanf(inputfile,"%f",&f_sicken_child);
	fscanf(inputfile,"%f",&f_sicken_young);
	fclose(inputfile);
	
	switch(f_sicken) {
		case 1:
			p_sicken[0] = 0.0280 * p_sicken_all;
			p_sicken[1] = 0.0220 * p_sicken_all;
			p_sicken[2] = 0.0177 * p_sicken_all;
			p_sicken[3] = 0.0134 * p_sicken_all;
			p_sicken[4] = 0.0093 * p_sicken_all;
			p_sicken[5] = 0.0060 * p_sicken_all;
			p_sicken[6] = 0.0036 * p_sicken_all;
		case 2:
			p_sicken[0] = 0.0250 * p_sicken_all;
			p_sicken[1] = 0.0250 * p_sicken_all;
			p_sicken[2] = 0.0100 * p_sicken_all;
			p_sicken[3] = 0.0100 * p_sicken_all;
			p_sicken[4] = 0.0100 * p_sicken_all;
			p_sicken[5] = 0.0100 * p_sicken_all;
			p_sicken[6] = 0.0100 * p_sicken_all;
		case 3:
			p_sicken[0] = 0.0285 * p_sicken_all;
			p_sicken[1] = 0.0215 * p_sicken_all;
			p_sicken[2] = 0.0163 * p_sicken_all;
			p_sicken[3] = 0.0123 * p_sicken_all;
			p_sicken[4] = 0.0093 * p_sicken_all;
			p_sicken[5] = 0.0070 * p_sicken_all;
			p_sicken[6] = 0.0053 * p_sicken_all;
		case 4:
			p_sicken[0] = 0.0250 * p_sicken_all;
			p_sicken[1] = 0.0250 * p_sicken_all;
			p_sicken[2] = 0.0177 * p_sicken_all;
			p_sicken[3] = 0.0126 * p_sicken_all;
			p_sicken[4] = 0.0089 * p_sicken_all;
			p_sicken[5] = 0.0045 * p_sicken_all;
			p_sicken[6] = 0.0034 * p_sicken_all;
		case 5:
			p_sicken[0] = 0.0270 * p_sicken_all;
			p_sicken[1] = 0.0230 * p_sicken_all;
			p_sicken[2] = 0.0188 * p_sicken_all;
			p_sicken[3] = 0.0145 * p_sicken_all;
			p_sicken[4] = 0.0101 * p_sicken_all;
			p_sicken[5] = 0.0056* p_sicken_all;
			p_sicken[6] = 0.0010 * p_sicken_all;
	}
	FILE *outputfile;
	outputfile = fopen("data/output.txt","w");

	int i, t, t_max = 365;
	int yearmax = 10;
	int j,k,count = 0;

	/* List for each state */
	List *Elist = malloc(sizeof(List));
	List *Ilist = malloc(sizeof(List));
	List *Tlist = malloc(sizeof(List));
	List *Rlist = malloc(sizeof(List));

	/* Nodes to read the list */
	ListNode *node, *temp;


	initialize_simulation(Elist, Ilist, Tlist, Rlist);

	/* Initialize grid */
	for(j = 0; j < CELLS; j++){
		for(k = 0; k < CELLS; k++){
			count += grid[j][k];
		}
	}
	
	/*
	printf("Tot: %d, ",count+Elist->logicalLength+Ilist->logicalLength+Tlist->logicalLength+Rlist->logicalLength);
	printf("S: %d, ",count);

	printf("E: %d, ", Elist->logicalLength);
	printf("I: %d, ", Ilist->logicalLength);
	printf("T: %d, ", Tlist->logicalLength);
	printf("R: %d\n", Rlist->logicalLength);
	*/

	for(i = 0; i < yearmax; i++){
		/*Years*/
		/* Restart counters each year */
		n_new_sick = 0;
		n_new_infected = 0;
		n_heal_infected = 0;
		n_dead_infected = 0;
		n_new_treatment = 0;
		n_new_treated = 0;
		n_recovered = 0;

		for(t = 0; t < t_max; t++){
			/* Days */
			/* Sick people dynamics */
			node = Ilist->head;


			while(node != NULL) {
				temp = node->next;
				sick_update(Ilist,node,Elist,Tlist);
				node = temp;
			}

			/* Infected people dynamics */
			node = Elist->head;

			while(node != NULL) {
				temp = node->next;
				infected_update(Elist,node,Ilist);
				node = temp;
			}

			/* Treatment people dynamics */
			node = Tlist->head;

			while(node != NULL) {
				temp = node->next;
				treatment_update(Tlist,node,Rlist);
				node = temp;
			}

			/* Treated people dynamics */
			node = Rlist->head;

			while(node != NULL) {
				temp = node->next;
				treated_update(Rlist,node,Ilist);
				node = temp;
			}

			/* Healthy people dynamics */
			count = 0;
			for(j = 0; j < CELLS; j++){
				for(k = 0; k < CELLS; k++){
					count += grid[j][k];
				}
			}
			move_healthy();
		}
		fprintf(outputfile,"%d %d %d %d %d %d %d\n", n_new_sick, n_new_infected, n_heal_infected, n_dead_infected, n_new_treatment, n_new_treated, n_recovered);
		/*printf("Infected: %d, ",Elist->logicalLength);
		printf("new sick: %d\n",n_new_sick);
		*/
	}
	fclose(outputfile);
	clock_t end = clock();
	double time_spent = (double)(end - begin) /CLOCKS_PER_SEC;
	printf("Elapsed: %f seconds \n",time_spent);
}

void print_num_states(List *Elist, List *Ilist, List *Tlist, List *Rlist, int count)
{
	printf("Tot: %d, ",count+Elist->logicalLength+Ilist->logicalLength+Tlist->logicalLength+Rlist->logicalLength);
	printf("S: %d, ",count);
	printf("E: %d, ", Elist->logicalLength);
	printf("I: %d, ", Ilist->logicalLength);
	printf("T: %d, ", Tlist->logicalLength);
	printf("R: %d\n", Rlist->logicalLength);
}

void print_yearly_parameters(void)
{
	printf("new sick: %d, ",n_new_sick);
	printf("new infected: %d, ", n_new_infected);
	printf("heal infected: %d, ", n_heal_infected);
	printf("new treatment %d, ", n_new_treatment);
	printf("new treated: %d, ", n_new_treated);
	printf("recovered: %d, ", n_recovered);
	printf("dead infected: %d\n", n_dead_infected);
}

void list_new(List *list, int elementSize, freeFunction freeFn)
{
  assert(elementSize > 0);
  list->logicalLength = 0;
  list->elementSize = elementSize;
  list->head = list->tail = NULL;
  list->freeFn = freeFn;
}
 
void list_destroy(List *list)
{
  ListNode *current;
  while(list->head != NULL) {
    current = list->head;
    list->head = current->next;

    if(list->freeFn) {
      list->freeFn(current->data);
    }

    free(current->data);
    free(current);
  }
}
 
void list_prepend(List *list, void *element)
{
  ListNode *node = malloc(sizeof(ListNode));
  node->data = malloc(list->elementSize);
  node->prev = NULL;

  memcpy(node->data, element, list->elementSize);

  if(list->logicalLength == 0) {
    list->head = list->tail = node;
  } else {
    node->next = list->head;
    list->head->prev = node;
    list->head = node;
  }

  list->logicalLength++;
}
 
void list_append(List *list, void *element)
{
  ListNode *node = malloc(sizeof(ListNode));
  node->data = malloc(list->elementSize);
  node->next = NULL;

  memcpy(node->data, element, list->elementSize);

  if(list->logicalLength == 0) {
    list->head = list->tail = node;
  } else {
    node->prev = list->tail;
    list->tail->next = node;
    list->tail = node;
  }

  list->logicalLength++;
}
 
void list_for_each(List *list, ListIterator iterator)
{
  assert(iterator != NULL);
 
  ListNode *node = list->head;
  bool result = TRUE;
  while(node != NULL && result) {
    result = iterator(node->data);
    node = node->next;
  }
}

void list_del_node(List *list, ListNode *node)
{
  assert(list->logicalLength > 0);

  if(node == list->head && node == list->tail) {
    list->head = NULL;
    list->tail = NULL;
  } else if(node == list->head) {
    list->head = node->next;
    list->head->prev = NULL;
  } else if(node == list->tail) {
    list->tail = node->prev;
    list->tail->next = NULL;
  } else {
    node->prev->next = node->next;
    node->next->prev = node->prev;
  }

  list->logicalLength--;

  free(node->data);
  free(node);
}
 
void list_head(List *list, void *element, bool removeFromList)
{
  assert(list->head != NULL);
 
  ListNode *node = list->head;
  memcpy(element, node->data, list->elementSize);
 
  if(removeFromList) {
    list->head = node->next;
    list->logicalLength--;
 
    free(node->data);
    free(node);
  }
}
 
void list_tail(List *list, void *element)
{
  assert(list->tail != NULL);
  ListNode *node = list->tail;
  memcpy(element, node->data, list->elementSize);
}
 
int list_size(List *list)
{
  return list->logicalLength;
}

void move(Position *pos){
	double r1, p_lin, p_tot;
	int r2;
	int lin_disp_x[4] = {0,1,0,-1};
	int lin_disp_y[4] = {1,0,-1,0};
	int diag_disp_x[4] = {1,1,-1,-1};
	int diag_disp_y[4] = {1,-1,1,-1};
	int i;

	p_tot = 4 + 4/sqrt(2);
	p_tot = 4 + 4;
	p_lin = 4/p_tot;

	r1 = randdb(0,1);
	r2 = randint(0,4);

	if (r1 < p_lin){
		pos->x += lin_disp_x[r2];
		pos->y += lin_disp_y[r2];
	} else{
		pos->x += diag_disp_x[r2];
		pos->y += diag_disp_y[r2];
	}

	boundary_condition(pos);
}

void move_healthy()
{
	int i,j,k,n;
	Position pos;

	for(i = 0; i < CELLS; i++){
		for(j = 0; j < CELLS; j++){
			n = grid[i][j];

			for(k = 0; k < n; k++){
				pos.x = i;
				pos.y = j;

				grid[i][j] -= 1;

				move(&pos);
				grid[pos.x][pos.y] += 1;
			}
		}
	}
}

void boundary_condition(Position *pos)
{
/* If a position is out of the grid, returns a new one
 * according to periodic boundary condition.
 */

	if (pos->x > (CELLS-1)){
		pos->x = 0;
	}

	else if (pos->x < 0){
		pos->x = CELLS - 1;
	}

	if (pos->y > (CELLS-1)){
		pos->y = 0;
	}

	else if (pos->y < 0){
		pos->y = CELLS - 1;
	}
}

double define_p_die(int age)
{
/* Define the probability to die according to age for all
 * individuals but undiagnosed sick.
 */
	double prob;

	if(age < p_die_age[0]) {
		prob = p_die[0];
	} else if(age < p_die_age[1]) {
		prob = p_die[1];
	} else {
		prob = p_die[2];
	}

	return prob;
}

/*************************************************************
 ****				  INFECTED DYNAMICS FUNCTIONS		  ****
 *************************************************************/

void infected_update(List *Elist, ListNode *node, List *Ilist)
{
/* Contains everything that has to be considered per infected node
 * each day.
 */

	Enode *infected = node->data;

	/* Check if it dies. If so, end function */
	if (infected_die(Elist, node)){
		return;
	}

	/* If not, grow and move */
	infected->age++;
	move(&(infected->pos));

	/* Check if it heals. If so, end function */
	if(infected_heal(Elist, node)) {
		return;
	}

	/* Check if it gets sick. If so, end function */
	if(infected_sicken(Elist, node, Ilist)) {
		return;
	}

	/* If it reaches here, advance by one the time infected */
	infected->t_infected++;
}

int infected_die(List *Elist, ListNode *node)
{
/* Check if an infected individual dies. If so, delete it, generate
 * a new healthy individual in a random position and return 1.
 * If it survives, return 0;
 */
	Enode *infected = node->data;
	double r = randdb(0,1);
	double prob = define_p_die(infected->age);

	if(r < prob) {
		Position pos = generate_position();
		new_healthy(pos);
		list_del_node(Elist,node);
		n_dead_infected++;
		return 1;
	} else {
		return 0;
	}
}

int infected_heal(List *Elist, ListNode *node)
{
/* Determine if an infected individual can be considered healthy
 * again. If so, generate a healthy in its position and return 1.
 * If not, return 0.
 */
	Enode *infected = node->data;

	if(infected->t_infected > t_infected_max) {
		new_healthy(infected->pos);
		list_del_node(Elist,node);
		n_heal_infected++;
		return 1;
	} else {
		return 0;
	}
}

int infected_sicken(List *Elist, ListNode *node, List *Ilist)
{
/* Given an infected individual determine if it gets sick.
 * If it does, delete infected node and generate sick node with
 * same characteristics.
 * Also determine if it can be considered healthy.
 */
	double prob, r;
	int i;
	float f; //factor to fit the desired behaviour

	r = randdb(0,1);
	Enode *infected = node->data;

	for(i = 0; i*365 < infected->t_infected; i++){
		;
	}

	prob = p_sicken[i-1]/365;

	if(infected->age < 5*365) {
		prob *= f_sicken_child;
	} else if(infected->age < 15*365) {
		prob *= f_sicken_young;
	}

	if(infected->HIV) {
		prob *= f_sicken_HIV;
	}
	if(infected->diabetes) {
		prob *= f_sicken_diabetes;
	}
	if(infected->smoking) {
		prob *= f_sicken_smoking;
	}

	f = 0.9;
	prob *= f;

	if(r < prob) {
		n_new_sick++;
		new_sick(Ilist, infected);
		list_del_node(Elist, node);
		return 1;
	} else {
		return 0;
	}
}

void new_sick(List *Ilist, Enode *infected)
{
/* Generate a sick node based on the characteristics
 * of a previously infected node.
 */
	Inode *new = malloc(sizeof(Inode));

	new->age = infected->age;
	new->foreign = infected->foreign;
	new->gender = infected->gender;
	new->smear = define_smear();
	new->t_sick = 0;
	define_diagnose(new);

	new->pos = infected->pos;

	list_append(Ilist, new);
}

/*************************************************************
 ****				 		SICK DYNAMICS FUNCTIONS				    ****
 *************************************************************/

void sick_update(List *Ilist, ListNode *node, List *Elist, List *Tlist)
{
/* Contains everything that has to be considered per sick node
 * each day.
 */

	Inode *sick = node->data;

	/* Check if it dies. If so, end function */
	if(sick_die(Ilist, node)) {
		return;
	}

	/* If not, grow and move */
	sick->age++;
	move(&(sick->pos));

	/* Check if it starts treatment. If so, end function */
	if(start_treatment(Ilist, node, Tlist)) {
		return;
	}

	/* If not, advance sick days and infect others */
	sick->t_sick++;
	infect(sick,Elist);
}

int sick_die(List *Ilist, ListNode *node)
{
/* Check if sick individual dies. If so, delete it, generate
 * a new healthy individual in a random position and return 1.
 * If it survives, return 0;
 */
	double r = randdb(0,1);
	Enode *infected = node->data;

	if(r < p_die_sick) {
		Position pos = generate_position();
		new_healthy(pos);
		list_del_node(Ilist,node);
		return 1;
	} else {
		return 0;
	}
}

int start_treatment(List *Ilist, ListNode *node, List *Tlist)
{
/* If the time sick of a sick individual has reached the 
 * diagnose delay time, move the node from sick to node under
 * treatment.
 */
	Inode *sick = node->data;

	if(sick->t_sick == sick->diagnose) {
		new_treatment(Tlist, sick);
		list_del_node(Ilist, node);
		n_new_treatment++;
		return 1;
	} else {
		return 0;
	}
}

void new_treatment(List *Tlist, Inode *sick)
{
/* Generate an individual under treatment from a sick one.
 * The sick evolves to treatment.
 */
	Tnode *new = malloc(sizeof(Tnode));

	new->age = sick->age;
	new->foreign = sick->foreign;
	new->gender = sick->gender;
	new->smear = sick->smear;
	new->t_treatment = 0;

	new->pos = sick->pos;

	list_append(Tlist, new);
}

void infect(Inode *sick, List *Elist)
{
/* Determine if there are healthy individuals susceptible
 * to be infected (in contact) and infect them with a given
 * probability.
 */
	Position look;
	double r, prob;
	int i, j, susceptibles;
	int square[4][2] = {{1,0},
					   	//{1,1},
					   	{0,1},
					   	//{-1,1},
					   	{-1,0},
					   	//{-1,-1},
					   	{0,-1}};
					   	//{1,-1}};

	prob = (1 + sick->smear)*p_infect;

	for(i = 0; i < 9; i++) {
		look = sick->pos;
		look.x += square[i][0];
		look.y += square[i][1];
		boundary_condition(&look);

		susceptibles = grid[look.x][look.y];

		for(j = 0; j < susceptibles; j++) {
			r = randdb(0,1);
			if (r < prob){
				new_infected(sick, Elist, look);
				grid[look.x][look.y] -= 1;
				n_new_infected++;
			}
		}
	}
}

void new_infected(Inode *sick, List *Elist, Position pos)
{
/* Generate a new infected from a healthy individual according to
 * characteristics determined by the case that infects.
 */
	Enode *new = malloc(sizeof(Enode));

	infected_age(new, sick);
	infected_gender_origin(new, sick);
	define_risk_factors(new);

	new->t_infected = 0;
	new->pos = pos;

	list_append(Elist, new);
}

void infected_age(Enode *node, Inode *infector)
{
/* Define age of a new infected node according
 * to the age of its infector.
 */
	int i = 0, j = 0;
	int age = infector->age;
	char gender = infector->gender;
	double r, prob;

	prob = 0;
	r = randdb(0,1);

/* Determine the age class. i-1 determines it */

	if (age < age_class_boundaries[0]){
		do {
			prob += p_age_class[i++][0];
		} while(prob < r);

	} else if (age < age_class_boundaries[1] && gender == 0) {
		do {
			prob += p_age_class[i++][1];
		} while(prob < r);

	} else if (age < age_class_boundaries[1] && gender == 1) {
		do {
			prob += p_age_class[i++][2];
		} while(prob < r);

	} else {
		do {
		prob += p_age_class[i++][3];
		} while(prob < r);
	}

/* Determine age group and age inside a class */

	prob = 0;
	r = randdb(0,1);

	switch(i-1){
		case 0:
			do {
				prob += p_young_age[j++];
			} while(prob < r);
			node->age = randint((young_age_groups[j-1]+1)*365, young_age_groups[j]*365);
			break;

		case 1:
			do {
				prob += p_adult_age[j++];
			} while(prob < r);
			node->age = randint((adult_age_groups[j-1]+1)*365, adult_age_groups[j]*365);
			break;

		case 2:
			do {
				prob += p_elder_age[j++];
			} while(prob < r);
			node->age = randint((elder_age_groups[j-1]+1)*365, elder_age_groups[j]*365);
			break;
	}
}

void infected_gender_origin(Enode *node, Inode *infector)
{
/* Define origin and gender of a new infected according to the
 * origin and gender of the infector.
 */
	int i = 0, aux;
	double r, prob;

	aux = infector->gender + 2*infector->foreign;
	// aux. 0: native male, 1: native female, 2: foreign male, 3: foreign female
			
	r = randdb(0.0,1.0);
	prob = 0;
	do {
		prob += p_origin_gender[i++][aux];
	} while(prob < r);

	switch(i-1){
		case 0:
			node->foreign = 0;
			node->gender = 0;
			break;

		case 1:
			node->foreign = 0;
			node->gender = 1;
			break;

		case 2:
			node->foreign = 1;
			node->gender = 0;
			break;
				
		case 3:
			node->foreign = 1;
			node->gender = 1;
			break;
	}
}

/*************************************************************
 ****				  TREATMENT DYNAMICS FUNCTIONS				 ****
 *************************************************************/

void treatment_update(List *Tlist, ListNode *node, List *Rlist)
{
/* Contains everything that has to be considered per node under
 * treatment each day.
 */

	Tnode *treatment = node->data;

	/* Check if it dies. If so, end function */
	if (treatment_die(Tlist, node)){
		return;
	}

	/* If not, grow and move */
	treatment->age++;
	move(&(treatment->pos));

	/* Check if it finishes treatment. If so, end function. */
	if(finish_treatment(Tlist, node, Rlist)) {
		return;
	}

	treatment->t_treatment++;
}

int treatment_die(List *Tlist, ListNode *node)
{
/* Check if an individual under treatment dies. If so, delete it, generate
 * a new healthy individual in a random position and return 1.
 * If it survives, return 0;
 */
	Tnode *treatment = node->data;
	double r = randdb(0,1);
	double prob = define_p_die(treatment->age);

	if(r < prob) {
		Position pos = generate_position();
		new_healthy(pos);
		list_del_node(Tlist,node);
		return 1;
	} else {
		return 0;
	}
}

int finish_treatment(List *Tlist, ListNode *node, List *Rlist)
{
/* Check if a given individual under treatment finishes or abandons
 * treatment. If so, evolve it to treated.
 */
	double r, prob;
	Tnode *treatment = node->data;

	r = randdb(0,1);
	prob = p_abandon/t_treatment_max; // Daily prob to abandon.

	if(r < prob || treatment->t_treatment == t_treatment_max) {
		new_treated(Rlist, treatment);
		list_del_node(Tlist, node);
		n_new_treated++;
		return 1;
	} else {
		return 0;
	}
}

void new_treated(List *Rlist, Tnode *treatment)
{
/* Move a node under treatment to treated.
 */
	Rnode *new = malloc(sizeof(Rnode));

	new->age = treatment->age;
	new->foreign = treatment->foreign;
	new->gender = treatment->gender;
	new->smear = treatment->smear;
	new->t_treatment = treatment->t_treatment;
	new->t_treated = 0;

	define_p_relapse(new);

	new->pos = treatment->pos;

	list_append(Rlist, new);
}

/*************************************************************
 ****				  TREATED DYNAMICS FUNCTIONS					 ****
 *************************************************************/

void treated_update(List *Rlist, ListNode *node, List *Ilist)
{
/* Contains everything that has to be considered per node under
 * treatment each day.
 */

	int t_left;
	double r, prob;
	Rnode *treated = node->data;

	/* Check if it dies. If so, end function */
	if(treated_die(Rlist, node)) {
		return;
	}

	/* If not, grow and move */
	treated->age++;
	move(&(treated->pos));

	/* Check either if it recovers, it relapses or it 
	 * continues as treated */
	t_left = t_to_healthy - treated->t_treatment;
	r = randdb(0,1);
	//prob = treated->p_relapse/t_left;
	prob = treated->p_relapse/714;

	if(treated->t_treated == t_left) {
		new_healthy(treated->pos);
		list_del_node(Rlist, node);
		n_recovered++;
	} else if(r < prob) {
		relapse(Ilist, treated);
		list_del_node(Rlist, node);
	} else {
		treated->t_treated++;
	}
}

int treated_die(List *Rlist, ListNode *node)
{
/* Check if a treated individual dies. If so, delete it, generate
 * a new healthy individual in a random position and return 1.
 * If it survives, return 0;
 */
	Tnode *treated = node->data;
	double r = randdb(0,1);
	double prob = define_p_die(treated->age);;

	if(r < prob) {
		Position pos = generate_position();
		new_healthy(pos);
		list_del_node(Rlist,node);
		return 1;
	} else {
		return 0;
	}
}


void relapse(List *Ilist, Rnode *treated)
{
/* Move a node under treatment back to sick due to
 * early abandon of treatment.
 */
	Inode *new = malloc(sizeof(Inode));

	new->age = treated->age;
	new->foreign = treated->foreign;
	new->gender = treated->gender;
	new->smear = treated->smear;
	new->t_sick = 0;
	define_diagnose(new);

	new->pos = treated->pos;

	list_append(Ilist, new);
}

void initialize_simulation(List *Elist, List *Ilist, List *Tlist, List *Rlist)
{
/* Initialize all type of individuals. Generate all the initial individuals.
 * Assign each one at the corresponding list or the grid.
 */

	int i;
	Position pos;

	setup_infected_list(Elist);
	setup_sick_list(Ilist);
	setup_treatment_list(Tlist);
	setup_treated_list(Rlist);

	for(i = 0; i < num_healthy; i++) {
		pos = generate_position();
		new_healthy(pos);
	}

	for(i = 0; i < num_infected; i++) {
		setup_new_infected(Elist);
	}

	for(i = 0; i < num_sick; i++) {
		setup_new_sick(Ilist);
	}

	for(i = 0; i < num_treatment; i++) {
		setup_new_treatment(Tlist);
	}

	for(i = 0; i < num_treated; i++) {
		setup_new_treated(Rlist);
	}
}

Position generate_position(void)
{
/* Generate a random position in the defined grid */
	Position new;

	new.x = randint(0, CELLS);
	new.y = randint(0, CELLS);

	return new;
}

void new_healthy(Position pos)
{
/* Generate a healthy individual in a given position.
 */
	grid[pos.x][pos.y] += 1;
}

void setup_infected_list(List *Elist)
{
/* Generate the list that will contain the infected individuals
 */
	list_new(Elist,sizeof(Enode),NULL);
}

void setup_new_infected(List *Elist)
{
/* Generate an infected individual with the characteristics
 * defined from the general population and append it to the
 * list of infected nodes.
 */
	Enode *new = malloc(sizeof(Enode));

	setup_infected_age(new);
	setup_infected_origin(new);
	setup_infected_gender(new);
	setup_t_infected(new);
	define_risk_factors(new);
	new->pos = generate_position();

	list_append(Elist, new);
}

void setup_infected_age(Enode *node)
{
/* Define age of a generated infected with the age distribution
 * obtained from data.
 */
	int i = 0;
	double prob = 0, r = randdb(0,1);
	
	do {
		prob += p_infected_age[i++];
	} while(prob < r);

	node->age = randint((age_groups[i-1]+1)*365, age_groups[i]*365);
}

void setup_infected_origin(Enode *node)
{
/* Define origin according to the probabilities defined
 */
	double r = randdb(0,1);

	node->foreign = 0;

	if(r < p_infected_foreign) {
		node->foreign = 1;
	}
}

void setup_infected_gender(Enode *node)
{
/* Define gender according to the probabilities defined
 */
	double r = randdb(0,1);

	node->gender = 1;

	if(r < p_infected_male) {
		node->gender = 0;
	}
}

void setup_t_infected(Enode *node)
{
/* Define the time that an individual has been in infected
 * state. Uniform between 0 and 7 years.
 */
	node->t_infected = randint(0,t_infected_max);
}

void define_risk_factors(Enode *node)
{
/* Defines the risk factors of an infected individual
 * according to the probabilities of the population
 */
	double r;

	node->HIV = 0;
	node->diabetes = 0;
	node->smoking = 0;

	r = randdb(0,1);
	if (r < p_HIV){
		node->HIV = 1;
	}

	r = randdb(0,1);
	if (r < p_diabetes){
		node->diabetes = 1;
	}

	r = randdb(0,1);
	if (r < p_smoking){
		node->smoking = 1;
	}
}

void setup_sick_list(List *Ilist)
{
/* Generate the list that will contain the sickened individuals
 */
	list_new(Ilist,sizeof(Inode),NULL);	
}

void setup_new_sick(List *Ilist)
{
/* Generate a sick individual with the characteristics
 * defined from the cases and append it to the
 * list of sickened nodes.
 */
	Inode *new = malloc(sizeof(Inode));

	new->age = setup_sickened_age();
	new->foreign = setup_sickened_origin();
	new->gender = setup_sickened_gender();
	new->smear = define_smear();
	define_diagnose(new);
	setup_t_sick(new);

	new->pos = generate_position();

	list_append(Ilist, new);
}

int setup_sickened_age(void)
{
/* Define age of a generated sickened with the age distribution
 * obtained from data.
 */
	int i = 0;
	double prob = 0, r = randdb(0,1);
	
	do {
		prob += p_sickened_age[i++];
	} while(prob < r);

	return randint((age_groups[i-1]+1)*365, age_groups[i]*365);
}

char setup_sickened_origin(void)
{
/* Define origin of sickened according to the probabilities defined
 */
	double r = randdb(0,1);

	if(r < p_sickened_foreign) {
		return 1;
	} else {
		return 0;
	}
}

char setup_sickened_gender(void)
{
/* Define gender according to the probabilities defined
 */
	double r = randdb(0,1);

	if(r < p_sickened_male) {
		return 0;
	} else {
		return 1;
	}
}

int define_smear(void)
{
/* Define if smear positive according to the probabilities defined
 */
	double r = randdb(0,1);

	if(r < p_smear) {
		return 1;
	} else {
		return 0;
	}
}

void define_diagnose(Inode *node)
{
/* Define the diagnose delay for a sick individual 
 * given its origin.
 */
	int mu = diagnose_mean[node->foreign];
	node->diagnose = randint_normal(mu,diagnose_std);
}

void setup_t_sick(Inode *node)
{
/* Define the time an individual have been sick.
 * Between 0 days and diagnose delay - 1.
 */
 	node->t_sick = randint(0,node->diagnose-1);
}

void setup_treatment_list(List *Tlist)
{
/* Generate the list that will contain the individuals
 * under treatment.
 */
	list_new(Tlist,sizeof(Tnode),NULL);	
}

void setup_new_treatment(List *Tlist)
{
/* Generate an individual under treatment with the characteristics
 * defined from the cases and append it to the
 * list of treatment nodes.
 */
	Tnode *new = malloc(sizeof(Tnode));

	new->age = setup_sickened_age();
	new->foreign = setup_sickened_origin();
	new->gender = setup_sickened_gender();
	new->smear = define_smear();
	setup_treatment_t_treatment(new);

	new->pos = generate_position();

	list_append(Tlist, new);
}

void setup_treatment_t_treatment(Tnode *node)
{
/* Define the time since starting treatment of an individual
 * under treatment.
 */

	node->t_treatment = randint(0,t_treatment_max);
}

void setup_treated_list(List *Rlist)
{
/* Generate the list that will contain the treated individuals
 */
	list_new(Rlist,sizeof(Rnode),NULL);	
}

void setup_new_treated(List *Rlist)
{
/* Generate a sick individual with the characteristics
 * defined from the cases and append it to the
 * list of sickened nodes.
 */
	Rnode *new = malloc(sizeof(Rnode));

	new->age = setup_sickened_age();
	new->foreign = setup_sickened_origin();
	new->gender = setup_sickened_gender();
	new->smear = define_smear();
	setup_treated_t_treatment(new);
	setup_t_treated(new);
	define_p_relapse(new);

	new->pos = generate_position();

	list_append(Rlist, new);
}

void setup_treated_t_treatment(Rnode *node)
{
/* Define the time since starting treatment of a treated
 * individual.
 */
	double r = randdb(0,1);

	if (r < p_abandon){
		node->t_treatment = randint(t_treatment_min, t_treatment_max);
	} else {
		node->t_treatment = t_treatment_max;
	}
}

void setup_t_treated(Rnode *node)
{
/* Define the time since finishing/abandoning treatment
 * for a treated individual.
 */
	int t_left = t_to_healthy - node->t_treatment;
	node->t_treated = randint(0,t_left);
}

void define_p_relapse(Rnode *node)
{
/* Define the probability to relapse of a treated individual.
 * If t_treated < min, will become sick again.
 * p_relapse straight line from 100% at 15 days to 1% at 180 days
 * of treatment.
 */
	int num = t_treatment_max - node->t_treatment;
	int den = t_treatment_max - t_treatment_min;

	node->p_relapse = ((double) num/den)*(1 - p_relapse_min) + p_relapse_min;
}

double randdb(double min, double max)
{
/* Generates a double pseudorandom number in [min,max).
 * Uniform distribution.
 */
	if(max < min) {
		printf("Error in random: min > max (min: %f, max: %f) \n",min,max);
		exit(1);
	}
		
	int r = rand();
	return min + (max - min) * (r / (double) RAND_MAX);	
}

int randint(int min, int max)
{
/* Generates an integer pseudorandom number in [min,max).
 * Uniform distribtion.
 */
	if(max < min) {
		printf("Error in random: min > max (min: %d, max: %d) \n",min,max);
		exit(1);
	}

	int r = rand();
	return min + (max - min) * (r / (double) RAND_MAX);
}

int randint_normal(int mu, int sig)
{
/* Generate an integer pseudorandom number with normal distribution.
 * Mean: mu; Std: sig.
 */
	double z1, z2, z, pi;

	z1 = randdb(0,1);
	z2 = randdb(0,1);

	pi = acos(-1.0);

	z = sqrt(-2.0 * log(z1)) * cos(2*pi*z2);

	return z*sig + mu;
}
