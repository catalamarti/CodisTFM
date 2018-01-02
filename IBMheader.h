#include "list.h"

#define CELLS 500

typedef struct position{
	int x;
	int y;
} Position;

typedef struct infected{ /* Infected individual. E node */
	int age;			 /* Age in days */
	int t_infected;		 /* Time infected */
	char foreign;		 /* 0: Native, 1: Foreign */
	char gender;		 /* 0: Man, 1: Woman */
	char HIV;			 /* 1: HIV-positive */
	char diabetes;		 /* 1: diabetes-positive */
	char smoking;		 /* 1: smoking-positive */
	Position pos;		 /* x,y coordinates in the grid */
} Enode;

typedef struct sick{ 	 /* Sick individual. I node */
	int age;			 /* Age in days */
	int t_sick;			 /* Time since sickening in days */
	int diagnose;		 /* Diagnose delay in days */
	char foreign;		 /* 0: Native, 1: Foreign */
	char gender;		 /* 0: Man, 1: Woman */
	int smear;			 /* 1: Smear positive */
	Position pos;		 /* x,y coordinates in the grid */
} Inode;

typedef struct treatment{/* Treatment individual. T node */
	int age;			 /* Age in days */
	int t_treatment;	 /* Time since diagnose in days */
	char foreign;		 /* 0: Native, 1: Foreign */
	char gender;		 /* 0: Man, 1: Woman */
	int smear;			 /* 1: Smear positive */
	Position pos;		 /* x,y coordinates in the grid */
} Tnode;

typedef struct treated{  /* Treated individual. R node */
	int age;			 /* Age in days */
	int t_treatment;	 /* Time spent in treatment in days */
	int t_treated;		 /* Time since finish/abandon treatment in days */
	char foreign;		 /* 0: Native, 1: Foreign */
	char gender;		 /* 0: Man, 1: Woman */
	int smear;			 /* 1: Smear positive */
	double p_relapse;	 /* Prob to relapse */
	Position pos;		 /* x,y coordinates in the grid */
} Rnode;

/* Declaration of random functions */
double randdb(double min, double max);
int randint(int min, int max);
int randint_normal(int mu, int sig);

/* Declaration of model setup functions */
void initialize_simulation(List *Elist, List *Ilist, List *Tlist, List *Rlist);
Position generate_position(void);
void new_healthy(Position pos);

/* Setup of infected nodes */
void setup_infected_list(List *Elist);
void setup_new_infected(List *Elist);
void setup_infected_age(Enode *node);
void setup_infected_origin(Enode *node);
void setup_infected_gender(Enode *node);
void setup_t_infected(Enode *node);
void define_risk_factors(Enode *node);

/* Setup of sick nodes */
void setup_sick_list(List *Ilist);
void setup_new_sick(List *Iist);
int setup_sickened_age(void);
char setup_sickened_origin(void);
char setup_sickened_gender(void);
int define_smear(void);
void define_diagnose(Inode *node);
void setup_t_sick(Inode *node);

/* Setup of treatment nodes */
void setup_treatment_list(List *Tlist);
void setup_new_treatment(List *Tlist);
void setup_treatment_t_treatment(Tnode *node);

/* Setup of treated nodes */
void setup_treated_list(List *Rlist);
void setup_new_treated(List *Rlist);
void setup_treated_t_treatment(Rnode *node);
void setup_t_treated(Rnode *node);
void define_p_relapse(Rnode *node);

/* Declaration of model dynamics functions */
/* Common functions */
void move(Position *pos);
void move_healthy();
void boundary_condition(Position *pos);
double define_p_die(int age);

/* Infected dynamics functions */
void infected_update(List *Elist, ListNode *node, List *Ilist);
int infected_die(List *Elist, ListNode *node);
int infected_heal(List *Elist, ListNode *node);
int infected_sicken(List *Elist, ListNode *node, List *Ilist);
void new_sick(List *Ilist, Enode *infected);

/* Sick dynamics functions */
void sick_update(List *Ilist, ListNode *node, List *Elist, List *Tlist);
int sick_die(List *Ilist, ListNode *node);
int start_treatment(List *Ilist, ListNode *node, List *Tlist);
void new_treatment(List *Tlist, Inode *sick);
void infect(Inode *sick, List *Elist);
void new_infected(Inode *sick, List *Elist, Position pos);
void infected_age(Enode *node, Inode *infector);
void infected_gender_origin(Enode *node, Inode *infector);

/* Treatment dynamics functions */
void treatment_update(List *Tlist, ListNode *node, List *Rlist);
int treatment_die(List *Tlist, ListNode *node);
int finish_treatment(List *Tlist, ListNode *node, List *Rlist);
void new_treated(List *Rlist, Tnode *treatment);

/* Treated dynamics functions */
void treated_update(List *Rlist, ListNode *node, List *Ilist);
int treated_die(List *Rlist, ListNode *node);
void relapse(List *Ilist, Rnode *treated);


extern int grid[CELLS][CELLS];

extern int num_healthy;
extern int num_infected;
extern int num_sick;
extern int num_treatment;
extern int num_treated;

extern int age_groups[11];
extern float p_infected_age[10];
extern float p_sickened_age[10];

extern float p_infected_male;
extern float p_infected_foreign;
extern float p_sickened_male;
extern float p_sickened_foreign;

extern int t_infected_max;
extern int t_treatment_min;
extern int t_treatment_max;
extern int t_to_healthy;

extern float p_HIV;
extern float p_diabetes;
extern float p_smoking;

extern float p_smear;

extern int diagnose_mean[2];
extern int diagnose_std;

extern float p_abandon;
extern float p_relapse_min;

extern float p_infect;

extern int age_class_boundaries[2];
extern float p_age_class[3][4];

extern int young_age_groups[5];
extern int adult_age_groups[5];
extern int elder_age_groups[4];

extern float p_young_age[4];
extern float p_adult_age[4];
extern float p_elder_age[3];

extern float p_origin_gender[4][4];

extern float p_sicken[7];
extern float f_sicken_child;
extern float f_sicken_young;
extern float f_sicken_HIV;
extern float f_sicken_diabetes;
extern float f_sicken_smoking;

extern int p_die_age[2];
extern double p_die[3];
extern double p_die_sick;

extern int n_new_sick;
extern int n_new_infected;
extern int n_heal_infected;
extern int n_dead_infected;
extern int n_new_treatment;
extern int n_new_treated;
extern int n_recovered;
