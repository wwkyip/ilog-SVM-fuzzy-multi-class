//Author: Wai Kuan Yip
//Date: 6 August 2010
//SoftMargin hyperplane SVM implementation
//Amended: 10 Aug Read from test and train files
//Amended: 11 Aug extension to m-class
//
//Amended: 14 Sept 2010
//Change to basic hyperplane fuzzy SVM according to Zhang et al., A novel fuzzy compensation multi-class support vector machine (2007)

#include <ilcplex/ilocplex.h>
#include "matrix.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

ILOSTLBEGIN

//Global variables

double polyKernelS = 2.0; //default
double gaussKernelQ = 1; //default
double rbfKernelD = 1;

const int noTrainSamples = 1360; 
const int noTestSamples =  884; 
const int dim = 135; 
string inpath = "/Users/motionlab_macpro/Documents/svmcode2/softmarginFuzzy/";

double C = 0.97;
const int noClass = 17; ;

double b[noClass];
int Ns;
double valAlpha[noTrainSamples][noClass]; //post solving solution
double m[noTrainSamples][noClass];

//prototype
double dotprod(matrix<double> & mat1, unsigned int i, matrix<double> & mat2, unsigned int j, int kernel);
void runSolver(int iClass);
int findMax(double ySVM[][noClass], int i); 
void updateM();

//data matrix
double y[noTrainSamples]; //original Y
double yByClass[noTrainSamples][noClass]; //Y by class, updated with 1, -1 according to class
double ySVM[noTestSamples][noClass]; //interim Y after each SVM rounds
double yFinal[noTestSamples]; //final Y
bool s[noTrainSamples][noClass];
double center[dim][noClass];

matrix <double> data(noTrainSamples, dim);
matrix <double> test(noTestSamples, dim);
matrix <double> dotprodMat(noTrainSamples,noTrainSamples);

//input file
ifstream trainFile((inpath + "train.txt").c_str());
ifstream testFile((inpath + "test.txt").c_str());

//output file
ofstream outFile((inpath + "testOutput.txt").c_str(),ios::app);

//dotprod function definition
void updateM(){
	
	double noD, max, dist;
	
	//for every class, find the mean
	for(int c=0; c<noClass; c++){
		//find the mean of the class c
		for(int d=0; d<dim; d++){
			center[d][c] = 0;
			noD = 0;
			for(int i=0; i<noTrainSamples; i++){
				if(y[i] == (c+1)){
					center[d][c] += data[i][d];
					noD ++;
				}
			}
			center[d][c] /= noD;
		}
		//find the maximum distance to the class center
		max = 0;
		for(int i=0; i<noTrainSamples; i++){
			dist = 0;
			if (y[i] == (c+1)){
				for(int d=0; d<dim; d++){
					dist += pow(data[i][d] - center[d][c],2.0);
				}
			}
			if(dist > max)
				max = dist;
		}
		//find the distance to the center of class c
		for(int i=0; i<noTrainSamples; i++){
			dist = 0;
			if(y[i] == (c+1)){
				for(int d=0; d<dim; d++){
					dist += pow(data[i][d] - center[d][c],2.0);
				}
			}
			m[i][c] = 1 - dist/max;
		}
	}
}

double dotprod(matrix<double> & mat1, unsigned int i, matrix<double> & mat2, unsigned int j, int kernel){
	double dotprod = 0.0;
	for(unsigned int k=0; k< dim; k++){
		if(kernel==1)
			dotprod += mat1[i][k]*mat2[j][k];
		else if(kernel==2)
			dotprod += pow(mat1[i][k]-mat2[j][k],2);
		else
			dotprod += pow(mat1[i][k]-mat2[j][k],2);			
	}
	if(kernel==1)
		return( pow(1 + dotprod, polyKernelS) );
	else if(kernel==2)
		return( exp(-1*gaussKernelQ*dotprod) );
	else
		return( exp(-0.5*dotprod/pow(rbfKernelD,2.0)) );	
}

int findMax(double ySVM[][noClass], int i){
	int maxIndex = 0;
	for(int iClass=0; iClass<noClass; iClass++){
		if(ySVM[i][iClass]>ySVM[i][maxIndex]){
			maxIndex = iClass;
		}
	}
	return maxIndex;
}

void runSolver(int iClass){
	
	//USE QP for optimization
	//FIND alpha based on MAX sum(alpha[i]) - 0.5*transpose(alpha)*y[i]*y[j]*dotproduct(x[i],x[j])alpha
	//s.t. alpha[i] >= 0 and sum(alpha[i]*y[i]) = 0
	
	IloEnv env; 
	try{ 
		
		IloModel model(env); 
		
		IloNumVarArray w(env); 
		for(int i=0; i<dim; i++) 
			w.add(IloNumVar(env,-1000.0,1000.0)); 
		
		IloNumVarArray alpha(env);
		for(int i=0; i<noTrainSamples; i++)
			alpha.add(IloNumVar(env,0,C*m[i][iClass])); 
		
		//objective function
		IloExpr expr1(env);
		for(int i=0; i<noTrainSamples; i++)
			expr1 += alpha[i];
		
		IloExpr expr2(env);
		for(int i=0; i<noTrainSamples; i++)
			for(int j=0; j<noTrainSamples; j++)
				expr2 += alpha[i]*alpha[j]*yByClass[i][iClass]*yByClass[j][iClass]*dotprodMat[i][j];		
		
		model.add(IloMaximize(env,expr1 - 0.5*expr2)); 
		
		//constraint sum(alpha[i]*y[i])=0
		IloExpr sumAlphaY(env);
		for(int i=0; i<noTrainSamples; i++){
			sumAlphaY += alpha[i]*yByClass[i][iClass];
		}
		model.add(sumAlphaY == 0.0);
		
		//Solve it!
		IloCplex cplex(model); 
		if(!cplex.solve()){ 
			env.error()<<"Failed to optimize LP."<<endl; 
			throw(-1); 
		} 
		
		//IloNumArray valAlpha(env); 
		//env.out()<<"Solution status="<<cplex.getStatus()<<endl; 
		//env.out()<<"Solution value="<<cplex.getObjValue()<<endl; 
		for(int i=0; i<noTrainSamples; i++){
			valAlpha[i][iClass] = cplex.getValue(alpha[i]); 
			//env.out() << "\nAlpha "<<i << ":" <<valAlpha[i][iClass];
		}
	} 
	
	catch(IloException&e){ 
		cerr<<"Concert exception caught:"<<e<<endl; 
	} 
	catch(...){ 
		cerr<<"Unknown exception caught"<<endl; 
	} 	
	env.end();
}

////////////////////////////////////////////
//MAIN
////////////////////////////////////////////

int main (int argc, char * const argv[]) {
	
	//input: (1) 1: polynomial, 2: for gaussian, 3: for RBF
	//		 (2) parameter
	//       (3) C
	int kernel = atoi(argv[1]);
	if(atoi(argv[1])==1)
		polyKernelS = atof(argv[2]);
	
	if(atoi(argv[1])==2)
		gaussKernelQ = atof(argv[2]);

	if(atoi(argv[1])==3)
		rbfKernelD = atof(argv[2]);

	C = atof(argv[3]);

	//read data into matrix
	string nextToken;
	int c = 0;
	while (trainFile>>nextToken){
		if((c % (dim+1)) != dim)
			data[(unsigned int)(c/(dim+1))][(unsigned int)(c % (dim+1))] = atof(nextToken.c_str());	
		else
			y[(unsigned int)(c/(dim+1))] = atoi(nextToken.c_str());
		c++;
	}

	c = 0;
	while (testFile>>nextToken){
		if((c % (dim+1)) != dim)
			test[(unsigned int)(c/(dim+1))][(unsigned int)(c % (dim+1))] = atof(nextToken.c_str());	
		c++;
	}
	
	//update the fuzzy membership m
	updateM();
	
	//initialize the final output vector
	for(int i=0; i<noTestSamples; i++)
		yFinal[i]=-1;
	
	//CREATE dotprod where dotprod[i,j] = x[i].x[j]
	for(unsigned int i=0; i<noTrainSamples; i++){
		for(unsigned int j=0; j<noTrainSamples; j++){
			dotprodMat[i][j] = dotprod(data, i, data, j, kernel);
		}
	}

	//M-CLASS EXTENSION
	for(int iClass=0; iClass<noClass; iClass++){
		
	//initialize the support indices to false
	for(int i=0; i<noTrainSamples; i++){
		s[i][iClass] = false;
		if((y[i]-1)==iClass)
			yByClass[i][iClass] = 1;
		else
			yByClass[i][iClass] = -1;	
	}
		
	runSolver(iClass);
	
	//FIND support vectors (indices s) s.t. alpha[i] > 0 
	Ns = 0;
	for(int i=0; i<noTrainSamples; i++){
		if((0 < valAlpha[i][iClass]) && (valAlpha[i][iClass] < C)){
			s[i][iClass] = true;
			Ns++;
		}
	}
	
	//calculate b = 1/Ns * sum(y[s] - sum(alpha[m]*y[m]*x[m].x[s]) )
	b[iClass] = 0.0;
	double temp = 0.0;
	for (int i=0; i<noTrainSamples; i++){
		if(s[i][iClass]==true){
			temp = 0.0;
			for(int m=0; m<noTrainSamples; m++){
				if(s[m][iClass]==true){
					temp = valAlpha[m][iClass]*yByClass[m][iClass]*dotprodMat[m][i];
				}
			}
			b[iClass] += yByClass[i][iClass] - temp;
		}	
	}
	b[iClass] = b[iClass]/Ns;
	
	//new point x is y = sgn(sum(alpha[i]*y[i]*test*x[i]) + b)
	double tempDP;
					 
	for(int i=0; i<noTestSamples; i++){
		tempDP = 0.0;
		for(int j=0; j<noTrainSamples; j++){			
			tempDP += valAlpha[j][iClass]*yByClass[j][iClass]*dotprod(test, i, data, j, kernel);
		}
		ySVM[i][iClass] = tempDP + b[iClass];
		
	}
	
	}//END FIND SVMs

	for(int i=0; i<noTestSamples; i++){
		yFinal[i] = findMax(ySVM,i);		
	}
	
	//this is to aggregate the results 
	int aggResults[noClass];
	int noSamplePerClass = noTestSamples/noClass;
	
	for(int i=0; i<noClass; i++)
		aggResults[i] = 0;
	
	outFile << "\n" << atoi(argv[1]) << "," << atof(argv[2]) << "," << atof(argv[3]);
	for(int i=0; i<noTestSamples; i++){
		if(yFinal[i]==(i/noSamplePerClass)){
			aggResults[i/noSamplePerClass]++;
		}
	}
	for(int i=0; i<noClass; i++)
		outFile<<","<<aggResults[i];
	outFile.close();

	return 0;
}
