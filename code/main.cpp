#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <stdio.h>

#define INPUT_NUM 9//输入层节点数
#define HIDE_NUM 3//隐含层节点数
#define OUTPUT_NUM 8//输出层节点数

using namespace std;

typedef struct
{
	double alpha;//激活函数参数
	double beta;//学习率
	double weight1[HIDE_NUM][INPUT_NUM];//输入层到隐含层的权重
	double weight2[OUTPUT_NUM][HIDE_NUM];//隐含层到输出层的权重
}N_Network;

//初始化神经网络
void Init_Network(N_Network *net)
{
	net->alpha=1;
	net->beta=0.1;
	srand((unsigned)time(NULL));
	for(int j=0; j<HIDE_NUM; j++)
		for(int k=0; k<INPUT_NUM; k++)
			net->weight1[j][k]=(double)2*rand()/RAND_MAX-1;
			//net->weight1[j][k]=(double)1/INPUT_NUM;

	for(int i=0; i<OUTPUT_NUM; i++)
		for(int j=0; j<HIDE_NUM; j++)
			net->weight2[i][j]=(double)2*rand()/RAND_MAX-1;
			//net->weight2[i][j]=(double)1/HIDE_NUM;
}

//预测当前样例所属类
void Predict(N_Network net, double input[], double output[])
{
	double hide[HIDE_NUM]={0};
	
	//得到当前网络的输出
	for(int j=0; j<HIDE_NUM; j++)
	{
		for(int k=0; k<INPUT_NUM; k++)
			hide[j]=hide[j]+input[k]*net.weight1[j][k];
		hide[j]=1/(1+exp(-net.alpha*hide[j]));
	}

	for(int i=0; i<OUTPUT_NUM; i++)
	{
		for(int j=0; j<HIDE_NUM; j++)
			output[i]=output[i]+hide[j]*net.weight2[i][j];
		output[i]=1/(1+exp(-net.alpha*output[i]));
	}
}

//训练神经网络
double Train_Network(N_Network *net, double input[][INPUT_NUM], double output[][OUTPUT_NUM], int sample_num)
{
	double weight1_diff[HIDE_NUM][INPUT_NUM]={0}, weight2_diff[OUTPUT_NUM][HIDE_NUM]={0};
	//double sum_error=0;
	double max_error=0;

	//计算对于一批训练数据的权重的修正值
	for(int s=0; s<sample_num; s++)
	{
		double hide[HIDE_NUM]={0}, pre_hide[HIDE_NUM]={0}, net_output[OUTPUT_NUM]={0}, pre_net_output[OUTPUT_NUM]={0};
		double error[OUTPUT_NUM]={0};	

		//得到当前网络的输出
		for(int j=0; j<HIDE_NUM; j++)
		{
			for(int k=0; k<INPUT_NUM; k++)
				pre_hide[j]=pre_hide[j]+input[s][k]*net->weight1[j][k];
			hide[j]=1/(1+exp(-net->alpha*pre_hide[j]));
		}
		for(int i=0; i<OUTPUT_NUM; i++)
		{
			for(int j=0; j<HIDE_NUM; j++)
				pre_net_output[i]=pre_net_output[i]+hide[j]*net->weight2[i][j];
			net_output[i]=1/(1+exp(-net->alpha*pre_net_output[i]));
		}
		//计算期望输出与实际输出之间的误差
		for(int i=0; i<OUTPUT_NUM; i++)
		{
			double square_error=0;
			
			error[i]=output[s][i]-net_output[i];
			square_error=0.5*pow(error[i], 2);
			if(square_error>max_error)
				max_error=square_error;
			//sum_error=sum_error+0.5*pow(error[i], 2);
		}

		//计算网络中各边的权重的修正值
		for(int i=0; i<OUTPUT_NUM; i++)
		{
			double temp=error[i]*(-net->alpha*exp(-net->alpha*pre_net_output[i])/pow(1+exp(-net->alpha*pre_net_output[i]), 2));

			for(int j=0; j<HIDE_NUM; j++)
			{
				for(int k=0; k<INPUT_NUM; k++)
				{
					weight1_diff[j][k]=weight1_diff[j][k]+temp*net->weight2[i][j]*(-net->alpha*exp(-net->alpha*pre_hide[j])/pow(1+exp(-net->alpha*pre_hide[j]), 2))*input[s][k];
				}
				weight2_diff[i][j]=weight2_diff[i][j]+temp*hide[j];
			}
		}
	}

	//修正网络中各边的权重
	for(int i=0; i<OUTPUT_NUM; i++)
		for(int j=0; j<HIDE_NUM; j++)			
			net->weight2[i][j]=net->weight2[i][j]+net->beta*weight2_diff[i][j];

	for(int j=0; j<HIDE_NUM; j++)
		for(int k=0; k<INPUT_NUM; k++)
				net->weight1[j][k]=net->weight1[j][k]+net->beta*weight1_diff[j][k];
	
	//sum_error=sum_error/(sample_num*OUTPUT_NUM);
	return max_error;
}

//判断当前的输出是否与期望输出一致
int Output_Judge(double net_output[], double test_output[])
{
	double net_max=0;
	int net_max_pos=0;

	for(int i=0; i<OUTPUT_NUM; i++)
	{
		if(net_output[i]>net_max)
		{
			net_max=net_output[i];
			net_max_pos=i;
		}
	}

	double test_max=0;
	int test_max_pos=0;

	for(int i=0; i<OUTPUT_NUM; i++)
	{
		if(test_output[i]>test_max)
		{
			test_max=test_output[i];
			test_max_pos=i;
		}
	}

	if(net_max_pos==test_max_pos && abs(net_max-test_max)<0.5)
		return 1;
	return 0;
}

int main()
{
	//FILE *train_file=fopen("../data/XOR_train.txt", "r");
	//FILE *test_file=fopen("../data/XOR_test.txt", "r");
	FILE *train_file=fopen("../data/IF_train.txt", "r");
	FILE *test_file=fopen("../data/IF_test.txt", "r");
	
	N_Network net;
	char str[1000];
	int train_num=0;
	
	if(train_file==NULL || test_file==NULL)
	{
		cout<<"Can't open file"<<endl;
		exit(-1);
	}

	double input[2000][INPUT_NUM];
	double output[2000][OUTPUT_NUM];
	//读取训练数据集
	while(!feof(train_file))
	{
		char* temp=NULL;
	
		str[0]='\0';
		fgets(str, 1000, train_file);
		temp=str;
		if(temp[0]=='I')
		{
			input[train_num][0]=1;//偏置输入1
			for(int n=1; n<INPUT_NUM; n++)
			{
				temp=strchr(temp, ' ');
				temp=strchr(temp, ':');
				input[train_num][n]=strtod(temp+1, 0);
			}
		}
		else if(temp[0]=='O')
		{
			for(int n=0; n<OUTPUT_NUM; n++)
			{
				temp=strchr(temp, ' ');
				temp=strchr(temp, ':');
				output[train_num][n]=strtod(temp+1, 0);
			}
			train_num++;
		}
	}
	fclose(train_file);
	cout<<"训练数据集样本数: "<<train_num<<endl;

	Init_Network(&net);//初始化网络
	/*
	for(int j=0; j<HIDE_NUM; j++)
		for(int k=0; k<INPUT_NUM; k++)
				cout<<net.weight1[j][k]<<' ';
	cout<<endl;
	for(int i=0; i<OUTPUT_NUM; i++)
		for(int j=0; j<HIDE_NUM; j++)			
			cout<<net.weight2[i][j]<<' ';
	cout<<endl;
	*/		
	cout<<"输入层神经元个数："<<INPUT_NUM<<endl;
	cout<<"隐含层神经元个数："<<HIDE_NUM<<endl;
	cout<<"输出层神经元个数："<<OUTPUT_NUM<<endl;
	cout<<"激活函数参数："<<net.alpha<<endl;
	cout<<"学习率："<<net.beta<<endl;

	long count=0;
	double max_error=1;
	clock_t start, finish;

	cout<<"\n开始训练"<<endl;
	//对训练数据集中所有数据进行批训练，使误差小于规定值
	start=clock();
	for(int i=0; max_error>0.1 && count<100000; i++)
	{
		//pos=rand()%train_num;
		max_error=Train_Network(&net, input, output, train_num);
		count++;
		//cout<<max_error<<endl;
	}
	finish=clock();
	cout<<"迭代次数: "<<count<<endl;
	cout<<"训练时间: "<<(double)(finish-start)/CLOCKS_PER_SEC<<"s"<<endl;

	double test_input[INPUT_NUM], test_output[OUTPUT_NUM], net_output[OUTPUT_NUM];
	int right_num=0, sum_num=0;
	
	cout<<"\n开始测试"<<endl;
	//读取测试数据
	while(!feof(test_file))
	{
		char* temp=NULL;
		
		str[0]='\0';
		fgets(str, 1000, test_file);
		temp=str;
		if(temp[0]=='I')
		{
			test_input[0]=1;//偏置输入1
			for(int n=1; n<INPUT_NUM; n++)
			{
				temp=strchr(temp, ' ');
				temp=strchr(temp, ':');
				test_input[n]=strtod(temp+1, 0);
			}
		}
		else if(temp[0]=='O')
		{
			for(int n=0; n<OUTPUT_NUM; n++)
			{
				temp=strchr(temp, ' ');
				temp=strchr(temp, ':');
				test_output[n]=strtod(temp+1, 0);
			}
			Predict(net, test_input, net_output);

			cout<<"输入:";
			for(int n=1; n<INPUT_NUM; n++)
				cout<<" "<<test_input[n];
			cout<<endl;
			cout<<"期望输出:";
			for(int n=0; n<OUTPUT_NUM; n++)
				cout<<" "<<test_output[n];
			cout<<endl;
			cout<<"实际输出:";
			for(int n=0; n<OUTPUT_NUM; n++)
				cout<<" "<<net_output[n];
			cout<<endl;

			if(Output_Judge(net_output, test_output))//判断输出结果与期望结果是否一致
				right_num++;
			sum_num++;
		}
	}
	fclose(test_file);
	cout<<"正确率: "<<(double)right_num/sum_num<<endl;
	/*
	for(int j=0; j<HIDE_NUM; j++)
		for(int k=0; k<INPUT_NUM; k++)
				cout<<net.weight1[j][k]<<' ';
	cout<<endl;
	for(int i=0; i<OUTPUT_NUM; i++)
		for(int j=0; j<HIDE_NUM; j++)			
			cout<<net.weight2[i][j]<<' ';
	cout<<endl;
	*/

	return 0;
}
