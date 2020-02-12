//呼叫函式庫
#include "opencv2/opencv.hpp"
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <ctime>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
#pragma warning(disable:4996)

//全域變數宣告
const int ITERATION_NUMBER=2; //要進行迭代的次數
const int GENE_LENGTH=56; //每個基因的長度
const int POPULATION_NUMBER=2; //有多少個基因
const double CROSSOVER_RATE=0.8; //發生交配的機率
const double MUTATION_RATE=0.001; //發生突變的機率


struct parent //一個母體
{
	int number;  //第幾號母體
	int fitness; //母體的適應函數
	double decs[5];//母體的小數
	bool gene[GENE_LENGTH];
	int gb;
	int shape1;
	int shape2;
};
//函數宣告
void cal_parent_dec(parent &p);
void cal_parent_fitness(parent &p);
void initial(void);  //初始化母體
void blob_detector(parent &p);
void blob_detector_best_gene(parent &p);
void print2(void);  //印出每個母體中的基因內容
void reproduction(void);  //使用輪盤式，將母體丟入交配池中
void crossover(void);  //進行交配
void mutation(void);  //進行突變
#define Rand()((double)rand()/(double)RAND_MAX) //可以隨機產生0~1均勻分布的亂數
#define select_one_body()(rand()%POPULATION_NUMBER)  //隨機取得母體編號
#define select_one_gene()(rand()%(GENE_LENGTH-1))  //隨機取得基因編號
parent population[POPULATION_NUMBER];
parent pool[POPULATION_NUMBER];
parent temp[POPULATION_NUMBER];
parent best_gene;

int counts;

Mat im;
Mat im_with_keypoints;
size_t x;
int i;
int k;
char filename[100];

int minArea;
int maxArea;
double avg_fitness = 0;
int gb, shape1, shape2;
void cal_parent_dec(parent &p)//將基因訊息轉為10進位小數
{
	double dec = 0;
	double dec2 = 0;
	double dec3 = 0;
	double dec4 = 0;
	double dec5 = 0;
	int geneLeng1 = GENE_LENGTH - 50;
	int geneLeng2 = GENE_LENGTH - 42;
	int geneLeng3 = GENE_LENGTH - 28;
	int geneLeng4 = GENE_LENGTH - 14;


	for(int i = geneLeng1 -1 ; i>=0; i--)
		{

			if(p.gene[i] == 1) dec +=  (1 << i);

		}

		for(int i = geneLeng2 -1; i>=6 ; i--)
		{
			if(p.gene[i] == 1) dec2 += (1 << i-6);
		}
		for(int i = geneLeng3 -1; i>=14 ; i--)
		{

			if(p.gene[i] == 1) dec3 +=  (1 << i-14);
		}


		for(int i = geneLeng4 -1; i>=28 ; i--)
		{

			if(p.gene[i] == 1) dec4 +=  (1 << i-28);
		}

		for(int i = GENE_LENGTH -1; i>=42 ; i--)
		{

			if(p.gene[i] == 1) dec5 +=  (1 << i-42);
		}

		//內插法將基因序列轉化為對應小數

		p.decs[0] = 35 + ((60 - 35) * dec) / 63;
		p.decs[1] = 1000 + ((1200 - 1000) * dec2) / 255;
		p.decs[2] = 0.0001+ ((0.9999 - 0.0001) * dec3) / 16383;
		p.decs[3] = 0.0001+ ((0.9999 - 0.0001) * dec4) / 16383;
		p.decs[4] = 0.0001+ ((0.9999 - 0.0001) * dec5) / 16383;
		p.gb = rand()%5*2+1;
		p.shape1 = rand()%3;
		p.shape2 = rand()%9+1;

}
//對每一條染色體[4]計算適應函數
void cal_parent_fitness(parent &p)//計算適應函數
{
	p.fitness = counts;

}
void blob_detector(parent &p){

	for (i = 1; i <= 218; i++)
			{
				//載入影像
				sprintf(filename, "D:/tests/test1/training/test%d.jpg", i);

				im = imread(filename, IMREAD_GRAYSCALE);

				if (!im.data)
				{
					break;
				}

				//影像前處理resize()

		if (im.cols >= 1024 && im.rows >= 1024){
			cv::resize(im, im, Size(1024/1.5, 1024/1.5));
		}else if(im.cols < 1024 && im.rows < 1024){
			cv::resize(im, im, Size(im.cols, im.rows));
		}

		//影像前處理:二值化，平滑，膨脹
		GaussianBlur(im, im, Size(p.gb,p.gb) ,0 ,0);
		Mat erodeStruct = getStructuringElement(p.shape1,Size(p.shape2,p.shape2));//MORPH_CROSS、MORPH_RECT
		dilate(im, im, erodeStruct);

		//斑點檢測
			SimpleBlobDetector::Params params;
			params.minThreshold = 10;
			params.maxThreshold = 255;


			params.filterByArea = true;
			params.minArea = p.decs[0];//
			params.maxArea = p.decs[1];//


			params.filterByCircularity = true;
			params.minCircularity = p.decs[2];
			params.maxCircularity = 1;


			params.filterByConvexity = true;
			params.minConvexity = p.decs[3];
			params.maxConvexity = 1;


			params.filterByInertia = true;
			params.minInertiaRatio = p.decs[4];
			params.maxInertiaRatio = 1;

			// Storage for blobs
			vector<KeyPoint> keypoints;

			// Set up detector with params
			Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

			// Detect blobs
			detector->detect( im, keypoints);

			x=keypoints.size();

			// the size of the circle corresponds to the size of blob

			drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

		std::ostringstream name;
		name << "test" << i << ".jpg";
			//目標函數：瑕疵片分類正確 +5,  正常片分類正確 +1, 瑕疵片分類錯誤 -2
			 if(x == 0){

				if(i > 113){
				counts++;
				}else{
				counts -= 2;

				}

			}else if(x > 0){

				if(i < 114){
				counts += 5;
				}

			}

	}
}
void blob_detector_best_gene(parent &p){

	for (i = 1; i <= 218; i++)
			{
				//載入影像
				sprintf(filename, "D:/tests/test1/training/test%d.jpg", i);

				im = imread(filename, IMREAD_GRAYSCALE);

				if (!im.data)
				{
					break;
				}

				//影像前處理resize()

		if (im.cols >= 1024 && im.rows >= 1024){
			cv::resize(im, im, Size(1024/1.5, 1024/1.5));
		}else if(im.cols < 1024 && im.rows < 1024){
			cv::resize(im, im, Size(im.cols, im.rows));
		}

		//影像前處理:二值化，平滑，膨脹
		GaussianBlur(im, im, Size(best_gene.gb,best_gene.gb) ,0 ,0);
		Mat erodeStruct = getStructuringElement(best_gene.shape1,Size(best_gene.shape2,best_gene.shape2));//MORPH_CROSS、MORPH_RECT
		dilate(im, im, erodeStruct);

		//斑點檢測
			SimpleBlobDetector::Params params;
			params.minThreshold = 10;
			params.maxThreshold = 255;


			// 通?判?blob的面?，也就是像素??是否在[minArea,maxArea)
			params.filterByArea = true;
			params.minArea = best_gene.decs[0];//
			params.maxArea = best_gene.decs[1];//


			// ?斑點圓度范??[0.50 ,maxCircularity )?，?中?blob
			params.filterByCircularity = true;
			params.minCircularity = best_gene.decs[2];
			params.maxCircularity = 1;


			// blob 凸度在範圍[minConvexity ,maxConvexity )?，?中?blob
			params.filterByConvexity = true;
			params.minConvexity = best_gene.decs[3];
			params.maxConvexity = 1;


			// 越接近0為直線，接近1為圓圈。??性率范??[0.20 ,maxInertiaRatio)?，?中?blob
			params.filterByInertia = true;
			params.minInertiaRatio = best_gene.decs[4];
			params.maxInertiaRatio = 1;

			// Storage for blobs
			vector<KeyPoint> keypoints;

			// Set up detector with params
			Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

			// Detect blobs
			detector->detect( im, keypoints);

			x=keypoints.size();

			// the size of the circle corresponds to the size of blob

			drawKeypoints( im, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

		std::ostringstream name;
		name << "test" << i << ".jpg";

			if(x == 0){

				cv::imwrite("D:/tests/negative/"+name.str(),im_with_keypoints);


			}else{
				cv::imwrite("D:/tests/positive/"+name.str(),im_with_keypoints);

			}
	}
}
//主程式
int main(){


	std::clock_t start;
	double duration;

	start = std::clock();
	cout << "B init() begin" << endl;
	using std::cout;
	srand(time(NULL));  //srand()函數設定初始的亂數種子

	initial();  //初始化母體
	cout<<"==========init===========\n";
	std::ofstream myfile;
	myfile.open ("D:/tests/testResult/testlog.csv");
	myfile << ",max_fitness,avg_fitness\n";
	//迭代
	for(int z=1;z<=ITERATION_NUMBER;z++){

		reproduction();  //使用輪盤式，將母體丟入交配池中
		crossover();  //進行交配
	    mutation();  //進行突變
	    myfile << z << ","<< best_gene.fitness << ","<< avg_fitness << "\n";
	    cout<<"==========Generation "<<z<<"============ \n";
	    cout<<"\n\n\n";
	}
	myfile.close();
	//double correct = double(best_gene.fitness)/218.00;
	cout<<"best value: ";
	printf("( %5.1f, %5.1f, %.4lf, %.4lf, %.4lf,gb: %d, shape1: %d, shape2: %d, fitness: %d )\n ",best_gene.decs[0],best_gene.decs[1],best_gene.decs[2],best_gene.decs[3],best_gene.decs[4], best_gene.gb, best_gene.shape1, best_gene.shape2,best_gene.fitness);

	blob_detector_best_gene(best_gene);
//	system("PAUSE");
	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout<<"time: "<< duration <<'\n';
	return 0;
}
//初始化母體
void initial(){
	using std::cout;
	for(k = 0 ; k < POPULATION_NUMBER ; ++k){

			population[k].number = k;
			cout<<"#"<<population[k].number<<"  ";
			for(int j = 0 ; j < GENE_LENGTH ; ++j)
			{
				population[k].gene[j] = rand() % 2;
			}
			cal_parent_dec(population[k]);

			blob_detector(population[k]);

			cal_parent_fitness(population[k]);
			counts = 0; //counts 歸零
			//correct = 0;

			if(i==0) best_gene=population[k];  //找出擁有最佳適應值的母體，複製到最佳母體
			else if (population[k].fitness>best_gene.fitness) best_gene=population[k];


			printf("( %5.1f, %5.1f, %.4lf, %.4lf, %.4lf,gb: %d, shape1: %d, shape2: %d, fitness: %d )\n ",population[k].decs[0],population[k].decs[1],population[k].decs[2],population[k].decs[3],population[k].decs[4], population[k].gb, population[k].shape1, population[k].shape2, population[k].fitness);

	}

}

//印出並計算每個母體中的內容
void print2(){
	using std::cout;
	double fitness_sum = 0;
	int num = POPULATION_NUMBER;
	for(k = 0 ; k < POPULATION_NUMBER ; ++k){

		population[k].number = k;
		cout<<"#"<<population[k].number<<"  ";

		    cal_parent_dec(population[k]);

		    blob_detector(population[k]);

		cal_parent_fitness(population[k]);
		counts = 0; //counts 歸零
		printf("( %5.1f, %5.1f, %.4lf, %.4lf, %.4lf,gb: %d, shape1: %d, shape2: %d, fitness: %d )\n ",population[k].decs[0],population[k].decs[1],population[k].decs[2],population[k].decs[3],population[k].decs[4], population[k].gb, population[k].shape1, population[k].shape2, population[k].fitness);

		fitness_sum += population[k].fitness;
		//找出擁有最佳適應值的母體，複製到最佳母體
		if (population[k].fitness>best_gene.fitness) {
			cout<<"population[k] > best_gene \n";
			best_gene=population[k];
		}
	}
	printf("fitness_sum: %5.2f\n",fitness_sum);
	cout<<"max value: ";
	cout<<" "<<best_gene.fitness<<"\n";
	avg_fitness = fitness_sum / num;
	cout<<"avg value: ";
	cout<<" "<<avg_fitness <<"\n";
}
//使用輪盤式，將母體丟入交配池中
void reproduction(){
	using std::cout;
	int choose;  //暫存被選中準備要被丟入交配池的母體編號(number)
	double total_fitness=0;  //全部母體的適應值加總
	double accumulate_probability[POPULATION_NUMBER];  //每個母體的所占比例累加
	//如果所占比例越大，那麼累加後他所占的區間也會越大，則被選到的機率更大
	double random;  //產生隨機0~1的數值
	//全部母體的適應值加總
	for(int i=0;i<POPULATION_NUMBER;i++)
	total_fitness+=population[i].fitness;
	//每個母體的所占比例累加
	accumulate_probability[0]=(double)population[0].fitness/(double)total_fitness;
	for(int i=1;i<POPULATION_NUMBER;i++)
	accumulate_probability[i]=accumulate_probability[i-1]+(double)population[i].fitness/(double)total_fitness;
	for(int k=0;k<POPULATION_NUMBER;k++){
		random=Rand();  //產生隨機0~1的數值
	    for(int j=0;j<POPULATION_NUMBER;j++){  //尋找到機率區間
		    if(random<=accumulate_probability[j]){
		    	choose=j;
		    	break;
			}
		    continue;
    	}
	pool[k]=population[choose];
	}
//	cout<<"chromo # to the pool:\n";
//	for(int l=0;l<POPULATION_NUMBER;l++){
//		cout<<"#"<<pool[l].number<<" ";
//	}
//    cout<<"\n";
    for(int k=0;k<POPULATION_NUMBER;k++){pool[k].number=k;}  //更新為在pool中的編號
    //印出目前在交配池中的母體
//    cout<<"after extract\n";
//    for(int i=0;i<POPULATION_NUMBER;i++){
//		cout<<"#"<<pool[i].number<<"  ";  //在pool中的編號
//
//		printf("( %5.2f, %5.2f,%.4lf,%.4lf,%.3lf,fitness: %d )\n ",pool[i].decs[0],pool[i].decs[1],pool[i].decs[2],pool[i].decs[3],pool[i].decs[4],pool[i].fitness);
//
//	}
}
//進行交配
void crossover(){
	using std::cout;
	int b1,b2;  //從母體編號中，隨機選出2個不相同的編號
	int g1,g2;  //從基因編號中，隨機選出g2>g1的編號
	int area;  //要進行交配的基因編號區間
	int temp;  //暫存要交換的基因元素
	for(int i=0;i<=POPULATION_NUMBER;i++){
		b1=select_one_body();
	    do b2=select_one_body();  //持續隨機產生編號，直到b2的值與b1不相同
	    while(b2==b1);
//	    cout<<"b1="<<b1<<" b2="<<b2<<" ";
	    //接下來隨機產生要交換基因的區間
	    //如果基因編號為0~4，那麼g1為0~3，而g2為1~4，並且g2>g1
	    g1=select_one_gene();
	    do g2=select_one_gene()+1;  //持續隨機產生編號，直到b2的值大於b1
	    while(g2<=g1);
//	    cout<<"g1="<<g1<<" g2="<<g2<<"\n";
	    double if_crossover = (double)rand()/(double)RAND_MAX;
//	    double if_crossover = 0.8;

	    if(if_crossover < CROSSOVER_RATE){  //決定是否要交配
//	    	cout<<"mating...\n";
	    	for(area=g1;area<=g2;area++){
		        temp=pool[b1].gene[area];
		        pool[b1].gene[area]=pool[b2].gene[area];
		        pool[b2].gene[area]=temp;
	        }
		}
	    else{
//	    	cout<<"no mating\n";
	    	continue;
		}
	}
	//印出交配後的基因內容
//    cout<<"after mating \n";
//    for(int i=0;i<POPULATION_NUMBER;i++){
//    	count=0;
//		cout<<"#"<<pool[i].number<<"  ";  //在pool中的編號
//		printf("( %5.2f, %5.2f,%.4lf,%.4lf,%.3lf,fitness: %d )\n ",pool[i].decs[0],pool[i].decs[1],pool[i].decs[2],pool[i].decs[3],pool[i].decs[4],pool[i].fitness);
//	}
    //將交配池pool中完成交換的所有母體複製回body
	for(int i=0;i<POPULATION_NUMBER;i++){
		population[i]=pool[i];
	}
}
//進行突變
void mutation(){
	int b1;  //從母體編號中，隨機選出1個編號
	int g1,g2;  //從基因編號中，隨機選出g2>g1的編號
	int temp;  //暫存要交換的基因元素

	using std::cout;
	for(int i=0;i<=POPULATION_NUMBER;i++){
		b1=select_one_body();
//	    cout<<"b1="<<b1<<" ";
	    //接下來隨機產生要進行交換的基因
	    //如果基因編號為0~4，那麼g1為0~3，而g2為1~4，並且g2>g1
	    g1=select_one_gene();
	    do g2=select_one_gene()+1;  //持續隨機產生編號，直到b2的值大於b1
	    while(g2<=g1);
//	    cout<<"g1="<<g1<<" g2="<<g2<<"\n";
	    double if_mutate = ((double)rand() / (double)RAND_MAX * (0.01 - 0.00001)) + 0.00001;
	    if(if_mutate < MUTATION_RATE){  //決定是否產生突變
//	    	cout<<"mutation...\n";
		    temp=population[b1].gene[g1];  //進行母體上的基因交換
		    population[b1].gene[g1]=population[b1].gene[g2];
		    population[b1].gene[g2]=temp;
		}
	    else{
//	    	cout<<"no mutation\n";
	    	continue;
		}

    }
	print2();

}
