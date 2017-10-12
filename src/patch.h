#ifndef PATCH_H
#define PATCH_H


#include <time.h>
#include <math.h>
#include <algorithm>
#define MAX_DEPTH 256
#define WIN_SIZE 3
using namespace std;
using namespace cv;
float weiFunc(float a, float b, float para, float distance=0, float radius=1){	
	return exp(-(abs(float(a-b))/para)+distance/radius);
}

void disp_stereo_color(IplImage *imageL, IplImage *imageR, IplImage *imageD, int depth, bool window=false, int bsize = 3,bool isLeft=true){
	int LR_minus=1;
	if (!isLeft){
		IplImage *temp = imageL;
		imageL = imageR;
		imageR = temp;
		LR_minus=-1;
	}


	int BSIZE = bsize;
	float alpha= 0.9;
	float gamma = 10;
	
	int colorBound = 10;
	int gradBound = 2;
	cvSet(imageD, cvScalar(0,0,0));
	int mindis;
	float minNum;
	int dif;
  	for(int i=0;i<imageL->height;i++){
		for(int j=1;j<imageL->widthStep;j=j+3){
			int centerColor = imageL->imageData[i*imageL->widthStep+j];
			minNum=100000;
			mindis=depth;
			for(int k=j,c=0;k>0,c<depth;k=k-(LR_minus*3),c++){
				//cerr<<(int)(imageL->imageData[i*imageL->widthStep+j]);
				if (!window){
					if((i*imageR->widthStep+k-1)>0){
						dif=abs((imageL->imageData[i*imageL->widthStep+j])-(imageR->imageData[i*imageR->widthStep+k]));
						int leftGradient = (imageL->imageData[i*imageL->widthStep+j+1])-(imageL->imageData[i*imageL->widthStep+j-1]);
						int rightGradient = (imageR->imageData[i*imageR->widthStep+k+1])-(imageR->imageData[i*imageR->widthStep+k-1]);
						leftGradient = abs(leftGradient);
						rightGradient = abs(rightGradient);
						float ww = weiFunc(dif,0,gamma);
						float rho = (1-alpha)*min(dif,colorBound) + alpha*min((abs(leftGradient-rightGradient)),gradBound);			
						float mome = ww*rho;
						if(mome<minNum){
							minNum=mome;
							mindis=c;
						}
					}					
				}
				else{
					float mome_w = 0;
					float ww_wall = 0;
					for(int p=0;p<BSIZE;++p){
						if(((i-BSIZE/2)+p)>0){
							for(int q=0;q<BSIZE;++q){
								if(((k-BSIZE/2)+q-1)>0){
									int centerNewColor = imageR->imageData[i*imageR->widthStep+k];
									int ii=(i-BSIZE/2+p);
									int jj=(j-(BSIZE/2)*3+3*q);
									int kk=(k-(BSIZE/2)*3+3*q);
									dif=abs((imageL->imageData[ii*imageL->widthStep+jj])-(imageR->imageData[ii*imageR->widthStep+kk]));
									int LGW=(imageL->imageData[ii*imageL->widthStep+jj+1])-(imageL->imageData[ii*imageL->widthStep+jj-1]);
									int RGW=(imageR->imageData[ii*imageR->widthStep+kk+1])-(imageR->imageData[ii*imageR->widthStep+kk-1]);
									LGW = abs(LGW);
									RGW = abs(RGW);
									float ww_w = weiFunc(centerColor,(imageR->imageData[ii*imageR->widthStep+kk]),gamma);
									ww_wall+=ww_w;
									float rho_w = (1-alpha)*min(dif,colorBound) + alpha*min((abs(LGW-RGW)),gradBound); 
									mome_w += ww_w*rho_w;
								}
								else{
									mome_w += 1000*BSIZE;
								}
							}
						}
						else{
							mome_w += 1000*BSIZE;
						}
					}
					mome_w=mome_w/ww_wall;
					if(mome_w<minNum){
						minNum=mome_w;
						mindis=c;
					}
				}
			}
			int newColor = (255/depth)*mindis;
			imageD->imageData[i*imageD->widthStep+j]=newColor;
	      	imageD->imageData[i*imageD->widthStep+j+1]=newColor;
     		imageD->imageData[i*imageD->widthStep+j+2]=newColor;
    	}
	}
}


void patch(IplImage *imgL, IplImage *imgR, IplImage *imgD ){
	bool window = false;
	float gamma = 10;
	float alpha = 0.9;

	
	srand (time(NULL));
	cerr<<"imgL height"<<imgL->height<<endl;
	cerr<<"imgL width"<<imgL->width<<endl;
	cerr<<"imgR height"<<imgR->height<<endl;
	cerr<<"imgR width"<<imgR->width<<endl;
	int height = imgL->height;
	int width = imgL->width;
	int *** pArr;
	pArr = new int ** [width];
	for(int i=0; i<width; ++i){
		pArr[i] = new int * [height];
		for(int j=0; j<height; ++j){
			pArr[i][j] = new int[3];
			//initialization//
			pArr[i][j][0] = 0;
			pArr[i][j][1] = 0;
			pArr[i][j][2] = 1;
		}
	}

	int maxFunctionTry = 10;
	int ax = 0;
	int by = 0;
	int cz = 1;
	
	int initRR = 9; //random range
	Grey(imgL);
	Grey(imgR);
	for(int iter = 0; iter<1; ++iter){
		for(int i = 0; i<width; ++i){
			for (int j = 0; j<height; ++j){
				float orig_mome = 9999999;
				int axTemp = ax;
				int byTemp = by;
				int czTemp = cz;
				for(int functionSet = 0; functionSet<maxFunctionTry; ++functionSet){	
					while(true){
						//random
						int aarandom = axTemp+rand()%initRR-(initRR/2);
						int bbrandom = byTemp+rand()%initRR-(initRR/2);
						int ccrandom = czTemp+rand()%initRR-(initRR/2);
						int newDepth = aarandom*i + bbrandom*j +ccrandom;
						if( newDepth <MAX_DEPTH && newDepth>=0){
							axTemp = aarandom;
							byTemp = bbrandom;
							czTemp = ccrandom;
							break;
						}
					}
					float mome = 0;
					for(int ch_j=j,c=0 ; ch_j >0,c<MAX_DEPTH ; ch_j=ch_j-3,c++){
						//different window things :D
						if(window){
								
						}
						else{
							
						
						}
					}

					if(mome<=orig_mome){
						pArr[i][j][0] = axTemp;
						pArr[i][j][1] = byTemp;
						pArr[i][j][2] = czTemp;
					}
				}			
			}
			cout<<i<<endl;
		}	
		// This will think which is the best function for specific pixel
	}

	for(int i=0; i<width; ++i){
		for(int j=0; j<height; ++j){
			delete [] pArr[i][j];
		}
		delete [] pArr[i];
	}
	delete pArr;
}
#endif
