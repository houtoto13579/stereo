#ifndef FRAME_H
#define FRAME_H


#include "filter.h"
#include <time.h>
#include <math.h>
#include <algorithm>
#define MAX_DEPTH 256
#define WIN_SIZE 3
using namespace std;
using namespace cv;

//Frame different function Wf(pt,pt-1)//
float FDF(float dif, float comm = 1, float constant=5, float gamma=1){
	 return constant*(float)(pow(0.5,dif*gamma/comm));
}

void disp_stereo_color_frame(IplImage *imageL,IplImage *imageR,IplImage *imageD,IplImage *imageL_pre,IplImage *imageD_pre,int depth,bool window=false,int bsize=3,bool isLeft=true, int iter=1){
	int clean = 10;
	int LR_minus=1;
	bool usingPreImg = true;
	if (!isLeft){
		IplImage *temp = imageL;
		imageL = imageR;
		imageR = temp;
		LR_minus=-1;
	}
	if (imageD_pre == NULL || iter%clean==0){
		usingPreImg = false;
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
			int preWeight = 0;
			int preDisp = 0;
			// this hasn't add the Right Picture Choice
			if(usingPreImg){
				int preCenterColor = imageL_pre->imageData[i*imageL->widthStep+j];
				preWeight = FDF(abs(preCenterColor-centerColor));
				preDisp = imageD_pre->imageData[i*imageL->widthStep+j];
			}
			minNum=100000;
			mindis=depth;
			for(int k=j,c=0;k>0,c<depth;k=k-(LR_minus*3),c++){
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
									mome_w += preWeight*(abs(c-preDisp));
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
	if(imageD_pre!=NULL)
		cvReleaseImage(&imageD_pre);
	if(imageL_pre!=NULL)
		cvReleaseImage(&imageL_pre);
	imageD_pre = cvCloneImage(imageD);
	imageL_pre = cvCloneImage(imageL);
}
//When using big frame, the previous image comparison will use a BSIZE square to calculate the "simularity"

void disp_stereo_color_big_frame(IplImage *imageL,IplImage *imageR,IplImage *imageD,IplImage *imageL_pre,IplImage *imageD_pre,int depth,bool window=false,int bsize=3,bool isLeft=true, int iter=1){
	int LR_minus=1;
	int clean = 10;
	bool usingPreImg = true;
	if (!isLeft){
		IplImage *temp = imageL;
		imageL = imageR;
		imageR = temp;
		LR_minus=-1;
	}
	if (imageD_pre == NULL || iter%clean==0){
		usingPreImg = false;
	}
	int BSIZE = bsize;
	float alpha= 0.9;
	float gamma = 10;
	int preRobust = 50;

	int colorBound = 10;
	int gradBound = 2;
	cvSet(imageD, cvScalar(0,0,0));
	int mindis;
	float minNum;
	int dif;
  	for(int i=0;i<imageL->height;i++){
		for(int j=1;j<imageL->widthStep;j=j+3){
			int centerColor = imageL->imageData[i*imageL->widthStep+j];
			int preWeight = 0;
			int preDif = 0;
			int preDisp = 0;
			int preCount = 0;
			// this hasn't add the Right Picture Choice
			if(usingPreImg){
				for(int p=0;p<BSIZE;++p){
					int k=j;
					if(((i-BSIZE/2)+p)>0){
						for(int q=0;q<BSIZE;++q){
							if(((k-BSIZE/2)+q-1)>0){
								int ii=(i-BSIZE/2+p);
								int jj=(j-(BSIZE/2)*3+3*q);
								int kk=(k-(BSIZE/2)*3+3*q);
								preDif+=abs((imageL->imageData[ii*imageL->widthStep+jj])-(imageL_pre->imageData[ii*imageL_pre->widthStep+kk]));
								preCount++;
							}
						}
					}
				}
				preWeight = FDF(preDif,preCount);
				preDisp = imageD_pre->imageData[i*imageL->widthStep+j];
			}
			minNum=100000;
			mindis=depth;
			for(int k=j,c=0;k>0,c<depth;k=k-(LR_minus*3),c++){
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
									mome_w += preWeight*(min(preRobust,abs(c-preDisp)));
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
	//do filter
	/*
	if(true){
		JointWMF jj;		
		Mat matL = cvarrToMat(imageL);
		Mat matD = cvarrToMat(imageD);
		//cvtColor(matL, matL, cv::COLOR_BGR2GRAY);
		//cvtColor(matD, matD, cv::COLOR_BGR2GRAY);
		Mat newResult = jj.filter(matD, matL,5);
		
		// This is a code that prevent you from core dump//
		IplImage* image2;
		image2 = cvCreateImage(cvSize(newResult.cols,newResult.rows),8,3);
		IplImage ipltemp=newResult;
		cvCopy(&ipltemp,imageD);

		Grey(imageD);
		//cvShowImage("After:", imageD);
		//cvWaitKey(0);
	}
	*/

	if(imageD_pre!=NULL)
		cvReleaseImage(&imageD_pre);
	if(imageL_pre!=NULL)
		cvReleaseImage(&imageL_pre);
	imageD_pre = cvCloneImage(imageD);
	imageL_pre = cvCloneImage(imageL);
}

#endif
