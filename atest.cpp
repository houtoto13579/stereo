#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <string>
#include <fstream>
#include <stdio.h>
#include <ctime>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
// Self Lib //
#include "src/test.h"
#include "src/census.h"
#include "src/gradient.h"
#include "src/patch.h"
#include "src/frame.h"
#include "src/filter.h"
#include "src/frame_cmotion.h"
#include "src/BP_motion.h"

// Include from toolkit file (which have to link png++ lib--> you could check in CmakeList.txt) //
// #include <math.h>
// #include "mail.h"
// #include "io_disp.h"
// #include "io_flow.h"
// #include "io_integer.h"
// #include "utils.h"

//============ Define ============//

#define HEIGHT 1000
#define WIDTH 1000

//TODO: DEPTH
#define DEPTH 64
#define BSIZE 13
#define NSIZE 3
#define JUMP 1
#define START 1
#define END 31

#define ITER_JUMP 1

#define CLEAN 1000
#define SPEED 1

#define FILTERSIZE 7

#define DISP_STEREO   0
#define DISP_WINDOW   1
#define DISP_CENSUS   2
#define DISP_GRADIENT 3
#define DISP_STEREO_COLOR	4
#define DISP_WINDOW_COLOR	5
#define DISP_PATCH    6
#define DISP_WINDOW_COLOR_FRAME	7
#define DISP_WINDOW_COLOR_BFRAME 8
#define DISP_WINDOW_COLOR_BFRAME_OPT 9
#define DISP_WINDOW_COLORASW_BFRAME_OPT 19
#define NEW_BP_BFRAME_OPT 20
#define NEW_BP 21
#define NEW_BP_TEMPORAL 22
#define DISP_WINDOW_CEN_BFRAME_OPT  10
#define DISP_WINDOW_BP_BFRAME_OPT  11
#define DISP_TRUE_AFFINE 12

// Testing will be higher than 100 to prevent saving image
#define TEST_OPTICAL 102
#define TEST_AFFINE  101

using namespace std;
using namespace cv;
//=========  MasterDebug ===========//
bool usrGrey = true;
bool autoParsing = false;
//========= Left or Right =========//
bool isLeft = true;
//===========  Filter =============//
bool useFilter=true;
bool refineUseFilter = true;
//=========== Compare =============//
bool Compare = true;
int compareTolerance = 1*(256/DEPTH);
bool deleteLeftDiscont = true; //true while make left BSIZE 
//=========  Directory ============//
bool outputOrig=true;
//TODO: Directory should change accordingly
//#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
/*
string img_directory = "input/Tsukuba/daylight/";
string img_left = "left/";
string img_right = "right/";

string img_name = "";
string child_directory = (isLeft)?img_left:img_right;
string img_output_directory = "output/Tsukuba_paper2/"+(child_directory)+"new_test/";
string img_discontinue_directory = "input/Tsukuba/discontinuity_maps/";
string img_compare_directory = "input/Tsukuba/disparity_maps/";
*/
//#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
string method = "dcb";

string img_directory = "input/Dcb/temple/";
string img_left = "left/";
string img_right = "right/";

string img_name = "";
string child_directory = (isLeft)?img_left:img_right;
string img_output_directory = "output/Dcb/"+(child_directory)+"temple/B17/BP/";
//string img_discontinue_directory = "";
string img_discontinue_directory = "input/Dcb/temple/occlution/";
string img_compare_directory = "input/Dcb/temple/ground/";

//#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
/*
string img_directory = "input/Kitti/2011_09_26_drive_0017_sync/";
string img_left = "image_02/data/";
string img_right = "image_03/data/";

string img_name = "";
string child_directory = (isLeft)?"left/":"right/";
string img_output_directory = "output/kitti_paper/2011_09_26_drive_0017_sync/"+(child_directory)+"window_opt_256d_25B_preASW/";
string img_discontinue_directory = "REMEMBER TO SET COMPARE TO FALSE";
string img_compare_directory = "ALSO, CHANGE THE CVLOADTYPE, FILENAME below as well as array prop in test.h";
*/
//#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
/*
string img_directory = "input/Nagoya/dog/";
string img_left = "030/";
string img_right = "035/";

string img_name = "";
string child_directory = (isLeft)?"left/":"right/";
string img_output_directory = "output/nagoya_paper/dog/"+(child_directory)+"figure/opt_7B_ADD03/";
string img_discontinue_directory = "REMEMBER TO SET COMPARE TO FALSE";
string img_compare_directory = "ALSO, CHANGE THE CVLOADTYPE, FILENAME below as well as array prop in test.h";
*/
//========== Disp_Type ============//

 /******************************
 * DISP_STEREO       			*
 * DISP_WINDOW       			*
 * DISP_CENSUS       			*
 * DISP_GRADIENT     			*
 * DISP_STEREO_COLOR 			*
 * DISP_WINDOW_COLOR 			*
 * DISP_PATCH	       			*
 * DISP_WINDOW_COLOR_FRAME		*
 * DISP_WINDOW_COLOR_BFRAME		*
 * DISP_WINDOW_COLOR_BFRAME_OPT * ->output frame
 * DISP_WINDOW_COLORASW_BFRAME_OPT * ->output frame
 * NEW_BP_BFRAME_OPT			*
 * NEW_BP						*
 * NEW_BP_TEMPORAL				*
 * DISP_WINDOW_CEN_BFRAME_OPT   *
 * DISP_WINDOW_BP_BFRAME_OPT    *
 * DISP_TRUE_AFFINE   			*
 * TEST_OPTICAL					*
 * TEST_AFFINE					*   
 ******************************/

int type = NEW_BP;

//========== Error_Rate ===========//

double error_sum = 0;

//========= Progress Bar ==========//

int totalPoint = 50;
float pcount = 0;
float pcountadd = (float)(END-START)/totalPoint;
bool Test_using_tsukuba = false;

//======== Main Function ==========//
	
int main(int argc, char** argv){
	cerr<<"Using: CV version: "<<CV_MAJOR_VERSION<<endl;
	cout<<"=== Disparity Checking ==="<<endl;
	cout<<"Output:"<<img_output_directory<<endl;
	cout<<"Method: " <<type<<endl;
	cout<<"Initiate Matching from "<<START<<" to "<<END<<endl;
	// Total Timer Begin
	const clock_t total_begin_time = clock();
	IplImage *imageD_pre=0;
	IplImage *imageL_pre=0;
	ofstream outfile;
	string fileRecord=img_output_directory+"record.csv";
	char fileRecordChar[1024];
	strcpy(fileRecordChar, fileRecord.c_str());
	outfile.open(fileRecordChar, ios_base::app);
	outfile <<"type,"<<type<<endl;
	outfile <<"depth,"<<DEPTH<<endl;
	outfile <<"compareTolerance,"<<compareTolerance<<endl;
	outfile <<"speed,"<<SPEED<<endl;
	outfile <<"boxsize,"<<BSIZE<<endl;
	outfile <<"nsize,"<<NSIZE<<endl;
	outfile <<"filterSize,"<<FILTERSIZE<<endl;
	for(int file_iter = START; file_iter<=END; file_iter+=ITER_JUMP){	
		// Timer Begin //
		const clock_t begin_time = clock();

		cerr<<"\rNow Processing("<<file_iter<<"/"<<END<<") ";
		pcount+=1;
		cerr<<"|";
		for(float i=0;i<totalPoint;i+=1)
			if(pcountadd == 0)
				cerr<<"#";
			else if(i+1<(pcount/(pcountadd)))
				cerr<<"#";
			else
				cerr<<"_";
		cerr<<"|";
		//TODO: FileName should change accordingly
		//img_name = "frame_"+num2str(file_iter); // this is for tsukuba
		img_name = num2file(file_iter,4); // this is for dcb
		//img_name=num2file(file_iter,6)+"_10";//this is for kitti
		//img_name=num2file(file_iter,10);// this is for kitti real and nagoya yuv

		//TODO: Left Right File Name
		//for leftname=rightname
		/*
		string img_Lname=img_name;
		string img_Rname=img_name;
		*/
		//for dcb
		
		string img_Lname= "L"+img_name;
		string img_Rname= "R"+img_name;
		string fileL = img_directory + img_left  + img_Lname + ".png";
		string fileR = img_directory + img_right  + img_Rname + ".png";
		
		if(Test_using_tsukuba){
			fileL="imL.png";
			fileR="imR.png";
		}
			
		char filename1[1024];
		char filename2[1024];
		strcpy(filename1, fileL.c_str());
		strcpy(filename2, fileR.c_str());
		// Image Size: 1000x669
		IplImage *imageL, *imageR, *imageD; //type of image
		IplImage *imageD_refine=0;
		//TODO: Need to change from LOAD_IMAGE_UNCHANGED <-> COLOR
 	 	//Tsukuba
		
		imageL = cvLoadImage(filename1,CV_LOAD_IMAGE_COLOR); // CVload type
 	 	imageR = cvLoadImage(filename2,CV_LOAD_IMAGE_COLOR); // CVload type
 	 	imageD = cvLoadImage(filename2,CV_LOAD_IMAGE_COLOR); // CVload type
		imageD_refine = cvLoadImage(filename2,CV_LOAD_IMAGE_COLOR);
		
		//Kitti or Nagoya
		/*
 	 	imageL = cvLoadImage(filename1,CV_LOAD_IMAGE_UNCHANGED); // CVload type
 	 	imageR = cvLoadImage(filename2,CV_LOAD_IMAGE_UNCHANGED); // CVload type
 	 	imageD = cvLoadImage(filename2,CV_LOAD_IMAGE_UNCHANGED); // CVload type
		imageD_refine = cvLoadImage(filename2,CV_LOAD_IMAGE_UNCHANGED);
		*/

		if(!imageL){
			cout<<"Error: Couldn't open the image L file.\n";
			cout<<"## filename: "<<filename1<<endl;
			return 0;
		}
		if(!imageR){
			cout<<"Error: Couldn't open the image R file.\n";
			cout<<"## filename: "<<filename2<<endl;
			return 0;
		}

		if(type==DISP_STEREO){
			if(usrGrey){
				Grey(imageL);
				Grey(imageR);
			}
			cvSet(imageD, cvScalar(0,0,0));
			int mindis;
			int min;
          	for(int i=0;i<imageL->height;i++){
				for(int j=1;j<imageL->widthStep;j=j+3){
					min=100;
					mindis=DEPTH;
					for(int k=j,c=0;k>0,c<DEPTH;k=k-3,c++){
						//cerr<<(int)(imageL->imageData[i*imageL->widthStep+j]);
						if((i*imageR->widthStep+k)>0){
							int dif=((imageL->imageData[i*imageL->widthStep+j])-(imageR->imageData[i*imageR->widthStep+k]));
							/*	
							if(dif<=1){
								mindis=c;
								break;
							}
							*/
							if(dif<0){
								dif=-dif;
							}
							if(dif<min){
								min=dif;
								mindis=c;
							}
						}
					}
					int newColor = (255/DEPTH)*mindis;
					imageD->imageData[i*imageD->widthStep+j]=newColor;
			      	imageD->imageData[i*imageD->widthStep+j+1]=newColor;
             		imageD->imageData[i*imageD->widthStep+j+2]=newColor;
            	}
          }
		  	/*
			cvShowImage("Depth", imageD);
          	cvWaitKey(0); // wait
          	cvReleaseImage(&imageL); // release IplImage
          	cvReleaseImage(&imageR); // release IplImage
			cvDestroyWindow("Left"); // destroy
			cvDestroyWindow("Right");
			*/
		}
		else if (type==DISP_WINDOW){
			if(usrGrey){
				Grey(imageL);
				Grey(imageR);
			}
			cvSet(imageD, cvScalar(0,0,0));
			int mindis;
			int min;
			int all;
			int dif;
          	for(int i=0;i<imageL->height;i+=JUMP){
				for(int j=0;j<imageL->widthStep;j=j+3*JUMP){
					min=1000;
					mindis=DEPTH;
					all=0;
					for(int k=j,c=0;k>0,c<DEPTH;k=k-3,c++){
						for(int p=0;p<BSIZE;++p){
							if(((i-BSIZE/2)+p)>0){
								for(int q=0;q<BSIZE;++q){
									if(((k-BSIZE/2)+q)>0){
										int ii=(i-BSIZE/2+p);
										int jj=(j-3*(BSIZE/2)+3*q);
										int kk=(k-3*(BSIZE/2)+3*q);
										dif=((imageL->imageData[ii*imageL->widthStep+jj])-(imageR->imageData[ii*imageR->widthStep+kk]));
										if(dif<0){
											dif=-dif;
										}
										all=all+dif;
									}
									else{
										all+=1000*BSIZE;
									}
								}
							}
							else{
								all+=1000*BSIZE;
							}
						}
						if(all<min){
							min=all;
							mindis=c;
						}
						all=0;
					}
					imageD->imageData[i*imageD->widthStep+j]=(255/DEPTH)*mindis;
			      	imageD->imageData[i*imageD->widthStep+j+1]=(255/DEPTH)*mindis;
             		imageD->imageData[i*imageD->widthStep+j+2]=(255/DEPTH)*mindis;
            	}
        	}
			
			/*
			cvShowImage("Depth", imageD);
        	cvWaitKey(0); // wait
        	cvReleaseImage(&imageL); // release IplImage
        	cvReleaseImage(&imageR); // release IplImage
			cvDestroyWindow("Left"); // destroy
			cvDestroyWindow("Right");	
			*/
		}

		else if(type == DISP_CENSUS){
			Mat matL = cvarrToMat(imageL);
			Mat matR = cvarrToMat(imageR);	
			Mat matD = cvarrToMat(imageD);
			Mat matTemp;
			Mat matTempTemp;
			cvtColor(matL, matL, cv::COLOR_BGR2GRAY);
			cvtColor(matR, matR, cv::COLOR_BGR2GRAY);
			cvtColor(matD, matTemp, cv::COLOR_BGR2GRAY);
			census_33(matL, matR, matTemp, DEPTH, BSIZE);
			
			cvtColor(matTemp, matTempTemp , cv::COLOR_GRAY2BGR);
			// This is a code that prevent you from core dump//
			IplImage* image2;
			image2 = cvCreateImage(cvSize(matTempTemp.cols,matTempTemp.rows),8,3);
			IplImage ipltemp=matTempTemp;
			cvCopy(&ipltemp,image2);

			imageD = image2;
			//Grey(imageD);
			//cerr<<"Grey finish"<<endl;
			//cvShowImage("After:", imageD);
			//cvWaitKey(0);
		}
		else if(type == DISP_GRADIENT){
				
			Mat matL = cvarrToMat(imageL);
			Mat matR = cvarrToMat(imageR);	
			Mat matD = cvarrToMat(imageD);
			Mat matTemp;
			Mat matTempTemp;
			cvtColor(matL, matL, cv::COLOR_BGR2GRAY);
			cvtColor(matR, matR, cv::COLOR_BGR2GRAY);
			cvtColor(matD, matTemp, cv::COLOR_BGR2GRAY);

			gradient_x(matL, matR, matTemp, DEPTH, BSIZE);
			
			cvtColor(matTemp, matTempTemp , cv::COLOR_GRAY2BGR);
			// This is a code that prevent you from core dump//
			IplImage* image2;
			image2 = cvCreateImage(cvSize(matTempTemp.cols,matTempTemp.rows),8,3);
			IplImage ipltemp=matTempTemp;
			cvCopy(&ipltemp,image2);

			imageD = image2;
			if(usrGrey)
				Grey(imageD);
		}
		else if (type == DISP_STEREO_COLOR){
			if(usrGrey){
				Grey(imageL);
				Grey(imageR);
			}
			disp_stereo_color(imageL,imageR,imageD,DEPTH,false,BSIZE,isLeft);
		
		}
		else if (type == DISP_WINDOW_COLOR){
			if(usrGrey){
				Grey(imageL);
				Grey(imageR);
			}
			disp_stereo_color(imageL,imageR,imageD,DEPTH,true,BSIZE,isLeft);
		
		}
		else if (type == DISP_PATCH){
			patch(imageL, imageR, imageD);
			
		}
		else if (type == DISP_WINDOW_COLOR_FRAME){
			if(usrGrey){
				Grey(imageL);
				Grey(imageR);
			}
			disp_stereo_color_frame(imageL,imageR,imageD,imageL_pre,imageD_pre,DEPTH,true,BSIZE,isLeft,file_iter);
		}
		else if (type == DISP_WINDOW_COLOR_BFRAME){
			if(usrGrey){
				Grey(imageL);
				Grey(imageR);
			}
			disp_stereo_color_big_frame(imageL,imageR,imageD,imageL_pre,imageD_pre,DEPTH,true,BSIZE,isLeft,file_iter);
		}
		else if (type == DISP_WINDOW_COLOR_BFRAME_OPT){
			if(usrGrey){
				Grey(imageL);
				Grey(imageR);
			}
			disp_stereo_color_big_frame_optical(imageL,imageR,imageD,imageL_pre,imageD_pre, imageD_refine, DEPTH,true,BSIZE,isLeft,file_iter, CLEAN, SPEED, 1, NSIZE);
		}
		else if (type == DISP_WINDOW_COLORASW_BFRAME_OPT){
			if(usrGrey){
				Grey(imageL);
				Grey(imageR);
			}
			disp_stereo_color_big_frame_optical(imageL,imageR,imageD,imageL_pre,imageD_pre, imageD_refine, DEPTH,true,BSIZE,isLeft,file_iter, CLEAN, SPEED, 2, NSIZE);
		}
		else if (type == NEW_BP_BFRAME_OPT){
			if(usrGrey){
				Grey(imageL);
				Grey(imageR);
			}
			new_bp_frame_optical(imageL,imageR,imageD,imageL_pre,imageD_pre, imageD_refine, DEPTH,true,BSIZE,isLeft,file_iter, CLEAN, SPEED, 2, NSIZE);
		}
		else if (type == NEW_BP){
			if(usrGrey){
				Grey(imageL);
				Grey(imageR);
			}
			new_bp(imageL,imageR,imageD,imageL_pre,imageD_pre, imageD_refine, DEPTH,true,BSIZE,isLeft,file_iter, CLEAN, SPEED, 2, NSIZE);
		}
		else if (type == NEW_BP_TEMPORAL){
			if(usrGrey){
				Grey(imageL);
				Grey(imageR);
			}
			new_bp_temporal(imageL,imageR,imageD,imageL_pre,imageD_pre, imageD_refine, DEPTH,true,BSIZE,isLeft,file_iter, CLEAN, SPEED, 2, NSIZE);
		}
		// else if (type == DISP_WINDOW_BP_BFRAME_OPT){
		// 	if(usrGrey){
		// 		Grey(imageL);
		// 		Grey(imageR);
		// 	}
		// 	useFilter=false; // Because BP have filter!!!!!!
		// 	disp_stereo_bp_big_frame_optical(imageL,imageR,imageD,imageL_pre,imageD_pre,DEPTH,true,BSIZE,isLeft,file_iter, CLEAN, SPEED);
		// }
		else if (type==DISP_TRUE_AFFINE){
			if(usrGrey){
				Grey(imageL);
				Grey(imageR);
			}
			useFilter=false; // we do it here!!
			//Do original Stereo Matching and set Clean everytime!!
			disp_stereo_census_big_frame_optical(imageL,imageR,imageD,imageL_pre,imageD_pre,DEPTH,true,BSIZE,isLeft,file_iter, 1, 0, false);
			JointWMF jj;		
			Mat matL = cvarrToMat(imageL);
			Mat matD = cvarrToMat(imageD);
			//cvtColor(matL, matL, cv::COLOR_BGR2GRAY);
			//cvtColor(matD, matD, cv::COLOR_BGR2GRAY);
			Mat newResult = jj.filter(matD, matL,5);
			IplImage* image2;
			image2 = cvCreateImage(cvSize(newResult.cols,newResult.rows),8,3);
			IplImage ipltemp=newResult;
			cvCopy(&ipltemp,imageD); 
			Grey(imageD);
			true_affine(imageL, imageR, imageD, imageL_pre, imageD_pre, imageD_refine, DEPTH, BSIZE, isLeft, file_iter, CLEAN, false);
		}
		else if (type == TEST_AFFINE){
			if(usrGrey){
				Grey(imageL);
				Grey(imageR);
			}
			disp_stereo_census_big_frame_optical(imageL,imageR,imageD,imageL_pre,imageD_pre,DEPTH,true,BSIZE,isLeft,file_iter, 1, 0, false);
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
			testing_affine(imageL, imageD, imageL_pre, imageD_pre, imageD_refine, file_iter);
			//recalculate
		}

		else if (type == TEST_OPTICAL){
			if(usrGrey){
				Grey(imageL);
				Grey(imageR);
			}
			testing_optical(imageL, imageD, imageL_pre, file_iter, SPEED);
		}


		//post Processing

		JointWMF jj;
		Mat matL = cvarrToMat(imageL);
		if(useFilter){
			Mat matD = cvarrToMat(imageD);
			//cvtColor(matL, matL, cv::COLOR_BGR2GRAY);
			//cvtColor(matD, matD, cv::COLOR_BGR2GRAY);
			Mat newResult = jj.filter(matD, matL,FILTERSIZE);
			
			// This is a code that prevent you from core dump//
			IplImage* image2;
			image2 = cvCreateImage(cvSize(newResult.cols,newResult.rows),8,3);
			IplImage ipltemp=newResult;
			cvCopy(&ipltemp,imageD); 
			Grey(imageD);
			//cvShowImage("After:", imageD);
			//cvWaitKey(0);
		}
		if(refineUseFilter && imageD_refine!=0){
			Mat matD_refine = cvarrToMat(imageD_refine);
			Mat newResult_refine = jj.filter(matD_refine, matL, FILTERSIZE);
			IplImage ipltemp_refine=newResult_refine;
			cvCopy(&ipltemp_refine,imageD_refine); 
			Grey(imageD_refine);
		}
		if(imageD_pre==0) // first time
			imageD_refine = cvCloneImage(imageD);
		//?!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		if(type==DISP_TRUE_AFFINE||type==DISP_WINDOW_COLOR_BFRAME_OPT||type==DISP_WINDOW_COLORASW_BFRAME_OPT)
			imageD_pre = cvCloneImage(imageD_refine);
		else{
			cvReleaseImage(&imageD_pre);
			imageD_pre = cvCloneImage(imageD);
		}
		cvReleaseImage(&imageL_pre);	
		imageL_pre = cvCloneImage(imageL);

		//This Part is for testing in TSUKUBA FILTER//
		//Remember to comment these for real data running!!!!!!!!!!!//
		/*
		if(file_iter==1){
			string fileFilter = "Tsukuba/filter_1.png";
			char filenamefileFilter[1024];
			strcpy(filenamefileFilter, fileFilter.c_str());
			imageD_pre = cvLoadImage(filenamefileFilter, CV_LOAD_IMAGE_COLOR);
		}
		*/
		
		// Save Output Image //
		string fileO=img_output_directory + img_name + "_o.png";
		char outputfilename[1024];
		strcpy(outputfilename, fileO.c_str());
		//imwrite(outputfilename,imageD);
		if(type<100 && outputOrig)
			cvSaveImage(outputfilename, imageD);

		//cvShowImage("Output", imageD);
		//cvWaitKey(0); // wait
		fileO=img_output_directory + "refine/" + img_name + "_r.png";
		strcpy(outputfilename, fileO.c_str());
		// save image//
		if(type<100)
			cvSaveImage(outputfilename, imageD_refine);
		if(Compare){       
			IplImage *imageDC, *imageCP; //type of image
			/*
			string fileDC = img_discontinue_directory + "left/"+ img_name + ".png";
			string fileCP = img_compare_directory + "left/" + img_name + ".png";
			*/
			
			// for DCB
			string fileDC = img_discontinue_directory + "TL_MASK"+ img_name + ".png";
			string fileCP = img_compare_directory + "left/TL" + img_name + ".png";
			
			char filenameDC[1024];
			char filenameCP[1024];
			strcpy(filenameDC, fileDC.c_str());
			strcpy(filenameCP, fileCP.c_str());
            if(img_discontinue_directory==""){
				imageDC = cvLoadImage(filenameCP, CV_LOAD_IMAGE_UNCHANGED);
				cvSet(imageDC, CV_RGB(23,23,23));
			}
			else
				imageDC = cvLoadImage(filenameDC, CV_LOAD_IMAGE_UNCHANGED);
			imageCP = cvLoadImage(filenameCP, CV_LOAD_IMAGE_UNCHANGED);
			if(!imageDC){
				cerr<<"Help, no DC for name:"<<filenameDC<<endl;
				return 0;
			}
			
			//Grey(imageCP);
		
			//cvShowImage("imageDC", imageDC);
            //cvWaitKey(0); // wait
			
			// Error //
			double total_pixel = 0;
			double error_pixel = 0;
			double frame_error;
			for(int i=0;i<imageD->height;i+=JUMP){
				for(int j=0;j<imageD->widthStep;j=j+3*JUMP){
					if((int)(imageDC->imageData[i*imageDC->widthStep+j*(imageDC->nChannels)/3]) != 0 && !(j<(BSIZE*3) + 30 && deleteLeftDiscont)){
						total_pixel++;
						int colorint=22;
						int dif=abs((imageD->imageData[i*imageD->widthStep+j])-(imageCP->imageData[i*imageCP->widthStep+j*(imageCP->nChannels)/3]));	
						if(dif>compareTolerance){
							colorint=200;
							imageD->imageData[i*imageD->widthStep+j] = 0;
			          		imageD->imageData[i*imageD->widthStep+j+1] = 0;
             	 			imageD->imageData[i*imageD->widthStep+j+2] = colorint;
							error_pixel++;
						}	
					}
					// if j<BSIZE*3 && deleteLeftDiscont
					else{
						imageD->imageData[i*imageD->widthStep+j] = 0;
			          	imageD->imageData[i*imageD->widthStep+j+1] = 200;
             	 		imageD->imageData[i*imageD->widthStep+j+2] = 0;
					}
				}
			}
			frame_error = (error_pixel/total_pixel);
			error_sum += frame_error;
			string fileO=img_output_directory + "compare/" + img_name + "_o_c.png";
			char outputfilename[1024];
			strcpy(outputfilename, fileO.c_str());
			// cvShowImage("Output", imageD);
            // cvWaitKey(0); // wait
			// save image//put
			if(type<100)
				cvSaveImage(outputfilename, imageD);

	
			cerr<<"Error: "<< frame_error*100<<"%;";

			double refine_frame_error = 0;
			if(imageD_refine!=NULL && imageD_refine!=0){
				double refine_total_pixel = 0;
				double refine_error_pixel = 0;
				for(int i=0;i<imageD_refine->height;i+=JUMP){
					for(int j=0;j<imageD_refine->widthStep;j=j+3*JUMP){
						if((int)(imageDC->imageData[i*imageDC->widthStep+j*(imageDC->nChannels)/3]) != 0 && !(j<BSIZE*3 +30 && deleteLeftDiscont)){
							refine_total_pixel++;
							int colorint=22;
							int dif=abs((imageD_refine->imageData[i*imageD_refine->widthStep+j])-(imageCP->imageData[i*imageCP->widthStep+j*(imageCP->nChannels)/3]));	
							if(dif>compareTolerance){				
								colorint=200;
								imageD_refine->imageData[i*imageD->widthStep+j] = 0;
								imageD_refine->imageData[i*imageD->widthStep+j+1] = 0;
								imageD_refine->imageData[i*imageD->widthStep+j+2] = colorint;
								refine_error_pixel++;
							}	
						}
						else{
							imageD_refine->imageData[i*imageD->widthStep+j] = 0;
							imageD_refine->imageData[i*imageD->widthStep+j+1] = 200;
							imageD_refine->imageData[i*imageD->widthStep+j+2] = 0;
						}
					}
				}
				refine_frame_error = (refine_error_pixel/refine_total_pixel);
				cerr<<"\nRefine Error: "<<refine_frame_error*100<<"%";
				// cvShowImage("Output", imageD_refine);
            	// cvWaitKey(0); // wait
				string fileO=img_output_directory + "refine_compare/" + img_name + "_r_c.png";
				char outputfilename[1024];
				strcpy(outputfilename, fileO.c_str());
				// save image//
				if(type<100)
					cvSaveImage(outputfilename, imageD_refine);
			}
						// Timer End //


			//output record file
			ofstream outfile;
			string fileRecord=img_output_directory+"record.csv";
			char fileRecordChar[1024];
			strcpy(fileRecordChar, fileRecord.c_str());
  			outfile.open(fileRecordChar, ios_base::app);
  			outfile <<img_name<<","<<frame_error<<","<<refine_frame_error<<endl; 
			//
		}
		float time_per_frame = (float)(clock() - begin_time)/1000000;
		cerr<<"; SPF: "<<time_per_frame <<"s";
		cerr<<"; Time Left: "<<(END-file_iter)*time_per_frame;	
		cerr<<"   "<<endl;
	}
	int total_time = (clock()-total_begin_time)/1000000;
	cout<<"Finish!!\n"<<endl;
	cout<<"==============SUMMARY=============="<<endl;
	cout<<endl;
	cout<<"Total Time: "<<total_time/60<<" mins "<<total_time%60<<" secs ("<<total_time<<" secs)"<<endl;
	cout<<"Average Time: "<<(float)total_time/(END-START)<<"(secs)"<<endl;
	cout<<"Average Correct Rate: "<<(float)(1-(error_sum)/(END-START+1))*100<<"%"<<endl;
	cout<<endl;
	cout<<"==================================="<<endl;
	return EXIT_SUCCESS;
}


/*

Fmaj7 F7 Cmaj7 Am Fmaj7 G C7 Am

F G C Am

F Fm C

*/
