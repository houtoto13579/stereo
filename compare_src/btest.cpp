#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <string>
#include <stdio.h>
#include <ctime>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
// Self Lib //
#include "test.h"

//============ Define ============//

#define HEIGHT 1000
#define WIDTH 1000
#define DEPTH 128
#define BSIZE 5
#define JUMP 1
#define START 1
#define END 30
using namespace std;
using namespace cv;

bool Compare = true;
int compareTolerance = 4;

//========== Error_Rate ===========//

double error_sum = 0;

//========= Progress Bar ==========//

int totalPoint = 50;
float pcount = 0;
float pcountadd = (float)(END-START)/totalPoint;


//=========  Directory ============//
string category = "book/";
string noise = "";

string img_directory = "input/Dcb/"+category;
string img_left = "left"+noise+"/";
string img_right = "right"+noise+"/";

string img_name = "";
string child_directory = img_left;
string img_A_dir = "../output/Dcb_paper_nf/"+category+"a0_005_T6/";
string img_B_dir = "../output/Dcb_paper_nf/"+category+"a0_01_T6/";
string img_discontinue_directory = "../input/Dcb/"+category+"occlution/";
string img_compare_directory = "../input/Dcb/"+category+"ground/";

// string img_discontinue_directory = "Tsukuba/discontinuity_maps/";
// string img_compare_directory = "Tsukuba/disparity_maps/";
// string img_A_dir = "output/Tsukuba/left/disp_window_color_B5_filter/";
// string img_B_dir = "output/Tsukuba/left/disp_window_color_frame_big_B5_sp5_filter/";
	
int main(int argc, char** argv){
	float totalStability_A = 0;
	float totalStability_B = 0;
	float totalError_A = 0;
	float totalError_B = 0;
	
	cout<<"=== Comparison Checking ==="<<endl;
	cout<<"Initiate Comparison from "<<START<<" to "<<END<<endl;
	// Total Timer Begin
	const clock_t total_begin_time = clock();
	for(int file_iter = START; file_iter<=END; file_iter++){	
		// Timer Begin //
		const clock_t begin_time = clock();

		cerr<<"\r*Now Processing("<<file_iter<<"/"<<END<<") ";
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

	
		// string img_name_1 = "frame_"+num2str(file_iter);
		// string img_name_2 = "frame_"+num2str(file_iter+1);
		string img_name_1 = num2file(file_iter,4);
		string img_name_2 = num2file(file_iter+1,4);

		string file_A_1 = img_A_dir + img_name_1 +"_o.png";
		string file_A_2 = img_A_dir + img_name_2 +"_o.png";	
		string file_B_1 = img_B_dir + img_name_1 +"_o.png";
		string file_B_2 = img_B_dir + img_name_2 +"_o.png";	
		char filename_A_1[1024];
		char filename_A_2[1024];
		char filename_B_1[1024];
		char filename_B_2[1024];
		strcpy(filename_A_1, file_A_1.c_str());
		strcpy(filename_A_2, file_A_2.c_str());
		strcpy(filename_B_1, file_B_1.c_str());
		strcpy(filename_B_2, file_B_2.c_str());
		// Image Size: 1000x669
		IplImage *imgA1,*imgA2,*imgB1,*imgB2; //type of image
 	 	imgA1 = cvLoadImage(filename_A_1,CV_LOAD_IMAGE_COLOR); // CVload type
 	 	imgA2 = cvLoadImage(filename_A_2,CV_LOAD_IMAGE_COLOR); // CVload type
 	 	imgB1 = cvLoadImage(filename_B_1,CV_LOAD_IMAGE_COLOR); // CVload type
 	 	imgB2 = cvLoadImage(filename_B_2,CV_LOAD_IMAGE_COLOR); // CVload type
		int width = imgA1->width;
		int height = imgA1->height;
		if(!imgA1){
			cout<<"Error: Couldn't open the image A1 file.\n";
			return 0;
		}
		if(!imgA2){
			cout<<"Error: Couldn't open the image A2 file.\n";
			return 0;
		}
		if(!imgB1){
			cout<<"Error: Couldn't open the image B1 file.\n";
			return 0;
		}
		if(!imgB2){
			cout<<"Error: Couldn't open the image B2 file.\n";
			return 0;
		}
		if(Compare){       
			
			IplImage *imgDC1, *imgDC2, *imgCP1, *imgCP2; //type of image
			// string fileDC_1 = img_discontinue_directory + "left/"+ img_name_1 + ".png";
			// string fileCP_1 = img_compare_directory + "left/" + img_name_1 + ".png";
			// string fileDC_2 = img_discontinue_directory + "left/"+ img_name_2 + ".png";
			// string fileCP_2 = img_compare_directory + "left/" + img_name_2 + ".png";

			string fileDC = img_discontinue_directory + "TL_MASK"+ img_name + ".png";
			string fileCP = img_compare_directory + "left/TL" + img_name + ".png";

			string fileDC_1 = img_discontinue_directory + "TL_MASK"+ img_name_1 + ".png";
			string fileCP_1 = img_compare_directory + "left/TL" + img_name_1 + ".png";
			string fileDC_2 = img_discontinue_directory + "TL_MASK"+ img_name_2 + ".png";
			string fileCP_2 = img_compare_directory + "left/TL" + img_name_2 + ".png";
			char filenameDC_1[1024];
			char filenameCP_1[1024];
			char filenameDC_2[1024];
			char filenameCP_2[1024];
			strcpy(filenameDC_1, fileDC_1.c_str());
			strcpy(filenameCP_1, fileCP_1.c_str());
			strcpy(filenameDC_2, fileDC_2.c_str());
			strcpy(filenameCP_2, fileCP_2.c_str());
            
			imgDC1 = cvLoadImage(filenameDC_1, CV_LOAD_IMAGE_UNCHANGED);
			imgCP1 = cvLoadImage(filenameCP_1, CV_LOAD_IMAGE_UNCHANGED);
			imgDC2 = cvLoadImage(filenameDC_1, CV_LOAD_IMAGE_UNCHANGED);
			imgCP2 = cvLoadImage(filenameCP_1, CV_LOAD_IMAGE_UNCHANGED);
			if(!imgDC1){
				cerr<<"Help, no DC for name:"<<filenameDC_1<<endl;
				return 0;
			}
			// Error //
			double total_pixel = 0;
			double frame_error;
			double A_error_pixel = 0;
			double B_error_pixel = 0;
			int improvement_A = 0;
			int improvement_B = 0;
			int errorment_A = 0;
			int errorment_B = 0;
			float remain_A = 0;
			float remain_B = 0;
			float total_pre = 0;
			int channel = imgCP1->nChannels;
			for(int i=0;i<imgA1->height;i+=JUMP){
				for(int j=0;j<imgA1->widthStep;j=j+3*JUMP){
					if((int)(imgDC1->imageData[i*imgDC1->widthStep+j*(imgDC1->nChannels)/3]) != 0){
						total_pixel++;
						int difA1=abs((uchar)(imgA1->imageData[i*imgA1->widthStep+j])-(uchar)(imgCP1->imageData[i*imgCP1->widthStep+j*channel/3]));	
						if(difA1<=compareTolerance){
						}
						else{
							A_error_pixel++;
						}
						if((int)(imgDC2->imageData[i*imgDC2->widthStep+j*(imgDC1->nChannels)/3]) != 0){
							total_pre++;
							int difA2=abs((uchar)(imgA2->imageData[i*imgA2->widthStep+j])-(uchar)(imgCP2->imageData[i*imgCP2->widthStep+j*channel/3]));	
							if (difA2<difA1)
								improvement_A++;
							else if (difA2>difA1)
								errorment_A++;
							else
								remain_A++;
						}	
						int difB1=abs((uchar)(imgB1->imageData[i*imgB1->widthStep+j])-(uchar)(imgCP1->imageData[i*imgCP1->widthStep+j*channel/3]));	
						if(difB1>compareTolerance){
							B_error_pixel++;
						}
						if((int)(imgDC2->imageData[i*imgDC2->widthStep+j*(imgDC1->nChannels)/3]) != 0){
							int difB2=abs((uchar)(imgB2->imageData[i*imgB2->widthStep+j])-(uchar)(imgCP2->imageData[i*imgCP2->widthStep+j*channel/3]));	
							if (difB2<difB1)
								improvement_B++;
							else if (difB2>difB1)
								errorment_B++;
							else
								remain_B++;
						}
					}
				}
			}			
			// Timer End //
			float time_per_frame = (float)(clock() - begin_time)/1000000;	
			
			
			//##################Print Function###################//

			//cerr<<" Error: "<< frame_error*100<<"%";
			//cerr<<"; SPF: "<<time_per_frame <<"sec";
			//cerr<<"; Time Left: "<<(END-file_iter)*time_per_frame;
			
			float stability_A = remain_A*100/total_pre;
			float stability_B = remain_B*100/total_pre;
			

			totalError_A += A_error_pixel*100/total_pixel;
			totalError_B += B_error_pixel*100/total_pixel;
			totalStability_A += stability_A;
			totalStability_B += stability_B;
			cerr<<"Error_A: "<<A_error_pixel*100/total_pixel<<"%, _B:"<<B_error_pixel*100/total_pixel<<"%"<<endl;
			// cerr<<"Improve_A: "<<improvement_A<<", _B:"<<improvement_B;	
			// cerr<<"\nBeError_A: "<<errorment_A<<", _B:"<<errorment_B;
			// cerr<<"\nRemain_A: "<<remain_A<<", _B:"<<remain_B;
			// cerr<<"\nStability_A:"<<stability_A<<"%, _B:"<<stability_B<<"%";
			// cerr<<"   \n";
		}
	}
	cout<<"Finish!!\n"<<endl;
	cout<<"==============SUMMARY=============="<<endl;
	cout<<"dir:"<<img_A_dir<<endl;
	cout<<"Average Error: "<<totalError_A/(END-START+1)<<"%"<<endl;
	cout<<"Average Stability: "<<totalStability_A/(END-START+1)<<"%"<<endl;
	cout<<endl;
	cout<<"#-#-#-#-#-#"<<endl;
	cout<<endl;
	cout<<"dir:"<<img_B_dir<<endl;
	cout<<"Average Error: "<<totalError_B/(END-START+1)<<"%"<<endl;
	cout<<"Average Stability: "<<totalStability_B/(END-START+1)<<"%"<<endl;
	cout<<"==================================="<<endl;
	return EXIT_SUCCESS;
}


/*

Fmaj7 F7 Cmaj7 Am Fmaj7 G C7 Am

F G C Am

F Fm C

*/
