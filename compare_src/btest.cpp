#include <iostream>
#include <cv.h>
#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
// Self Lib //
#include "test.h"
#include <string>

//============ Define ============//

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

bool print_progress = false;

//=========  Directory ============//
string category = "tanks/";
string noise = "";
int method = 0; //1 for tsukuba
	
int main(int argc, char** argv){
	if(argc==2){
		category = argv[1];	
		category += "/";
	}
	string img_left, img_right, child_directory, parent_dir, img_discontinue_directory, img_compare_directory;
	if(method==0){
		img_left = "left"+noise+"/";
		img_right = "right"+noise+"/";
		child_directory = img_left;
		parent_dir = "../output/Dcb_paper_final/"+category;
		img_discontinue_directory = "../input/Dcb/"+category+"occlution/";
		img_compare_directory = "../input/Dcb/"+category+"ground/";
	}
	else if (method == 1){
		img_left = "left"+noise+"/";
		img_right = "right"+noise+"/";
		child_directory = img_left;
		parent_dir = "../output/Tsukuba_paper_1/";
		img_discontinue_directory = "../input/Tsukuba/discontinuity_maps/";
		img_compare_directory = "../input/Tsukuba/disparity_maps/";
	}
	ifstream my_list("dir_list.txt");
	string line;
	vector<string> dir_vec;
	while (std::getline(my_list, line))
		dir_vec.push_back(line);
	int dir_count = dir_vec.size();
	vector<float> total_sta_arr(dir_count, 0);
	vector<float> total_err_arr(dir_count, 0);
	
	cout<<"=== Comparison Checking ==="<<endl;
	cout<<"Initiate Comparison from "<<START<<" to "<<END<<endl;
	for(int file_iter = START; file_iter<=END; file_iter++){	
		if(print_progress){
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
		}
		string img_name_1, img_name_2;
		if(method==0){
			img_name_1 = num2file(file_iter,4);
			img_name_2 = num2file(file_iter+1,4);
		}
		else if (method==1){
			img_name_1 = "frame_"+num2str(file_iter);
			img_name_2 = "frame_"+num2str(file_iter+1);
		}
		// Image Size: 1000x669
		if(Compare){
			IplImage *imgDC1, *imgDC2, *imgCP1, *imgCP2; //type of image
			string fileDC_1,fileDC_2,fileCP_1,fileCP_2;
			if(method==0){
				fileDC_1 = img_discontinue_directory + "TL_MASK"+ img_name_1 + ".png";
				fileCP_1 = img_compare_directory + "left/TL" + img_name_1 + ".png";
				fileDC_2 = img_discontinue_directory + "TL_MASK"+ img_name_2 + ".png";
				fileCP_2 = img_compare_directory + "left/TL" + img_name_2 + ".png";
			}
			else if (method==1){
				fileDC_1 = img_discontinue_directory + "left/"+ img_name_1 + ".png";
				fileCP_1 = img_compare_directory + "left/" + img_name_1 + ".png";
				fileDC_2 = img_discontinue_directory + "left/"+ img_name_2 + ".png";
				fileCP_2 = img_compare_directory + "left/" + img_name_2 + ".png";
			}
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
			IplImage **img1 = new IplImage*[dir_count];
			IplImage **img2 = new IplImage*[dir_count];
			for (int f=0; f<dir_count; ++f){
				// load image //
				string img_name_1_i = parent_dir+dir_vec[f]+"/"+img_name_1+"_o.png";
				string img_name_2_i = parent_dir+dir_vec[f]+"/"+img_name_2+"_o.png";
				//cerr<<"file"<<img_name_1<<endl;
				char filename_1[1024];
				char filename_2[1024];
				strcpy(filename_1, img_name_1_i.c_str());
				strcpy(filename_2, img_name_2_i.c_str());
				img1[f] = cvLoadImage(filename_1,CV_LOAD_IMAGE_COLOR); 
				img2[f] = cvLoadImage(filename_2,CV_LOAD_IMAGE_COLOR);
				if(!img1[f]){
					cerr<<"Help, no file for name:"<<filename_1<<endl;
					return 0;
				}
				if(!img2[f]){
					cerr<<"Help, no file for name:"<<filename_2<<endl;
					return 0;
				}
			}
			int img_height = img1[0]->height;
			int img_widstep = img1[0]->widthStep;

			vector<int> err_pix_arr(dir_count, 0);
			vector<float> improvement_arr(dir_count, 0);
			vector<float> errorment_arr(dir_count, 0);
			vector<float> remain_arr(dir_count, 0);
			float total_pre_pixel = 0;
			int CP_chan = imgCP1->nChannels;
			int DC_chan = imgDC1->nChannels;
			for(int i=0;i<img_height;i+=JUMP){
				for(int j=0;j<img_widstep;j=j+3*JUMP){
					if((int)(imgDC1->imageData[i*imgDC1->widthStep+j*(DC_chan)/3]) != 0){
						total_pixel++;
						for (int f=0; f<dir_count; ++f){
							int dif1=abs((uchar)(img1[f]->imageData[i*img1[f]->widthStep+j])-(uchar)(imgCP1->imageData[i*imgCP1->widthStep+j*CP_chan/3]));
							if(dif1>compareTolerance)
								err_pix_arr[f]++;
							if((int)(imgDC2->imageData[i*imgDC2->widthStep+j*(DC_chan)/3]) != 0){
								total_pre_pixel++;
								int dif2=abs((uchar)(img2[f]->imageData[i*img2[f]->widthStep+j])-(uchar)(imgCP2->imageData[i*imgCP2->widthStep+j*CP_chan/3]));	
								if (dif2<dif1)
									improvement_arr[f]++;
								else if (dif2>dif1)
									errorment_arr[f]++;
								else
									remain_arr[f]++;
							}
						}
					}
				}
			}
			//##################Print Function###################//
			for(int f=0; f<dir_count; ++f){
				total_sta_arr[f] += remain_arr[f]*100/(total_pre_pixel)*dir_count;
				total_err_arr[f] += err_pix_arr[f]*100/total_pixel;
			}
		}
	}
	cout<<"==============SUMMARY=============="<<endl;
	for(int f=0; f<dir_count; ++f){
		cout<<"dir: "<< dir_vec[f] <<endl;
		cout<<"Avg Err: "<<total_err_arr[f]/(END-START+1)<<"%"<<endl;
		cout<<"Avg Sta: "<<total_sta_arr[f]/(END-START+1)<<"%"<<endl;
		cout<<"#-#-#-#-#-#-#-#-#-#-#-#"<<endl;
	}
	return EXIT_SUCCESS;
}


/*

Fmaj7 F7 Cmaj7 Am Fmaj7 G C7 Am

F G C Am

F Fm C

*/
