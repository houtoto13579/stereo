#ifndef TEST_H
#define TEST_H


#include <sstream>
#include <iostream>

using namespace std;

class cmakeTest{
	public:
		cmakeTest(){
			a = 5;
		}
	
		int getA(){return a;}
	private:
		int a;
};

#endif
//Tsukuba

#define GREY_WIDTH 640
#define GREY_HEIGHT 480

//Kitti
/*
#define GREY_WIDTH 1242
#define GREY_HEIGHT 375
*/
//Nagoya-Champagne, Dog
/*
#define GREY_WIDTH 1280
#define GREY_HEIGHT 960
*/
//Nagoya-Kendo	
/*
#define GREY_WIDTH 1024
#define GREY_HEIGHT 768
*/
void Grey(IplImage *Image1){

	uchar Blue[GREY_HEIGHT][GREY_WIDTH];
	uchar Green[GREY_HEIGHT][GREY_WIDTH];
	uchar Red[GREY_HEIGHT][GREY_WIDTH];
	uchar Gray[GREY_HEIGHT][GREY_WIDTH];

	/* Load Image RGB Values */
	for(int i=0;i<Image1->height;i++){
      for(int j=0;j<Image1->widthStep;j=j+3){
         Blue[i][(int)(j/3)]=Image1->imageData[i*Image1->widthStep+j];
         Green[i][(int)(j/3)]=Image1->imageData[i*Image1->widthStep+j+1];
         Red[i][(int)(j/3)]=Image1->imageData[i*Image1->widthStep+j+2];
   		}
   	}
 /* Implement Algorithms */
   for(int i=0;i<Image1->height;i++){
      for(int j=0;j<Image1->width;j++){
         Gray[i][j]=(uchar)(0.299*Red[i][j] + 0.587*Green[i][j] + 0.114*Blue[i][j]);
         Red[i][j]=Gray[i][j];
         Green[i][j]=Gray[i][j];
         Blue[i][j]=Gray[i][j];
  	   }
 	}
 /* Save Image RGB Values */
   for(int i=0;i<Image1->height;i++){
		for(int j=0;j<Image1->widthStep;j=j+3){
			Image1->imageData[i*Image1->widthStep+j]=Blue[i][(int)(j/3)];
	      	Image1->imageData[i*Image1->widthStep+j+1]=Green[i][(int)(j/3)];
      		Image1->imageData[i*Image1->widthStep+j+2]=Red[i][(int)(j/3)];
     	}
   }
	//cvShowImage("Test", Image1);
    //cvWaitKey(0); // wait
}

string num2file(int num, int amount)
{
	stringstream ss;
	// the number is converted to string with the help of stringstream
	ss << num; 
	string ret;
	ss >> ret;						
	// Append zero chars
	int str_length = ret.length();
	for (int i = 0; i < amount - str_length; i++)
		ret = "0" + ret;
	return ret;
}
string num2str(int num)
{
	stringstream ss;
	// the number is converted to string with the help of stringstream
	ss << num; 
	string ret;
	ss >> ret;						
	return ret;
}
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
