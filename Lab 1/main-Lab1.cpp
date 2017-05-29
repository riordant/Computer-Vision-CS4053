#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <stdio.h>
#include <string>   


using namespace std;

const int NUM_IMAGES = 6;
const int NUM_TEST_IMAGES = 2;
const int NUM_CHANNELS = 3;
const int NUM_BOTTLES_PER_IMAGE = 5;
const int BOTTLE_OFFSET = 125;
const int FIRST_X_IMG_POS = 175;
const int SECOND_X_IMG_POS = 265;
const int FIRST_Y_IMG_POS = 30;
const int SECOND_Y_IMG_POS = 70;

void drawImg(cv::Mat labelArea,int img,int bottle){
    
    string out = "Image " + to_string(img) + " Bottle " + to_string(bottle);
    cv::namedWindow(out,cv::WINDOW_AUTOSIZE);
    cv::moveWindow(out,0,500);
    imshow(out,labelArea);
}

void drawHistogram(cv::Mat labelArea,int img,int bottle){
    /*
     * this function is taken from http://docs.opencv.org/doc/tutorials/imgproc/histograms/histogram_calculation/histogram_calculation.html
     */
    vector<cv::Mat> bgr_planes;
    split( labelArea, bgr_planes );
    cv:: Mat b_hist, g_hist, r_hist;
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    int histSize = 256;
    bool uniform = true; bool accumulate = false;
    
    calcHist( &bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
    
    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    
    cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
    
    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
    
    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
             cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
             cv::Scalar(255, 0, 0), 2, 8, 0);
        line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
             cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
             cv::Scalar(0, 255, 0), 2, 8, 0);
        line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
             cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
             cv::Scalar(0, 0, 255), 2, 8, 0  );
    }
    
    /// Display
    cv::namedWindow("", CV_WINDOW_AUTOSIZE );
    imshow("", histImage );
    cv::waitKey();
}

int main(int argc, char** argv) {
    
    //load images.
    char filename[100];
    cv::Mat images[NUM_IMAGES];
    for (int file_num=1; file_num <= NUM_IMAGES; file_num++)
    {
        sprintf(filename,"./Glue%d.jpg",file_num);
        images[file_num-1] = cv::cvarrToMat(cvLoadImage(filename,-1)); //have to use cvLoadImage and convert as my installation of openCV is having problems with imread.
    }
    
    //images are now loaded.
    //I am going to use two images as test data to deduce my standard deviation threshold value, ie. I know where the labels/non-labels are in these images.
    //I will just use the first two images from the total set of 6 for this process, getting the standard deviations of each.
    //I will use the highest value of the no label bottles and the lowest value of the labelled bottles, and get the average of the two for the threshold.
    
    cv::Mat currLabelArea,labelImageRed,labelImageGreen,labelImageBlue;
    cv::Mat channels[NUM_CHANNELS];
    int img = 0;
    int noLabel[NUM_TEST_IMAGES][2] = {{0,4},{1,1}};//image and bottle pointers for the bottles with no label.
    
    cv::Scalar meanR,stddevR,meanG,stddevG,meanB,stddevB;
    double avgStdDev=0,noLabelHighest=0,labelLowest = 0;
    
    for(; img < NUM_TEST_IMAGES; img++)
    {
        for(int bottle=0; bottle<NUM_BOTTLES_PER_IMAGE;bottle++)
        {
            //the following code selects the label area from each subsequent bottle.
            currLabelArea = images[img](cv::Range(FIRST_X_IMG_POS, SECOND_X_IMG_POS), cv::Range(FIRST_Y_IMG_POS +(BOTTLE_OFFSET*bottle), SECOND_Y_IMG_POS +(BOTTLE_OFFSET*bottle)));
            
            cv::split(currLabelArea, channels);
            
            labelImageRed = channels[2];
            labelImageGreen = channels[1];
            labelImageBlue = channels[0]; //split image into RGB values
            
            cv::meanStdDev(labelImageRed,meanR,stddevR);
            cv::meanStdDev(labelImageGreen,meanG,stddevG);
            cv::meanStdDev(labelImageBlue,meanB,stddevB); // calculate stddev of each.
            avgStdDev = (stddevR.val[0] + stddevG.val[0] + stddevB.val[0])/NUM_CHANNELS; //get overall average
            
            printf("average standard deviation for image %d and bottle %d: %f\n",img,bottle,avgStdDev);
            
            //the first condition here is satisfied if the current label area has no label.
            if((img == noLabel[img][0]) && (bottle == noLabel[img][1])){ if(noLabelHighest < avgStdDev) noLabelHighest = avgStdDev;}
            else { if((avgStdDev < labelLowest) || (labelLowest == 0)) labelLowest = avgStdDev;} //labelLowest is initialised at zero so the second condition handles that
            
            drawImg(currLabelArea,img,bottle);
            drawHistogram(currLabelArea,img,bottle); //draw the label area and a histogram output of
            
        }
    }
    
    double threshold = (ceil(noLabelHighest) + floor(labelLowest))/2; //calculate a threshold value for the standard deviation.
    
    printf("highest value of bottles with no label: %f\n", noLabelHighest);
    printf("lowest value of bottles with a label: %f\n", labelLowest);
    printf("threshold: %f\n", threshold);
    
    //we now use the threshold value. we cycle the rest of the images and bottles, performing standard deviation on each and asserting whether the bottle has a label or not.
    for(; img < NUM_IMAGES; img++)
    {
        for(int bottle=0; bottle<NUM_BOTTLES_PER_IMAGE;bottle++)
        {
            currLabelArea = images[img](cv::Range(FIRST_X_IMG_POS, SECOND_X_IMG_POS), cv::Range(FIRST_Y_IMG_POS +(BOTTLE_OFFSET*bottle), SECOND_Y_IMG_POS +(BOTTLE_OFFSET*bottle)));
            
            cv::split(currLabelArea, channels);
            
            labelImageRed = channels[2];
            labelImageGreen = channels[1];
            labelImageBlue = channels[0];
            
            cv::meanStdDev(labelImageRed,meanR,stddevR);
            cv::meanStdDev(labelImageGreen,meanG,stddevG);
            cv::meanStdDev(labelImageBlue,meanB,stddevB);
            avgStdDev = (stddevR.val[0] + stddevG.val[0] + stddevB.val[0])/NUM_CHANNELS;
            string isLabel = " ";
            if(avgStdDev < threshold) isLabel = " not ";
            printf("the average standard deviation for this label area is %f.\n", avgStdDev);
            printf("therefore bottle %d in image %d does%shave a label.\n",bottle,img,isLabel.c_str());
            drawImg(currLabelArea,img,bottle);
            drawHistogram(currLabelArea,img,bottle);
        }
    }
    
}