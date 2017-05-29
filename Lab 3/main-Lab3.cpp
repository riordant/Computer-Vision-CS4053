#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <vector> // STL dynamic memory.
#include <cmath>
#include <fstream>

using namespace cv;
using namespace std;

int getFrameCount(String path)
{
    /*
     * regular getframes function was not working. This function is referenced from the following resource:
     * http://www.ymer.org/amir/2007/06/04/getting-the-number-of-frames-of-an-avi-video-file-for-opencv/
     */
    int  nFrames;
    char tempSize[4];
    
    // Trying to open the video file
    ifstream  videoFile(path, ios::in | ios::binary );
    // Checking the availablity of the file
    if ( !videoFile ) {
        cout << "Couldn't open the input file " << path << endl;
        exit( 1 );
    }
    
    // get the number of frames
    videoFile.seekg( 0x30 , ios::beg );
    videoFile.read( tempSize , 4 );
    nFrames = (unsigned char ) tempSize[0] + 0x100*(unsigned char ) tempSize[1] + 0x10000*(unsigned char ) tempSize[2] +    0x1000000*(unsigned char ) tempSize[3];
    
    videoFile.close();
    return nFrames;
    
}

/*
 for every contoured object, calculate it's centroid and search against found_objects.
 global objects stored as: centroid, isDAbandoned, framesPresentFor.
 if framesPresentFor >= 4*FPS(4 seconds), isDetected becomes true.
 if found (margin error of 1), increment that objects' frame count.
 if not, add to global objects with moments and a count of 1
*/

struct contour
{
    Point2f centroid;
    double area;
    vector<Point> points;
    bool foundThisFrame;
    bool isAbandoned;
    bool isRemoved;
    int framesAbandonedFor;
    int framesRemovedFor;
};

vector<contour> found_contours;
int FPS;
Mat color_img;

void display_abandoned_contours(vector<Vec4i> hierarchy)
{
    Scalar color = Scalar(255);
    for(int i=0;i<found_contours.size();i++)
    {
        if(found_contours[i].framesAbandonedFor>=(FPS*4)) //if the object is present for over 3 seconds
        {
            std::vector<std::vector<cv::Point>> contourVec;
            contourVec.push_back(found_contours[i].points);
            drawContours(color_img, contourVec, 0, color, 2, 8, hierarchy, 0, Point());
            cout << "found contour number: " << i << endl;
            cout << "contour area: " << found_contours[i].area << endl;
            cout << "frames abandoned for: " << found_contours[i].framesAbandonedFor << endl;
            if(found_contours[i].isRemoved==false)
            {
                putText(color_img,"Object detected.",Point(found_contours[i].centroid.x,found_contours[i].centroid.y),FONT_HERSHEY_PLAIN,1.0,Scalar(255),1);
            }
        }
    }

}

void set_found_this_frame_false()
{
    //for(found_object contour: found_contours) contour.foundThisFrame=false;
    for(int i=0; i<found_contours.size(); i++) {
        found_contours[i].foundThisFrame=false;
    }
}

bool check_and_update_contour(vector<Point> points, Point2f centroid)
{
    bool present=false;
    for(int i=0;i<found_contours.size() && !present;i++)
    {
        if(fabs(centroid.x-found_contours[i].centroid.x)<=1 && fabs(centroid.y-found_contours[i].centroid.y)<=1) //centroids within 1 of each other
        {
            found_contours[i].centroid = centroid;
            found_contours[i].points = points;
            found_contours[i].framesAbandonedFor++;
            //cout << "frames abandoned for: " << found_contours[i].framesAbandonedFor << endl;
            found_contours[i].foundThisFrame= true;
            present = true;
        }
    }
    
    return present;
}

void add_contour(vector<Point> points, Point2f centroid,double area)
{
    contour new_contour;
    new_contour.centroid = centroid;
    new_contour.area = area;
    new_contour.points = points;
    new_contour.isAbandoned = false;
    new_contour.isRemoved = false;
    new_contour.foundThisFrame = true;
    new_contour.framesAbandonedFor=1;
    new_contour.framesRemovedFor=0;
    
    found_contours.push_back(new_contour);
}

bool remove_old_contours()
{
    for(int i=0;i<found_contours.size();i++)
    {
        if(found_contours[i].foundThisFrame==false)
        {
            if(found_contours[i].framesAbandonedFor>=FPS*4)
            {
                if(found_contours[i].framesRemovedFor==0) found_contours[i].isRemoved=true;
                found_contours[i].framesRemovedFor++;
                cout << "frames removed for: " << found_contours[i].framesRemovedFor << endl;

                putText(color_img,"Object Removed.",Point(found_contours[i].centroid.x,found_contours[i].centroid.y),FONT_HERSHEY_PLAIN,1.0,Scalar(255),1);
                if(found_contours[i].framesRemovedFor>=1*FPS)
                {
                    if(i == found_contours.size()-1) found_contours.pop_back();
                    else for(int j=i;j<found_contours.size()-1;j++) found_contours[j] = found_contours[j+1];
                }
                return true;
            }
            
            if(i == found_contours.size()-1) found_contours.pop_back();
            else for(int j=i;j<found_contours.size()-1;j++) found_contours[j] = found_contours[j+1];
        }
    }
    return true;
}

int main()
{
    
    String path = "ObjectAbandonmentAndRemoval2.avi";
    //String path = "aban3.mp4";
    
    VideoCapture cap(path);
    int row_size=cap.get(CV_CAP_PROP_FRAME_WIDTH),col_size = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    
    Mat frame,running_avg,background(row_size,col_size,CV_8UC1,Scalar(0)),abandoned;
    printf("video captured.\n");
    FPS = cap.get(CV_CAP_PROP_FPS);
    cout << "FPS: " << FPS << endl;
    
    int row=0,col=0,frame_count=0;
    int totalFrames = getFrameCount(path);
    
    int running_avg_frames=100;// number of frames to take for the initial background, and the number of frames to take for the running average frames.
    
    bool get_bg=true;
    
    //The following 3d vector is used to represent the list of points for every position in the frame.
    //the positions are initialsed by the row and column size of the frame. points can then be pushed to each position.
    vector<vector<vector<int>>> frames_points_list(row_size,vector<vector<int> >(col_size,vector <int>(0,0)));
    
    
    int frames_points_total[row_size][col_size]; // running total of each position. the value at each index is the total sum of it's respective list at the same position in frames_points_list.
    float frames_points_average[row_size][col_size]; //uses the total to calculate an average over the last 'running_avg_frames' frames
    
    for(row=0;row<row_size;row++)
         for(int col=0;col<col_size;col++)
            frames_points_total[row][col] = 0; frames_points_average[row][col] = 0; //initialisation
    
    
    while(frame_count<totalFrames)
    {
        cap>>frame; //store next frame in image
        
        color_img = frame;
        cvtColor(frame,frame,CV_BGR2GRAY); //convert frame to greyscale
        running_avg=frame.clone();
        absdiff(running_avg,running_avg,running_avg);
        
        for(row=0;row<row_size;row++)
        {
            for(col=0;col<col_size;col++)
            {
                frames_points_list[row][col].push_back(frame.at<uchar>(Point(row,col))); // push newest point to list
                frames_points_total[row][col]+=frame.at<uchar>(Point(row,col));          // add latest pount to total
                
                if(frame_count<running_avg_frames) //If background has not yet been acquired
                {
                    frames_points_average[row][col]=0;	  
                }
                if(frame_count>=running_avg_frames)//If background has been acquired
                {
                    frames_points_total[row][col]-=frames_points_list[row][col].front();           //subtract oldest point from total
                    frames_points_list[row][col].erase(frames_points_list[row][col].begin());      //pop oldest point from list
                    
                    frames_points_average[row][col]=frames_points_total[row][col]/running_avg_frames; //get average of last 'last_bg_frame' points.
                }
                
                running_avg.at<uchar>(Point(row,col))=frames_points_average[row][col];//get averaged background
            }
        }
        
        if(frame_count<running_avg_frames)
        {
            putText(running_avg,"Getting Background..",Point(160,180),FONT_HERSHEY_PLAIN,2.5,Scalar(255),1);
            absdiff(running_avg,running_avg,background);//initialize background.
        }
        
        if(frame_count>=running_avg_frames && get_bg==true) //loop run once. sets the first background difference as the background
        {
            background=running_avg.clone();
            get_bg = false;
        }
      
        absdiff(running_avg,background,abandoned); //absolute difference between background and background_diff. this gives the abandoned matrix
        
        Mat thresholded;
        threshold(abandoned,thresholded,50,255,THRESH_BINARY);
        vector<vector<Point>> contours;
        
        moveWindow("Abandoned",row_size,0);
        imshow("Abandoned",abandoned);
        moveWindow("Running Average",0,col_size);
        imshow("Running Average",running_avg);
        moveWindow("Thresholded",row_size,col_size);
        imshow("Thresholded",thresholded);
        
        
        set_found_this_frame_false(); //new contours will be set foundThisFrame to true as they are added/updated, which makes it possiblr for old ones to be removed
        
        vector<Vec4i> hierarchy;
        findContours(thresholded, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
        
        for(int i=0;i<contours.size();i++)
        {
            Moments moment = moments(contours[i],true);
            Point2f centroid = Point2f( moment.m10/moment.m00 , moment.m01/moment.m00 );
            double area = contourArea(contours[i]);
            if (!check_and_update_contour(contours[i],centroid) && (frame_count>=running_avg_frames) && (centroid.x>=0) && (centroid.y>=0)) add_contour(contours[i],centroid,area);
            
        }
        
        remove_old_contours();
        display_abandoned_contours(hierarchy);
        
        imshow("Original",color_img);
        frame_count++; // increment frame count
        cout << "frame count: " << frame_count << endl;
        waitKey(1);
        
    }
    
    
    return 0;
}
