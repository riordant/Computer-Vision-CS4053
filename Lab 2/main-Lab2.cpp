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

#define PI 3.14159265

using namespace cv;
using namespace std;

const int NUM_OBJECTS = 13;
const int NUM_SCENES = 25;
const int NUM_TEST_IMAGES = 8;

const int SLOPE_TOP = 0;
const int SLOPE_RIGHT = 1;
const int SLOPE_BOTTOM = 2;
const int SLOPE_LEFT = 3;
const int ANGLE_TOP_LEFT = 4;
const int ANGLE_TOP_RIGHT = 5;
const int ANGLE_BOTTOM_RIGHT = 6;
const int ANGLE_BOTTOM_LEFT = 7;
const int NUM_GOOD_MATCHES = 8;
const int PAGE = 9;

const int STANDARD_DEV_MARGIN = 2;

//8 images will be used as test data.
//these images will be the image scenes 1-4 and 14-17
//position 0 is the scene index and position 1 is the page in the scene.
const int testSceneIndices[NUM_TEST_IMAGES][2] = {{0,0},{2,2},{4,4},{6,6},{13,1},{16,3},{19,7},{21,10}};


vector<float> testMeanAngles;
vector<float> testStdDevAngles; //global vectors for the means and standard deviations of the test data used.



vector<float> getMeanAngles(vector<vector<float>> attributes){
    vector<float> result;
    float totalAngleTopLeft = 0;
    float totalAngleTopRight = 0;
    float totalAngleBottomRight = 0;
    float totalAngleBottomLeft = 0;
    for(int i=0;i<attributes.size();i++)
    {
        totalAngleTopLeft += attributes[i][ANGLE_TOP_LEFT];
        totalAngleTopRight += attributes[i][ANGLE_TOP_RIGHT];
        totalAngleBottomRight += attributes[i][ANGLE_BOTTOM_RIGHT];
        totalAngleBottomLeft += attributes[i][ANGLE_BOTTOM_LEFT];
    }
    result.push_back(totalAngleTopLeft/attributes.size());
    result.push_back(totalAngleTopRight/attributes.size());
    result.push_back(totalAngleBottomRight/attributes.size());
    result.push_back(totalAngleBottomLeft/attributes.size());
    
    return result;
    
}

vector<float> getStdDevAngles(vector<vector<float>> attributes, vector<float> means){
    vector<float> result;
    float totalAngleTopLeft = 0;
    float totalAngleTopRight = 0;
    float totalAngleBottomRight = 0;
    float totalAngleBottomLeft = 0;
    for(int i=0;i<attributes.size();i++)
    {
        totalAngleTopLeft += pow(attributes[i][ANGLE_TOP_LEFT]-means[0],2);
        totalAngleTopRight += pow(attributes[i][ANGLE_TOP_RIGHT]-means[1],2);
        totalAngleBottomRight += pow(attributes[i][ANGLE_BOTTOM_RIGHT]-means[2],2);
        totalAngleBottomLeft += pow(attributes[i][ANGLE_BOTTOM_LEFT]-means[3],2);

    }
    result.push_back(sqrt(totalAngleTopLeft/attributes.size()));
    result.push_back(sqrt(totalAngleTopRight/attributes.size()));
    result.push_back(sqrt(totalAngleBottomRight/attributes.size()));
    result.push_back(sqrt(totalAngleBottomLeft/attributes.size()));
    
    return result;
}

bool isTestScene(int scene){
    
    for(int i=0;i<NUM_TEST_IMAGES;i++){
        if(testSceneIndices[i][0]==scene) return true;
    }
    return false;
}

bool anglesWithinMargin(vector<float> currObject){
    //if the round of the angle is within the floor of the mean-stddev or the ceiling of the mean+stddev: correct.
    
    if(round(currObject[ANGLE_TOP_LEFT])     >= (floor(testMeanAngles[0]-testStdDevAngles[0])-STANDARD_DEV_MARGIN) && round(currObject[ANGLE_TOP_LEFT])     <= (ceil(testMeanAngles[0]+testStdDevAngles[0])+STANDARD_DEV_MARGIN)
    && round(currObject[ANGLE_TOP_RIGHT])    >= (floor(testMeanAngles[1]-testStdDevAngles[1])-STANDARD_DEV_MARGIN) && round(currObject[ANGLE_TOP_RIGHT])    <= (ceil(testMeanAngles[1]+testStdDevAngles[1])+STANDARD_DEV_MARGIN)
    && round(currObject[ANGLE_BOTTOM_RIGHT]) >= (floor(testMeanAngles[2]-testStdDevAngles[2])-STANDARD_DEV_MARGIN) && round(currObject[ANGLE_BOTTOM_RIGHT]) <= (ceil(testMeanAngles[2]+testStdDevAngles[2])+STANDARD_DEV_MARGIN)
    && round(currObject[ANGLE_BOTTOM_LEFT])  >= (floor(testMeanAngles[3]-testStdDevAngles[3])-STANDARD_DEV_MARGIN) && round(currObject[ANGLE_BOTTOM_LEFT])  <= (ceil(testMeanAngles[3]+testStdDevAngles[3])+STANDARD_DEV_MARGIN))
    {
        return true;
    }
    return false;
}

//this function is passed the attributes for each page matched against a scene.
//it return the correct page number for that scene.
int findObject(vector<vector<float>> attributes){
    
    
    while(attributes.size()>0)
    {
        int maxMatchesIndex=0;
        int maxMatches=0;
        for(int i=0;i<attributes.size();i++)
        {
            
            float currMaxMatches = attributes[i][NUM_GOOD_MATCHES];
            if(currMaxMatches>maxMatches)
            {
                maxMatches = currMaxMatches;
                maxMatchesIndex = i;
            }
            
        }

        vector<float> currObject = attributes[maxMatchesIndex];
        
        attributes[maxMatchesIndex].erase(attributes[maxMatchesIndex].begin(),attributes[maxMatchesIndex].end()); //remove
        for(int j=maxMatchesIndex;j<attributes.size()-1;j++) attributes[j] = attributes[j+1]; //shift remaining elements in vector to the left.
        
        if(anglesWithinMargin(currObject)) return currObject[PAGE];
        
    }
    return -1;
}

//returns a number of attributes of the found object in a vector.
vector<float> getObjectAttributes(vector<Point2f> scene_corners){
    
    vector<float> attributes;
    float topSlope    = (scene_corners[1].y - scene_corners[0].y)/(scene_corners[1].x - scene_corners[0].x);
    float rightSlope  = (scene_corners[2].y - scene_corners[1].y)/(scene_corners[2].x - scene_corners[1].x);
    float bottomSlope = (scene_corners[2].y - scene_corners[3].y)/(scene_corners[2].x - scene_corners[3].x);
    float leftSlope   = (scene_corners[3].y - scene_corners[0].y)/(scene_corners[3].x - scene_corners[0].x);
    
    attributes.push_back(topSlope);
    attributes.push_back(rightSlope);
    attributes.push_back(bottomSlope);
    attributes.push_back(leftSlope);
    
    float topLeftAngle = (atan(fabs((leftSlope -topSlope)       /1+(leftSlope*topSlope)))     * 180) / PI;
    float topRightAngle = (atan(fabs((rightSlope-topSlope)      /1+(rightSlope*topSlope)))    * 180) / PI;
    float bottomRightAngle = (atan(fabs((rightSlope-bottomSlope)/1+(rightSlope*bottomSlope))) * 180) / PI;
    float bottomLeftAngle = (atan(fabs((leftSlope -bottomSlope) /1+(leftSlope*bottomSlope)))  * 180) / PI;
    
    attributes.push_back(topLeftAngle);
    attributes.push_back(topRightAngle);
    attributes.push_back(bottomRightAngle);
    attributes.push_back(bottomLeftAngle);
    
    return attributes;

}

//this is the function where the SURF feature detection algorithm is implemented.
//heavily references http://morf.lv/modules.php?name=tutorials&lasit=2
vector<float> getObject(Mat sceneMat, Mat objectMat,int sceneNum, int pageNum)
{
	
    float nndrRatio = 0.7f;
    int hessianValue = 10;
    int k = 2;
    //vector of keypoints
    vector<cv::KeyPoint> keypointsO;
    vector<cv::KeyPoint> keypointsS;
    
    Mat descriptors_object, descriptors_scene;
    
    //Extract keypoints
    SurfFeatureDetector surf(hessianValue);
    surf.detect(sceneMat,keypointsS);
    surf.detect(objectMat,keypointsO);
    
    //Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;
    extractor.compute( sceneMat, keypointsS, descriptors_scene );
    extractor.compute( objectMat, keypointsO, descriptors_object );
    
    //Matching descriptors by k nearest neighbour
    FlannBasedMatcher matcher;
    vector<vector<DMatch>> matches;
    matcher.knnMatch(descriptors_object, descriptors_scene, matches, k);
    vector< DMatch> good_matches;
    
    for (int i = 0; i < matches.size(); ++i)
    {
        if (matches[i].size() < 2)
            continue;
        
        const DMatch &m1 = matches[i][0];
        const DMatch &m2 = matches[i][1];
        
        if(m1.distance <= nndrRatio * m2.distance)
            good_matches.push_back(m1);
    }
    
    Mat img_matches;
    drawMatches(objectMat, keypointsO, sceneMat, keypointsS, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    vector<Point2f> obj;
    vector<Point2f> scene;
        
    for(unsigned int i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypointsO[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypointsS[ good_matches[i].trainIdx ].pt );
    }
        
    Mat H = findHomography( obj, scene, CV_RANSAC );
         
    //-- Get the corners from the image_1 ( the object to be "detected" )
    vector< Point2f > obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( objectMat.cols, 0 );
    obj_corners[2] = cvPoint( objectMat.cols, objectMat.rows );
    obj_corners[3] = cvPoint( 0, objectMat.rows );
        
    std::vector<Point2f> scene_corners(4);
        
    perspectiveTransform( obj_corners, scene_corners, H);
    
    int lineWidth = 2;
        
    line( img_matches, scene_corners[0]+ Point2f( objectMat.cols, 0), scene_corners[1]+ Point2f( objectMat.cols, 0), Scalar(255,0,0), lineWidth);
    line( img_matches, scene_corners[1]+ Point2f( objectMat.cols, 0), scene_corners[2]+ Point2f( objectMat.cols, 0), Scalar(255,0,0), lineWidth);
    line( img_matches, scene_corners[2]+ Point2f( objectMat.cols, 0), scene_corners[3]+ Point2f( objectMat.cols, 0), Scalar(255,0,0), lineWidth);
    line( img_matches, scene_corners[3]+ Point2f( objectMat.cols, 0), scene_corners[0]+ Point2f( objectMat.cols, 0), Scalar(255,0,0), lineWidth);
    
    printf("Matches found: %lu\n", matches.size());
 	printf("Good Matches found: %lu\n", good_matches.size());
    string output = "page: " + to_string(pageNum)  + " scene: " + to_string(sceneNum);
    
    Mat img_matches_resize;
    resize(img_matches, img_matches_resize, Size(), .66, .66, INTER_CUBIC);
    
    imshow(output, img_matches_resize);
    vector<float> objAttributes = getObjectAttributes(scene_corners);
    objAttributes.push_back(good_matches.size());
    return objAttributes;
}

void getResults(){
    char filename_obj[100];
    char filename_scene[100];
    cv::Mat img_objects[NUM_OBJECTS];
    cv::Mat img_scenes[NUM_SCENES];
    for(int i=1; i<=NUM_SCENES;i++)
    {
        if(!isTestScene(i-1))
        {
            sprintf(filename_scene,"./BookView%d.JPG",i);
            img_scenes[i-1] = cvarrToMat(cvLoadImage(filename_scene,-1));
            vector<vector<float>> attributes;
            
            for (int j=1; j <= NUM_OBJECTS; j++)
            {
                sprintf(filename_obj,"./Page%d.jpg",j);
                img_objects[j-1] = cvarrToMat(cvLoadImage(filename_obj,-1)); //have to use cvLoadImage and convert as my installation of openCV is having problems with imread.
                vector<float> objAttributes = getObject(img_scenes[i-1],img_objects[j-1],i,j);
                objAttributes.push_back(j); //this value acts as a key so that we know the original page in the case that the "attributes" vector is altered.
                attributes.push_back(objAttributes);
                waitKey(0);
            }
            int bestMatchImg = findObject(attributes);
            if(bestMatchImg==-1) printf("no image found for scene %d",i);
            else printf("Scene %d page: %d\n", i, bestMatchImg);
        }
    }

}
vector<vector<float>> getTestData(){
    char filename_obj[100];
    char filename_scene[100];
    cv::Mat img_objects[NUM_OBJECTS];
    cv::Mat img_scenes[NUM_SCENES];
    
    vector<vector<float>> attributes;
    
    for(int i=0;i<NUM_TEST_IMAGES;i++){
        sprintf(filename_scene,"./BookView%d.JPG",testSceneIndices[i][0]+1);
        img_scenes[i] = cvarrToMat(cvLoadImage(filename_scene,-1));
        
        sprintf(filename_obj,"./Page%d.jpg",testSceneIndices[i][1]+1);
        img_objects[i] = cvarrToMat(cvLoadImage(filename_obj,-1)); //have to use cvLoadImage and convert as my installation of openCV is having problems with imread.
        vector<float> objAttributes = getObject(img_scenes[i],img_objects[i],testSceneIndices[i][0]+1,testSceneIndices[i][1]+1);
        objAttributes.push_back(testSceneIndices[i][1]+1); //this value acts as a key so that we know the original page in the case that the "attributes" vector is altered.
        attributes.push_back(objAttributes);
        waitKey(0);
    }
    
    return attributes;
}

int main(int argc, char** argv)
{

    vector<vector<float>> testAttributes = getTestData();
    
    testMeanAngles = getMeanAngles(testAttributes);
    testStdDevAngles = getStdDevAngles(testAttributes, testMeanAngles);
    
    getResults();
    
    return 0;
}
