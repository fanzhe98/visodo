#include "vo_features.h"
#include <omp.h>

using namespace cv;
using namespace std;

#define MAX_FRAME 1000
#define MIN_NUM_FEAT 2000

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal) {

    string line;
    int i = 0;
    ifstream myfile("06.txt");
    double x = 0, y = 0, z = 0;
    double x_prev, y_prev, z_prev;
    if (myfile.is_open()) {
        while ((getline(myfile, line)) && (i <= frame_id)) {
            z_prev = z;
            x_prev = x;
            y_prev = y;
            std::istringstream in(line);
            //cout << line << '\n';
            for (int j = 0; j < 12; j++) {
                in >> z;
                if (j == 7) y = z;
                if (j == 3) x = z;
            }

            i++;
        }
        myfile.close();
    } else {
        cout << "Unable to open file";
        return 0;
    }

    return sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev));

}


int main(int argc, char **argv) {

    Mat img_1, img_2;
    Mat R_f, t_f; //the final rotation and tranlation vectors containing the

    R_f = Mat::zeros(3, 3, CV_32FC1);
    t_f = Mat::zeros(3, 1, CV_32FC1);

    FileStorage storage("test.xml", cv::FileStorage::WRITE);
    storage << "R0" << R_f;
    storage << "T0" << t_f;


    double scale = 1.00;
    char filename1[200];
    char filename2[200];
    sprintf(filename1, argv[1], 0);
    sprintf(filename2, argv[1], 1);

    char text[100];
    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(10, 50);

    //read the first two frames from the dataset
    Mat img_1_c = imread(filename1);
    Mat img_2_c = imread(filename2);

    if (!img_1_c.data || !img_2_c.data) {
        std::cout << " --(!) Error reading images " << std::endl;
        return -1;
    }

    // we work with grayscale images
    cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
    cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

    // feature detection, tracking
    vector<Point2f> points1, points2;        //vectors to store the coordinates of the feature points
    featureDetection(img_1, points1);        //detect features in img_1
    vector<uchar> status;
    featureTracking(img_1, img_2, points1, points2, status); //track those features to img_2

    double focal = 718.8560;
    cv::Point2d pp(607.1928, 185.2157);
    //recovering the pose and the essential matrix
    Mat E, R, t, mask;
    E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points2, points1, R, t, focal, pp, mask);

    Mat prevImage = img_2;
    Mat currImage;
    vector<Point2f> prevFeatures = points2;
    vector<Point2f> currFeatures;

    char filename[100];


    R_f = R.clone();
    t_f = t.clone();
    storage << "R1" << R_f;
    storage << "T1" << t_f;

    clock_t begin = clock();

    namedWindow("Camera", WINDOW_AUTOSIZE);// Create a window for display.
    namedWindow("Trajectory", WINDOW_AUTOSIZE);// Create a window for display.

    Mat traj = Mat::zeros(600, 600, CV_8UC3);
//    VideoWriter v("traj.avi", CV_FOURCC('M','J','P','G'),20, Size(600,600));
    clock_t temp = clock();


    for (int numFrame = 2; numFrame < 300; numFrame++) {
        sprintf(filename, argv[1], numFrame);
        //cout << numFrame << endl;
        Mat currImage_c = imread(filename);
        cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
        vector<uchar> status;
        featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

        E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
        recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

        Mat prevPts(2, prevFeatures.size(), CV_64F), currPts(2, currFeatures.size(), CV_64F);


        #pragma omp parallel for
        for (int i = 0; i < prevFeatures.size(); i++) {
            prevPts.at<double>(0, i) = prevFeatures.at(i).x;
            prevPts.at<double>(1, i) = prevFeatures.at(i).y;

            currPts.at<double>(0, i) = currFeatures.at(i).x;
            currPts.at<double>(1, i) = currFeatures.at(i).y;
        }

        scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));

        //cout << "Scale is " << scale << endl;

        if ((scale > 0.1) && (t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {

            t_f = t_f + scale * (R_f * t);
            R_f = R * R_f;

        } else {
            //cout << "scale below 0.1, or incorrect translation" << endl;
        }


        // a redetection is triggered in case the number of feautres being trakced go below a particular threshold
        if (prevFeatures.size() < MIN_NUM_FEAT) {
            featureDetection(prevImage, prevFeatures);
            featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

        }

        prevImage = currImage.clone();
        prevFeatures = currFeatures;

        int x = int(t_f.at<double>(0)) + 300;
        int y = int(t_f.at<double>(2)) + 100;
        circle(traj, Point(x, y), 1, CV_RGB(255, 0, 0), 2);

        rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
        sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1),
                t_f.at<double>(2));
        putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

        imshow("Camera", currImage_c);
        imshow("Trajectory", traj);

//        v.write(traj);
        waitKey(1);
        clock_t n = clock();
        cout << "Trjectory Calculation: " << double(n - temp) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
        temp = clock();
        string rname = "R" + to_string(numFrame);
        string tname = "T" + to_string(numFrame);
        storage << rname << R_f;
        storage << tname << t_f;
    }
//    v.release();
    storage.release();
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Total time taken: " << elapsed_secs << "s" << endl;

    cout << R_f << endl;
    cout << t_f << endl;
    storage.release();
    return 0;
}