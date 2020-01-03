#include <iostream>
#include <cstdlib>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

struct conpoment {
    Point2d coordinate;
    int height;
    int width;
};

class LicensePlate {
public:
    string srcImagePath;
    Mat srcImage;
    Mat licensePlateImage;
    Rect2d licenseRect;
    vector<Rect2d> licenseRectangles;

    LicensePlate() {

    }

    string getSourcesImagePath() {
        return this->srcImagePath;
    }

    Mat getSourceImage() {
        return this->srcImage;
    }

    Mat getLicensePlate() {
        return this->licensePlateImage;
    }

    Rect getRect() {
        return this->licenseRect;
    }

    Rect getLicenseRect(int idx) {
        return this->licenseRectangles.at(idx);
    }

    vector<Rect2d> getLicenseRects() {
        return this->licenseRectangles;
    }
};

Mat getGaussianBlurImage(Mat& image) {
    Mat out;
    GaussianBlur(image, out, Size(17, 17), 0, 0, BORDER_DEFAULT);
    return out;
}

Mat getGaussianBlurImage(Mat& image, int size) {
    Mat out;
    GaussianBlur(image, out, Size(size, size), 0, 0, BORDER_DEFAULT);
    return out;
}

Mat getGrayScaleImage(Mat& image) {
    Mat out;
    cvtColor(image, out, CV_RGB2GRAY);
    return out;
}

Mat getSobelImage(Mat& image) {
    Mat out;
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    //X方向
    Sobel(image, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    //Sobel(img, img, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
    convertScaleAbs(image, out);

    //Y方向
    Sobel(image, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);
    convertScaleAbs(image, out);

    //合並
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, out);
    return out;
}

Mat getBinaryImage(Mat& image) {
    Mat out;
    threshold(image, out, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
    threshold(image, out, 100, 255, CV_THRESH_BINARY);
    return out;
}


Mat Close(Mat& image) {
    Mat out;
    Mat element = getStructuringElement(MORPH_RECT, Size(17, 5));
    morphologyEx(image, out, MORPH_CLOSE, element);
    return out;
}

Mat getOpening(Mat& image, int size) {
    Mat out;
    Mat element = getStructuringElement(MORPH_RECT, Size(size, size));
    morphologyEx(image, out, MORPH_OPEN, element);
    return out;
}

Mat getClosing(Mat& image, int size) {
    Mat out;
    Mat element = getStructuringElement(MORPH_RECT, Size(size, size));
    morphologyEx(image, out, MORPH_CLOSE, element);
    return out;
}

Mat getEdgeDetectionImage(Mat& image) {
    Mat out;
    Canny(image, out, 500, 200, 3);
    return out;
}

Mat getCanny(cv::Mat& img) {
    Mat input_image1;
    img.copyTo(input_image1);
    cvtColor(input_image1, input_image1, CV_BGR2GRAY);
    input_image1.convertTo(input_image1, CV_32FC1);
    Mat sobelx = (Mat_<float>(3, 3) << -0.125, 0, 0.125, -0.25, 0, 0.25, -0.125, 0, 0.125);
    filter2D(input_image1, input_image1, input_image1.type(), sobelx);
    Mat mul_image;
    multiply(input_image1, input_image1, mul_image);
    const int scaleValue = 4;
    double threshold = scaleValue * mean(mul_image).val[0];//4 * img_s的平均值
    Mat resultImage = Mat::zeros(mul_image.size(), mul_image.type());
    float* pDataimg_s = (float*)(mul_image.data);
    float* pDataresult = (float*)(resultImage.data);
    const int height = input_image1.rows;
    const int width = input_image1.cols;
    //--- 非極大值抑制 + 閥值分割
    for (size_t i = 1; i < width - 1; i++)
    {
        for (size_t j = 1; j < height - 1; j++)
        {
            bool b1 = (pDataimg_s[i * height + j] > pDataimg_s[i * height + j - 1]);
            bool b2 = (pDataimg_s[i * height + j] > pDataimg_s[i * height + j + 1]);
            bool b3 = (pDataimg_s[i * height + j] > pDataimg_s[(i - 1) * height + j]);
            bool b4 = (pDataimg_s[i * height + j] > pDataimg_s[(i + 1) * height + j]);
            pDataresult[i * height + j] = 255 * ((pDataimg_s[i * height + j] > threshold) && ((b1 && b2) || (b3 && b4)));
        }
    }
    resultImage.convertTo(resultImage, CV_8UC1);
    return resultImage;
}

Mat Swell(Mat& img) {
    //圖片膨脹處理
    Mat dilate_image, erode_image;
    //自定義核進行 x 方向的膨脹腐蝕
    Mat elementX = getStructuringElement(MORPH_RECT, Size(25, 1));
    Mat elementY = getStructuringElement(MORPH_RECT, Size(1, 19));
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 1));
    Point point(-1, -1);
    dilate(img, img, elementX, point, 2);
    erode(img, img, elementX, point, 4);
    dilate(img, img, elementX, point, 2);
    //自定義核進行 Y 方向的膨脹腐蝕
    erode(img, img, elementY, point, 1);
    dilate(img, img, elementY, point, 2);
    erode(img, img, element, point, 15);
    dilate(img, img, element, point, 15);
    //imshow("test", dilate_image);
    //waitKey(1000);
    //噪聲處理
    //平滑處理 中值濾波
    medianBlur(img, img, 15);
    medianBlur(img, img, 15);
    return img;
}
int checking(Mat& srcImage, string name) {
    Mat srcHist, dstHist;
    cv::threshold(srcImage, srcImage, 30, 255, 0);
    //imshow("Gray Scale", srcImage);
    cvtColor(srcImage, srcImage, CV_RGB2GRAY);
    GaussianBlur(srcImage, srcImage, Size(27, 27), 0, 0, BORDER_DEFAULT);
    srcImage = getSobelImage(srcImage);
    //imshow("原始图"+ name, srcImage);
    int dims = 1;
    float hranges[] = { 0, 255 };
    const float* ranges[] = { hranges };   // 这里需要為const類型  
    int size = 256;
    int channels = 0;
    calcHist(&srcImage, 1, &channels, Mat(), srcHist, dims, &size, ranges);
    calcHist(&srcImage, 1, &channels, Mat(), dstHist, dims, &size, ranges);

    Mat srcHistImage(size, size, CV_8U, Scalar(0));
    Mat dstHistImage(size, size, CV_8U, Scalar(0));
    //獲取最大值和最小值  
    double minValue = 0;
    double srcMaxValue = 0;
    minMaxLoc(srcHist, &minValue, &srcMaxValue, 0, 0);
    //繪製直方圖
    //saturate_cast函数的作用即是：當運算完之后，结果為负，則轉為0，結果超出255，則為255。  
    int hpt = saturate_cast<int>(0.9 * size);
    for (int i = 0; i < 256; i++)
    {
        float srcValue = srcHist.at<float>(i);           //   注意hist中是float類型 
        //拉伸到0-max  
        int srcRealValue = saturate_cast<int>(srcValue * (float)hpt / srcMaxValue);

        line(srcHistImage, Point(i, size - 1), Point(i, size - srcRealValue), Scalar(255));
    }
    //imshow("原圖直方圖" + name, srcHistImage);
    medianBlur(srcHistImage, srcHistImage, 15);
    //imshow("均衡後直方圖" + name, srcHistImage);
    Size s = srcHistImage.size();
    int Max = 0;
    int rows = s.height;
    int cols = s.width;
    int Data = 0;
    for (int j = 0; j < cols; j++) {
        Scalar colour = srcHistImage.at<uchar>(Point(30, j));
        if (colour.val[0] == 255) {
            Data++;
        }
    }
    Max = Data;
    //waitKey(0);
    return Max;
}

Mat QueryContour(Mat& img, Mat& out, Mat& origin, LicensePlate& license) {
    //查詢輪廓
    vector<vector<Point>>cont;
    findContours(img, cont, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
    vector<vector<Point>>contours_poly(cont.size());
    vector<Rect>boundRect(cont.size());
    vector<RotatedRect> minRect(cont.size());
    for (int i = 0; i < cont.size(); i++) {
        approxPolyDP(Mat(cont[i]), contours_poly[i], 3, true);//用指定精度逼近多邊形曲線
        boundRect[i] = boundingRect(Mat(contours_poly[i])); //外接矩形
        minRect[i] = minAreaRect(Mat(cont[i])); //最小外接矩形
    }
    Mat res = img.clone();
    RNG g_rng(12345);
    int coordinate[500][2];
    int Candidate = 0;
    int Data = 0;
    for (int i = 0; i < cont.size(); i++) {
        Point2f rect_points[4];
        minRect[i].points(rect_points);
        int X = rect_points[2].x - rect_points[1].x;
        int Y = rect_points[0].y - rect_points[1].y;
        //if (rect_points[0].x < 800 || rect_points[0].y > 800)continue;
        if (X * Y < 12000)continue;
        if (sqrt(pow((rect_points[0].x - rect_points[1].x), 2)) > 30)continue;
        if (sqrt(pow((rect_points[1].y - rect_points[2].y), 2)) > 30)continue;
        Rect area(rect_points[1].x, rect_points[1].y, X, Y);
        Mat imgChecking = out(area);
        coordinate[Data][0] = checking(imgChecking, to_string(i));
        coordinate[Data][1] = i;
        Data++;
    }
    int Max = 0;
    for (int i = 0; i < Data; i++) {
        if (coordinate[i][0] > Max) {
            Max = coordinate[i][0];
            Candidate = coordinate[i][1];
        }
    }
    Point2f rect_points[4];
    minRect[Candidate].points(rect_points);
    int X = rect_points[2].x - rect_points[1].x;
    int Y = rect_points[0].y - rect_points[1].y;
    int locationI[4][2];
    for (int i = 0; i < 4; i++) {
        locationI[i][0] = rect_points[i].x;
        locationI[i][1] = rect_points[i].y;
    }

    Mat Cutimg;
    Rect area;

    if (Y > 70) {//單層車牌的長闊比
        if ((double)X / (double)Y - (double)1.7 < 0) { //雙層車牌的長闊比
            area = Rect(locationI[1][0] - 10, locationI[1][1], X + 30, Y);
            Cutimg = origin(area);
        }
        else {
            area = Rect(locationI[1][0] + 5, locationI[1][1], X - 10, Y);
            Cutimg = origin(area);
        }
    }
    else {
        if ((double)X / (double)Y - (double)4.3 < 0) {//單層車牌的長闊比
            area = Rect(locationI[1][0] - 10, locationI[1][1], X + 40, Y);
            Cutimg = origin(area);
        }
        else {
            area = Rect(locationI[1][0] - 25, locationI[1][1] - 10, X + 35, Y + 10);
            Cutimg = origin(area);
        }
    }
    license.licensePlateImage = Cutimg.clone();
    license.licenseRect = area;

    return Cutimg;
}

Mat getLicensePlate(string imgPath, LicensePlate& license) {
    Mat image;
    Mat originImage, outputImage;
    image = imread(imgPath);
    image.copyTo(originImage);
    image.copyTo(outputImage);

    license.srcImagePath = imgPath;
    license.srcImage = image.clone();

    // for gaussian blur
    image = getGaussianBlurImage(image);
    //imshow("圖片膨脹處理+2", image);
    /*
    // for gray scale
    image = getGrayScaleImage(image);
    //imshow("Gray Scale", img);

    // for binary
    image = getBinaryImage(image);
    //imshow("binary", image);

    // for edge detection
    image = getEdgeDetectionImage(image);
    //imshow("Edge Detection", image);
    */
    image = getCanny(image);
    //imshow("圖片膨脹處理+1", image);
    image = Swell(image);
    //imshow("圖片膨脹處理", image);

    image = QueryContour(image, outputImage, originImage, license);
    //imshow("查詢輪廓", image);

    return image;
}

Mat getLabeledLicensePlate(LicensePlate& license) {
    int threshold = 0;
    int labels;
    Mat image = license.licensePlateImage.clone();
    Mat outputImage;
    Mat imageStates, imageCentroid, imageLabel;
    Mat labeldImage;
    //Mat element = getStructuringElement(MORPH_RECT, Size(3, 1));
    Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
    //Mat elementX = getStructuringElement(MORPH_RECT, Size(3, 1));
    //Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));


    outputImage = getGrayScaleImage(image);
    //imshow("testG", outputImage);

    erode(outputImage, outputImage, element, Point(-1, -1), 1);
    //dilate(outputImage, outputImage, element, Point(-1, -1), 1);

    cv::threshold(outputImage, outputImage, 50, 255, 0);


    int pixels = outputImage.cols * outputImage.rows;
    if (countNonZero(outputImage) > pixels / 2) {
        bitwise_not(outputImage, outputImage);
    }
    //imshow("testB", outputImage);
    labels = connectedComponentsWithStats(outputImage, imageLabel, imageStates, imageCentroid);

    std::vector<cv::Vec3b> colors(labels);
    colors[0] = cv::Vec3b(0, 0, 0);


    int count = 0;
    double ratio;
    vector<conpoment> position(labels);
    // 0 for the whole image background
    for (int label = 1; label < labels; label++) {
        colors[label] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
        ratio = double(imageStates.at<int>(label, CC_STAT_WIDTH)) / double(imageStates.at<int>(label, CC_STAT_HEIGHT));
        if (imageStates.at<int>(label, CC_STAT_WIDTH) > 1 && imageStates.at<int>(label, CC_STAT_HEIGHT) > 35 &&
            imageStates.at<int>(label, CC_STAT_WIDTH) < 80 && imageStates.at<int>(label, CC_STAT_HEIGHT) < 100 &&
            imageStates.at<int>(label, CC_STAT_AREA) > 150 && ratio > 0.2 && ratio < 1.3) {
            position.at(count).coordinate = Point(imageStates.at<int>(label, CC_STAT_LEFT), imageStates.at<int>(label, CC_STAT_TOP));
            position.at(count).height = imageStates.at<int>(label, CC_STAT_HEIGHT);
            position.at(count).width = imageStates.at<int>(label, CC_STAT_WIDTH);
            count++;

            //cout << "CC_STAT_LEFT   = " << imageStates.at<int>(label, cv::CC_STAT_LEFT) << endl;
            //cout << "CC_STAT_TOP    = " << imageStates.at<int>(label, cv::CC_STAT_TOP) << endl;
            //cout << "CC_STAT_WIDTH  = " << imageStates.at<int>(label, cv::CC_STAT_WIDTH) << endl;
            //cout << "CC_STAT_HEIGHT = " << imageStates.at<int>(label, cv::CC_STAT_HEIGHT) << endl;
            //cout << "CC_STAT_AREA   = " << imageStates.at<int>(label, cv::CC_STAT_AREA) << endl;
            //cout << endl;
        }
    }
    license.licenseRectangles.resize(count);
    cout << license.srcImagePath << " has " << count << " characters." << endl;


    cv::Mat dst(outputImage.size(), CV_8UC3);
    for (int r = 0; r < dst.rows; ++r) {
        for (int c = 0; c < dst.cols; ++c) {
            int label = imageLabel.at<int>(r, c);
            cv::Vec3b& pixel = dst.at<cv::Vec3b>(r, c);
            pixel = colors[label];
        }
    }
    //imshow("cc", dst);


    // draw the rectangle for each character

    Scalar color = Scalar(0, 255, 255);
    Rect rect;
    for (int i = 0; i < count; i++) {
        rect = Rect(position.at(i).coordinate.x + license.licenseRect.tl().x, position.at(i).coordinate.y + license.licenseRect.tl().y, position.at(i).width, position.at(i).height);
        rectangle(license.srcImage, rect, color, 2, 8, 0);
        rectangle(license.licensePlateImage, Rect(position.at(i).coordinate.x, position.at(i).coordinate.y, position.at(i).width, position.at(i).height), color, 2, 8, 0);
        license.licenseRectangles.at(i) = Rect(position.at(i).coordinate.x, position.at(i).coordinate.y, position.at(i).width, position.at(i).height);
    }
    rectangle(license.srcImage, license.licenseRect, Scalar(0, 0, 255), 5, 8, 0);

    return license.srcImage;
}

vector<LicensePlate> getLabeledLicensePlate(vector<string> carImage) {
    Mat tmp;
    vector<LicensePlate> licensePlates;
    for (int i = 0; i < carImage.size(); i++) {
        licensePlates.push_back(LicensePlate());
    }
    cout << "Total has " << carImage.size() << " cars." << endl;

    for (int i = 0; i < carImage.size(); i++) {
        tmp = getLicensePlate(carImage.at(i), licensePlates.at(i));
        tmp = getLabeledLicensePlate(licensePlates.at(i));
    }

    return licensePlates;
}

void sortLicensePlateNumbers(vector<LicensePlate>& licensePlates) {
    vector<int> plates;
    for (int i = 0; i < licensePlates.size(); i++) {

    }
}

void saveFile(vector<LicensePlate>& carImage, string name) {
    Rect2d rect;

    ofstream ans(name);

    if (!ans) {
        cout << "file can't open!" << endl;
    }
    else {
        for (int i = 0; i < carImage.size(); i++) {
            ans << carImage[i].srcImagePath << endl;
            for (int j = 0; j < carImage[i].licenseRectangles.size(); j++) {
                ans << carImage[i].licenseRect.tl().x + carImage[i].licenseRectangles[j].tl().x << " ";
                ans << carImage[i].licenseRect.tl().y + carImage[i].licenseRectangles[j].tl().y << " ";

                ans << carImage[i].licenseRect.tl().x + carImage[i].licenseRectangles[j].br().x << " ";
                ans << carImage[i].licenseRect.tl().y + carImage[i].licenseRectangles[j].br().y << endl;
            }
        }
    }
    ans.close();
}

int main() {
    int totalCars = 20;
    Mat img;
    Mat out;
    string dir = "car/";
    string carImg[20] = { dir + "001.jpg", dir + "002.jpg", dir + "003.jpg", dir + "004.jpg",
                          dir + "005.jpg", dir + "006.jpg", dir + "007.jpg", dir + "008.jpg",
                          dir + "009.jpg", dir + "010.jpg", dir + "011.jpg", dir + "012.jpg",
                          dir + "013.jpg", dir + "014.jpg", dir + "015.jpg", dir + "016.jpg",
                          dir + "017.jpg", dir + "018.jpg", dir + "019.jpg", dir + "020.jpg" };

    /*for (int i = 0; i < totalCars; i++) {
        string test =carImg[i];
        LicensePlate l = LicensePlate();
        Mat a = getLicensePlate(test, l);
        imshow(carImg[i], a);
    }*/

    /*
    string test = dir + "016.jpg";
    LicensePlate l = LicensePlate();
    Mat a = getLicensePlate(test, l);
    imshow("A", a);
    */

    /*
    out = getLabeledLicensePlate(l);
    namedWindow("B", 0);
    cvResizeWindow("B", 1000, 500);
    imshow("B", out);
    waitKey(0);
    */
     
    vector<string> carImage;
    carImage.assign(carImg, carImg + totalCars);
    vector<LicensePlate> licensePlates = getLabeledLicensePlate(carImage);

    saveFile(licensePlates, dir + "output.txt");

    for (int i = 0; i < totalCars; i++) {
        const char* cstr = carImage[i].c_str();
        namedWindow(carImage[i], 0);
        cvResizeWindow(cstr, 1000, 500);
        imshow(carImage[i], licensePlates.at(i).srcImage);
        cout << carImage[i] << endl;
        cout << "Rect " << licensePlates.at(i).licenseRect << endl;
        cout << "sizes " << i << " : " << licensePlates.at(i).licenseRectangles.size() << endl;
        for (int j = 0; j < licensePlates.at(i).licenseRectangles.size(); j++) {
            cout << "Rects " << j+1 << licensePlates.at(i).licenseRectangles.at(j) << endl;
        }
        //waitKey(0);
    }
    waitKey(0);
    cvDestroyAllWindows();
    
}