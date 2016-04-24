/*
	相比“单手控制”，此代码固定了操作窗口的位置，不再实时更新大小和位置

备注：	当鼠标的移动阈值（深度空间的像素）为1个像素时，抖动程度可以接受，但是精度不够高，当为0时，精度足够，但是抖动太剧烈.
	也许转换为用彩色空间的坐标来控制鼠标会好一些，因为彩色空间的分辨率较高，但是深度空间与彩色空间的转换不是很精确.
*/


#include <iostream>
#include <cstdio>
#include <Kinect.h>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <math.h>

using	namespace	std;
using	namespace	cv;

const	double	OK_LEVEL = 0.15;	//判断跟手是否在一个平面上的容错值,单位米
const	int	HAND_UP = 150;		//手掌上可能存在指尖的区域，单位毫米
const	int	HAND_LEFT_RIGHT = 100;	//手掌左右可能存在指尖的区域，单位毫米
const	int	OK_MOUSE = 0;		//鼠标开始移动的阈值，越大越稳定，越小越精确
Vec3b	COLOR_TABLE[] = { Vec3b(255,0,0),Vec3b(0,255,0),Vec3b(0,0,255),Vec3b(255,255,255),Vec3b(0,0,0) };
enum { BLUE, GREEN, RED, WHITE, BLACK };

bool	depth_rage_check(int, int, int, int);
bool	level_check(const CameraSpacePoint &, const CameraSpacePoint &);
bool	distance_check(const CameraSpacePoint &, const CameraSpacePoint &);
bool	check_new_point(DepthSpacePoint &, DepthSpacePoint &, int, int);
void	draw_line(Mat &, const DepthSpacePoint &, DepthSpacePoint &);
void	draw_body(Mat &, BYTE *, int, int);
void	draw_Hand(Mat &, const DepthSpacePoint &);
void	draw_circle(Mat &, int, int);
int	main(void)
{
	IKinectSensor	* mySensor = nullptr;
	GetDefaultKinectSensor(&mySensor);
	mySensor->Open();

	IFrameDescription	* myDescription = nullptr;

	int	depthHeight = 0, depthWidth = 0;
	IDepthFrameSource	* myDepthSource = nullptr;
	IDepthFrameReader	* myDepthReader = nullptr;
	IDepthFrame		* myDepthFrame = nullptr;
	mySensor->get_DepthFrameSource(&myDepthSource);
	myDepthSource->get_FrameDescription(&myDescription);
	myDescription->get_Height(&depthHeight);
	myDescription->get_Width(&depthWidth);
	myDepthSource->OpenReader(&myDepthReader);			//以上为Depth帧的准备，直接开好Reader


	IBodyIndexFrameSource	* myBodyIndexSource = nullptr;
	IBodyIndexFrameReader	* myBodyIndexReader = nullptr;
	IBodyIndexFrame		* myBodyIndexFrame = nullptr;
	mySensor->get_BodyIndexFrameSource(&myBodyIndexSource);
	myBodyIndexSource->OpenReader(&myBodyIndexReader);		//以上为BodyIndex帧的准备,直接开好Reader


	IBodyFrameSource	* myBodySource = nullptr;
	IBodyFrameReader	* myBodyReader = nullptr;
	IBodyFrame		* myBodyFrame = nullptr;
	mySensor->get_BodyFrameSource(&myBodySource);
	myBodySource->OpenReader(&myBodyReader);			//以上为Body帧的准备，直接开好Reader

	ICoordinateMapper	* myMapper = nullptr;
	mySensor->get_CoordinateMapper(&myMapper);			//Maper的准备

	DepthSpacePoint		front = { 0,0 };				//用来记录上一次鼠标的位置
	DepthSpacePoint		depthUpLeft = { 1,1 };			//操作窗口的左上角和右下角，要注意这两个X和X、Y和Y的差会作为除数，所以不能都为0
	DepthSpacePoint		depthDownRight = { 0,0 };
	bool	gotEdge = false;
	while (1)
	{
		while (myDepthReader->AcquireLatestFrame(&myDepthFrame) != S_OK);
		UINT	depthBufferSize = 0;
		UINT16	* depthBuffer = nullptr;;
		myDepthFrame->AccessUnderlyingBuffer(&depthBufferSize, &depthBuffer);		//读取Depth数据

		while (myBodyIndexReader->AcquireLatestFrame(&myBodyIndexFrame) != S_OK);	//读取BodyIndex数据
		UINT	bodyIndexBufferSize = 0;
		BYTE	* bodyIndexBuffer = nullptr;
		myBodyIndexFrame->AccessUnderlyingBuffer(&bodyIndexBufferSize, &bodyIndexBuffer);
		Mat	img(depthHeight, depthWidth, CV_8UC3);
		draw_body(img, bodyIndexBuffer, depthHeight, depthWidth);

		while (myBodyReader->AcquireLatestFrame(&myBodyFrame) != S_OK);			//读取Body数据
		int	bodyBufferSize = 0;
		myBodySource->get_BodyCount(&bodyBufferSize);
		IBody	** bodyArray = new IBody *[bodyBufferSize];
		for (int i = 0; i < bodyBufferSize; i++)
			bodyArray[i] = nullptr;
		myBodyFrame->GetAndRefreshBodyData(bodyBufferSize, bodyArray);

		for (int i = 0; i < bodyBufferSize; i++)					//遍历6个人
		{
			BOOLEAN		result = false;
			if (bodyArray[i]->get_IsTracked(&result) == S_OK && result)		//将关节点输出，正式开始处理
			{
				Joint	jointArray[JointType_Count];
				bodyArray[i]->GetJoints(JointType_Count, jointArray);

				//以下为确定操作窗口的左上角和右下角，左右手的窗口位置不同。以Head关节到Neck关节的距离作为单位长度.最后用的深度数据，所以要转换
				if (!gotEdge)
					if (jointArray[JointType_Neck].TrackingState == TrackingState_Tracked && jointArray[JointType_Head].TrackingState == TrackingState_Tracked)
					{
						CameraSpacePoint	cameraNeck = jointArray[JointType_Neck].Position;
						CameraSpacePoint	cameraHead = jointArray[JointType_Head].Position;
						double	unite = sqrt(pow(cameraNeck.X - cameraHead.X, 2) + pow(cameraNeck.Y - cameraHead.Y, 2) + pow(cameraNeck.Z - cameraHead.Z, 2));
						CameraSpacePoint	cameraUpLeft = { cameraNeck.X + unite * 0.5,cameraNeck.Y + unite * 3,cameraNeck.Z };
						CameraSpacePoint	cameraDownRight = { cameraNeck.X + unite * 4,cameraNeck.Y + unite,cameraNeck.Z };
						myMapper->MapCameraPointToDepthSpace(cameraUpLeft, &depthUpLeft);
						myMapper->MapCameraPointToDepthSpace(cameraDownRight, &depthDownRight);
						gotEdge = true;
					}



				//指尖识别,记录最长的那根手指的指尖
				DepthSpacePoint		highestPoint = { depthWidth - 1,depthHeight - 1 };
				if (jointArray[JointType_HandRight].TrackingState == TrackingState_Tracked)
				{
					CameraSpacePoint	cameraHandRight = jointArray[JointType_HandRight].Position;
					DepthSpacePoint		depthHandRight;
					myMapper->MapCameraPointToDepthSpace(cameraHandRight, &depthHandRight);

					//开始测试右手手掌区域
					for (int i = depthHandRight.Y; i > depthHandRight.Y - HAND_UP; i--)
						for (int j = depthHandRight.X - HAND_LEFT_RIGHT; j < depthHandRight.X + 100; j++)	//确定要检查的范围
						{
							if (!depth_rage_check(j, i, depthWidth, depthHeight))				//判合法
								continue;
							int	index = i * depthWidth + j;
							CameraSpacePoint	cameraTemp;
							DepthSpacePoint		depthTemp = { j,i };
							myMapper->MapDepthPointToCameraSpace(depthTemp, depthBuffer[index], &cameraTemp);

							if (bodyIndexBuffer[index] > 5 || (bodyIndexBuffer[index] <= 5 && !level_check(cameraHandRight, cameraTemp)))	//看此像素是否不属于人体(指尖上方一点)，或者属于人体但是和手不在同一平面（胸）
							{
								bool	flag = true;
								for (int k = 1; k <= 5; k++)	//看时候此点下面连续5个像素都属于人体，且和手腕在同一平面,且距离合适
								{
									int	index_check = (i + k) * depthWidth + j;
									depthTemp.X = j;
									depthTemp.Y = i + k;
									myMapper->MapDepthPointToCameraSpace(depthTemp, depthBuffer[index_check], &cameraTemp);
									if (bodyIndexBuffer[index_check] <= 5 && level_check(cameraHandRight, cameraTemp) && distance_check(cameraHandRight, cameraTemp))
										continue;
									else
									{
										flag = false;
										break;
									}
								}
								if (flag && i < highestPoint.Y)
								{
									highestPoint.X = j;
									highestPoint.Y = i;
								}
							}
						}
				}


				int	windowWidth = (int)depthDownRight.X - (int)depthUpLeft.X;	//计算操作窗口的尺寸
				int	windowHeight = (int)depthDownRight.Y - (int)depthUpLeft.Y;
				draw_line(img, depthUpLeft, depthDownRight);
				if (check_new_point(front, highestPoint, depthHeight, depthWidth))
				{
					draw_circle(img, highestPoint.X, highestPoint.Y);		//碰到边缘就停止，不反弹
					if (highestPoint.X < depthUpLeft.X)
						highestPoint.X = depthUpLeft.X;
					if (highestPoint.X > depthDownRight.X)
						highestPoint.X = depthDownRight.X;
					if (highestPoint.Y > depthDownRight.Y)
						highestPoint.Y = depthDownRight.Y;
					if (highestPoint.Y < depthUpLeft.Y)
						highestPoint.Y = depthUpLeft.Y;
					int	mouseX = fabs(highestPoint.X - depthUpLeft.X);
					int	mouseY = fabs(highestPoint.Y - depthUpLeft.Y);
					mouse_event(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE, 65535 * mouseX / windowWidth, 65535 * mouseY / windowHeight, 0, 0);	//计算公式：小窗口的点/小窗口尺寸 = 大窗口的点/大
					front = highestPoint;
				}
				else							//如果和上一帧相比移动的幅度小于阈值，则保持上一帧的状态
				{
					draw_circle(img, front.X, front.Y);
					int	mouseX = fabs(front.X - depthUpLeft.X);
					int	mouseY = fabs(front.Y - depthUpLeft.Y);
					mouse_event(MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE, 65535 * mouseX / windowWidth, 65535 * mouseY / windowHeight, 0, 0);
				}
			}
		}

		imshow("TEST", img);
		if (waitKey(30) == VK_ESCAPE)
			break;

		myDepthFrame->Release();
		myBodyIndexFrame->Release();
		myBodyFrame->Release();
		delete[] bodyArray;
	}
	myBodySource->Release();
	myBodyIndexSource->Release();
	myDepthSource->Release();
	myBodyReader->Release();
	myBodyIndexReader->Release();
	myDepthReader->Release();
	myDescription->Release();
	myMapper->Release();
	mySensor->Close();
	mySensor->Release();


	return	0;
}

void	draw_body(Mat & img, BYTE * buffer, int height, int width)
{
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			int	index = buffer[i * width + j];
			if (index <= 5)
				img.at<Vec3b>(i, j) = COLOR_TABLE[WHITE];
			else
				img.at<Vec3b>(i, j) = COLOR_TABLE[BLACK];
		}
}

void	draw_circle(Mat & img, int x, int y)
{
	Point	center = { x,y };
	circle(img, center, 5, COLOR_TABLE[GREEN], -1, 0, 0);
}

void	draw_Hand(Mat & img, const DepthSpacePoint & hand)
{
	Point	center = { (int)hand.X,(int)hand.Y };
	circle(img, center, 5, COLOR_TABLE[BLUE], -1, 0, 0);
}

void	draw_line(Mat & img, const DepthSpacePoint & UpLeft, DepthSpacePoint & DownRight)
{
	Point	a = { (int)UpLeft.X,(int)DownRight.Y };
	circle(img, a, 5, COLOR_TABLE[RED], -1, 0, 0);
	Point	b = { (int)UpLeft.X,(int)UpLeft.Y };
	circle(img, b, 5, COLOR_TABLE[GREEN], -1, 0, 0);
	Point	c = { (int)DownRight.X,(int)UpLeft.Y };
	circle(img, c, 5, COLOR_TABLE[BLUE], -1, 0, 0);
	Point	d = { (int)DownRight.X,(int)DownRight.Y };
	circle(img, d, 5, COLOR_TABLE[WHITE], -1, 0, 0);
	line(img, a, b, COLOR_TABLE[RED], 1, 8, 0);
	line(img, b, c, COLOR_TABLE[RED], 1, 8, 0);
	line(img, c, d, COLOR_TABLE[RED], 1, 8, 0);
	line(img, a, d, COLOR_TABLE[RED], 1, 8, 0);
}

bool	level_check(const CameraSpacePoint & hand, const CameraSpacePoint & temp)
{
	if (fabs(temp.Z - hand.Z) <= OK_LEVEL)
		return	true;
	return	false;
}

bool	distance_check(const CameraSpacePoint & hand, const CameraSpacePoint & temp)
{
	double	ans = sqrt(pow(hand.X - temp.X, 2) + pow(hand.Y - temp.Y, 2) + pow(hand.Z - temp.Z, 2));
	if (ans <= 0.2 && ans >= 0.06)
		return	true;
	return	false;
}

bool	depth_rage_check(int x, int y, int depthWidth, int depthHeight)
{
	if (x >= 0 && x < depthWidth && y >= 0 && y < depthHeight)
		return	true;
	return	false;
}

bool	check_new_point(DepthSpacePoint & front, DepthSpacePoint & now, int height, int width)
{
	if (now.X == width - 1 && now.Y == height - 1 && (front.X || front.Y))
		return	false;
	else	if (fabs(now.X - front.X) <= OK_MOUSE && fabs(now.Y - front.Y) <= OK_MOUSE)
		return	false;
	return	true;
}