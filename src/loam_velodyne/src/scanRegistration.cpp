// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.



/*****************************Before You Read********************************************/
/*imu is a right-handed coordinate system with the x-axis forward, the y-axis left, and the z-axis upward,
The velodyne lidar is mounted as a right-handed coordinate system with the x-axis forward, the y-axis left, and the z-axis up,
scanRegistration will unify the two to the right-handed coordinate system with the z-axis forward, the x-axis left, and the y-axis up
, which is the coordinate system used in J. Zhang's paper
After swapping: R = Ry(yaw)*Rx(pitch)*Rz(roll)
*******************************************************************************/

#include <cmath>
#include <vector>

#include <loam_velodyne/common.h>
#include <opencv/cv.h>
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::sin;
using std::cos;
using std::atan2;

//Scanning period, velodyne frequency 10Hz, period 0.1s
const double scanPeriod = 0.1;

//Initialize control variables
const int systemDelay = 20;//Deprecate the first 20 frames of initial data
int systemInitCount = 0;
bool systemInited = false;

//Lidar line count
const int N_SCANS = 16;

//Point cloud curvature, 40000 is the maximum number of points in a frame of point cloud
float cloudCurvature[40000];
//The serial number corresponding to the curvature point
int cloudSortInd[40000];
//Whether the point is filtered flag: 0 unfiltered, 1 filtered
int cloudNeighborPicked[40000];
//Point classification label: 2 represents a large curvature, 1 represents a relatively large curvature, 1 represents a small curvature, 
//and 0 represents a relatively small curvature (where 1 contains 2, 0 contains 1, 0 and 1 constitute all the points of the point cloud )
int cloudLabel[40000];

//Where the Imu timestamp is greater than the current point cloud timestamp
int imuPointerFront = 0;
//The position in the array of the most recent point received by Imu
int imuPointerLast = -1;
//Imu circular queue length
const int imuQueLength = 200;

//The displacement/velocity/Eulerian angle of the first point in the point cloud data
float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0;
float imuRollCur = 0, imuPitchCur = 0, imuYawCur = 0;

float imuVeloXStart = 0, imuVeloYStart = 0, imuVeloZStart = 0;
float imuShiftXStart = 0, imuShiftYStart = 0, imuShiftZStart = 0;

//Velocity and displacement information of the current point
float imuVeloXCur = 0, imuVeloYCur = 0, imuVeloZCur = 0;
float imuShiftXCur = 0, imuShiftYCur = 0, imuShiftZCur = 0;

//Distortion displacement of the current point of the point cloud data relative to the first point at the beginning, speed
float imuShiftFromStartXCur = 0, imuShiftFromStartYCur = 0, imuShiftFromStartZCur = 0;
float imuVeloFromStartXCur = 0, imuVeloFromStartYCur = 0, imuVeloFromStartZCur = 0;

//Imu information
double imuTime[imuQueLength] = {0};
float imuRoll[imuQueLength] = {0};
float imuPitch[imuQueLength] = {0};
float imuYaw[imuQueLength] = {0};

float imuAccX[imuQueLength] = {0};
float imuAccY[imuQueLength] = {0};
float imuAccZ[imuQueLength] = {0};

float imuVeloX[imuQueLength] = {0};
float imuVeloY[imuQueLength] = {0};
float imuVeloZ[imuQueLength] = {0};

float imuShiftX[imuQueLength] = {0};
float imuShiftY[imuQueLength] = {0};
float imuShiftZ[imuQueLength] = {0};

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubImuTrans;

//Calculate the displacement distortion caused by the acceleration and deceleration of the point 
//in the point cloud relative to the first starting point in the local coordinate system
void ShiftToStartIMU(float pointTime)
{
  //Calculate the distortion displacement due to acceleration and deceleration relative to the first 
  //point (distortion displacement delta_Tg in the global coordinate system)
  //imuShiftFromStartCur = imuShiftCur -(imuShiftStart + imuVeloStart *pointTime)
  imuShiftFromStartXCur = imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime;
  imuShiftFromStartYCur = imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
  imuShiftFromStartZCur = imuShiftZCur - imuShiftZStart - imuVeloZStart * pointTime;

  /********************************************************************************
  Rz(pitch).inverse * Rx(pitch).inverse * Ry(yaw).inverse * delta_Tg
  transfrom from the global frame to the local frame
  *********************************************************************************/

  //Rotate around the y axis ( imu yaw start ), ie ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuShiftFromStartXCur - sin(imuYawStart) * imuShiftFromStartZCur;
  float y1 = imuShiftFromStartYCur;
  float z1 = sin(imuYawStart) * imuShiftFromStartXCur + cos(imuYawStart) * imuShiftFromStartZCur;

  //Rotate around the x-axis (imu pitch start), ie rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;

  //Rotate around the z-axis (imu roll start), ie rz(pitch).inverse
  imuShiftFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuShiftFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuShiftFromStartZCur = z2;
}

//Calculate the velocity distortion (increment) of the point in the point cloud relative to the 
//first starting point due to acceleration and deceleration in the local coordinate system
void VeloToStartIMU()
{
  //Calculate the distortion speed due to acceleration and deceleration relative to the first point (distortion speed increment delta vg in the global coordinate system)
  imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart;
  imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
  imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;

  /********************************************************************************
    Rz(pitch).inverse * Rx(pitch).inverse * Ry(yaw).inverse * delta_Vg
    transfrom from the global frame to the local frame
  *********************************************************************************/
  
  //Rotate around the y axis ( imu yaw start ), ie ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuVeloFromStartXCur - sin(imuYawStart) * imuVeloFromStartZCur;
  float y1 = imuVeloFromStartYCur;
  float z1 = sin(imuYawStart) * imuVeloFromStartXCur + cos(imuYawStart) * imuVeloFromStartZCur;

  //Rotate around the x-axis (imu pitch start), ie rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;

  //Rotate around the z-axis (imu roll start), ie rz(pitch).inverse
  imuVeloFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuVeloFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuVeloFromStartZCur = z2;
}

//Remove displacement distortion caused by point cloud acceleration and deceleration
void TransformToStartIMU(PointType *p)
{
  /********************************************************************************
  Ry*Rx*Rz*Pl, transform point to the global frame
  *********************************************************************************/
  //Rotate around the z-axis (imuRollCur)
  float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;
  float y1 = sin(imuRollCur) * p->x + cos(imuRollCur) * p->y;
  float z1 = p->z;

  //Rotate around the x-axis (imu pitch cur)
  float x2 = x1;
  float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
  float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;

  //Rotate around the y-axis (imu yaw cur)
  float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
  float y3 = y2;
  float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;

  /********************************************************************************
    Rz(pitch).inverse * Rx(pitch).inverse * Ry(yaw).inverse * Pg
    transfrom global points to the local frame
  *********************************************************************************/
  
  //Rotate around the y-axis ( imu yaw start )
  float x4 = cos(imuYawStart) * x3 - sin(imuYawStart) * z3;
  float y4 = y3;
  float z4 = sin(imuYawStart) * x3 + cos(imuYawStart) * z3;

  //Rotate around the x-axis (imu pitch start)
  float x5 = x4;
  float y5 = cos(imuPitchStart) * y4 + sin(imuPitchStart) * z4;
  float z5 = -sin(imuPitchStart) * y4 + cos(imuPitchStart) * z4;

  //Rotate around the z-axis ( imu roll start ), then superimpose the translation
  p->x = cos(imuRollStart) * x5 + sin(imuRollStart) * y5 + imuShiftFromStartXCur;
  p->y = -sin(imuRollStart) * x5 + cos(imuRollStart) * y5 + imuShiftFromStartYCur;
  p->z = z5 + imuShiftFromStartZCur;
}

//Integrating Velocity and Displacement
void AccumulateIMUShift()
{
  float roll = imuRoll[imuPointerLast];
  float pitch = imuPitch[imuPointerLast];
  float yaw = imuYaw[imuPointerLast];
  float accX = imuAccX[imuPointerLast];
  float accY = imuAccY[imuPointerLast];
  float accZ = imuAccZ[imuPointerLast];

  //Rotate the current acceleration value around the exchanged ZXY fixed axis (original XYZ) by (roll, pitch, yaw) angles, and convert to obtain the acceleration value in the world coordinate system (right hand rule)
  //rotate around the z-axis (roll)
  float x1 = cos(roll) * accX - sin(roll) * accY;
  float y1 = sin(roll) * accX + cos(roll) * accY;
  float z1 = accZ;
  //rotate around the x-axis (pitch)
  float x2 = x1;
  float y2 = cos(pitch) * y1 - sin(pitch) * z1;
  float z2 = sin(pitch) * y1 + cos(pitch) * z1;
  //rotate around the y-axis (yaw)
  accX = cos(yaw) * x2 + sin(yaw) * z2;
  accY = y2;
  accZ = -sin(yaw) * x2 + cos(yaw) * z2;

  //previous imu point
  int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;
  //The time from the previous point to the current point, that is, the calculation of the imu measurement period
  double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack];
  //The frequency of imu is required to be at least higher than that of lidar. Only such imu information is used, and subsequent correction is meaningful.
  if (timeDiff < scanPeriod) {//(implies movement from rest)
    //Find the displacement and velocity of each imu time point, between the two points as a uniform acceleration linear motion
    imuShiftX[imuPointerLast] = imuShiftX[imuPointerBack] + imuVeloX[imuPointerBack] * timeDiff 
                              + accX * timeDiff * timeDiff / 2;
    imuShiftY[imuPointerLast] = imuShiftY[imuPointerBack] + imuVeloY[imuPointerBack] * timeDiff 
                              + accY * timeDiff * timeDiff / 2;
    imuShiftZ[imuPointerLast] = imuShiftZ[imuPointerBack] + imuVeloZ[imuPointerBack] * timeDiff 
                              + accZ * timeDiff * timeDiff / 2;

    imuVeloX[imuPointerLast] = imuVeloX[imuPointerBack] + accX * timeDiff;
    imuVeloY[imuPointerLast] = imuVeloY[imuPointerBack] + accY * timeDiff;
    imuVeloZ[imuPointerLast] = imuVeloZ[imuPointerBack] + accZ * timeDiff;
  }
}

//Receiving point cloud data, the velodyne radar coordinate system is installed as a right-handed 
//coordinate system with the x-axis forward, the y-axis left, and the z-axis upward
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
  if (!systemInited) {//Discard the first 20 point cloud data
    systemInitCount++;
    if (systemInitCount >= systemDelay) {
      systemInited = true;
    }
    return;
  }

  //Record the start and end index of each scan point with curvature
  std::vector<int> scanStartInd(N_SCANS, 0);
  std::vector<int> scanEndInd(N_SCANS, 0);
  
  //current point cloud time
  double timeScanCur = laserCloudMsg->header.stamp.toSec();
  pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
  //The message is converted into pcl data storage
  pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
  std::vector<int> indices;
  //remove empty dots
  pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
  //Number of point cloud points
  int cloudSize = laserCloudIn.points.size();
  //The rotation angle of the starting point of the lidar scan, the atan2 range is [-pi, +pi], the 
  //negative sign is taken when calculating the rotation angle because the velodyne rotates clockwise
  float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
  //The rotation angle of the end point of lidar scan, add 2*pi to make the point cloud rotation period 2*pi
  float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                        laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;

  //The difference between the end azimuth and the start azimuth is controlled in the range of (PI, 3*PI), 
  //allowing lidar to not be a circular scan
  //In this range under normal circumstances: pi < endOri -startOri < 3*pi, if it is abnormal, it will be corrected
  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }
  //Whether the lidar scan line is rotated more than half
  bool halfPassed = false;
  int count = cloudSize;
  PointType point;
  std::vector<pcl::PointCloud<PointType> > laserCloudScans(N_SCANS);
  /**********************************************************************************************************/
  //Me parece que este For lo hacen para ordenar la nube a un sistema XYZ y los coloca el orden segun el BEAM
  /**********************************************************************************************************/ 
  for (int i = 0; i < cloudSize; i++) {
    //The coordinate axis is exchanged, and the coordinate system of the velodyne lidar is also converted 
    //to the right-hand coordinate system with the z-axis forward and the x-axis left.
    point.x = laserCloudIn.points[i].y;
    point.y = laserCloudIn.points[i].z;
    point.z = laserCloudIn.points[i].x;

    //Calculate the elevation angle of the point (according to the vertical angle calculation formula of 
    //the lidar document), arrange the laser line numbers according to the elevation angle, and the interval
    //between each two scans of velodyne is 2 degrees
    float angle = atan(point.y / sqrt(point.x * point.x + point.z * point.z)) * 180 / M_PI;
    int scanID;
    //Elevation angle is rounded (adding or subtracting 0.5 truncation effect is equal to rounding)
    int roundedAngle = int(angle + (angle<0.0?-0.5:+0.5)); 
    if (roundedAngle > 0){
      scanID = roundedAngle;
    }
    else {
      scanID = roundedAngle + (N_SCANS - 1);
    }
    //Filter points, only select points in the range of [15 degrees, +15 degrees], scan id belongs to [0,15]
    if (scanID > (N_SCANS - 1) || scanID < 0 ){
      count--;
      continue;
    }

    //The rotation angle of the point
    float ori = -atan2(point.x, point.z);
    if (!halfPassed) {//According to whether the scan line is rotated more than half, select and calculate the 
                      //difference between the start position or the end position, so as to compensate
      //make sure -pi/2 < ori -startOri < 3*pi/2
      if (ori < startOri - M_PI / 2) {
        ori += 2 * M_PI;
      } else if (ori > startOri + M_PI * 3 / 2) {
        ori -= 2 * M_PI;
      }

      if (ori - startOri > M_PI) {
        halfPassed = true;
      }
    } else {
      ori += 2 * M_PI;

      //-3 *pi /2 <ori -endOri <pi /2
      if (ori < endOri - M_PI * 3 / 2) {
        ori += 2 * M_PI;
      } else if (ori > endOri + M_PI / 2) {
        ori -= 2 * M_PI;
      } 
    }

    //-1/2 < relTime < 3/2 (the ratio of the rotation angle of the point to the rotation angle of the whole cycle, 
    //that is, the relative time of the point in the point cloud)
    float relTime = (ori - startOri) / (endOri - startOri);
    //Point intensity = line number + point relative time (that is, an integer + a decimal, the integer part is 
    //the line number, and the decimal part is the relative time of the point), uniform scan: Calculate the relative 
    //scan start based on the current scan angle and scan cycle time at the starting position
    point.intensity = scanID + scanPeriod * relTime;

    //point time = point cloud time + cycle time
    if (imuPointerLast >= 0) {//If imu data is received, use imu to correct point cloud distortion
      float pointTime = relTime * scanPeriod;//Calculate the cycle time of the point
      //Look for the IMU position where the timestamp of the point cloud is less than the timestamp of the IMU: imuPointerFront
      while (imuPointerFront != imuPointerLast) {
        if (timeScanCur + pointTime < imuTime[imuPointerFront]) {
          break;
        }
        imuPointerFront = (imuPointerFront + 1) % imuQueLength;
      }

      if (timeScanCur + pointTime > imuTime[imuPointerFront]) {//Not found, at this time imu pointer front==imt pointer last, 
                                                               //only the speed, displacement and Euler angle of the latest imu 
                                                               //received can only be used as the speed, displacement and Euler 
                                                               //angle of the current point
        imuRollCur = imuRoll[imuPointerFront];
        imuPitchCur = imuPitch[imuPointerFront];
        imuYawCur = imuYaw[imuPointerFront];

        imuVeloXCur = imuVeloX[imuPointerFront];
        imuVeloYCur = imuVeloY[imuPointerFront];
        imuVeloZCur = imuVeloZ[imuPointerFront];

        imuShiftXCur = imuShiftX[imuPointerFront];
        imuShiftYCur = imuShiftY[imuPointerFront];
        imuShiftZCur = imuShiftZ[imuPointerFront];
      } else {//If the imu position of the point cloud timestamp is less than the imu timestamp, the point must be between the imu 
              //pointer back and the imu pointer front. According to this linear interpolation, the speed, displacement and Euler 
              //angle of the point cloud point are calculated.
        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
        //Calculate the weight distribution ratio according to the time distance, that is, linear interpolation
        float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) 
                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) 
                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

        imuRollCur = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
        imuPitchCur = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
        if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] > M_PI) {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] + 2 * M_PI) * ratioBack;
        } else if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] < -M_PI) {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] - 2 * M_PI) * ratioBack;
        } else {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + imuYaw[imuPointerBack] * ratioBack;
        }

        //Essence: imuVeloXCur = imuVeloX[imuPointerback] + (imuVelX[imuPointerFront]-imuVelX[imuPoniterBack])*ratioFront
        imuVeloXCur = imuVeloX[imuPointerFront] * ratioFront + imuVeloX[imuPointerBack] * ratioBack;
        imuVeloYCur = imuVeloY[imuPointerFront] * ratioFront + imuVeloY[imuPointerBack] * ratioBack;
        imuVeloZCur = imuVeloZ[imuPointerFront] * ratioFront + imuVeloZ[imuPointerBack] * ratioBack;

        imuShiftXCur = imuShiftX[imuPointerFront] * ratioFront + imuShiftX[imuPointerBack] * ratioBack;
        imuShiftYCur = imuShiftY[imuPointerFront] * ratioFront + imuShiftY[imuPointerBack] * ratioBack;
        imuShiftZCur = imuShiftZ[imuPointerFront] * ratioFront + imuShiftZ[imuPointerBack] * ratioBack;
      }

      if (i == 0) {//If it is the first point, remember the velocity, displacement, Euler angle of the starting position of the point cloud
        imuRollStart = imuRollCur;
        imuPitchStart = imuPitchCur;
        imuYawStart = imuYawCur;

        imuVeloXStart = imuVeloXCur;
        imuVeloYStart = imuVeloYCur;
        imuVeloZStart = imuVeloZCur;

        imuShiftXStart = imuShiftXCur;
        imuShiftYStart = imuShiftYCur;
        imuShiftZStart = imuShiftZCur;
      } else {//After calculating the displacement velocity distortion of each point relative to the first point due to non-uniform acceleration 
              //and deceleration motion, and re-compensating and correcting the position information of each point in the point cloud
        ShiftToStartIMU(pointTime);
        VeloToStartIMU();
        TransformToStartIMU(&point);
      }
    }
    laserCloudScans[scanID].push_back(point);//Put each point of compensation correction into the container of the corresponding line number
  }

  //Get the number of points in the valid range
  cloudSize = count;

  pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
  for (int i = 0; i < N_SCANS; i++) {//Put all the points into a container according to the line number from small to large
    *laserCloud += laserCloudScans[i];
  }
  int scanCount = -1;
  /**********************************************************************************************************/
  //En este For para cada punto realizan la formula de la curvatura, pero solo para los puntos de un mismo BEAM
  //Tambien, crea unas variables adicionales como Sort, Label, Neighborpicked... Ya veremos de que trata..
  //Los valores de curvatura los guardan en el vector CloudCurvature
  /**********************************************************************************************************/
  for (int i = 5; i < cloudSize - 5; i++) {//The curvature is calculated using the five points before and after each point, so the first and last 
                                           //five points are skipped
    float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x 
                + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x 
                + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x 
                + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x
                + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x
                + laserCloud->points[i + 5].x;
    float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y 
                + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y 
                + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y 
                + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y
                + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y
                + laserCloud->points[i + 5].y;
    float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z 
                + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z 
                + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z 
                + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z
                + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z
                + laserCloud->points[i + 5].z;
    //curvature calculation
    cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
    //record the index of the curvature point
    cloudSortInd[i] = i;
    //Initially, the points are not filtered
    cloudNeighborPicked[i] = 0;
    //Initialize to less flat point
    cloudLabel[i] = 0;

    //For each scan, only the first matching point will come in, because the points of each scan are stored together
    if (int(laserCloud->points[i].intensity) != scanCount) {
      scanCount = int(laserCloud->points[i].intensity);//Control each scan to only enter the first point

      //The curvature is only calculated by the same scan. The curvature calculated across scans is illegal. Exclude, that is, exclude the five 
      //points before and after each scan.
      if (scanCount > 0 && scanCount < N_SCANS) {
        scanStartInd[scanCount] = i + 5;
        scanEndInd[scanCount - 1] = i - 5;
      }
    }
  }
  //The valid point sequence of the first scan curvature point starts from the 5th, and the end point sequence of the last laser line is size 5
  scanStartInd[0] = 5;
  scanEndInd.back() = cloudSize - 5;

  /**********************************************************************************************************/
  //Hacen el cálculo de los puntos que son conflictivos (Sit1,Sit2,Sit3) y los guardan en NeighborPicked
  /**********************************************************************************************************/
  //Select points to exclude the points that are easily blocked by the inclined plane and outliers. Some points are easy to be blocked by the 
  //inclined plane, and the outliers may appear by chance, which may cause the two scans before and after the scan to not be seen at the same time.
  for (int i = 5; i < cloudSize - 6; i++) {//and the latter pip value, so subtract 6
    float diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x;
    float diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y;
    float diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z;
    //Calculate the sum of squared distances between the effective curvature point and the next point
    float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;

    if (diff > 0.1) {//Premise: The distance between two points should be greater than 0.1

      //point depth
      float depth1 = sqrt(laserCloud->points[i].x * laserCloud->points[i].x + 
                     laserCloud->points[i].y * laserCloud->points[i].y +
                     laserCloud->points[i].z * laserCloud->points[i].z);

      //the depth of the next point
      float depth2 = sqrt(laserCloud->points[i + 1].x * laserCloud->points[i + 1].x + 
                     laserCloud->points[i + 1].y * laserCloud->points[i + 1].y +
                     laserCloud->points[i + 1].z * laserCloud->points[i + 1].z);

      //According to the ratio of the depth of the two points, the distance is calculated after pulling the point with a larger depth back
      if (depth1 > depth2) {
        diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x * depth2 / depth1;
        diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y * depth2 / depth1;
        diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z * depth2 / depth1;

        //The side length ratio is also the radian value. If it is less than 0.1, it means that the angle is relatively small, the slope is relatively 
        //steep, and the point depth changes sharply. The point is on the slope that is approximately parallel to the laser beam.
        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1) {//Exclude points that are easily blocked by slopes
          //This point and the previous five points (roughly on the slope) are all set as filtered
          cloudNeighborPicked[i - 5] = 1;
          cloudNeighborPicked[i - 4] = 1;
          cloudNeighborPicked[i - 3] = 1;
          cloudNeighborPicked[i - 2] = 1;
          cloudNeighborPicked[i - 1] = 1;
          cloudNeighborPicked[i] = 1;
        }
      } else {
        diffX = laserCloud->points[i + 1].x * depth1 / depth2 - laserCloud->points[i].x;
        diffY = laserCloud->points[i + 1].y * depth1 / depth2 - laserCloud->points[i].y;
        diffZ = laserCloud->points[i + 1].z * depth1 / depth2 - laserCloud->points[i].z;

        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1) {
          cloudNeighborPicked[i + 1] = 1;
          cloudNeighborPicked[i + 2] = 1;
          cloudNeighborPicked[i + 3] = 1;
          cloudNeighborPicked[i + 4] = 1;
          cloudNeighborPicked[i + 5] = 1;
          cloudNeighborPicked[i + 6] = 1;
        }
      }
    }

    float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
    float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
    float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
    //sum of squared distances from the previous point
    float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

    //sum of squares of point depths
    float dis = laserCloud->points[i].x * laserCloud->points[i].x
              + laserCloud->points[i].y * laserCloud->points[i].y
              + laserCloud->points[i].z * laserCloud->points[i].z;

    //The sum of squares with the front and rear points is greater than 2/10,000 of the sum of the squares of the depth. These points are 
    //regarded as outliers, including points on steep slopes, strong convex and concave points and some points in open areas, which are set as 
    //filtered , deprecated
    if (diff > 0.0002 * dis && diff2 > 0.0002 * dis) {
      cloudNeighborPicked[i] = 1;
    }
  }


  pcl::PointCloud<PointType> cornerPointsSharp;
  pcl::PointCloud<PointType> cornerPointsLessSharp;
  pcl::PointCloud<PointType> surfPointsFlat;
  pcl::PointCloud<PointType> surfPointsLessFlat;

  /**********************************************************************************************************/
  //Para rayos de una misa fila SCAN/BEAM se realiza la subdivisión de regiones. Además, se ordena la nube con 
  //base al valor de su curvatura de menor a mayor
  /**********************************************************************************************************/
  //Divide the points on each line into the corresponding categories: edge points and plane points
  for (int i = 0; i < N_SCANS; i++){
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
    
    /**********************************************************************************************************/
    //Para cada sector, analizamos si el punto será borde o plano o estará en la categoria de los sobrantes
    /**********************************************************************************************************/
    //Divide the curvature points of each scan into 6 equal parts to ensure that all surrounding points are selected as feature points
    for (int j = 0; j < 6; j++) {
      //Six equal starting points: sp = scanStartInd + (scanEndInd -scanStartInd)*j/6
      int sp = (scanStartInd[i] * (6 - j)  + scanEndInd[i] * j) / 6;
      //Six equal end points: ep = scanStartInd -1 + (scanEndInd -scanStartInd)*(j+1)/6
      int ep = (scanStartInd[i] * (5 - j)  + scanEndInd[i] * (j + 1)) / 6 - 1;

      //sort by curvature from small to large
      for (int k = sp + 1; k <= ep; k++) {
        for (int l = k; l >= sp + 1; l--) {
          //If the back curvature point is greater than the front, then swap
          if (cloudCurvature[cloudSortInd[l]] < cloudCurvature[cloudSortInd[l - 1]]) {
            int temp = cloudSortInd[l - 1];
            cloudSortInd[l - 1] = cloudSortInd[l];
            cloudSortInd[l] = temp;
          }
        }
      }
      /********************************************************************************************************/
      //Para cada punto de cada region se le analiza si es borde o no. ADemás se le clasifica en E1 o E2. 
      //Adicional, para evitar que los puntos caracteristícos se junten, por cada punto guardado se le analiza
      //sus vencidarios y se filtran como conflictivos, solo para evitar tener muchos FEaturesPoints en un único 
      //lugar.
      /********************************************************************************************************/
      //pick points with large and relatively large curvature of each segment
      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--) {
        int ind = cloudSortInd[k];//The point sequence of the maximum curvature point

        //If the curvature is large, the curvature is indeed large, and it is not filtered out
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] > 0.1) {//!!!Esta es la condición del diagrama de flujo "EdgePointsDetection"
        
          largestPickedNum++;
          if (largestPickedNum <= 2) {//Pick the top 2 points with the largest curvature and put them into the sharp point set
            cloudLabel[ind] = 2;//2 means the point has a large curvature
            cornerPointsSharp.push_back(laserCloud->points[ind]);
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else if (largestPickedNum <= 20) {//Pick the top 20 points with the largest curvature and put them into the less sharp point set
            cloudLabel[ind] = 1;//1 means the point curvature is relatively sharp
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else {
            break;//Ya superamos los umbrales. No debemos guardar más puntos
          }

          cloudNeighborPicked[ind] = 1;//filter flag set
          //Prestar atencion a lo de abajo!!
          //Filter out 5 consecutive points with relatively close distances before and after the point with relatively large curvature to prevent 
          //the feature points from gathering, so that the feature points are distributed as evenly as possible in each direction
          for (int l = 1; l <= 5; l++) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      /********************************************************************************************************/
      //Similar que el anterior, hacemos el proceso de PlanarPointsDetection
      /********************************************************************************************************/
      //Select points with small curvature for each segment
      int smallestPickedNum = 0;
      for (int k = sp; k <= ep; k++) {
        int ind = cloudSortInd[k];

        //If the curvature is indeed small and not filtered out
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] < 0.1) {

          cloudLabel[ind] = -1;//1 represents a point with little curvature
          surfPointsFlat.push_back(laserCloud->points[ind]);

          smallestPickedNum++;
          if (smallestPickedNum >= 4) {//Only the smallest four are selected, and the remaining label==0, they are all with relatively small curvature
            break;
          }

          cloudNeighborPicked[ind] = 1;
          for (int l = 1; l <= 5; l++) {//Also prevent feature points from gathering
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      //The remaining points (including the previously excluded points) are all classified into the less flat category of plane points
      for (int k = sp; k <= ep; k++) {
        if (cloudLabel[k] <= 0) {//Los valores de cloudLabel son -1,0,1,2
          surfPointsLessFlatScan->push_back(laserCloud->points[k]);
        }
      }
    }

    //Interesante lo que hacen, para reducir el tiempo de procesamiento, hacen un downsampling de los puntos
    //Since the less flat points are the most, perform voxel grid filtering on each segmented less flat point
    pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
    pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setInputCloud(surfPointsLessFlatScan);
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.filter(surfPointsLessFlatScanDS);

    //less flat point summary
    surfPointsLessFlat += surfPointsLessFlatScanDS;
  }

  //public eliminates all points after non-uniform motion distortion
  sensor_msgs::PointCloud2 laserCloudOutMsg;
  pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
  laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
  laserCloudOutMsg.header.frame_id = "/camera";
  pubLaserCloud.publish(laserCloudOutMsg);

  //public plane point and edge point after eliminating non-uniform motion distortion
  sensor_msgs::PointCloud2 cornerPointsSharpMsg;
  pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
  cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsSharpMsg.header.frame_id = "/camera";
  pubCornerPointsSharp.publish(cornerPointsSharpMsg);

  sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
  pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
  cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsLessSharpMsg.header.frame_id = "/camera";
  pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

  sensor_msgs::PointCloud2 surfPointsFlat2;
  pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
  surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsFlat2.header.frame_id = "/camera";
  pubSurfPointsFlat.publish(surfPointsFlat2);

  sensor_msgs::PointCloud2 surfPointsLessFlat2;
  pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
  surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsLessFlat2.header.frame_id = "/camera";
  pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

  //publish IMU message, because the cycle is at the end, so Cur is the last point, that is, the Euler angle of the last point, the distortion 
  //displacement and the speed of a point cloud period increase
  pcl::PointCloud<pcl::PointXYZ> imuTrans(4, 1);
  //Start point Euler angle
  imuTrans.points[0].x = imuPitchStart;
  imuTrans.points[0].y = imuYawStart;
  imuTrans.points[0].z = imuRollStart;

  //Euler angles of the last point
  imuTrans.points[1].x = imuPitchCur;
  imuTrans.points[1].y = imuYawCur;
  imuTrans.points[1].z = imuRollCur;

  //Distortion displacement and velocity of the last point relative to the first point
  imuTrans.points[2].x = imuShiftFromStartXCur;
  imuTrans.points[2].y = imuShiftFromStartYCur;
  imuTrans.points[2].z = imuShiftFromStartZCur;

  imuTrans.points[3].x = imuVeloFromStartXCur;
  imuTrans.points[3].y = imuVeloFromStartYCur;
  imuTrans.points[3].z = imuVeloFromStartZCur;

  sensor_msgs::PointCloud2 imuTransMsg;
  pcl::toROSMsg(imuTrans, imuTransMsg);
  imuTransMsg.header.stamp = laserCloudMsg->header.stamp;
  imuTransMsg.header.frame_id = "/camera";
  pubImuTrans.publish(imuTransMsg);
}

//Receive the imu message, the imu coordinate system is the right-handed coordinate system of the x-axis forward, the y-axis to the right, 
//and the z-axis upward
void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
{
  double roll, pitch, yaw;
  tf::Quaternion orientation;
  //convert Quaternion msg to Quaternion
  tf::quaternionMsgToTF(imuIn->orientation, orientation);
  //This will get the roll pitch and yaw from the matrix about fixed axes X, Y, Z respectively. That's R = Rz(yaw)*Ry(pitch)*Rx(roll).
  //Here roll pitch yaw is in the global frame
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

  //Subtract the influence of gravity, find the actual value of the acceleration in the xyz direction, and exchange the coordinate axes, unify 
  //them to the right-hand coordinate system with the z-axis forward and the x-axis left. After the exchange, RPY corresponds to the fixed axes 
  //ZXY(RPY---ZXY). Now R = Ry(yaw)*Rx(pitch)*Rz(roll).
  float accX = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81;
  float accY = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81;
  float accZ = imuIn->linear_acceleration.x + sin(pitch) * 9.81;

  //Circular shift effect to form a circular array
  imuPointerLast = (imuPointerLast + 1) % imuQueLength;

  imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
  imuRoll[imuPointerLast] = roll;
  imuPitch[imuPointerLast] = pitch;
  imuYaw[imuPointerLast] = yaw;
  imuAccX[imuPointerLast] = accX;
  imuAccY[imuPointerLast] = accY;
  imuAccZ[imuPointerLast] = accZ;

  AccumulateIMUShift();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "scanRegistration");
  ros::NodeHandle nh;

  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2> 
                                  ("/velodyne_points", 2, laserCloudHandler);

  ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu> ("/imu/data", 50, imuHandler);

  pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>
                                 ("/velodyne_cloud_2", 2);

  pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>
                                        ("/laser_cloud_sharp", 2);

  pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>
                                            ("/laser_cloud_less_sharp", 2);

  pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>
                                       ("/laser_cloud_flat", 2);

  pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>
                                           ("/laser_cloud_less_flat", 2);

  pubImuTrans = nh.advertise<sensor_msgs::PointCloud2> ("/imu_trans", 5);

  ros::spin();

  return 0;
}

