#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>

struct point_type{
    cv::Point p_pixel;
    float range;
    float x,y,z;
    float intensity;
};

// typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZI PointT;

void pointcloud_visualizer(pcl::PointCloud<PointT>::Ptr cloud)
{
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    std::cout<<"size:"<<cloud->size()<<std::endl;
    viewer.showCloud(cloud);
    while (!viewer.wasStopped())
    {
    }
}

void pointcloud_visualizer(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    std::cout<<"size:"<<cloud->size()<<std::endl;
    viewer.showCloud(cloud);
    while (!viewer.wasStopped())
    {
    }
}

int main(int argc, char  *argv[])
{
    //读取参数 图片、点云
    if (argc != 8)
    {
        std::cout<<"输入正确的图片和点云"<<std::endl;
        return -1;
    }
    std::string flag = argv[3];
    float x_ = std::stof(argv[4]);
    float y_ = std::stof(argv[5]);
    float u_ = std::stof(argv[6]);
    float v_ = std::stof(argv[7]);

    //读取图像
    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    std::cout<<"image_rows"<<image.rows<<std::endl;
    std::cout<<"image_cols"<<image.cols<<std::endl;

    //读取点云
    pcl::PointCloud<PointT>::Ptr cloud_in(new pcl::PointCloud<PointT>());
    pcl::io::loadPCDFile<PointT>(argv[2], *cloud_in);

    //雷达坐标系下，点云滤波处理+点云显示
    pcl::PointCloud<PointT>::Ptr cloud_filter (new pcl::PointCloud<PointT>());
    for(const auto& p:(*cloud_in)){
        // if (p.x <= 0 || std::abs(p.y)>=10)
        if (p.x <= 0 )
        {
            continue;
        }
        cloud_filter->points.push_back(p);
    }

    //雷达坐标系转换到相机坐标系
    Eigen::Matrix3d R;
    R << 1.0712590905646691e-02, -9.7589883155074775e-04, 9.9994214233502454e-01,
        -9.9967732139170673e-01, 2.3023302826578718e-02, 1.0732223537442631e-02,
        -2.3032444316447132e-02, -9.9973445231642677e-01, -7.2894470539575096e-04;
    Eigen::Vector3d t;
    t << 1.6590545127807857e-01, 7.8359582766119231e-03, -1.7705054196958453e-01;
    
    Eigen::Matrix4d Transformed;
    Transformed.block<3, 3>(0, 0) = R;
    Transformed.topRightCorner(3, 1) = t;
    Eigen::Matrix4d Transformed_transpose = Transformed.transpose();
    std::cout << Transformed.matrix()<< std::endl;

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*cloud_filter, *cloud_transformed, Transformed_transpose); //点云转换

    std::cout<<"相机坐标系下的点云数目为:"<<cloud_transformed->points.size()<<std::endl;
    
    //相机坐标系下点云投影
    //设置Ｋ内参矩阵

    Eigen::Matrix3d K;
    K << 8.8625938251091372e+02*x_, 0., 664.26225408794346+u_,
        0., 8.8756691147521019e+02*y_, 366.76158911118756+v_,
        0., 0., 1.0;

    double k1 = 1.7176426571616824e-01, k2 =-5.5378313614637031e-01, k3 = 5.2299542084624961e-01;
    double p1 = 4.4955851302517285e-03, p2 = 1.9956894972634263e-03;

    // Eigen::Matrix3d K;
    // K << 903.6628, 0., 645.4777,
    //     0., 903.8976, 367.2172,
    //     0., 0., 1.0;

    // double k1 = 0.1223, k2 = -0.301, k3 = 0;
    // double p1 = 0, p2 = 0;

    cv::Mat image_lidar(720, 1280, CV_8UC3);

    float max_intensity = 0, min_intensity = 1000;
    float max_range = 0, min_range = 1000;
    float max_x = 0, min_x = 100;
    float max_y = 0, min_y = 100;
    float max_z = 0, min_z = 100;
    std::vector<point_type> pixel_sum;

    for (const auto &p : (*cloud_transformed))    //从相机坐标系到像素坐标系，记录xyzir
    {
        Eigen::Vector3d p_normalizated(p.x / p.z, p.y / p.z, 1);        //每个点进行归一化处理

        double x = p_normalizated[0];        //归一化平面上激光点去畸变
        double y = p_normalizated[1];
        double r = x * x + y * y;
        Eigen::Vector3d p_undisted(x * (1 + k1 * r + k2 * r * r + k3 * r * r * r) + 2 * p1 * x * y + p2 * (r + 2 * x * x),
                                   y * (1 + k1 * r + k2 * r * r + k3 * r * r * r) + p1 * (r + 2 * y * y) + 2 * p2 * x * y,
                                   1.0);

        Eigen::Vector3d p_projected = K * p_undisted;        //进行像素平面投影
        cv::Point p_pixel;
        p_pixel.x = p_projected[0];
        p_pixel.y = p_projected[1];

        if (p_pixel.x <= image.cols && p_pixel.y <= image.rows)
        {
            point_type point_tmp;
            point_tmp.x = p.x;
            point_tmp.y = p.y;
            point_tmp.z = p.z;
            point_tmp.intensity = p.intensity;
            float range = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
            point_tmp.range = range;
            point_tmp.p_pixel = p_pixel;
            pixel_sum.push_back(point_tmp);

            if(p.x>=max_x)  max_x = p.x;
            if(p.x<= min_x) min_x = p.x;
            if(p.y>=max_y)  max_y = p.y;
            if(p.y<= min_y) min_y = p.y;
            if(p.z>=max_z) max_z = p.z;
            if(p.z<=min_z) min_z = p.z;
            if(p.intensity >= max_intensity) max_intensity = p.intensity;
            if(p.intensity <= min_intensity) min_intensity = p.intensity;
            if(range>=max_range) max_range = range;
            if(range<=min_range) min_range = range;
        }
    }

    std::cout<<"投影在图像上的点云数目为:"<<pixel_sum.size()<<std::endl;
    if(true){
            std::cout<<"ｘ的范围"<<min_x<<"~"<<max_x<<std::endl;
            std::cout<<"ｙ的范围"<<min_y<<"~"<<max_y<<std::endl;
            std::cout<<"ｚ的范围"<<min_z<<"~"<<max_z<<std::endl;
            std::cout<<"intensity的范围"<<min_intensity<<"~"<<max_intensity<<std::endl;
            std::cout<<"range的范围"<<min_range<<"~"<<max_range<<std::endl;
    }

    //对提取的点进行处理
    for (const auto &p : pixel_sum)
    {
        int cur_val;
        if (flag == "intensity")
        {
            float val = p.intensity;
            float minVal = min_intensity;
            float maxVal = max_intensity;
            cur_val = (int)(255 * (val - minVal) / (maxVal - minVal));
        }
        else if (flag == "range")
        {
            float val = p.range;
            float minVal = min_range;
            float maxVal = max_range;
            cur_val = (int)(255 * (val - minVal) / (maxVal - minVal));
        }
        else if (flag == "x")
        {
            float val = p.x;
            float minVal = min_x;
            float maxVal = max_x;
            cur_val = (int)(255 * (val - minVal) / (maxVal - minVal));
        }
        else if (flag == "y")
        {
            float val = p.y;
            float minVal = min_y;
            float maxVal = max_y;
            cur_val = (int)(255 * (val - minVal) / (maxVal - minVal));
        }
        else if (flag == "z")
        {
            float val = p.z;
            float minVal = min_z;
            float maxVal = max_z;
            cur_val = (int)(255 * (val - minVal) / (maxVal - minVal));
        }
        else
        {
            std::cout << "===========" << std::endl;
            std::cout << "字段输入错误" << std::endl;
            return -1;
        }
        
        //上色方案１
        // int red, green, blue = 0;
        // red = cur_val;
        // green = 255 - cur_val;

        //上色方案２
        // int red = std::abs(255 - cur_val);
        // int green = std::abs(127 - cur_val);
        // int blue = std::abs(0 - cur_val);

        //上色方式３gray
        int red, green, blue = 0;
        if (cur_val <= 51)
			{
				blue = 255;
				green = cur_val * 5;
				red = 0;
			}
			else if (cur_val <= 102)
			{
				cur_val -= 51;
				blue = 255 - cur_val * 5;
				green = 255;
				red = 0;
			}
			else if (cur_val <= 153)
			{
				cur_val -= 102;
				blue = 0;
				green = 255;
				red = cur_val * 5;
			}
			else if (cur_val <= 204)
			{
				cur_val -= 153;
				blue = 0;
				green= 255 - static_cast<unsigned char>(cur_val * 128.0 / 51 + 0.5);
				red= 255;
			}
			else if (cur_val <= 255)
			{
				cur_val -= 204;
				blue = 0;
				green = 127 - static_cast<unsigned char>(cur_val * 127.0 / 51 + 0.5);
				red= 255;
			}

        cv::circle(image_lidar, p.p_pixel, 2, cv::Scalar( red,green, blue), -1);
    }

    //显示图片
    cv::Mat image_result;
    cv::addWeighted(image_lidar, 0.6, image, 0.4, 0, image_result);
    cv::imshow("image_result", image_result);
    cv::imwrite("result.png", image_result); 
    cv::waitKey(0);
	system("pause");

    return 0;
}
