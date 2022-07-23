#pragma once

#include "mesh.h"
#include <embree3/rtcore.h>
#include <embree3/rtcore_ray.h>
#include <vector>
#include <opencv2/opencv.hpp>

struct R_Mesh;

struct EMB_Vertex { float x, y, z; };
struct EMB_Triangle { int v0, v1, v2; };

inline void getPixelInterInfor(const cv::Vec2f& imgPos, int imgH, int imgW,
	cv::Vec2i& p0, cv::Vec2i& p1, cv::Vec2i& p2, cv::Vec2i& p3, cv::Vec4f& ef)
{
	float myPixelSize = 1.;
	float x0 = floor(imgPos[0] / myPixelSize) * myPixelSize;
	float x1 = x0 + myPixelSize;
	float y0 = floor(imgPos[1] / myPixelSize) * myPixelSize;
	float y1 = y0 + myPixelSize;

	float aa = (imgPos[0] - x0) / myPixelSize;
	float bb = (imgPos[1] - y0) / myPixelSize;

	cv::Vec2f pp0 = cv::Vec2f(x0, y0); cv::Vec2f pp1 = cv::Vec2f(x1, y0);
	cv::Vec2f pp2 = cv::Vec2f(x0, y1); cv::Vec2f pp3 = cv::Vec2f(x1, y1);

	p0 = cv::Vec2i((int)floor(pp0[0] + 0.5), (int)floor(pp0[1] + 0.5));
	p0[0] = MIN((imgW - 1), MAX(p0[0], 0));
	p0[1] = MIN((imgH - 1), MAX(p0[1], 0));
	float a0 = (1. - aa) * (1. - bb);

	p1 = cv::Vec2i((int)floor(pp1[0] + 0.5), (int)floor(pp1[1] + 0.5));
	p1[0] = MIN((imgW - 1), MAX(p1[0], 0));
	p1[1] = MIN((imgH - 1), MAX(p1[1], 0));
	float a1 = aa * (1. - bb);

	p2 = cv::Vec2i((int)floor(pp2[0] + 0.5), (int)floor(pp2[1] + 0.5));
	p2[0] = MIN((imgW - 1), MAX(p2[0], 0));
	p2[1] = MIN((imgH - 1), MAX(p2[1], 0));
	float a2 = (1. - aa) * bb;

	p3 = cv::Vec2i((int)floor(pp3[0] + 0.5), (int)floor(pp3[1] + 0.5));
	p3[0] = MIN((imgW - 1), MAX(p3[0], 0));
	p3[1] = MIN((imgH - 1), MAX(p3[1], 0));
	float a3 = aa * bb;

	ef = cv::Vec4f(a0, a1, a2, a3) / (a0 + a1 + a2 + a3);
}

inline cv::Vec3f  getInfoFromMat_3f(const cv::Vec2f& imgPos, const cv::Mat& colorMat)
{
	int imgH = colorMat.rows;
	int imgW = colorMat.cols;

	if (imgPos[0] < 0 || imgPos[0] > imgW - 1 || imgPos[1] < 0 || imgPos[1] > imgH - 1)
		return cv::Vec3f(-1., -1., -1.);

	float myPixelSize = 1.;
	float x0 = floor(imgPos[0] / myPixelSize) * myPixelSize;
	float x1 = x0 + myPixelSize;
	float y0 = floor(imgPos[1] / myPixelSize) * myPixelSize;
	float y1 = y0 + myPixelSize;

	cv::Vec2f p0 = cv::Vec2f(x0, y0); cv::Vec2f p1 = cv::Vec2f(x1, y0);
	cv::Vec2f p2 = cv::Vec2f(x0, y1); cv::Vec2f p3 = cv::Vec2f(x1, y1);

	float aa = (imgPos[0] - x0) / myPixelSize;
	float bb = (imgPos[1] - y0) / myPixelSize;

	cv::Point pp0 = cv::Point((int)floor(p0[0] + 0.5), (int)floor(p0[1] + 0.5));
	pp0.x = MIN((imgW - 1), MAX(pp0.x, 0));
	pp0.y = MIN((imgH - 1), MAX(pp0.y, 0));
	cv::Vec3f c0 = colorMat.at<cv::Vec3f>(pp0);
	float a0 = (1. - aa) * (1. - bb);

	cv::Point pp1 = cv::Point((int)floor(p1[0] + 0.5), (int)floor(p1[1] + 0.5));
	pp1.x = MIN((imgW - 1), MAX(pp1.x, 0));
	pp1.y = MIN((imgH - 1), MAX(pp1.y, 0));
	cv::Vec3f c1 = colorMat.at<cv::Vec3f>(pp1);
	float a1 = aa * (1. - bb);

	cv::Point pp2 = cv::Point((int)floor(p2[0] + 0.5), (int)floor(p2[1] + 0.5));
	pp2.x = MIN((imgW - 1), MAX(pp2.x, 0));
	pp2.y = MIN((imgH - 1), MAX(pp2.y, 0));
	cv::Vec3f c2 = colorMat.at<cv::Vec3f>(pp2);
	float a2 = (1. - aa) * bb;

	cv::Point pp3 = cv::Point((int)floor(p3[0] + 0.5), (int)floor(p3[1] + 0.5));
	pp3.x = MIN((imgW - 1), MAX(pp3.x, 0));
	pp3.y = MIN((imgH - 1), MAX(pp3.y, 0));
	cv::Vec3f c3 = colorMat.at<cv::Vec3f>(pp3);
	float a3 = aa * bb;

	cv::Vec3f cc = (a0 * c0 + a1 * c1 + a2 * c2 + a3 * c3) / (a0 + a1 + a2 + a3);
	return cc;
}


class RayIntersection
{
private:
	RTCDevice device;
	RTCScene scene;
	R_Mesh* pShapeMesh;
	bool ifSceneComm;

	EMB_Vertex* vertices;
	EMB_Triangle* triangles;

public:
	RayIntersection();
	~RayIntersection();

	unsigned int addObj(R_Mesh* pMesh);

	unsigned int addObj(std::vector<cv::Vec3f> verts, std::vector<cv::Vec3i> fInds);

	RTCHit rayIntersection(cv::Vec3f& ori, cv::Vec3f& dir);

	cv::Vec3f interPos(RTCHit& h);

private:
	void setSceneCommit();
};