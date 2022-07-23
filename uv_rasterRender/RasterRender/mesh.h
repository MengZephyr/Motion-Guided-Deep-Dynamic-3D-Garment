#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

struct R_Mesh
{
	std::vector<cv::Vec3f> verts;
	std::vector<cv::Vec3f> vnormals;
	std::vector<cv::Vec3i> faceInds;

	int numV() { return verts.size(); }
	int numF() { return faceInds.size();}

	void scaleMesh(float sx, float sy, float sz)
	{
		for (int v = 0; v < numV(); v++)
		{
			cv::Vec3f pos = verts[v];
			pos = cv::Vec3f(pos[0] * sx, pos[1] * sy, pos[2] * sz);
			verts[v] = pos;
		}
	}

	void normalizeVertNorm()
	{
		assert(vnormals.size() == verts.size());
		for (int vi = 0; vi < vnormals.size(); vi++)
			vnormals[vi] = normalize(vnormals[vi]);
	}

	std::vector<cv::Vec3f> calcVertNorm()
	{
		std::vector<cv::Vec3f> normArray(numV(), cv::Vec3f(0., 0., 0.));
		std::vector<double> sumWei(numV(), 0.);
		for (int f = 0; f < numF(); f++)
		{
			cv::Vec3i find = faceInds[f];
			cv::Vec3f d0 = verts[find[1]] - verts[find[0]];
			cv::Vec3f d1 = verts[find[2]] - verts[find[1]];
			cv::Vec3f fN = normalize(d0.cross(d1));
			float area = norm(fN) * 0.5;
			for (int d = 0; d < 3; d++)
			{
				normArray[find[d]] += area * fN;
				sumWei[find[d]] += area;
			}
		}
		for (int v = 0; v < numV(); v++)
			if (sumWei[v] < 1.e-8)
				continue;
			else
				normArray[v] = normalize(normArray[v] / sumWei[v]);

		return normArray;
	}

	std::vector<std::vector<int>> calcAdjVertInds()
	{
		std::vector<std::vector<int>> adjPnts(numV(), std::vector<int>());
		for (int f = 0; f < numF(); f++)
		{
			cv::Vec3i FF = faceInds[f];
			for (int d = 0; d < 3; d++)
			{
				int vID = FF[d];
				int a1 = FF[(d + 1) % 3];
				int a2 = FF[(d + 2) % 3];
				if (std::find(adjPnts[vID].begin(), adjPnts[vID].end(), a1) == adjPnts[vID].end())
					adjPnts[vID].push_back(a1);
				if (std::find(adjPnts[vID].begin(), adjPnts[vID].end(), a2) == adjPnts[vID].end())
					adjPnts[vID].push_back(a2);
			} // end for d
		} // end for f
		return adjPnts;
	}

	void calcVertLaplacian(std::vector<std::vector<int>>& adjVInds, 
		std::vector<cv::Vec3f>& neiVector, std::vector<cv::Vec3f>& lapVector)
	{
		neiVector = std::vector<cv::Vec3f>(numV(), cv::Vec3f(0., 0., 0.));
		lapVector = std::vector<cv::Vec3f>(numV(), cv::Vec3f(0., 0., 0.));
		
		for (int v = 0; v < numV(); v++)
		{
			std::vector<int>& neiAdj = adjVInds[v];
			cv::Vec3f llv(0., 0., 0.);
			for (int a = 0; a < neiAdj.size(); a++)
				llv = llv - verts[neiAdj[a]];
			
			llv = llv / float(neiAdj.size());
			neiVector[v] = llv;
			lapVector[v] = verts[v] + llv;

		} // end for v
	}
};


inline void calcTSpace(cv::Vec3f p, cv::Vec3f p1, cv::Vec3f p2,
	cv::Vec2f uv, cv::Vec2f uv1, cv::Vec2f uv2,
	cv::Vec3f& TVector)
{
	cv::Vec3f E1 = p1 - p;
	cv::Vec3f E2 = p2 - p;
	float Du1 = uv1[0] - uv[0];
	float Dv1 = uv1[1] - uv[1];
	float Du2 = uv2[0] - uv[0];
	float Dv2 = uv2[1] - uv[1];

	TVector = cv::Vec3f(0., 0., 0.);
	//BVector = cv::Vec3f(0., 0., 0.);
	float f = 1. / (Du1 * Dv2 - Du2 * Dv1);
	TVector[0] = f * (Dv2 * E1[0] - Dv1 * E2[0]);
	TVector[1] = f * (Dv2 * E1[1] - Dv1 * E2[1]);
	TVector[2] = f * (Dv2 * E1[2] - Dv1 * E2[2]);
	/*BVector[0] = f * (-Du2 * E1[0] + Du1 * E2[0]);
	BVector[1] = f * (-Du2 * E1[1] + Du1 * E2[1]);
	BVector[2] = f * (-Du2 * E1[2] + Du1 * E2[2]);*/
	TVector = normalize(TVector);
	//BVector = normalize(BVector);
}

struct ScanCameraModel
{
	ScanCameraModel(const std::vector<cv::Vec3f> & matArray, int ImgH, int ImgW, bool ifFS=false)
	{
		cv::Matx33f& R = this->RotMat;
		cv::Vec3f& T = this->TransVec;
		for (int pi = 0; pi < 3; pi++)
		{
			R(pi, 0) = matArray[pi][0];
			R(pi, 1) = matArray[pi][1];
			R(pi, 2) = matArray[pi][2];
			T[pi] = matArray[3][pi]/1000.;
		}
		std::cout << T << std::endl;
		std::cout << R << std::endl;

		T = -R.t() * T;
		R = R.t();
		std::cout << T << std::endl;
		std::cout << R << std::endl;

		this->rx = ImgW;
		this->ry = ImgH;
		this->cx = matArray[4][0];
		this->cy = matArray[4][1];
		this->fs = 0.00465000000;
		this->fx = ifFS ? matArray[5][0]/fs : matArray[5][0];
		this->fy = ifFS ? matArray[5][1]/fs : matArray[5][1];
		std::cout << fx << " " << fy << std::endl;
	}

	cv::Matx33f RotMat;
	cv::Vec3f   TransVec;
	float fx, fy, cx, cy;
	int rx, ry;
	float fs;

	cv::Vec3f proj(const cv::Vec3f & p, float scale = 1.)
	{
		cv::Vec3f pos = p;
		pos = RotMat * pos + TransVec;
		pos = cv::Vec3f(-fx * pos[0] / pos[2] + cx, fy * pos[1] / pos[2] + cy, pos[2]);
		pos[0] *= scale;
		pos[1] *= scale;
		return pos;
	}

	std::vector<cv::Vec3f> projVertArray(const std::vector<cv::Vec3f> & verts, float scale=1.)
	{
		std::vector<cv::Vec3f> projVertArray(verts.size());
		for (int i = 0; i < verts.size(); i++)
			projVertArray[i] = proj(verts[i], scale);

		return projVertArray;
	}
};

struct CameraModel
{
	CameraModel(const std::vector<cv::Vec4f>& matArray, int ImgH, int ImgW)
	{
		cv::Matx33f& R = this->RotMat;
		cv::Vec3f& T = this->TransVec;
		for (int pi = 0; pi < 3; pi++)
		{
			R(pi, 0) = matArray[pi][0];
			R(pi, 1) = matArray[pi][1];
			R(pi, 2) = matArray[pi][2];
			T[pi] = matArray[pi][3];
		}
		this->rx = ImgW;
		this->ry = ImgH;
		this->cx = 0.5 * (this->rx - 1.);
		this->cy = 0.5 * (this->ry - 1.);
		this->fx = -matArray[4][0] * 0.5 * (this->rx - 1.);
		this->fy = matArray[5][1] * 0.5 * (this->ry - 1.);
		//printf("%f, %f\n", fx, fy);
	}

	cv::Matx33f RotMat;
	cv::Vec3f   TransVec;
	float fx, fy, cx, cy;
	int rx, ry;

	cv::Vec3f proj(const cv::Vec3f& p, bool ifLeft=true)
	{
		cv::Vec3f pos = ifLeft ? cv::Vec3f(p[0], -p[2], p[1]) : cv::Vec3f(p[0], p[1], p[2]);
		pos = RotMat * pos + TransVec;
		pos = cv::Vec3f(fx * pos[0] / pos[2] + cx, fy * pos[1] / pos[2] + cy, pos[2]);
		return pos;
	}

	cv::Vec3f orthProj(const cv::Vec3f& p, bool ifLeft = false)
	{
		float fs = 10. * 512. / 36.;
		cv::Vec3f pos = ifLeft ? cv::Vec3f(p[0], -p[2], p[1]) : cv::Vec3f(p[0], p[1], p[2]);
		pos = RotMat * pos + TransVec;
		pos = cv::Vec3f(-1.* fx * pos[0] + cx, ry - 1.* fy* pos[1] - cy, pos[2]);
		return pos;
	}

	cv::Vec3f projNorm(const cv::Vec3f& n, bool ifLeft = true)
	{
		cv::Vec3f nn = ifLeft ? cv::Vec3f(n[0], -n[2], n[1]) : cv::Vec3f(n[0], n[1], n[2]);
		nn = RotMat * n;
		nn = normalize(n);
		return nn;
	}

	std::vector<cv::Vec3f> projVertArray(const std::vector<cv::Vec3f>& verts, bool ifOri = false)
	{
		std::vector<cv::Vec3f> projVertArray(verts.size());
		for (int i = 0; i < verts.size(); i++)
			projVertArray[i] = ifOri? orthProj(verts[i]) : proj(verts[i]);
		
		return projVertArray;
	}
};